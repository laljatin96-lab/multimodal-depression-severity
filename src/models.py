import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class AVClassifier(nn.Module):
    def __init__(self, audio_dim, visual_dim, hidden_size=32, dropout=0.3, gender_dim=0):
        super().__init__()
        self.audio = nn.LSTM(audio_dim, hidden_size, batch_first=True)
        self.visual = nn.LSTM(visual_dim, hidden_size, batch_first=True)
        self.gender_dim = gender_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden_size + gender_dim, 5))

    def forward(self, audio, visual, lengths, gender=None):
        reps = []
        for sequence, model in [(audio, self.audio), (visual, self.visual)]:
            packed = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h, _) = model(packed)
            reps.append(h[-1])
        if self.gender_dim:
            if gender is None or gender.shape[1] != self.gender_dim:
                raise ValueError("Gender encoding missing or wrong size")
            reps.append(gender)
        return self.head(torch.cat(reps, dim=1))


def aggregate_probabilities(probabilities, owners, participants):
    result = probabilities.new_zeros(participants, 5)
    result.index_add_(0, owners, probabilities)
    counts = torch.bincount(owners, minlength=participants).to(probabilities.dtype)
    if (counts == 0).any():
        raise ValueError("Participant without windows")
    return result / counts[:, None]


class IndependentAVClassifier(AVClassifier):
    """Encode each modality's windows, pool by participant, then fuse.

    Audio and visual window counts/order may differ. Only participant identity
    joins the modalities; there is no cross-modal timestamp correspondence.
    """

    def forward(self, audio, visual, gender):
        participants = len(gender)
        reps = []
        for batch, encoder in [(audio, self.audio), (visual, self.visual)]:
            packed = pack_padded_sequence(batch['values'], batch['lengths'].cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = encoder(packed)
            owners = batch['owners']
            counts = torch.bincount(owners, minlength=participants).to(hidden.dtype)
            if len(counts) != participants or (counts == 0).any():
                raise ValueError('Each participant requires windows for both modalities')
            pooled = hidden.new_zeros(participants, hidden.shape[-1]).index_add(0, owners, hidden[-1])
            reps.append(pooled / counts[:, None])
        if self.gender_dim:
            if gender.shape[1] != self.gender_dim:
                raise ValueError('Gender encoding missing or wrong size')
            reps.append(gender)
        return self.head(torch.cat(reps, dim=1))
