"""Saved-model demonstration for a reserved validation participant."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import digest, joined_manifest, write_json
from .exploratory_baselines import load_audited_sequences
from .independent_datasets import independent_loader
from .independent_train import PROTOCOL, build_model, predictions
from .train import choose_device


def predict(run, participant_id, output):
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f'Preserving existing demonstration: {destination}')
    torch.set_num_threads(2)
    state = torch.load(Path(run)/'best.pt',map_location='cpu',weights_only=False)
    config = state['config']
    manifest = joined_manifest(config)
    row = manifest[manifest.participant_id == str(participant_id)].reset_index(drop=True)
    if len(row) != 1 or row.iloc[0].split != 'validation':
        raise ValueError('This exploratory demo requires a validation participant; training/test participants are excluded')
    sequences, source_provenance = load_audited_sequences(config,manifest)
    provenance = {'protocol':PROTOCOL,'manifest_hash':digest(manifest.to_dict('records')),'features':source_provenance}
    if state['protocol'] != PROTOCOL or state['provenance'] != provenance:
        raise ValueError('Checkpoint data/protocol mismatch')
    device = choose_device(config['device'])
    model = build_model(sequences,config,state['gender']).to(device)
    model.load_state_dict(state['model'])
    ids, probs = predictions(model,independent_loader(row,sequences,state['scalers'],config),device)
    report = {'participant_id':ids[0],'split':'validation','model':state['model_name'],'seed':config['seed'],'predicted_class':int(probs.argmax(1)[0]),'probabilities':probs[0].astype(float).tolist(),'questionnaire_class':int(row.iloc[0].severity_class),'questionnaire_phq8_score':int(row.iloc[0].phq8_score),'protocol':PROTOCOL,'interpretation':'Exploratory research prediction; softmax values are not calibrated clinical certainty'}
    write_json(destination,report)
    return {'saved':str(destination),'split':'validation','probability_sum':float(np.sum(probs[0]))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--participant',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.run,args.participant,args.output),indent=2))
