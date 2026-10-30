"""Participant-ID guard for the unchanged, numeric 49.28% inference bundle."""
import argparse
import json
from pathlib import Path
import re


def validate_identity(transcript, audio_csv, visual_csv):
    suffixes = ['_Transcript.csv', '_OpenSMILE2.3.0_mfcc.csv', '_OpenFace2.1.0_Pose_gaze_AUs.csv']
    ids = []
    for path, suffix in zip([transcript, audio_csv, visual_csv], suffixes):
        match = re.fullmatch(r'(\d+)' + re.escape(suffix), Path(path).name)
        if match is None:
            raise ValueError('Expected participant-prefixed file name for each specified modality')
        ids.append(match.group(1))
    if len(set(ids)) != 1:
        raise ValueError('Transcript/audio/visual participant IDs differ')
    return ids[0]


def predict(output, transcript, audio_csv, visual_csv):
    from .best_observed_inference import predict as original, BUNDLE
    participant = validate_identity(transcript, audio_csv, visual_csv)
    return {**original(BUNDLE, output, transcript, audio_csv, visual_csv), 'participant_id': participant}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ['output', 'transcript', 'audio-csv', 'visual-csv']:
        p.add_argument('--'+flag, required=True)
    a = p.parse_args()
    print(json.dumps(predict(a.output, a.transcript, a.audio_csv, a.visual_csv), indent=2))


if __name__ == '__main__':
    main()
