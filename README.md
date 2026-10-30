# Project_V2 — approximately 50% observed CV procedure

The best observed participant-wise model-selection procedure achieved
**49.8978% (244/489 repeated predictions)**, which rounds to 50% at whole-percent
precision. It did **not** achieve an exact accuracy of at least 50%.
The packaged retained checkpoint's associated procedure achieved
**49.2843% (241/489)**. The 49.90% procedure selected this existing checkpoint
when fitted on all training participants; it did not produce a separately
verified 49.90% deployment checkpoint. No new training or accuracy improvement
is claimed by this release.

The evaluation has 163 original participants, five classes, five outer folds,
three inner folds and three partition seeds (42, 137, 2026). The 489 predictions
are repeated measurements of 163 people. All development results are exploratory
after extensive model selection; they are not independent test accuracy.
The paired gain over 49.28% is +0.6135 percentage points, with participant-level
95% interval [-0.8180, +2.0450] percentage points. Severe recall remains zero.

## Included

- Original Python source and configuration files, including the WavLM selection
  procedure and fixed regularized MiniLM/AV/language checkpoint recipe.
- Unchanged numeric retained model, preprocessing arrays and integrity metadata.
- Frozen MiniLM encoder/tokenizer assets, so inference needs no model download.
- Aggregate CV metrics, confusion matrices, class results and paired uncertainty.
- Portable inference entry point, exact feature schema and release verification.

Participant dataset records, transcripts, recordings, features, questionnaire
responses, labels, per-person predictions, original datasets and Python
environments are not included. Original source/tests contain example IDs. The
WavLM encoder is not included because the retained checkpoint
does not use it. Historical experiment scripts need their original licensed
datasets, frozen result caches and any additional encoders; copying this ZIP
alone cannot rerun those private-data experiments. Existing tests under `tests/`
are original project tests; many require those private artifacts.

## Setup

Python **3.12 on macOS/Linux** is the supported release environment. Historical
project imports use the Unix `fcntl` module. The tested Python version and direct
dependency versions are recorded in `environment-tested.json`; a new environment installation on a
different machine has not been tested.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python verify_release.py
```

Inference runs locally. Required source CSVs must belong to the same participant
and have the original supplied schemas. The audio input is an **MFCC CSV**, not
a WAV file. The visual input is an **OpenFace CSV**, not a video file.

```bash
python predict.py \
  --transcript /absolute/path/123_Transcript.csv \
  --audio-csv /absolute/path/123_OpenSMILE2.3.0_mfcc.csv \
  --visual-csv /absolute/path/123_OpenFace2.1.0_Pose_gaze_AUs.csv \
  --output /absolute/path/prediction.json
```

Use your actual participant ID in all three filenames. Output must be a new
filename in an existing directory. The entry point also works when launched
from another working directory. It does not accept questionnaire scores,
answers, severity labels or age as inference inputs.

Prepared-feature inference is available for features created by this exact
pipeline, with the contract declared in `FEATURE_SCHEMA.json`:

```bash
python predict.py --features /absolute/path/features.npz \
  --output /absolute/path/prediction.json
```

Arrays are semantic384 + AV128 + language16 in the documented fixed order.
Do not substitute arbitrary embeddings or fabricate missing modalities. The
class order is 0 None/minimal, 1 Mild, 2 Moderate, 3 Moderately severe, 4 Severe.
Outputs are uncalibrated research scores, not an established clinical diagnosis
or a guaranteed accuracy for a new participant.

## Model and evidence

The retained model uses training-fitted scaling and PCA32 on MiniLM384 + AV128,
then appends 16 standardized language features scaled by 0.5. The five-class
logistic head has C=0.001 and square-root class-frequency weights. All fitted
preprocessing is already stored in `checkpoints/best_observed_4928/weights.npz`.
No refit is needed for inference.

See `ACCURACY.json` and `evidence/` for exact aggregate results. The majority-class
baseline is 47.2393%. Neither this archive nor the approximate 50% name claims
that the 53–56% target was achieved. The original project and every previous
model/result are preserved.

See `THIRD_PARTY_NOTICES.md` for the bundled public encoder's source and license.
