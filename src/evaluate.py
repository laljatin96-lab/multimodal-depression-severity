import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .common import VERSION, write_json


def metrics(y, probs):
    y, pred = np.asarray(y, int), np.asarray(probs).argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
    present = support > 0
    pred_counts = np.bincount(pred, minlength=5)
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall[present].mean()), "participants": len(y), "macro_f1_definition": "All five declared classes; undefined class F1 contributes zero", "per_class": [{"class": i, "precision": float(precision[i]) if pred_counts[i] else None, "recall": float(recall[i]) if support[i] else None, "f1": float(f1[i]) if support[i] + pred_counts[i] else None, "support": int(support[i])} for i in range(5)], "confusion_matrix": confusion_matrix(y, pred, labels=range(5)).tolist(), "unsupported_classes": np.flatnonzero(~present).tolist()}


def write_report(frame, probabilities, output, model_name, seed, split):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if len(frame) != len(probabilities) or not len(frame):
        raise ValueError("Prediction/participant mismatch or empty evaluation")
    result = metrics(frame.severity_class, probabilities)
    result["group_metrics"] = {str(gender or "UNKNOWN"): metrics(group.severity_class, probabilities[frame.index.get_indexer(group.index)]) for gender, group in frame.groupby("gender", dropna=False)}
    result.update({"model": model_name, "seed": seed, "split": split, "version": VERSION})
    write_json(out / "metrics.json", result)
    predictions = frame[["participant_id", "gender", "severity_class"]].reset_index(drop=True).copy()
    predictions["predicted_label"] = probabilities.argmax(1)
    for i in range(5):
        predictions[f"p{i}"] = probabilities[:, i]
    predictions["split"], predictions["seed"], predictions["model_version"] = split, seed, VERSION + ":" + model_name
    predictions.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(result["per_class"]).to_csv(out / "per_class.csv", index=False)
    pd.DataFrame(result["confusion_matrix"]).to_csv(out / "confusion_matrix.csv", index=False)
    for i, (group, report) in enumerate(result["group_metrics"].items()):
        pd.DataFrame(report["confusion_matrix"]).to_csv(out / f"group_{i}_confusion_matrix.csv", index=False)
    return result


def paired_bootstrap(first, second, repetitions=2000, seed=42):
    a = pd.read_csv(first, dtype={"participant_id": str})
    b = pd.read_csv(second, dtype={"participant_id": str})
    if a.participant_id.duplicated().any() or b.participant_id.duplicated().any():
        raise ValueError("Bootstrap requires one row per participant")
    if set(a.participant_id) != set(b.participant_id):
        raise ValueError("Paired bootstrap requires identical participant sets")
    a, b = a.sort_values("participant_id"), b.sort_values("participant_id")
    if not np.array_equal(a.severity_class, b.severity_class) or not np.array_equal(a.split, b.split):
        raise ValueError("Mismatched labels or evaluation splits")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        ix = rng.integers(0, len(a), len(a))
        ma = metrics(a.severity_class.to_numpy()[ix], a[[f"p{i}" for i in range(5)]].to_numpy()[ix])
        mb = metrics(b.severity_class.to_numpy()[ix], b[[f"p{i}" for i in range(5)]].to_numpy()[ix])
        differences.append([mb[k] - ma[k] for k in ["accuracy", "macro_f1", "balanced_accuracy"]])
    return {k: {"delta_second_minus_first_ci95": np.percentile(np.array(differences)[:, i], [2.5, 97.5]).tolist()} for i, k in enumerate(["accuracy", "macro_f1", "balanced_accuracy"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    args = p.parse_args()
    from .train import evaluate_saved
    print(evaluate_saved(args.run, args.split))


if __name__ == "__main__":
    main()
