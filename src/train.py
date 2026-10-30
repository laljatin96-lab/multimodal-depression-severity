"""Bounded participant-weighted training; test evaluation is a separate command."""
import argparse
from functools import partial
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from .common import VERSION, digest, encode_gender, joined_manifest, load_config, write_json
from .datasets import ParticipantDataset, collate_participants, fit_scalers, load_sequences, summarize
from .evaluate import write_report, metrics
from .models import AVClassifier, aggregate_probabilities


def choose_device(request):
    if request != "auto":
        return torch.device(request)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(c, sequences, gender):
    seq = next(iter(sequences.values()))
    return AVClassifier(seq["audio"].shape[1] + 1, seq["visual"].shape[1] + 1, c["hidden_size"], c["dropout"], len(c["gender_categories"]) + 1 if gender else 0)


def model_probs(model, batch, device):
    owners = batch["owner"].to(device)
    gender = batch["gender"].to(device)[owners]
    logits = model(batch["audio"].to(device), batch["visual"].to(device), batch["lengths"], gender)
    return logits, aggregate_probabilities(logits.softmax(-1), owners, len(batch["ids"]))


def predict_loader(model, loader, device):
    model.eval()
    ids, probs = [], []
    with torch.no_grad():
        for batch in loader:
            _, p = model_probs(model, batch, device)
            probs.append(p.cpu().numpy())
            ids.extend(batch["ids"])
    return ids, np.concatenate(probs)


def provenance(manifest, c):
    cache = Path(c["cache"]) / "participants"
    return {"manifest_hash": digest(manifest.to_dict("records")), "cache_fingerprints": {pid: json.loads((cache / f"{pid}.json").read_text())["fingerprint"] for pid in manifest.participant_id}, "version": VERSION}


def record_result(run, report, config):
    path = Path(config.get("experiment_log", "EXPERIMENT_LOG.csv"))
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"run": str(run), **{k: report[k] for k in ["model", "seed", "split", "accuracy", "macro_f1", "balanced_accuracy", "participants"]}, "group_metrics": json.dumps(report["group_metrics"])}
    df = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    if len(df):
        df = df[~((df.run == str(run)) & (df.split == report["split"]))]
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def loader_for(frame, sequences, scalers, c, shuffle=False, generator=None):
    return DataLoader(ParticipantDataset(frame, sequences, scalers, c), batch_size=c["batch_size"], shuffle=shuffle, num_workers=0, collate_fn=partial(collate_participants, categories=c["gender_categories"]), generator=generator)


def validate_gender(c, manifest, gender):
    if gender and (not c["gender_categories"] or not manifest.loc[manifest.split == "train", "gender"].isin(c["gender_categories"]).any()):
        raise ValueError("Gender experiments require recorded metadata and explicitly configured categories; never infer gender from voices")


def train(c, run, gender=False, balanced=False, resume=False):
    out = Path(run)
    if out.exists() and not resume:
        raise FileExistsError(f"Preserving existing run {out}; choose a new run directory or --resume")
    seed_all(c["seed"])
    torch.set_num_threads(2)
    manifest = joined_manifest(c)
    validate_gender(c, manifest, gender)
    for split in ["train", "validation", "test"]:
        if not (manifest.split == split).any():
            raise ValueError(f"Missing {split} participants")
    sequences = load_sequences(manifest, c)
    train_frame = manifest[manifest.split == "train"].reset_index(drop=True)
    val_frame = manifest[manifest.split == "validation"].reset_index(drop=True)
    scalers = fit_scalers(sequences, train_frame.participant_id.tolist())
    prov = provenance(manifest, c)
    device = choose_device(c["device"])
    model = make_model(c, sequences, gender).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    counts = np.bincount(train_frame.severity_class, minlength=5)
    weights = np.ones(5, np.float32)
    if balanced:
        present = counts > 0
        weights[present] = len(train_frame) / (present.sum() * counts[present])
        weights[~present] = 0
    weight_tensor = torch.tensor(weights, device=device)
    generator = torch.Generator().manual_seed(c["seed"])
    start, best, stale, history = 0, -1.0, 0, []
    model_name = "AV" + ("-gender" if gender else "") + ("-balanced" if balanced else "")
    if resume:
        state = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        if state["config"] != c or state["provenance"] != prov or state["gender"] != gender or state["balanced"] != balanced:
            raise ValueError("Resume configuration/data mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for opt_state in optimizer.state.values():
            for key, value in opt_state.items():
                if torch.is_tensor(value):
                    opt_state[key] = value.to(device)
        start, best, stale, history = state["epoch"] + 1, state["best_metric"], state["stale"], state["history"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        generator.set_state(state["rng"]["loader"])
        if device.type == "cuda" and state["rng"].get("cuda"):
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        if device.type == "mps" and state["rng"].get("mps") is not None:
            torch.mps.set_rng_state(state["rng"]["mps"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", c)
    train_loader = loader_for(train_frame, sequences, scalers, c, True, generator)
    val_loader = loader_for(val_frame, sequences, scalers, c)
    write_json(out / "contributions.json", {row["participant_id"]: {"windows": len(windows), "participant_loss_weight": 1 / len(train_frame)} for row, _, windows in train_loader.dataset})
    for epoch in range(start, c["epochs"]):
        if stale >= c["patience"]:
            break
        tick = time.monotonic()
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, _ = model_probs(model, batch, device)
            owners = batch["owner"].to(device)
            labels = batch["labels"].to(device)
            # Average within participant before batch averaging, so long interviews
            # never gain extra weight simply from producing more windows.
            losses = torch.nn.functional.cross_entropy(logits, labels[owners], reduction="none")
            person_loss = losses.new_zeros(len(labels)).index_add(0, owners, losses)
            person_loss /= torch.bincount(owners, minlength=len(labels))
            loss = (person_loss * weight_tensor[labels]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip"])
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        ids, probs = predict_loader(model, val_loader, device)
        if ids != val_frame.participant_id.tolist():
            raise ValueError("Validation feature/label order mismatch")
        report = metrics(val_frame.severity_class, probs)
        val_loss = float(-np.log(np.maximum(probs[np.arange(len(probs)), val_frame.severity_class.to_numpy()], 1e-12)).mean())
        improved = report["macro_f1"] > best
        best = max(best, report["macro_f1"])
        stale = 0 if improved else stale + 1
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_frame), "validation_loss": val_loss, "validation_macro_f1": report["macro_f1"], "validation_accuracy": report["accuracy"], "seconds": time.monotonic() - tick})
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": None, "epoch": epoch, "best_metric": best, "stale": stale, "history": history, "config": c, "gender": gender, "balanced": balanced, "model_name": model_name, "scalers": scalers, "provenance": prov, "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "loader": generator.get_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "mps": torch.mps.get_rng_state() if device.type == "mps" else None}}
        torch.save(state, out / "latest.pt")
        if improved:
            torch.save(state, out / "best.pt")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"{model_name} seed={c['seed']} epoch={epoch} val_macro_f1={report['macro_f1']:.4f}", flush=True)
    plot_history(history, out)
    return evaluate_saved(out, "validation")


def plot_history(history, out):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for key in ["train_loss", "validation_loss"]:
        axes[0].plot(frame.epoch, frame[key], label=key)
    axes[1].plot(frame.epoch, frame.validation_macro_f1, label="participant validation macro-F1")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=150)
    plt.close(fig)


def expanded_probs(model, x):
    p = model.predict_proba(x)
    out = np.zeros((len(x), 5))
    out[:, model.classes_.astype(int)] = p
    return out


def baselines(c, run):
    out = Path(run)
    if out.exists():
        raise FileExistsError(out)
    manifest = joined_manifest(c)
    sequences = load_sequences(manifest, c)
    train_frame = manifest[manifest.split == "train"]
    val_frame = manifest[manifest.split == "validation"]
    if not len(train_frame) or not len(val_frame):
        raise ValueError("Empty train or validation split")
    x = np.stack([summarize(sequences[p]) for p in train_frame.participant_id])
    vx = np.stack([summarize(sequences[p]) for p in val_frame.participant_id])
    y = train_frame.severity_class.to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Logistic baseline requires at least two training classes")
    prov = provenance(manifest, c)
    for name, estimator in [("majority", DummyClassifier(strategy="most_frequent")), ("logistic", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=c["seed"])) )]:
        estimator.fit(x, y)
        report = write_report(val_frame.reset_index(drop=True), expanded_probs(estimator, vx), out / name / "validation", name, c["seed"], "validation")
        report["data_kind"] = c.get("data_kind", "provided_dataset")
        write_json(out / name / "validation" / "metrics.json", report)
        record_result(out / name, report, c)
        dest = out / name
        with (dest / "baseline.pkl").open("wb") as f:
            pickle.dump({"estimator": estimator, "config": c, "provenance": prov, "model_name": name}, f)
    return "Saved majority and regularized logistic validation baselines"


def evaluate_saved(run, split):
    out = Path(run)
    if (out / "baseline.pkl").exists():
        with (out / "baseline.pkl").open("rb") as f:
            state = pickle.load(f)
    else:
        # Load only trusted local checkpoints: pickle-based formats execute code.
        state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    c = state["config"]
    manifest = joined_manifest(c)
    if provenance(manifest, c) != state["provenance"]:
        raise ValueError("Data/split/preprocessing changed since training")
    sequences = load_sequences(manifest, c)
    frame = manifest[manifest.split == split].reset_index(drop=True)
    if not len(frame):
        raise ValueError(f"Empty evaluation split {split}")
    if "estimator" in state:
        probabilities = expanded_probs(state["estimator"], np.stack([summarize(sequences[p]) for p in frame.participant_id]))
    else:
        device = choose_device(c["device"])
        model = make_model(c, sequences, state["gender"]).to(device)
        model.load_state_dict(state["model"])
        ids, probabilities = predict_loader(model, loader_for(frame, sequences, state["scalers"], c), device)
        if ids != frame.participant_id.tolist():
            raise ValueError("Prediction/label mismatch")
    report = write_report(frame, probabilities, out / split, state["model_name"], c["seed"], split)
    report["data_kind"] = c.get("data_kind", "provided_dataset")
    write_json(out / split / "metrics.json", report)
    record_result(out, report, c)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--run", required=True)
    p.add_argument("--gender", action="store_true")
    p.add_argument("--balanced", action="store_true")
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    c = load_config(args.config)
    print(baselines(c, args.run) if args.baselines else train(c, args.run, args.gender, args.balanced, args.resume))


if __name__ == "__main__":
    main()
