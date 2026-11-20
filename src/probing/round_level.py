"""Compare causal Input22 probes on a patient-level 60/20/20 split."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from src.probing.data import load_vectors, pack, packet_path, preprocess
from src.utils import create_request_directory, digest, read, save

FORMAT = Path(__file__).resolve().parents[2] / "results/current_diagnosis_gru/source_manifests/format_calls_manifest.json"


@dataclass(frozen=True)
class Configuration:
    hidden: int
    dropout: float
    weight_decay: float


class RegularizedGRU(nn.Module):
    def __init__(self, config: Configuration, input_size: int = 2560, layers: int = 1,
                 skip: bool = True) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, config.hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(config.hidden, 1)
        self.direct = nn.Linear(input_size, 1, bias=False) if skip else None
        self.input_dropout = nn.Dropout(config.dropout)
        self.state_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dropped = self.input_dropout(x)
        states, _ = self.gru(dropped)
        logits = self.head(self.state_dropout(states))
        if self.direct is not None:
            logits = logits + self.direct(dropped)
        return logits.squeeze(-1)


def loss_value(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    element = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return ((element * mask).sum(1) / mask.sum(1)).mean()


def fit(xs: list[np.ndarray], ys: list[np.ndarray], train: np.ndarray, val: np.ndarray | None,
        config: Configuration, seed: int, epochs: int = 100, *, layers: int = 1,
        skip: bool = True) -> tuple[RegularizedGRU, int, list[dict[str, float | int]]]:
    torch.manual_seed(seed)
    model = RegularizedGRU(config, xs[train[0]].shape[-1], layers, skip)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=config.weight_decay)
    x, target, mask = pack(xs, ys, train)
    validation = pack(xs, ys, val) if val is not None else None
    best, best_epoch, stale = float("inf"), 1, 0
    state = copy.deepcopy(model.state_dict())
    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_value(model(x), target, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row: dict[str, float | int] = {"epoch": epoch, "train_bce": float(loss.detach())}
        if validation is not None:
            model.eval()
            vx, vt, vm = validation
            with torch.no_grad():
                vl = float(loss_value(model(vx), vt, vm))
            row["validation_bce"] = vl
            if vl < best - 0.0001:
                best, best_epoch, stale = vl, epoch, 0
                state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= 12:
                curve.append(row)
                break
        curve.append(row)
    if validation is not None:
        model.load_state_dict(state)
    else:
        best_epoch = epochs
    model.eval()
    return model, best_epoch, curve


def predict(model: RegularizedGRU, xs: list[np.ndarray], indices: np.ndarray) -> list[np.ndarray]:
    with torch.no_grad():
        return [model(torch.from_numpy(xs[i])[None]).sigmoid().numpy().reshape(-1) for i in indices]


def metrics(y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray,
            weights: np.ndarray | None = None) -> dict[str, object]:
    pred = scores >= thresholds
    return {"precision": float(precision_score(y, pred, sample_weight=weights, zero_division=0)),
            "recall": float(recall_score(y, pred, sample_weight=weights, zero_division=0)),
            "f1": float(f1_score(y, pred, sample_weight=weights, zero_division=0)),
            "f1_correct": float(f1_score(y, pred, pos_label=0, sample_weight=weights, zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", sample_weight=weights, zero_division=0)),
            "accuracy": float(accuracy_score(y, pred, sample_weight=weights)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred, sample_weight=weights)),
            "auroc": float(roc_auc_score(y, scores, sample_weight=weights)) if len(np.unique(y)) == 2 else None,
            "confusion_true_rows_0_1": confusion_matrix(y, pred, labels=[0, 1], sample_weight=weights).tolist(),
            "n": len(y)}


@dataclass
class Candidate:
    name: str
    features: str
    model: LogisticRegression | RegularizedGRU
    settings: dict[str, object]


def flat_labels(labels: list[np.ndarray], indices: np.ndarray) -> np.ndarray:
    return np.concatenate([labels[i] for i in indices])


def history_features(vectors: np.ndarray) -> np.ndarray:
    previous = np.zeros_like(vectors)
    previous[1:] = vectors.cumsum(axis=0)[:-1] / np.arange(1, len(vectors))[:, None]
    return np.concatenate([vectors, previous], axis=1)


def scores(candidate: Candidate, xs: list[np.ndarray], indices: np.ndarray) -> np.ndarray:
    if isinstance(candidate.model, RegularizedGRU):
        return np.concatenate(predict(candidate.model, xs, indices))
    return candidate.model.predict_proba(np.concatenate([xs[i] for i in indices]))[:, 1]


def evaluate(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, object]:
    known = y >= 0
    output = metrics(y[known], score[known], np.full(known.sum(), threshold))
    return {**output, "threshold": threshold, "alarms": int((score[known] >= threshold).sum()),
            "average_precision": float(average_precision_score(y[known], score[known]))}


def select_thresholds(y: np.ndarray, score: np.ndarray) -> dict[str, dict[str, object] | None]:
    # Only validation predictions are passed here. The grid is fixed before looking at scores.
    rows = [evaluate(y, score, i / 100) for i in range(5, 100)]
    accuracy = max(rows, key=lambda m: (m["accuracy"], m["balanced_accuracy"], -abs(m["threshold"] - 0.5)))
    eligible = [m for m in rows if m["precision"] >= 0.70 and m["alarms"] >= 20]
    precision = max(eligible, key=lambda m: (m["recall"], m["precision"], m["accuracy"]), default=None)
    return {"fixed": rows[45], "accuracy": accuracy, "precision70": precision}


def audit_errors(source: Path, ids: list[str], y: np.ndarray, score: np.ndarray,
                 threshold: float, output: Path) -> None:
    format_labels = {row["request_id"]: row["label"] for row in read(FORMAT)["rows"]}
    patients = {p["patient_id"]: p for p in read(source / "manifest.json")["patients"]}
    calls = [(pid, row) for pid in ids for row in patients[pid]["calls"]]
    records = []
    for (pid, row), label, value in zip(calls, y, score, strict=True):
        if label < 0:
            continue
        alarm = value >= threshold
        case = ("TP" if label == 1 else "FP") if alarm else ("FN" if label == 1 else "TN")
        records.append({"patient_id": pid, "call": row["call"], "format_error": format_labels[row["request_id"]],
                        "case": case, "score": float(value), "packet": str(packet_path(source, Path(row["packet"]))),
                        "diagnosis": row["current_diagnosis"], "reference": row["reference"],
                        "annotation_reason": row.get("rationale")})
    counts = {}
    for column in ("call", "format_error"):
        counts[column] = {str(group): {case: sum(r[column] == group and r["case"] == case for r in records)
                                      for case in ("TP", "FP", "TN", "FN")}
                          for group in sorted({r[column] for r in records})}
    extremes = {}
    for case, reverse in (("FP", True), ("FN", False)):
        ordered = sorted((r for r in records if r["case"] == case), key=lambda r: r["score"], reverse=reverse)
        unique = {}
        for row in ordered:
            unique.setdefault(row["patient_id"], row)
        extremes[case] = list(unique.values())[:10]
    save(output, {"threshold": threshold, "counts": counts, "extremes": extremes})


def fit_features(xs: list[np.ndarray], train: np.ndarray, output: Path) -> dict[str, list[np.ndarray]]:
    """Fit preprocessing on training patients and transform every sequence."""
    scaler, standardized = preprocess(xs, train)
    if int(scaler.n_samples_seen_) != sum(len(xs[i]) for i in train):
        raise ValueError("Scaler fit coverage mismatch")
    joblib.dump(scaler, output / "scaler.joblib")
    features = {"full": standardized, "history": [history_features(x) for x in standardized]}
    for size in (32, 64):
        pca = PCA(n_components=size, svd_solver="randomized", random_state=42)
        pca.fit(np.concatenate([standardized[i] for i in train]))
        name = f"pca{size}"
        features[name] = [pca.transform(x).astype(np.float32) for x in standardized]
        joblib.dump(pca, output / f"{name}.joblib")
    return features


def fit_candidates(features: dict[str, list[np.ndarray]], ys: list[np.ndarray],
                   train: np.ndarray, val: np.ndarray, output: Path) -> tuple[dict[str, Candidate], dict[str, str]]:
    candidates = {}
    families = {}
    y_train = flat_labels(ys, train)
    known = y_train >= 0
    weights = np.concatenate([np.full(len(ys[i]), 1 / (ys[i] >= 0).sum()) for i in train])[known]
    weights /= weights.mean()
    for feature in ("full", "history", "pca32", "pca64"):
        x = np.concatenate([features[feature][i] for i in train])[known]
        for c in (0.0001, 0.001, 0.01):
            name = f"linear_{feature}_{c:g}"
            model = LogisticRegression(C=c, solver="liblinear", max_iter=1000, random_state=42)
            model.fit(x, y_train[known], sample_weight=weights)
            candidates[name] = Candidate(name, feature, model, {"C": c})
            families[name] = "linear_" + feature if feature in ("full", "history") else "linear_pca"
    gru_specs = [
        ("gru_d96", "full", 8, .30, .01, 1, True, False),
        ("gru_full_early", "full", 8, .30, .01, 1, True, True),
        ("gru_full_small", "full", 4, .30, .10, 1, True, True),
        ("gru_full_no_skip", "full", 4, .30, .10, 1, False, True),
        ("gru_pca32_h4", "pca32", 4, .10, .01, 1, False, True),
        ("gru_pca32_h8", "pca32", 8, .30, .01, 1, False, True),
        ("gru_pca64_h4", "pca64", 4, .10, .01, 1, False, True),
        ("gru_pca64_h8", "pca64", 8, .30, .10, 1, False, True),
        ("gru_pca64_skip", "pca64", 4, .10, .01, 1, True, True),
        ("gru_pca32_two_layers", "pca32", 4, .10, .01, 2, False, True),
    ]
    curves = {}
    for name, feature, hidden, dropout, decay, layers, skip, early in gru_specs:
        config = Configuration(hidden, dropout, decay)
        model, epoch, curve = fit(features[feature], ys, train, val if early else None, config, seed=42,
                                  epochs=200 if feature != "full" else (100 if early else 50), layers=layers, skip=skip)
        settings = {**asdict(config), "layers": layers, "skip": skip, "epochs": epoch,
                    "parameters": sum(p.numel() for p in model.parameters())}
        candidates[name] = Candidate(name, feature, model, settings)
        families[name] = ("gru_full" if feature == "full" else "gru_pca_two_layers" if layers == 2 else "gru_pca")
        curves[name] = curve
        print(f"{name} trained, epoch {epoch}", flush=True)
    save(output / "learning_curves.json", curves)
    return candidates, families


def run(source: Path, split_seed: int = 42) -> None:
    output = create_request_directory(source.parent / "checker_holdout60_20_20")
    print(f"Output: {output}", flush=True)
    patients, xs, ys = load_vectors(source)
    ids = [p["patient_id"] for p in patients]
    if len({p["subject_id"] for p in patients}) != len(patients):
        raise ValueError("Patient split requires one sequence per subject")
    strata = [f"{p['cohort']}:{int(y[y >= 0].mean() >= 0.5)}" for p, y in zip(patients, ys, strict=True)]
    development, test = train_test_split(np.arange(len(ids)), test_size=0.2, stratify=strata, random_state=split_seed)
    train, val = train_test_split(development, test_size=len(test), stratify=[strata[i] for i in development], random_state=split_seed + 1)
    train, val, test = np.sort(train), np.sort(val), np.sort(test)
    if len(set(train) | set(val) | set(test)) != len(ids) or sum(map(len, (train, val, test))) != len(ids):
        raise ValueError("Invalid split")
    splits = {"train": train, "validation": val, "test": test}
    save(output / "split.json", {name: [ids[i] for i in indices] for name, indices in splits.items()})
    save(output / "manifest.json", {"source": str(source.resolve()), "source_sha256": digest(source / "manifest.json"),
         "seeds": [split_seed, split_seed + 1], "split_rule": "stratified by source cohort and patient majority label",
         "rule": "Freeze per-family and overall validation winners before evaluating on test. No test-based refit."})
    features = fit_features(xs, train, output)
    candidates, families = fit_candidates(features, ys, train, val, output)

    y_val = flat_labels(ys, val)
    val_scores = {name: scores(c, features[c.features], val) for name, c in candidates.items()}
    validation = {name: {"features": c.features, "settings": c.settings, "family": families[name],
                         **select_thresholds(y_val, val_scores[name])} for name, c in candidates.items()}
    save(output / "validation_candidates.json", validation)
    rank = sorted(candidates, key=lambda name: (validation[name]["accuracy"]["accuracy"],
                  validation[name]["accuracy"]["balanced_accuracy"], validation[name]["accuracy"]["auroc"]), reverse=True)
    policies = {"d96_fixed": ("gru_d96", 0.5), "overall_accuracy": (rank[0], validation[rank[0]]["accuracy"]["threshold"])}
    for family in sorted(set(families.values())):
        winner = next(name for name in rank if families[name] == family)
        policies[family] = (winner, validation[winner]["accuracy"]["threshold"])
    eligible = [name for name in rank if validation[name]["precision70"] is not None]
    if eligible:
        winner = max(eligible, key=lambda name: (validation[name]["precision70"]["recall"], validation[name]["precision70"]["precision"]))
        policies["precision70"] = (winner, validation[winner]["precision70"]["threshold"])
    save(output / "frozen_selection.json", {"policies": policies, "accuracy_rank": rank,
         "precision_policy": "Validation precision >= 0.70 and at least 20 alarms. Maximize validation recall."})
    winner, threshold = policies["overall_accuracy"]
    audit_errors(source, [ids[i] for i in val], y_val, val_scores[winner], threshold, output / "validation_error_audit.json")
    audit_errors(source, [ids[i] for i in val], y_val, val_scores["gru_d96"], 0.5, output / "baseline_error_audit.json")
    print(f"Validation choices frozen: {policies}", flush=True)

    results = {}
    predictions = {}
    for name in dict.fromkeys(candidate for candidate, _ in policies.values()):
        candidate = candidates[name]
        folder = output / name
        folder.mkdir()
        if isinstance(candidate.model, RegularizedGRU):
            torch.save({"state_dict": candidate.model.state_dict(), "features": candidate.features,
                        "input_size": features[candidate.features][0].shape[-1], **candidate.settings}, folder / "model.pt")
        else:
            joblib.dump(candidate.model, folder / "model.joblib")
        predictions[name] = {part: scores(candidate, features[candidate.features], indices) for part, indices in splits.items()}
    for policy, (name, threshold) in policies.items():
        results[policy] = {"candidate": name, "threshold": threshold,
                           **{part: evaluate(flat_labels(ys, indices), predictions[name][part], threshold)
                              for part, indices in splits.items()}}
    counts = {part: {"patients": len(indices), "calls": len(flat_labels(ys, indices)),
                     "label0": int((flat_labels(ys, indices) == 0).sum()), "label1": int((flat_labels(ys, indices) == 1).sum())}
              for part, indices in splits.items()}
    save(output / "summary.json", {"counts": counts, "results": results})
    save(output / "test_predictions.json", {"patient_ids": [ids[i] for i in test],
         "label_sequences": [ys[i].tolist() for i in test],
         "scores": {name: values["test"].tolist() for name, values in predictions.items()}})
    print("Completed held-out evaluation", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/current_diagnosis_gru/expanded_regularized/8cf72a3c5b7244158f9f9942bc2bf0ad"))
    parser.add_argument("--split-seed", type=int, default=42, help="Patient split seed; model training seeds remain fixed")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        run(args.source, args.split_seed)


if __name__ == "__main__":
    main()
