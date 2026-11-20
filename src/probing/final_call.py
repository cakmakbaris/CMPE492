"""Compare 66 fixed linear probes on a shared reviewed final-call cohort."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "results/linear_probes/matplotlib_cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score

from src.utils import create_request_directory, digest, read

MODULES = tuple(f"model.layers.{i}" for i in range(32)) + ("model.norm",)
STAGES = ("input", "output")
SETTINGS = {"C": 0.001, "class_weight": None, "threshold": 0.5, "seed": 42,
            "validation_fraction": 0.4, "bootstrap_samples": 500}


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    predicted = (probability >= 0.5).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    return {"accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "auroc": float(roc_auc_score(y, probability)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "correct_recall": float(matrix[1, 1] / matrix[1].sum()),
            "incorrect_recall": float(matrix[0, 0] / matrix[0].sum()),
            "confusion_matrix_true_rows_0_1": matrix.tolist()}


def write_new_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def checkpoint(output: Path, stage: str, completed: list[str]) -> None:
    path = output / "checkpoint.json"
    state = read(path) if path.exists() else {}
    state.update(stage=stage, completed_models=completed,
                 updated_at=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_vectors(review: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, np.ndarray], np.ndarray]:
    rows: list[dict[str, object]] = []
    values: dict[str, list[np.ndarray]] = {stage: [] for stage in STAGES}
    seen: set[str] = set()
    for patient in sorted(review["patients"], key=lambda row: row["patient_id"]):
        if patient["label"] not in {"correct", "incorrect"}:
            continue
        patient_id = patient["patient_id"]
        subject = patient_id.split("-")[0]
        if subject in seen:
            raise ValueError(f"Repeated subject requires grouped split: {subject}")
        seen.add(subject)
        run = Path(patient["run_dir"])
        result = read(run / "run.json")
        if result["termination_reason"] != "final_diagnosis_accepted" or result["final_diagnosis"] != patient["prediction"]:
            raise ValueError(f"Review/run disagreement: {patient_id}")
        calls = [json.loads(line) for line in (run / "activations.jsonl").read_text(encoding="utf-8").splitlines()]
        final = max(calls, key=lambda call: call["call"])
        if final["call"] != result["rounds"]:
            raise ValueError(f"Final-call disagreement: {patient_id}")
        folder = Path(final["activations_path"])
        metadata = read(folder / "metadata.json")
        if metadata["capture_version"] != 3 or metadata["model"] != "qwen3.5-4b" or metadata["request_id"] != final["request_id"]:
            raise ValueError(f"Unexpected capture: {patient_id}")
        tokens = metadata["generated_token_ids"]
        if len(tokens) != final["generated_tokens"] or metadata["output_position"] != metadata["input_position"] + len(tokens):
            raise ValueError(f"Token-position disagreement: {patient_id}")
        if metadata["output_last_token_id"] != tokens[-1]:
            raise ValueError(f"Last-token disagreement: {patient_id}")
        states = torch.load(folder / "activations.pt", map_location="cpu", weights_only=True)
        for stage in STAGES:
            expected = set(MODULES) | ({"model.embed_tokens"} if stage == "input" else set())
            if set(states[stage]) != expected or set(metadata["shapes"][stage]) != expected:
                raise ValueError(f"Module mismatch: {patient_id}/{stage}")
            vector_list: list[np.ndarray] = []
            for module in MODULES:
                tensor = states[stage][module]
                if tuple(tensor.shape) != (1, 2560) or metadata["shapes"][stage][module] != [1, 2560] or not torch.isfinite(tensor).all():
                    raise ValueError(f"Invalid tensor: {patient_id}/{stage}/{module}")
                vector_list.append(tensor.float().numpy().reshape(-1))
            values[stage].append(np.stack(vector_list))
        rows.append({"patient_id": patient_id, "subject_id": subject,
                     "label": int(patient["label"] == "correct"), "run": str(run),
                     "request_id": final["request_id"], "call": final["call"],
                     "capture": str(folder), "input_last_token_id": metadata["input_last_token_id"],
                     "output_last_token_id": metadata["output_last_token_id"],
                     "output_last_is_eos": metadata["output_last_is_eos"],
                     "input_position": metadata["input_position"], "output_position": metadata["output_position"],
                     "conversation_sha256": metadata["conversation_sha256"],
                     "model_config_sha256": metadata["model_config_sha256"],
                     "chat_template_sha256": metadata["chat_template_sha256"],
                     "system_prompt_sha256": metadata["system_prompt_sha256"],
                     "uncertainty": patient.get("uncertainty", [])})
        if len(rows) % 40 == 0:
            print(f"Loaded {len(rows)} final-call captures", flush=True)
    if len(rows) < 20:
        raise ValueError("Too few eligible patients")
    return rows, {stage: np.stack(value) for stage, value in values.items()}, np.array([row["label"] for row in rows])


def auc_interval(labels: np.ndarray, probability: np.ndarray) -> list[float]:
    rng = np.random.default_rng(42)
    samples: list[float] = []
    for _ in range(500):
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) == 2:
            samples.append(float(roc_auc_score(labels[indices], probability[indices])))
    return np.quantile(samples, [0.025, 0.975]).tolist()


def make_probe(c: float = 0.001) -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("probe", LogisticRegression(C=c, class_weight=None, solver="liblinear", max_iter=5000, random_state=42))])


def plot_results(output: Path, results: list[dict[str, object]], c: float = 0.001) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for stage, color in [("input", "#7864d9"), ("output", "#159e91")]:
        selected = [row for row in results if row["stage"] == stage]
        for ax, metric in zip(axes, ["accuracy", "auroc"]):
            ax.plot(range(1, 34), [row["validation"][metric] for row in selected], marker=".", label=stage, color=color)
            ax.set_title(f"Validation {metric}")
            ax.set_xlabel("Decoder block (1-based); N = final normalization")
            ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32, 33], ["1", "4", "8", "12", "16", "20", "24", "28", "32", "N"])
            ax.set_ylim(0.35, 1.0)
            ax.grid(alpha=0.25)
    axes[0].axhline(results[0]["majority_baseline"], color="gray", linestyle="--", label="Majority baseline")
    axes[1].axhline(0.5, color="gray", linestyle="--", label="Chance ranking")
    for ax in axes:
        ax.legend()
    fig.suptitle(f"Qwen3.5-4B: shared subjects/split; C={c:g}; exploratory validation")
    fig.tight_layout()
    fig.savefig(output / "layer_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for stage in STAGES:
        selected = [row for row in results if row["stage"] == stage]
        for split, style in [("train", "--"), ("validation", "-")]:
            ax.plot(range(1, 34), [row[split]["accuracy"] for row in selected], style, label=f"{stage} {split}")
    ax.set(xlabel="1..32 decoder blocks, 33 final normalization", ylabel="Accuracy", title="Train / validation gaps across layers (not epoch learning curves)")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output / "train_validation_gaps.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=Path("results/linear_probes/all_layers_528_20261031_0800/clinical_review_528.json"))
    parser.add_argument("--output", type=Path, help="Destination or unfinished run; defaults to a fresh directory")
    parser.add_argument("--c", type=float, default=0.001, help="Inverse logistic regularization strength")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    if not np.isfinite(args.c) or args.c <= 0:
        parser.error("--c must be finite and positive")
    settings = {**SETTINGS, "C": args.c, "seed": args.split_seed}
    output = args.output.resolve() if args.output else create_request_directory(Path("results/linear_probes"))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError("Experiment already finalized; preserve prior results")
    review = read(args.review)
    rows, features, labels = load_vectors(review)
    train, validation = train_test_split(np.arange(len(rows)), test_size=0.4, stratify=labels, random_state=args.split_seed)
    identity = {"settings": settings, "patients": rows, "train_indices": train.tolist(), "validation_indices": validation.tolist(), "modules": list(MODULES), "review_sha256": digest(args.review)}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if read(manifest_path)["identity"] != identity:
            raise ValueError("Resume identity/split/config changed")
    else:
        write_new_json(manifest_path, {"identity": identity, "review": str(args.review.resolve()),
                       "cohort_counts": dict(Counter(int(v) for v in labels)),
                       "target": "Preliminary primary-reference agreement, final accepted calls only",
                       "validation": "Exploratory reused subjects; no independent blind test; no validation tuning",
                       "capture": "v3; last serialized input token; actual final output token including EOS; all blocks + final norm"})
    print(f"Cohort {len(rows)}; train {len(train)}; validation {len(validation)}", flush=True)
    completed: list[str] = []
    results: list[dict[str, object]] = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baseline = float(max((labels[validation] == 0).mean(), (labels[validation] == 1).mean()))
    with threadpool_limits(limits=1):
        for stage in STAGES:
            for layer, module in enumerate(MODULES):
                name = f"{stage}_{'norm' if layer == 32 else f'block_{layer + 1:02d}'}"
                result_path = output / f"{name}_results.json"
                model_path = output / f"{name}_classifier.joblib"
                if result_path.exists():
                    result = read(result_path)
                    if not model_path.exists() or result["stage"] != stage or result["module"] != module:
                        raise ValueError(f"Partial or mismatched saved model: {name}")
                else:
                    if model_path.exists():
                        raise ValueError(f"Orphan model must be preserved/reviewed: {name}")
                    x = features[stage][:, layer, :]
                    model = make_probe(args.c)
                    scores = cross_val_score(model, x[train], labels[train], cv=cv, scoring="balanced_accuracy", n_jobs=1)
                    model.fit(x[train], labels[train])
                    probability = model.predict_proba(x[validation])[:, 1]
                    result = {"name": name, "stage": stage, "module": module, "block_one_based": layer + 1 if layer < 32 else None,
                              "settings": settings, "train": metrics(labels[train], model.predict_proba(x[train])[:, 1]),
                              "validation": metrics(labels[validation], probability),
                              "validation_auroc_bootstrap95": auc_interval(labels[validation], probability),
                              "train_cv_balanced_accuracy_mean": float(scores.mean()), "train_cv_balanced_accuracy_std": float(scores.std()),
                              "majority_baseline": baseline, "train_patients": len(train), "validation_patients": len(validation),
                              "validation_predictions": [{"patient_id": rows[i]["patient_id"], "true_label": int(labels[i]), "score": float(p), "prediction": int(p >= 0.5)} for i, p in zip(validation, probability)]}
                    joblib.dump(model, model_path)
                    write_new_json(result_path, result)
                completed.append(name); results.append(result)
                checkpoint(output, "training", completed)
                print(f"{len(completed)}/66 {name}: accuracy={result['validation']['accuracy']:.3f}, AUROC={result['validation']['auroc']:.3f}", flush=True)
    assert len(results) == 66
    for result in results:
        model = joblib.load(output / f"{result['name']}_classifier.joblib")
        assert model.named_steps["probe"].coef_.shape == (1, 2560)
        assert model.named_steps["scale"].n_samples_seen_ == len(train)
        assert model.named_steps["probe"].C == args.c
    plot_results(output, results, args.c)
    fields = ["stage", "module", "train_accuracy", "validation_accuracy", "validation_balanced_accuracy", "validation_auroc", "train_cv_balanced_accuracy", "majority_baseline"]
    with (output / "layer_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for r in results:
            writer.writerow({"stage": r["stage"], "module": r["module"], "train_accuracy": r["train"]["accuracy"], "validation_accuracy": r["validation"]["accuracy"], "validation_balanced_accuracy": r["validation"]["balanced_accuracy"], "validation_auroc": r["validation"]["auroc"], "train_cv_balanced_accuracy": r["train_cv_balanced_accuracy_mean"], "majority_baseline": baseline})
    summary = {"clinical_counts": review["counts"], "patients": len(rows), "train_patients": len(train), "validation_patients": len(validation),
               "train_counts": dict(Counter(int(v) for v in labels[train])), "validation_counts": dict(Counter(int(v) for v in labels[validation])),
               "models_completed": 66, "settings": settings, "majority_baseline": baseline,
               "best_exploratory": {stage: {metric: max([r for r in results if r["stage"] == stage], key=lambda r: r["validation"][metric])["name"] for metric in ["accuracy", "auroc"]} for stage in STAGES},
               "results": results}
    write_new_json(output / "summary.json", summary)
    checkpoint(output, "models_complete_documentation_pending", completed)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
