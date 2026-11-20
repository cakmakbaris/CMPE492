"""Compare case-level feature 6324 means using explicitly annotated groups."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, TypeAdapter

from src.sae.analysis import measure
from src.sae.checkpoint import CHECKPOINT, MODULE, WEIGHT_SHA256, load_weights
from src.utils import create_request_directory, digest, save


class Case(BaseModel):
    patient_id: str
    capture: Path
    cancer_first: bool
    explicit_terms: bool
    indirect_clues: bool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="Reviewed CSV, one input capture per patient")
    parser.add_argument("--feature", type=int, default=6324, help="Zero-based feature ID of the pinned SAE")
    args = parser.parse_args()
    if not 0 <= args.feature < 65536:
        parser.error("--feature must be between 0 and 65535")
    with args.cases.open(encoding="utf-8-sig", newline="") as handle:
        cases = TypeAdapter(list[Case]).validate_python(list(csv.DictReader(handle)))
    captures = [(args.cases.parent / case.capture).resolve() for case in cases]
    if (not cases or len({c.patient_id for c in cases}) != len(cases)
            or len(set(captures)) != len(cases)):
        raise ValueError("Expected one unique patient and input capture per row")
    weights = load_weights(CHECKPOINT)
    conditions = ("All cases", "Explicit terms excluded", "Indirect clues only", "No identified clues")
    groups: dict[str, dict[bool, list[float]]] = {name: {True: [], False: []} for name in conditions}
    rows = []
    for case, capture in zip(cases, captures, strict=True):
        result = measure(capture, [args.feature], "input", weights)
        value = result["mean_including_zeros"][0]
        rows.append({"patient_id": case.patient_id, "mean": value, "capture_sha256": result["capture_sha256"]})
        groups["All cases"][case.cancer_first].append(value)
        if not case.explicit_terms:
            groups["Explicit terms excluded"][case.cancer_first].append(value)
            condition = "Indirect clues only" if case.indirect_clues else "No identified clues"
            groups[condition][case.cancer_first].append(value)

    summary = {name: {"cancer_first" if cancer else "other": {
        "patients": len(values), "mean": mean(values) if values else None}
        for cancer, values in group.items()} for name, group in groups.items()}
    output = create_request_directory(Path(__file__).resolve().parents[2] / "results/sae_analysis")
    save(output / "comparison.json", {"feature": args.feature, "module": MODULE, "stage": "input",
         "checkpoint_sha256": WEIGHT_SHA256, "cases_path": str(args.cases.resolve()),
         "cases_sha256": digest(args.cases), "groups": summary, "case_means": rows})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for key, offset, label in (("cancer_first", -0.2, "Cancer-related first diagnosis"),
                               ("other", 0.2, "Other first diagnosis")):
        values = [summary[name][key] for name in conditions[:3]]
        bars = ax.barh([i + offset for i in range(3)],
                       [v["mean"] if v["patients"] else float("nan") for v in values], 0.36, label=label)
        ax.bar_label(bars, labels=[f'n={v["patients"]}' if v["patients"] else "" for v in values])
    ax.set_yticks(range(3), conditions[:3])
    ax.invert_yaxis()
    ax.set_xlabel(f"Mean input activation of feature {args.feature}")
    ax.margins(x=0.2)
    ax.legend()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
