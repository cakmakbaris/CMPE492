"""Summarize saved judgments on matched all-tests cases and stepwise subgroups."""
import argparse
import json
from pathlib import Path

from pydantic import BaseModel


class AllTestsReview(BaseModel):
    patient_id: str
    top1_category: str
    best_top3_category: str


def all_tests(path: Path) -> dict[str, tuple[bool, bool]]:
    categories = {"tam_uyumlu", "kismen_uyumlu", "uyumsuz", "tamamen_uyumsuz"}
    success = {"tam_uyumlu", "kismen_uyumlu"}
    outcomes: dict[str, tuple[bool, bool]] = {}
    for record in json.loads(path.read_text(encoding="utf-8")):
        row = AllTestsReview.model_validate(record)
        if row.patient_id in outcomes or {row.top1_category, row.best_top3_category} - categories:
            raise ValueError(f"Duplicate patient or unknown agreement category in {path}")
        outcomes[row.patient_id] = row.top1_category in success, row.best_top3_category in success
    return outcomes


def counts(rows: list[tuple[bool, bool]]) -> dict[str, int]:
    return {"patients": len(rows), "top1": sum(first for first, _ in rows),
            "top3": sum(best for _, best in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    reviews = Path("results/reference_diagnosis_agreement")
    parser.add_argument("--baseline", type=Path, default=reviews / "codex_manual_top3_all_679/patient_agreement.json")
    parser.add_argument("--findings", type=Path, default=reviews / "codex_manual_evidence_top3_671/patient_agreement.json")
    parser.add_argument("--stepwise", type=Path, default=reviews / "stepwise_batch4_manual_top3_675/top3_scored.json")
    args = parser.parse_args()
    baseline, findings = all_tests(args.baseline), all_tests(args.findings)
    shared = sorted(baseline.keys() & findings.keys())
    if not shared:
        raise ValueError("The all-tests reviews have no shared patients")
    groups: dict[bool, list[tuple[bool, bool]]] = {True: [], False: []}
    seen: set[str] = set()
    for row in json.loads(args.stepwise.read_text(encoding="utf-8"))["patients"]:
        labels = row["labels"]
        if (row["patient"] in seen or type(row["gave_final"]) is not bool or
                not labels or set(labels) - {"F", "P", "R", "U"}):
            raise ValueError("Invalid stepwise review")
        seen.add(row["patient"])
        groups[row["gave_final"]].append((labels[0] in {"F", "P"}, bool({"F", "P"}.intersection(labels[:3]))))
    print(json.dumps({"baseline_shared": counts([baseline[k] for k in shared]),
                      "findings_shared": counts([findings[k] for k in shared]),
                      "stepwise_all": counts(groups[True] + groups[False]),
                      "stepwise_final": counts(groups[True]), "stepwise_other": counts(groups[False])}, indent=2))


if __name__ == "__main__":
    main()
