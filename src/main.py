"""Run diagnosis, evaluation and activation experiments from the project root."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "server": ("src.model_server.main", "Serve one local Qwen model"),
    "all-tests": ("src.diagnosis.all_tests", "Baseline or findings-list diagnosis"),
    "stepwise": ("src.diagnosis.stepwise", "Stepwise diagnosis with per-round capture"),
    "agreement": ("src.evaluation.agreement", "Score saved baseline, findings and stepwise judgments"),
    "final-probe": ("src.probing.final_call", "Fit 66 final-call logistic probes"),
    "round-probe": ("src.probing.round_level", "Compare round-level linear and GRU probes"),
    "probe-steering": ("src.probing.steering", "Prepare, replay or assess probe-guided steering"),
    "sae": ("src.sae.analysis", "Measure selected SAE features on one capture"),
    "cancer-feature": ("src.sae.feature_6324", "Compare feature 6324 across reviewed patient groups"),
    "historical": ("src.historical", "Re-run archived architectures and SAE experiments"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repeat", type=int, default=1, help="Number of independent fresh runs")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without starting it")
    commands = parser.add_subparsers(dest="experiment", required=True)
    for name, (_, description) in COMMANDS.items():
        commands.add_parser(name, add_help=False, help=description, allow_abbrev=False)
    args, options = parser.parse_known_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    flags = {value.split("=", 1)[0] for value in options}
    if args.repeat > 1 and (args.experiment in {"server", "probe-steering"} or
                            flags & {"--resume", "--output", "--batch-dir", "--results"}):
        parser.error("--repeat requires independent runs with automatically allocated destinations")
    if args.experiment == "server" and "--model" not in flags:
        options = ["--model", "qwen3.5", "--structured-sampling", *options]
    command = [sys.executable, "-m", COMMANDS[args.experiment][0], *options]
    for index in range(args.repeat):
        print(f"[{index + 1}/{args.repeat}] {subprocess.list2cmdline(command)}", flush=True)
        if not args.dry_run:
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
