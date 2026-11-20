"""Run a historical protocol in a fresh local directory using the desktop source archive."""
import argparse
import re
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from src.utils import create_request_directory, save

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "CMPE492-archive"
MODULES = {
    "gemini-team": "src.gemini_multiagent.main",
    "agent": "src.main_legacy",
    "architecture": "src.architecture_experiments.main",
    "history": "src.architecture_experiments.pilot",
    "verifier": "src.architecture_experiments.all_tests_verifier",
    "sae-discovery": "src.sae.run_sae_discovery",
    "sae-steering": "src.sae.run_sae_steering_pair",
    "qwenscope": "src.sae.qwenscope_cancer_comparison",
    "qwenscope-tokens": "src.sae.qwenscope_token_comparison",
}


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def stage(run: Path, archive: Path, experiment: str) -> None:
    compact = archive / "compact-layout-20261120"
    previous = archive / "source-refactor-20261120"
    for name in ("__init__", "config", "dataset", "retrieval"):
        copy(ROOT / f"src/{name}.py", run / f"src/{name}.py")
    for name, snapshot in (("schemas", compact), ("utils", compact), ("model", previous)):
        copy(snapshot / f"src/{name}.py", run / f"src/{name}.py")
    if experiment in {"architecture", "agent", "history", "verifier", "gemini-team"}:
        for folder in ("src/agents", "src/architecture_experiments"):
            for path in (compact / folder).glob("*.py"):
                copy(path, run / folder / path.name)
        for folder in ("architecture_experiments/prompts", "prompts/team"):
            shutil.copytree(compact / folder, run / folder)
        copy(ROOT / "prompts/diagnosis/agent.txt", run / "prompts/diagnosis.txt")
        if experiment == "gemini-team":
            for path in (archive / "src/gemini_multiagent").glob("*.py"):
                copy(path, run / "src/gemini_multiagent" / path.name)
            shutil.copytree(archive / "gemini_multiagent_1/prompts", run / "gemini_multiagent_1/prompts")
            copy(archive / "gemini_multiagent_1/schemas.json", run / "gemini_multiagent_1/schemas.json")
        elif experiment == "history":
            pilot = (archive / "src/architecture_experiments/pilot.py").read_text(encoding="utf-8")
            pilot = pilot.replace('parser.add_argument("--prepare",', 'parser.add_argument("--run", action="store_true", help="Prepare and execute a fresh batch")\n    parser.add_argument("--prepare",')
            pilot = pilot.replace("    if args.prepare:", "    if args.run:\n        execute(prepare(args.dataset))\n    elif args.prepare:")
            (run / "src/architecture_experiments/pilot.py").write_text(pilot, encoding="utf-8")
        elif experiment == "agent":
            copy(archive / "src/graph.py", run / "src/graph.py")
            # Synthetic demo fixtures retired with the old CLI. Real mode never imports them.
            script = (archive / "src/main_legacy.py").read_text(encoding="utf-8")
            script = script.replace("from src.demo import mock_case, mock_model, mock_team_model", "")
            script = script.replace("case = mock_case()", 'raise ValueError("Historical agent replay requires --real")')
            (run / "src/main_legacy.py").write_text(script, encoding="utf-8")
        elif experiment == "verifier":
            for name in ("__init__", "structured", "diagnosis_spans"):
                copy(ROOT / f"src/model_server/{name}.py", run / f"src/model_server/{name}.py")
        return

    if experiment.startswith("qwenscope"):
        for path in (compact / "src/sae").glob("qwenscope*.py"):
            copy(path, run / "src/sae" / path.name)
        pooled = run / "src/sae/qwenscope_cancer_comparison.py"
        code = pooled.read_text(encoding="utf-8")
        for constant, default in (("BASELINE", ROOT / "qwen9b_all_tests_baseline"), ("MODEL", ROOT / "models/qwen3.5-9b"),
                                  ("CHECKPOINT", ROOT / "models/sae/qwenscope-qwen35-9b-layer15-k100/layer15.sae.pt")):
            code = "\n".join(f"{constant} = Path({str(default)!r})" if line.startswith(f"{constant} = ") else line
                             for line in code.split("\n"))
        pooled.write_text(code, encoding="utf-8")
        token = run / "src/sae/qwenscope_token_comparison.py"
        code = token.read_text(encoding="utf-8")
        before, body = code.split("    args = parser.parse_args(argv)", 1)
        for key in ("baseline", "model", "checkpoint"):
            before += f'    parser.add_argument("--{key}", type=Path, default={key.upper()})\n'
            body = re.sub(rf"\b{key.upper()}\b", f"args.{key}", body)
        token.write_text(before + "    args = parser.parse_args(argv)" + body, encoding="utf-8")
        return

    fixtures = archive / "SAE_Steering"
    copy(fixtures / "prompt_pairs.json", run / "SAE_Steering/prompt_pairs.json")
    for path in (archive / "src/sae").glob("*.py"):
        copy(path, run / "src/sae" / path.name)
    with (run / "src/schemas.py").open("a", encoding="utf-8") as handle:
        handle.write("\nfrom src.sae.synthetic_schemas import SAESteeringPair, SAESteeringPromptSet\n")
    # Remap immutable inputs only. The archived algorithms still allocate new outputs under run.
    for name in ("run_sae_discovery", "run_sae_steering_pair"):
        path = run / f"src/sae/{name}.py"
        code = path.read_text(encoding="utf-8")
        for old, target in (
            ('ROOT / "artifacts"', ROOT / "models/sae"),
            ('ROOT.parent / "models/qwen3.5-4b"', ROOT / "models/qwen3.5-4b"),
            ('PROJECT / "models/qwen3.5-4b"', ROOT / "models/qwen3.5-4b"),
        ):
            code = code.replace(old, f"Path({str(target)!r})")
        for prefix in ("runs/discovery/", "runs/steering_pair/"):
            code = code.replace(f'ROOT / "{prefix}', f'Path({str(fixtures)!r}) / "{prefix}')
        path.write_text(code, encoding="utf-8")


def launch(run: Path, archive: Path, experiment: str, options: list[str]) -> None:
    stage(run, archive, experiment)
    module = MODULES[experiment]
    flags = {value.split("=", 1)[0] for value in options}
    defaults = {"--dataset": ROOT / "data/DiagBench_MIMICIV.json"} if experiment in {
        "agent", "architecture", "history", "verifier", "gemini-team"} else {}
    if experiment == "verifier":
        defaults.update({"--model": ROOT / "models/qwen3.5-9b", "--output": run / "outputs",
                         "--baseline-prompt": ROOT / "prompts/diagnosis/baseline.txt"})
    for flag, value in defaults.items():
        if flag not in flags:
            options.extend([flag, str(value)])
    save(run / "launch.json", {"module": module, "arguments": options, "archive": str(archive)})
    print(f"Historical workspace: {run}", flush=True)
    bootstrap = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module(sys.argv.pop(1),run_name='__main__')"
    if experiment == "gemini-team" or (experiment == "agent" and "gemini" in options):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    result = subprocess.run([sys.executable, "-c", bootstrap, str(run), module, *options], cwd=ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("experiment", choices=MODULES)
    parser.add_argument("options", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    archive = args.archive.resolve()
    help_only = any(arg in ("-h", "--help") for arg in args.options)
    if args.experiment == "agent" and not help_only and "--real" not in args.options:
        parser.error("agent requires --real and an explicit backend")
    if args.experiment == "history" and not help_only and ("--prepare" not in args.options or "--run" in args.options) and "--without-capture" not in args.options:
        parser.error("The frozen history replay has no activation capture. Pass --without-capture to select this protocol.")
    options = [arg for arg in args.options if arg != "--without-capture"]
    if help_only:
        with TemporaryDirectory() as temporary:
            launch(Path(temporary), archive, args.experiment, options)
    else:
        launch(create_request_directory(ROOT / "results/historical" / args.experiment), archive, args.experiment, options)


if __name__ == "__main__":
    main()
