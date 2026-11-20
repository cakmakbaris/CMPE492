# CMPE492

Clinical diagnosis experiments with Qwen3.5-4B. The project compares diagnoses produced from all recorded findings with diagnoses produced by requesting tests step by step. It measures Top-1 and Top-3 agreement with saved judgments and analyzes activations with linear probes, GRUs and a sparse autoencoder (SAE). The judgments are AI-assisted and lack independent clinical adjudication.

## Quick start

From the project root, activate `.venv` and install the dependencies.

```powershell
python -m pip install -r requirements.txt
python -m src.main --help
```

The PowerShell launcher `.\scripts\run.ps1` selects the project's virtual environment and accepts the same arguments. Real diagnosis runs need `data/DiagBench_MIMICIV.json` and local weights under `models/`. Requesting CLI help does not load a model.

```powershell
# All-tests: start the server in a separate terminal.
python -m src.main server
python -m src.main all-tests --experiment baseline --limit 4
python -m src.main stepwise --limit 4
```

Use `--experiment evidence_findings` for the findings variant and `--resume RUN_DIR` to resume either diagnosis batch. Stepwise runs directly on a free GPU. It retrieves only the exact recorded results requested, with up to three tests per round and four rounds. The separate general-purpose agent allows two tests per turn and defaults to five diagnosis attempts. The current batch protocols capture BF16 activations at block 15 for unpadded inputs and selected diagnosis output spans with preceding tokens.

## Analysis and local files

Run `python -m src.main COMMAND --help` for options. `agreement` scores saved judgments. `final-probe` and `round-probe` train on saved activations and reviewed labels. `probe-steering` replays saved prompts. `sae` and `cancer-feature` require local captures and the pinned SAE checkpoint. `historical` uses the sibling `../CMPE492-archive/` directory and accepts `--archive PATH`.

Diagnosis and analysis code live under `src/`, prompt templates under `prompts/diagnosis/`, and synthetic tests under `tests/`. Patient data, research documents, model weights, credentials and experiment outputs stay local.
