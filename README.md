# CMPE492 · Clinical diagnosis and activation analysis

The project runs Qwen3.5-4B on clinical cases, compares diagnostic predictions with saved reference judgments, and analyzes model activations using linear probes, GRUs and a pretrained sparse autoencoder (SAE). Patient data, labels, model weights and run outputs stay local.

## Repository

| Location | Responsibility |
| --- | --- |
| `src/main.py`, `src/historical.py` | Experiment CLI and archived-runner launcher |
| `src/diagnosis/`, `prompts/diagnosis/` | Experiment runners, output schemas and frozen prompt templates |
| `src/dataset.py`, `src/schemas.py`, `src/retrieval.py` | Case adaptation, shared contracts and exact test-result lookup |
| `src/model.py`, `src/config.py`, `src/model_server/` | Model adapters, generation settings, local HTTP server, constrained decoding and activation capture |
| `src/evaluation/` | Export manual judgments and count Top-1/Top-3 agreement |
| `src/probing/` | Final-call linear probes, round-level linear/GRU comparison and probe-guided steering |
| `src/sae/` | Checkpoint validation, JumpReLU encoding, feature comparison and a decoder-feature steering hook |
| `src/utils.py`, `scripts/`, `tests/` | Run and logging helpers, Windows launchers and synthetic checks |
| `docs/` (local) | Project metadata, literature notes and SAE feature registry |

Git includes the Python source and tests, four diagnosis prompt templates, the PowerShell launcher, dependency and test configuration, and this README. Data, weights, credentials, research documents and experiment outputs stay on the local machine. A fresh checkout needs the local inputs listed for each experiment below.

## Run experiments

From PowerShell in the project directory, use `.\scripts\run.ps1`. It selects the project's virtual environment. With the virtual environment already active, `python -m src.main` accepts the same arguments. Run the launcher with `--help` to list experiments and append `--help` to any experiment to see its parameters.

| Experiment | Example arguments after `.\scripts\run.ps1` |
| --- | --- |
| Start the local Qwen3.5-4B server for all-tests, in a separate terminal | `server` |
| Baseline on the first four dataset records | `all-tests --experiment baseline --limit 4` |
| Findings-list variant on the same records | `all-tests --experiment evidence_findings --limit 4` |
| Four-case stepwise batch on a free GPU | `stepwise --limit 4 --seed 492` |
| Score the existing three judgment files | `agreement` |
| Final-call logistic probes on the reviewed cohort | `final-probe --c 0.001 --split-seed 42` |
| Linear and GRU round-level comparison | `round-probe --split-seed 42` |
| Prepare a probe-guided steering run | `probe-steering --phase prepare` |
| Replay that run, summarize predictions, then assess reviewed labels | `probe-steering --phase run --output RUN_DIR`, then use `summarize` and `assess` with the same directory |
| Inspect multiple features of one saved capture | `sae --capture CAPTURE_DIR --features 6324 49723 --stage input` |
| Compare groups by first diagnosis | `cancer-feature --cases REVIEWED_CASES.csv --feature 6324` |

Use `--all` for a full all-tests cohort. Omit `--limit` for the full stepwise cohort. All-tests applies the limit in dataset order before leakage screening. Stepwise applies it after screening and shuffling. For a custom dataset, both accept `--dataset PATH` and record its path in the protocol. For interrupted inference use `all-tests --resume RUN_DIR` or `stepwise --resume RUN_DIR`.

Place launcher options before the experiment name. For example, `.\scripts\run.ps1 --repeat 3 final-probe --c 0.01` creates three independent output directories. Identical seeds intentionally reproduce the same split. To explore split sensitivity, vary `--split-seed` explicitly. `.\scripts\run.ps1 --dry-run stepwise --limit 4` prints the command. `agreement` prints scores to the terminal. Probe steering prints a run directory at preparation time for its later phases.

### Historical experiments

`historical` reads runner sources and prompts from the sibling desktop archive. Each invocation creates an isolated workspace under `results/historical/EXPERIMENT/` with its launch arguments and local outputs. Help uses a temporary directory. Point to a relocated archive with `historical --archive PATH EXPERIMENT ...`. Dataset and model path overrides are interpreted from the project root.

| Experiment | Example arguments after `.\scripts\run.ps1 historical` |
| --- | --- |
| General diagnosis agent, one case | `agent --real --patient-id ID --model-url http://127.0.0.1:8000` |
| Single diagnosis vs. test/diagnosis/verifier team | `architecture --real --architecture single_diagnosis --patient-id ID` and repeat with `test_diagnosis_verifier` |
| All-evidence vs. matched diagnosis-history replay | `history --real --run --without-capture` |
| 9B all-tests diagnosis, verification and revision | `verifier --limit 4` |
| Early MedGemma and Qwen3 runs | Start `.\scripts\run.ps1 server --model medgemma` or `server --model qwen3`, then use the general agent command above |
| Gemini diagnosis or Gemini team | `agent --real --diagnosis-provider gemini --patient-id ID` or `gemini-team --patient-id ID --model MODEL_NAME` |
| Frozen synthetic SAE discovery pairs | `sae-discovery --run` |
| Matched synthetic disease-prompt steering | `sae-steering --run --disease49723 --feature 49723 --raw-strength 5` |
| 9B Qwen-Scope pooled analysis | `qwenscope --baseline BASELINE_DIR --checkpoint CHECKPOINT --feature 54347` |
| 9B Qwen-Scope token analysis | `qwenscope-tokens --pooled-run POOLED_DIR --baseline BASELINE_DIR --checkpoint CHECKPOINT --feature 54347` |

Historical HTTP experiments need the matching local server running separately. For the greedy Qwen3.5 endpoint protocol with compact activations, start `python -m src.model_server.main --model qwen3.5` and supply `--capture-activations` to the general agent. The frozen ten-case history replay explicitly runs without activation capture. The Gemini commands send case inputs to the configured Google API and require an authorized dataset and API credentials. SAE steering reads the original discovery and reference artifacts in the archive. The 9B Qwen-Scope analysis needs its own baseline captures and SAE checkpoint. The 4B group comparison needs the reviewed CSV described below.

## Diagnosis

Run commands from the project root using the virtual environment in `.venv`. Dependencies are listed in `requirements.txt`. The real-data runners expect `data/DiagBench_MIMICIV.json` and local weights under `models/`. Importing a module or requesting CLI help does not start inference.

```powershell
python -m src.main --help
python -m src.main all-tests --help
python -m src.main stepwise --help
```

**All-tests.** In a separate terminal, start the Qwen server with `python -m src.model_server.main --model qwen3.5 --structured-sampling`. The all-tests runner sends requests through `src/model_server/client.py`. The server keeps one model at `127.0.0.1:8000`. Both variants receive the presentation and every recorded nonmissing result. The baseline returns ranked diagnoses with reasoning. The findings variant returns supporting and contradicting findings for each diagnosis. Python validates each structured response. Presentation leakage flags exclude cases before inference, and reference diagnoses are saved separately for evaluation.

```powershell
python -m src.main all-tests --all --experiment baseline
python -m src.main all-tests --all --experiment evidence_findings
python -m src.main all-tests --patient-id PATIENT_ID --experiment baseline
python -m src.main all-tests --resume BATCH_DIRECTORY
```

New batches freeze the prompt, protocol and patient order in a fresh directory. Resume uses the saved protocol, skips verified completions and preserves previous attempts. The attempt limit is three per patient. Activation capture stores BF16 outputs of zero-based block 15 for all unpadded input tokens and the selected output diagnosis spans, including their preceding tokens.

**Stepwise.** Run `python -m src.main stepwise` on a free GPU. This runner calls its shared Qwen instance directly and requires FLA and per-patient XGrammar. Seed 492 fixes the order of eligible cases. Groups contain up to four patients. On each round the model sees the presentation, the remaining test IDs and the results revealed so far. Python checks requested IDs against that patient's current catalog before revealing their exact recorded values. The frozen batch protocol allows three requests per round and four rounds. Model-declared final, round limit and generation/validation failure are recorded separately. Each valid round retains block-15 BF16 input and diagnosis-span activations. Use `--resume RUN_DIRECTORY` to continue a batch.

The separate general-purpose diagnosis agent uses one `DiagnosisOutput`, up to two tests per turn and a default limit of five attempts. Its current diagnosis must name one of its hypotheses and uses that hypothesis's reasoning. These engineering bounds carry no clinical stopping interpretation.

## Evaluation

`src.evaluation.agreement` reads saved judgments. It matches baseline and findings by patient ID before counting full and partial matches. Stepwise counts include all evaluated cases and separate subgroups by model-declared final action. The judgments are AI-assisted and lack independent clinical adjudication.

```powershell
$r = "results/reference_diagnosis_agreement"
python -m src.evaluation.agreement --baseline "$r/codex_manual_top3_all_679/patient_agreement.json" --findings "$r/codex_manual_evidence_top3_671/patient_agreement.json" --stepwise "$r/stepwise_batch4_manual_top3_675/top3_scored.json"
```

## Probes and steering

**Final call.** `src.probing.final_call` fits 66 logistic regressions at input and output endpoints across 32 blocks and final normalization. It uses reviewed final diagnoses, keeps patients in a single split and fits scaling on training data. Pass a fresh output directory, or an unfinished directory with the same experiment identity.

**Round level.** `src.probing.round_level` compares linear models, including a history-summary variant, with GRUs on per-round diagnosis agreement. It checks hashes of the saved Input22 activations and annotation packets before training. Model selection uses validation patients, with held-out patients reserved for evaluation.

```powershell
python -m src.probing.final_call --review results/linear_probes/all_layers_528_20261031_0800/clinical_review_528.json --output NEW_OUTPUT_DIRECTORY
python -m src.probing.round_level --source results/current_diagnosis_gru/expanded_regularized/8cf72a3c5b7244158f9f9942bc2bf0ad
python -m src.probing.steering --help
```

The steering runner replays saved prompts with score-gated directions and matched controls. The active round-level comparison keeps its annotation packets in `packets/` beside its source manifest under `results/current_diagnosis_gru/expanded_regularized/`. The loader resolves the former round-error packet paths there and verifies their original hashes. Keep `results/internal_medicine_528/` and local model-server activation directories at their recorded locations. Existing manifests contain machine-local paths.

## Sparse autoencoder

The Decode Research Matryoshka SAE uses 65,536 features on the 2,560-dimensional output of Qwen3.5-4B block 15. Its pinned weights live under `models/sae/`. The loader checks the model, layer, tensor shapes and weight hash. The feature registry is in `docs/sae-features.md`.

```powershell
python -m src.sae.analysis --capture CAPTURE_DIRECTORY --features 6324 --stage input
python -m src.sae.feature_6324 --cases REVIEWED_CASES.csv
```

The first command summarizes one saved capture. The second averages feature 6324 over every input token, including zeros, then averages the patient means within each group. The CSV requires `patient_id,capture,cancer_first,explicit_terms,indirect_clues`. Capture paths can be absolute or relative to the CSV. Group labels describe the first predicted diagnosis and oncology clues in the input. The command saves case means, group counts, a plot and input/checkpoint provenance to a fresh local directory. The original patient-level grouping file for the historical comparison has not been located. Provide reviewed labels to run a new comparison.

`src.sae.steering.FeatureSteering` applies a decoder-feature direction during prefill, decode or both. Its caller owns the generation, matched baseline and capture.

## Local records

`data/`, `models/`, `results/`, model-server activation directories and `.env` are Git-ignored. Saved prompts, per-patient activations and annotations may contain patient information. Keep them local. Use synthetic inputs for development and run the test suite only when requested.

Retired runners, prompts, pilot results and restoration manifests are in the desktop sibling directory `../CMPE492-archive/`. Historical runs retain the commands and paths recorded when they were created.
