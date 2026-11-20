"""Resumable all-patient baseline, immutable per-attempt logs and bounded retries."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from src.diagnosis.schemas import BaselineDiagnosisOutput, EvidenceDiagnosisOutput
from src.config import Settings
from src.dataset import load_case, load_cases
from src.model_server.client import HttpModel
from src.schemas import ChatMessage, PatientCase, is_missing
from src.utils import create_request_directory, extract_final_json, replace_json

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/DiagBench_MIMICIV.json"
DiagnosisOutputType = type[BaselineDiagnosisOutput] | type[EvidenceDiagnosisOutput]


def complete(folder: Path, output_type: DiagnosisOutputType = BaselineDiagnosisOutput) -> bool:
    try:
        output_type.model_validate_json((folder / "output.json").read_text(encoding="utf-8"))
        meta = json.loads((folder / "activations/metadata.json").read_text(encoding="utf-8"))
        return (folder / "activations/activations.pt").stat().st_size > 0 and meta['module'] == 'model.layers.15' and bool(meta['diagnosis_spans'])
    except (OSError, ValueError, KeyError):
        return False


def prepare(experiment: str, patient_id: str | None, dataset: Path = DATASET, limit: int | None = None) -> Path:
    cases = load_cases(dataset)[:limit] if patient_id is None else [load_case(dataset, patient_id)]
    patients = [case.patient_id for case in cases]
    if not patients or len(patients) != len(set(patients)):
        raise ValueError("Expected a nonempty cohort with distinct patient IDs")
    prompt = ROOT / "prompts/diagnosis" / f"{experiment}.txt"
    output_type = EvidenceDiagnosisOutput if experiment == "evidence_findings" else BaselineDiagnosisOutput
    batch = create_request_directory(ROOT / f"results/all_patients_all_tests_layer15_{experiment}")
    shutil.copyfile(prompt, batch / "prompt_template.txt")
    replace_json(batch / "protocol.json", {
        "experiment": experiment, "prompt_name": prompt.name, "output_schema": output_type.__name__,
        "patient_ids": patients, "dataset": str(dataset.resolve()), "limit": limit,
        "selection": "explicit patient" if patient_id else "dataset order",
        "module": "model.layers.15", "dtype": "bfloat16",
        "capture": "all input tokens and output diagnosis spans including preceding tokens",
        "max_attempts_per_patient": 3, "minimum_free_bytes": 6 * 1024**3,
        "max_new_tokens": Settings().max_new_tokens, "xgrammar": True, "thinking": False,
        "server_sampling": "temperature1 top_p0.95 top_k20 min_p0 presence1.5 repetition1 seed492",
    })
    print(f"Run {batch}", flush=True)
    return batch


def generate(case: PatientCase, model: HttpModel, folder: Path, template: str,
             output_type: DiagnosisOutputType = BaselineDiagnosisOutput, *,
             link_activations: bool = False) -> BaselineDiagnosisOutput | EvidenceDiagnosisOutput:
    model.set_output_log(folder / 'model_output.txt')
    tests = [{'test_name': k, 'result': v} for k, v in case.examinations.items() if not is_missing(v)]
    messages = [ChatMessage(role='user', content=template.format(patient_info=case.presentation,
                               test_results=json.dumps(tests, ensure_ascii=False, indent=2)))]
    schema = output_type.model_json_schema()
    replace_json(folder / 'messages.json', [m.model_dump() for m in messages])
    replace_json(folder / 'response_schema.json', schema)
    raw = model.generate_messages(messages, response_schema=schema, capture_diagnosis_tokens=True)
    (folder / 'raw_output.txt').write_text(raw, encoding='utf-8')
    output = output_type.model_validate_json(extract_final_json(raw))
    (folder / 'output.json').write_text(output.model_dump_json(indent=2), encoding='utf-8')
    record = json.loads((folder / 'activations.jsonl').read_text().splitlines()[-1])
    shutil.copytree(Path(record['activations_path']), folder / 'activations',
                    copy_function=os.link if link_activations else shutil.copy2)
    (folder / 'reference_for_evaluation.json').write_text(case.reference.model_dump_json(indent=2), encoding='utf-8')
    if not complete(folder, output_type):
        raise ValueError('Missing output or activation artifacts')
    return output


def execute(batch: Path) -> None:
    import msvcrt

    # OS lock is released even on process termination; a stale PID cannot cause duplicate work.
    with (batch / 'runner.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        replace_json(batch / 'runner.json', {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()})
        protocol = json.loads((batch / 'protocol.json').read_text(encoding='utf-8'))
        evidence = protocol.get('experiment') == 'evidence_findings'
        output_type = EvidenceDiagnosisOutput if evidence else BaselineDiagnosisOutput
        ids = protocol['patient_ids']
        dataset = Path(protocol.get('dataset', DATASET))
        records = [load_case(dataset, ids[0])] if len(ids) == 1 else load_cases(dataset)
        cases = {c.patient_id: c for c in records}
        template = (batch / 'prompt_template.txt').read_text(encoding='utf-8')
        model = HttpModel('http://127.0.0.1:8000', Settings())
        states: list[dict[str, str]] = []
        for index, pid in enumerate(protocol['patient_ids'], 1):
            case = cases[pid]
            patient = batch / pid
            patient.mkdir(exist_ok=True)
            attempts = sorted(patient.glob('attempt*'))
            done = next((p for p in attempts if complete(p, output_type)), None)
            if case.leakage_flags:
                state = 'skipped_presentation_leakage'
            elif done:
                state = 'completed'
            elif len(attempts) >= protocol['max_attempts_per_patient']:
                state = 'failed_attempt_limit'
            else:
                if shutil.disk_usage(batch).free < protocol['minimum_free_bytes']:
                    replace_json(batch / 'attention.json', {'reason': 'low_disk_space', 'patient_id': pid})
                    raise RuntimeError('Disk space reserve reached; do not auto-retry without freeing space')
                folder = patient / f'attempt{len(attempts)+1:03d}'
                folder.mkdir()
                replace_json(batch / 'progress.json', {'patient_id': pid, 'index': index, 'attempt': str(folder),
                    'started_utc': datetime.now(timezone.utc).isoformat()})
                print(f'START {index}/{len(protocol["patient_ids"])} {pid} {folder.name}', flush=True)
                try:
                    generate(case, model, folder, template, output_type, link_activations=evidence)
                    state = 'completed'
                except Exception as exc:
                    state = 'error'
                    replace_json(folder / 'error.json', {'error_type': type(exc).__name__, 'message': str(exc)})
                    print(f'ERROR {pid}: {type(exc).__name__}', flush=True)
                    if isinstance(exc, (ConnectionError, TimeoutError, OSError, RuntimeError)):
                        states.append({'patient_id': pid, 'status': state})
                        replace_json(batch / 'status.json', states)
                        raise
            states.append({'patient_id': pid, 'status': state})
            replace_json(batch / 'status.json', states)
            print(f'DONE {index}/{len(protocol["patient_ids"])} {pid}: {state}', flush=True)
        replace_json(batch / 'finished.json', {'time_utc': datetime.now(timezone.utc).isoformat(),
              'counts': {s: sum(r['status']==s for r in states) for s in {r['status'] for r in states}}})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--all", action="store_true", help="Run every dataset record in original order")
    action.add_argument("--patient-id", help="Run one named case")
    action.add_argument("--resume", type=Path, help="Continue an existing batch with its frozen protocol")
    parser.add_argument("--experiment", choices=("baseline", "evidence_findings"))
    parser.add_argument("--dataset", type=Path)
    action.add_argument("--limit", type=int, help="Run the first N dataset records")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.resume and args.dataset:
        parser.error("--resume uses the saved dataset")
    if args.resume and args.experiment:
        parser.error("--resume uses the experiment recorded in protocol.json")
    execute(args.resume or prepare(args.experiment or "baseline", args.patient_id, args.dataset or DATASET, args.limit))


if __name__ == "__main__":
    main()
