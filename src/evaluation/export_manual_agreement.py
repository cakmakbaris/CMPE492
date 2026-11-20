"""Format hand-written Codex judgments; no inference, label prediction or model calls."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


LABELS = {'T': 'tam_uyumlu', 'P': 'kismen_uyumlu', 'U': 'uyumsuz', 'X': 'tamamen_uyumsuz'}
EXPLANATIONS = {
    'T': 'Aynı tıbbi tanı veya eşanlamlı ifade; anlamlı tanısal ayrım saptanmadı.',
    'P': 'Gerçek tanısal/bileşen örtüşmesi var; kapsam, neden, alt tip veya özgüllük farklı.',
    'U': 'Aynı tanı değil; klinik olarak ilişkili alan veya ayırıcı tanı içinde.',
    'X': 'Referansla tanısal örtüşme ve anlamlı klinik alan ilişkisi saptanmadı.',
}


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def export(snapshot: Path, destination: Path) -> None:
    patients = json.loads(snapshot.read_text(encoding='utf-8'))
    notes = json.loads((destination / 'manual_review_notes.json').read_text(encoding='utf-8'))
    codes: dict[int, str] = {}
    for path in sorted(destination.glob('manual_codes_*.txt')):
        for line in path.read_text(encoding='utf-8').splitlines():
            index, code = line.split()
            if int(index) in codes:
                raise ValueError(f'Duplicate patient index: {index}')
            codes[int(index)] = code
    if set(codes) != set(range(len(patients))):
        raise ValueError('Manual label coverage differs from frozen snapshot')
    rows = []
    patient_rows = []
    review_rows = []
    for index, patient in enumerate(patients):
        hypotheses = patient['hypotheses'][:3]
        code = codes[index]
        if len(code) != len(hypotheses) or set(code) - LABELS.keys():
            raise ValueError(f'Invalid supplied manual labels: {index}')
        note = notes.get(str(index), {})
        local_rows = []
        for rank, (hypothesis, category) in enumerate(zip(hypotheses, code), 1):
            row = {
                'snapshot_index': index, 'patient_id': patient['patient_id'], 'rank': rank,
                'reference_diagnosis': patient['reference']['diagnosis_results'],
                'reference_full': patient['reference']['final_diagnosis'],
                'diagnosis': hypothesis['diagnosis'], 'reasoning': hypothesis['reasoning'],
                'category': LABELS[category], 'category_code': category,
                'category_definition': EXPLANATIONS[category],
                'manual_review_note': note.get('note', ''),
                'needs_user_review': bool(note), 'review_priority': note.get('priority', ''),
                'annotator': 'Codex direct semantic review; no Qwen labeling',
                'source': patient['source'],
            }
            rows.append(row)
            local_rows.append(row)
        record = {
            'snapshot_index': index, 'patient_id': patient['patient_id'],
            'reference_diagnosis': patient['reference']['diagnosis_results'],
            'top1_category': LABELS[code[0]],
            'best_top3_category': LABELS[min(code, key='TPUX'.index)],
            'hypothesis_count': len(code), 'hypotheses': local_rows,
            'needs_user_review': bool(note), 'manual_review_note': note.get('note', ''),
            'review_priority': note.get('priority', ''),
        }
        patient_rows.append(record)
        if note:
            review_rows.append(record)
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'snapshot': str(snapshot.resolve()), 'patients': len(patients), 'hypotheses': len(rows),
        'hypothesis_counts': dict(Counter(row['category'] for row in rows)),
        'top1_patient_counts': dict(Counter(row['top1_category'] for row in patient_rows)),
        'best_top3_patient_counts': dict(Counter(row['best_top3_category'] for row in patient_rows)),
        'patients_needing_user_review': len(review_rows),
        'high_priority_review_patients': sum(row['review_priority'] == 'high' for row in review_rows),
        'method': 'All supplied labels selected directly by Codex, then all reference/name pairs reread. Exporter only checks coverage and formats labels. No Qwen/other inference calls for this manual review.',
        'limitations': [
            'Provisional semantic judgments; no independent clinical adjudication.',
            'Partial may match a secondary component while missing the principal condition.',
            'Unsupported subtype/etiology and internally contradictory labels need review.',
            'A category definition is a shared template, not an individually written justification.',
            'Coverage is limited to the patients in the supplied snapshot.',
            'Reference agreement is separate from evidence sufficiency or medical truth.',
            'Treatment offered as diagnosis is forced into X and flagged; inspect separately.',
        ],
    }
    save_json(destination / 'hypothesis_agreement.json', rows)
    save_json(destination / 'patient_agreement.json', patient_rows)
    save_json(destination / 'needs_user_review.json', review_rows)
    save_json(destination / 'summary.json', summary)
    with (destination / 'hypothesis_agreement.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f'# {len(patients)} hasta: ilk üç tanının elle referans uyumu', '',
             'Codex tarafından doğrudan etiketlendi; Qwen etiketleri kullanılmadı. Klinik uzman doğrulaması beklenir.', '',
             'T = tam uyumlu; P = kısmen uyumlu; U = uyumsuz/ilişkili; X = tamamen uyumsuz.', '',
             'Kısmi uyum ana hastalık yerine yalnız ek hastalık eşleşmesi olabilir; notları okuyun.', '',
             '## Sayımlar', '', '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```', '',
             '## Hastalar', '']
    for record in patient_rows:
        lines.extend([f"### {record['snapshot_index']} — {record['patient_id']}", '',
                      f"Referans: {record['reference_diagnosis']}", ''])
        for row in record['hypotheses']:
            lines.append(f"- {row['rank']}. **{row['category']}** — {row['diagnosis']}")
        lines.append('')
        if record['manual_review_note']:
            lines.extend([f"Kontrol notu: {record['manual_review_note']}", ''])
    (destination / 'review.md').write_text('\n'.join(lines), encoding='utf-8')
    review_lines = ['# Kullanıcı kontrolü önerilen eşleşmeler', '',
                    'Sıralar modelin hipotez sırası; snapshot indeksleri sıfırdan başlar.', '']
    for record in sorted(review_rows, key=lambda row: row['review_priority'] != 'high'):
        review_lines.extend([f"## {record['patient_id']} — {record['review_priority']}", '',
                             f"Referans: {record['reference_diagnosis']}", ''])
        for row in record['hypotheses']:
            review_lines.append(f"- {row['rank']}. {row['diagnosis']} → {row['category']}")
        review_lines.extend(['', record['manual_review_note'], ''])
    (destination / 'needs_user_review.md').write_text('\n'.join(review_lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    export(args.snapshot, args.destination)


if __name__ == '__main__':
    main()
