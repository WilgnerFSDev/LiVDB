import os
import re
import sys
import pandas as pd
from pathlib import Path

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.dataset_builder import calculate_hash, calculate_size
from scripts.ui import UI
from scripts.utils import strip_c_comments


def clean_c_code(code: str) -> str:
    if not isinstance(code, str) or not code.strip():
        return ""

    code = code.replace('\r\n', '\n').replace('\r', '\n')

    code = strip_c_comments(code)

    lines = [line.rstrip() for line in code.split('\n') if line.strip()]
    return "\n".join(lines).strip()


def transform_reveal(input_path: str = None, output_name: str = "dataset_reveal.csv") -> None:
    if input_path is None:
        candidates = [
            os.path.join(_ROOT, "datasets_reveal_devign", "reveal.xlsx"),
            os.path.join(_ROOT, "datasets_reveal_devign", "reveal.csv"),
            os.path.join(_ROOT, "datasets", "reveal_original.csv"),
            os.path.join(_ROOT, "datasets", "reveal.xlsx"),
        ]
        for c in candidates:
            if os.path.exists(c):
                input_path = c
                break
        if input_path is None:
            input_path = os.path.join(_ROOT, "datasets_reveal_devign", "reveal.xlsx")

    output_path = os.path.join(_ROOT, "datasets", output_name)

    if not os.path.exists(input_path):
        UI.error(f"Input file not found: {input_path}")
        return

    UI.header("ReVeal Dataset Conversion (Chakraborty et al. 2020)")
    UI.info(f"Reading input dataset: {input_path}...")

    if input_path.endswith('.xlsx') or input_path.endswith('.xls'):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    total_raw = len(df)
    vuln_raw = sum(1 for _, r in df.iterrows() if int(float(r.get('label', 0))) == 1)
    clean_raw = total_raw - vuln_raw
    projects_raw = df['project'].nunique() if 'project' in df.columns else 0

    UI.header("Dataset Stats: Before Processing")
    print(f"Total Rows in Source : {total_raw:,}")
    print(f"Vulnerable (1)       : {vuln_raw:,} ({(vuln_raw / total_raw) * 100:.2f}%)")
    print(f"Non-Vulnerable (0)   : {clean_raw:,} ({(clean_raw / total_raw) * 100:.2f}%)")
    print(f"Unique Projects      : {projects_raw}")
    print("-" * 50)

    UI.info(f"Cleaning C source code and filtering corrupt rows...")

    processed_rows = []
    skipped_nan = 0
    vuln_count = 0
    clean_count = 0

    for idx, row in df.iterrows():
        raw_code = row.get('functionSource')

        if pd.isna(raw_code) or not isinstance(raw_code, str):
            skipped_nan += 1
            continue

        clean_code = clean_c_code(raw_code)
        if not clean_code or clean_code.lower() == "nan":
            skipped_nan += 1
            continue

        try:
            label = int(float(row.get('label', 0)))
        except (ValueError, TypeError):
            label = 0

        if label == 1:
            vuln_count += 1
        else:
            clean_count += 1

        project = str(row.get('project', 'unknown')).strip()
        raw_hash = str(row.get('hash', 'unknown')).strip()
        clean_hash = str(abs(int(raw_hash))) if raw_hash.lstrip('-').isdigit() else raw_hash

        processed_rows.append({
            "cve_id": f"REVEAL-{project.upper()}-{clean_hash[:8]}",
            "filename": f"{project}_func.c",
            "hash": calculate_hash(clean_code),
            "lines": calculate_size(clean_code),
            "label": label,
            "function": clean_code,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df = pd.DataFrame(processed_rows)
    out_df.to_csv(output_path, index=False)

    total_final = len(processed_rows)
    projects_final = sorted(list(set(r['cve_id'].split('-')[1] for r in processed_rows)))

    UI.success(f"Conversion complete! Saved to: {output_path}")
    if skipped_nan > 0:
        UI.warning(f"Discarded {skipped_nan} corrupt/empty functions (e.g. NaN in source dataset).")

    UI.header("Dataset Stats: After Processing (Final Clean CSV)")
    print(f"Total Valid Functions : {total_final:,}")
    print(f"Vulnerable (1)        : {vuln_count:,} ({(vuln_count / total_final) * 100:.2f}%)")
    print(f"Non-Vulnerable (0)    : {clean_count:,} ({(clean_count / total_final) * 100:.2f}%)")
    print(f"Unique Projects ({len(projects_final)}) : {', '.join(projects_final)}")
    print("-" * 50)


if __name__ == "__main__":
    in_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "dataset_reveal.csv"
    transform_reveal(in_arg, out_arg)
