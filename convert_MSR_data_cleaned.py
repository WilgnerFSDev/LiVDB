import argparse
import difflib
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import uuid
from typing import Dict, List, Optional

import pandas as pd

try:
    from unidiff import PatchSet
except ImportError:
    PatchSet = None

try:
    from scripts.ui import UI
except ImportError:
    class UI:
        @staticmethod
        def header(t: str) -> None:
            print(f"\n{'=' * 50}\n{t}\n{'=' * 50}")

        @staticmethod
        def info(t: str) -> None:
            print(f"[*] {t}")

        @staticmethod
        def step(t: str) -> None:
            print(f"[+] {t}")

        @staticmethod
        def success(t: str) -> None:
            print(f"[OK] {t}")

        @staticmethod
        def warning(t: str) -> None:
            print(f"[WARN] {t}")

        @staticmethod
        def error(t: str) -> None:
            print(f"[ERROR] {t}")

try:
    from scripts.utils import strip_c_comments
except ImportError:
    _C_COMMENT_RE = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )

    def strip_c_comments(code: str) -> str:
        if not isinstance(code, str) or not code:
            return ""

        def _replacer(match: re.Match) -> str:
            s = match.group(0)
            if s.startswith("/") or s.startswith("/*"):
                return " "
            return s

        return _C_COMMENT_RE.sub(_replacer, code)


def calculate_hash(code_content: str) -> str:
    clean_code = strip_c_comments(code_content)
    normalized = "".join(clean_code.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def calculate_size(code_content: str) -> int:
    return len(code_content.splitlines())


def clean_function_code(code: str) -> str:
    if not isinstance(code, str) or not code.strip():
        return ""

    code = code.replace("\r\n", "\n").replace("\r", "\n")

    clean = strip_c_comments(code)

    lines = [line.rstrip() for line in clean.split("\n") if line.strip()]
    return "\n".join(lines).strip()


def extract_filename(row: pd.Series) -> str:
    fn = str(row.get("file_name", ""))
    if fn and fn not in ("nan", "None", ""):
        return fn.split("/")[-1]

    for col in ("project_before", "project_after"):
        url = str(row.get(col, ""))
        if not url or url in ("nan", "None", ""):
            continue

        m1 = re.search(r"[;/]f=([^;&]+)", url)
        if m1:
            return m1.group(1).split("/")[-1]

        m2 = re.search(r"blob/[^/]+/(.+\.[a-zA-Z0-9_]+)", url)
        if m2:
            return m2.group(1).split("/")[-1]

        m3 = re.search(r"([^/]+\.[a-zA-Z0-9_]+)(?:[;?]|$)", url.split(";")[0])
        if m3:
            return m3.group(1)

    project = str(row.get("project", "unknown")).strip()
    return f"{project}_func.c"


def compute_aligned_diff(func_before: str, func_after: str):
    if PatchSet is not None:
        old = func_before.strip() + "\n"
        new = func_after.strip() + "\n"
        tmpdir = tempfile.gettempdir()
        u_id = uuid.uuid4().hex
        f1 = os.path.join(tmpdir, f"old_{u_id}.c")
        f2 = os.path.join(tmpdir, f"new_{u_id}.c")
        try:
            with open(f1, "w", encoding="utf-8") as fh1, open(f2, "w", encoding="utf-8") as fh2:
                fh1.write(old)
                fh2.write(new)

            u_val = len(old.splitlines()) + len(new.splitlines())
            cmd = ["git", "diff", "--no-index", "--no-prefix", f"-U{u_val}", f1, f2]
            res = subprocess.run(cmd, capture_output=True, text=True)

            if res.stdout:
                parsed_patch = PatchSet(res.stdout)
                if len(parsed_patch) > 0:
                    hunk = list(parsed_patch[0])[0]
                    diff_str = str(hunk).split("\n", 1)[1]
                    diff_lines = diff_str.splitlines()

                    lines_before = []
                    added = []
                    removed = []

                    for idx, li in enumerate(diff_lines, start=1):
                        if not li:
                            continue
                        if li[0] == "-":
                            lines_before.append(li[1:])
                            removed.append(idx)
                        elif li[0] == "+":
                            lines_before.append("// " + li[1:])
                            added.append(idx)
                        elif li.startswith("\\"):
                            continue
                        elif li.startswith(" "):
                            lines_before.append(li[1:])
                        else:
                            lines_before.append(li)

                    aligned_before = "\n".join(lines_before)
                    return aligned_before, added, removed, diff_lines
        except Exception:
            pass
        finally:
            if os.path.exists(f1):
                os.remove(f1)
            if os.path.exists(f2):
                os.remove(f2)

    old_lines = func_before.splitlines()
    new_lines = func_after.splitlines()
    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=max(len(old_lines), len(new_lines))))
    hunk_lines = [l for l in diff if not (l.startswith("---") or l.startswith("+++") or l.startswith("@@"))]

    lines_before = []
    added = []
    removed = []

    for idx, li in enumerate(hunk_lines, start=1):
        if not li:
            continue
        if li.startswith("-"):
            lines_before.append(li[1:])
            removed.append(idx)
        elif li.startswith("+"):
            lines_before.append("// " + li[1:])
            added.append(idx)
        elif li.startswith(" "):
            lines_before.append(li[1:])
        else:
            lines_before.append(li)

    aligned_before = "\n".join(lines_before)
    return aligned_before, added, removed, hunk_lines


def convert_msr_data(
    input_path: str = "MSR_data_cleaned.csv",
    output_path: str = "datasets/dataset_bigvul.csv",
    chunk_size: int = 15000,
    max_samples: Optional[int] = None,
    paper_mode: bool = True,
    splits_csv: Optional[str] = None,
) -> Optional[str]:
    if not os.path.exists(input_path):
        UI.error(f"Input file not found: {input_path}")
        return None

    file_size_gb = os.path.getsize(input_path) / (1024 ** 3)
    mode_str = "Paper Baseline Replication Mode (ISSTA'24)" if paper_mode else "Standard LiVDB Schema"
    UI.header(f"Big-Vul (MSR_data_cleaned) -> LiVDB Converter [{mode_str}]")
    UI.info(f"Input file : {input_path} ({file_size_gb:.2f} GB)")
    UI.info(f"Output file: {output_path}")
    UI.info(f"Chunk size : {chunk_size:,} records per chunk")

    splits_map = {}
    if paper_mode:
        candidate_splits = [
            splits_csv,
            "plugins/plugin_cfvd/bigvul_rand_splits.csv",
            "../naturalcc/examples/counterfactual-vulnerability-detection/cfexplainer/storage/external/bigvul_rand_splits.csv",
            os.path.join(os.path.dirname(__file__), "../naturalcc/examples/counterfactual-vulnerability-detection/cfexplainer/storage/external/bigvul_rand_splits.csv"),
        ]
        chosen_split = None
        for cand in candidate_splits:
            if cand and os.path.exists(cand):
                chosen_split = cand
                break

        if chosen_split:
            UI.info(f"Loading official LineVD/Big-Vul splits: {chosen_split}")
            sdf = pd.read_csv(chosen_split)
            splits_map = dict(zip(sdf["id"], sdf["label"]))
            UI.success(f"Loaded {len(splits_map):,} split assignments.")
        else:
            UI.warning("bigvul_rand_splits.csv not found; splits will default to 'train'.")

    cols_to_read = [
        "Unnamed: 0",
        "CVE ID",
        "project",
        "commit_id",
        "file_name",
        "project_before",
        "project_after",
        "func_before",
        "vul",
    ]
    if paper_mode:
        cols_to_read.append("func_after")

    processed_rows: List[Dict] = []
    total_raw = 0
    vuln_raw = 0
    clean_raw = 0
    skipped_empty = 0
    skipped_paper_filter = 0

    UI.step("Streaming and processing chunks from MSR_data_cleaned...")

    try:
        reader = pd.read_csv(
            input_path,
            usecols=cols_to_read,
            chunksize=chunk_size,
            low_memory=False,
            encoding="utf-8",
            on_bad_lines="skip",
        )

        chunk_idx = 0
        for chunk in reader:
            chunk_idx += 1
            chunk_len = len(chunk)
            total_raw += chunk_len

            for _, row in chunk.iterrows():
                try:
                    label = int(row.get("vul", 0))
                except (ValueError, TypeError):
                    label = 0

                if label == 1:
                    vuln_raw += 1
                else:
                    clean_raw += 1

                raw_code = row.get("func_before")
                if not isinstance(raw_code, str) or not raw_code.strip():
                    skipped_empty += 1
                    continue

                raw_id = row.get("Unnamed: 0")
                try:
                    sample_id = int(raw_id)
                except (ValueError, TypeError):
                    sample_id = total_raw

                split_label = splits_map.get(sample_id, "train")

                if paper_mode:
                    clean_before = clean_function_code(raw_code)
                    if not clean_before or len(clean_before.splitlines()) < 2:
                        skipped_empty += 1
                        continue

                    if label == 1:
                        raw_after = str(row.get("func_after", ""))
                        clean_after = clean_function_code(raw_after) if raw_after else ""
                        aligned_before, added, removed, diff_lines = compute_aligned_diff(clean_before, clean_after)

                        if len(added) == 0 and len(removed) == 0:
                            skipped_paper_filter += 1
                            continue
                        if not clean_before.strip().endswith(("}", ";")):
                            skipped_paper_filter += 1
                            continue
                        if clean_after and not clean_after.strip().endswith(("}", ";")):
                            skipped_paper_filter += 1
                            continue
                        if aligned_before.strip().endswith(");"):
                            skipped_paper_filter += 1
                            continue
                        if diff_lines:
                            mod_prop = len(added + removed) / len(diff_lines)
                            if mod_prop >= 0.7:
                                skipped_paper_filter += 1
                                continue
                        if len(aligned_before.splitlines()) <= 5:
                            skipped_paper_filter += 1
                            continue

                        final_func = aligned_before
                        removed_str = ",".join(map(str, removed))
                    else:
                        final_func = clean_before
                        removed_str = ""

                    code_hash = calculate_hash(final_func)
                    code_lines = calculate_size(final_func)

                    raw_cve = str(row.get("CVE ID", "")).strip()
                    project = str(row.get("project", "unknown")).strip()
                    commit_id = str(row.get("commit_id", "unknown")).strip()
                    cve_id = raw_cve if raw_cve and raw_cve not in ("nan", "None", "") else f"BIGVUL-{project.upper()}-{commit_id[:8]}"
                    filename = extract_filename(row)

                    processed_rows.append({
                        "id": sample_id,
                        "cve_id": cve_id,
                        "filename": filename,
                        "hash": code_hash,
                        "lines": code_lines,
                        "label": label,
                        "function": final_func,
                        "split": split_label,
                        "removed": removed_str,
                    })

                else:
                    clean_code = clean_function_code(raw_code)
                    if not clean_code or len(clean_code.splitlines()) < 2:
                        skipped_empty += 1
                        continue

                    raw_cve = str(row.get("CVE ID", "")).strip()
                    project = str(row.get("project", "unknown")).strip()
                    commit_id = str(row.get("commit_id", "unknown")).strip()
                    cve_id = raw_cve if raw_cve and raw_cve not in ("nan", "None", "") else f"BIGVUL-{project.upper()}-{commit_id[:8]}"
                    filename = extract_filename(row)
                    code_hash = calculate_hash(clean_code)
                    code_lines = calculate_size(clean_code)

                    processed_rows.append({
                        "id": sample_id,
                        "cve_id": cve_id,
                        "filename": filename,
                        "hash": code_hash,
                        "lines": code_lines,
                        "label": label,
                        "function": clean_code,
                        "split": split_label,
                    })

                if max_samples and len(processed_rows) >= max_samples:
                    break

            sys.stdout.write(
                f"\r[LiVDB] Chunk {chunk_idx}: {total_raw:,} rows scanned | "
                f"{len(processed_rows):,} valid functions extracted..."
            )
            sys.stdout.flush()

            if max_samples and len(processed_rows) >= max_samples:
                print("\n[LiVDB] Maximum sample limit reached (test mode).")
                break

        print()

    except Exception as e:
        UI.error(f"Error during dataset processing: {e}")
        return None

    if not processed_rows:
        UI.error("No valid functions were extracted.")
        return None

    UI.step("Constructing DataFrame...")
    df_final = pd.DataFrame(processed_rows)

    if not paper_mode:
        df_final = df_final.sort_values(by=["cve_id", "label"], ascending=[True, False]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    UI.step(f"Saving formatted dataset to: {output_path}...")
    df_final.to_csv(output_path, index=False, encoding="utf-8")

    total_final = len(df_final)
    vuln_final = int((df_final["label"] == 1).sum())
    clean_final = int((df_final["label"] == 0).sum())
    unique_cves = int(df_final["cve_id"].nunique())

    perc_vuln = (vuln_final / total_final * 100) if total_final > 0 else 0
    perc_clean = (clean_final / total_final * 100) if total_final > 0 else 0

    UI.success(f"Conversion completed successfully! Saved to: {output_path}")
    UI.header("Converted Dataset Stats")
    print(f"Total Raw Rows Read : {total_raw:,}")
    print(f"Skipped Empty/Short : {skipped_empty:,}")
    if paper_mode:
        print(f"Skipped Paper Filter: {skipped_paper_filter:,} (mod_prop, length, abnormal endings)")
    print(f"Total Valid Saved   : {total_final:,}")
    print(f"Vulnerable (1)      : {vuln_final:,} ({perc_vuln:.2f}%)")
    print(f"Non-Vulnerable (0)  : {clean_final:,} ({perc_clean:.2f}%)")
    print(f"Unique CVE IDs      : {unique_cves:,}")
    if "split" in df_final.columns:
        print("\nPartitions Breakdown:")
        for sp, cnt in df_final["split"].value_counts().items():
            sp_vuln = int(((df_final["split"] == sp) & (df_final["label"] == 1)).sum())
            sp_clean = cnt - sp_vuln
            print(f"  - {sp}: {cnt:,} functions ({sp_vuln:,} vuln, {sp_clean:,} clean)")
    print("-" * 50)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert MSR_data_cleaned.csv to LiVDB or Paper format."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="MSR_data_cleaned.csv",
        help="Path to MSR_data_cleaned.csv (default: MSR_data_cleaned.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="datasets/dataset_bigvul.csv",
        help="Destination path for the formatted CSV (default: datasets/dataset_bigvul.csv)",
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=15000,
        help="Chunk size for streaming memory efficiency (default: 15000)",
    )
    parser.add_argument(
        "--max-samples",
        "-m",
        type=int,
        default=None,
        help="Limit number of extracted functions (useful for quick testing)",
    )
    parser.add_argument(
        "--paper-mode",
        dest="paper_mode",
        action="store_true",
        default=True,
        help="Enable exact ISSTA'24 / LineVD replication mode (diff alignment, removed lines, official splits, no curation) (default: True)",
    )
    parser.add_argument(
        "--no-paper-mode",
        dest="paper_mode",
        action="store_false",
        help="Disable paper mode and use standard LiVDB schema without diff alignment",
    )
    parser.add_argument(
        "--splits-csv",
        type=str,
        default=None,
        help="Path to bigvul_rand_splits.csv (default: auto-detected)",
    )
    args = parser.parse_args()

    convert_msr_data(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        paper_mode=args.paper_mode,
        splits_csv=args.splits_csv,
    )


if __name__ == "__main__":
    main()
