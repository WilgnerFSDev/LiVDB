import hashlib
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from scripts.c_parser import extract_functions_from_code
from scripts.ui import UI


@dataclass
class PhaseTimer:

    name: str
    total: float = 0.0
    calls: int = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.total += time.perf_counter() - self._start
        self.calls += 1

    @property
    def avg(self):
        return self.total / self.calls if self.calls else 0.0


@dataclass
class Metrics:
    io_read: PhaseTimer = field(default_factory=lambda: PhaseTimer("io_read"))
    json_parse: PhaseTimer = field(default_factory=lambda: PhaseTimer("json_parse"))
    date_filter: PhaseTimer = field(default_factory=lambda: PhaseTimer("date_filter"))
    append: PhaseTimer = field(default_factory=lambda: PhaseTimer("append"))
    df_build: PhaseTimer = field(default_factory=lambda: PhaseTimer("df_build"))

    c_parse_vuln: PhaseTimer = field(default_factory=lambda: PhaseTimer("c_parse_vuln"))
    c_parse_patch: PhaseTimer = field(
        default_factory=lambda: PhaseTimer("c_parse_patch")
    )
    hashing: PhaseTimer = field(default_factory=lambda: PhaseTimer("hashing"))

    pool_wall_time: float = 0.0

    cve_dirs_total: int = 0
    cve_skipped_no_cache: int = 0
    cve_skipped_date: int = 0
    cve_skipped_score: int = 0
    cve_skipped_cwe: int = 0
    cve_skipped_tangled: int = 0
    cve_skipped_no_paths: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    functions_found: int = 0
    functions_changed: int = 0
    functions_unchanged: int = 0
    functions_removed: int = 0
    errors: int = 0

    date_index_used: bool = False
    cve_prefiltered: int = 0
    workers_used: int = 0
    chunks_processed: int = 0
    csv_rows_written: int = 0
    csv_flush_count: int = 0

    func_sizes: List[int] = field(default_factory=list)

    files_per_cve: List[int] = field(default_factory=list)

    wall_start: float = field(default_factory=time.perf_counter)

    @property
    def wall_elapsed(self) -> float:
        return time.perf_counter() - self.wall_start

    @property
    def parallelism_efficiency(self) -> Optional[float]:
        if self.pool_wall_time <= 0 or self.workers_used <= 0:
            return None
        total_cpu = (
            self.c_parse_vuln.total + self.c_parse_patch.total + self.hashing.total
        )
        ideal_cpu = self.pool_wall_time * self.workers_used
        return total_cpu / ideal_cpu if ideal_cpu > 0 else None

    def report(self) -> str:
        elapsed = self.wall_elapsed

        def _pct_wall(seconds: float) -> str:
            return f"{seconds / elapsed * 100:>7.1f}%" if elapsed > 0 else "    n/a"

        def _pct_pool_cpu(seconds: float) -> str:
            budget = self.pool_wall_time * self.workers_used
            return f"{seconds / budget * 100:>7.1f}%" if budget > 0 else "    n/a"

        lines = [
            "",
            "=" * 60,
            "  PERFORMANCE REPORT",
            "=" * 60,
            "",
            f"  Total wall time           : {elapsed:.2f}s",
            f"  Pool wall time            : {self.pool_wall_time:.2f}s"
            f"  ({_pct_wall(self.pool_wall_time)} of wall)",
            "",
            "  ── CVE-level stats ──────────────────────────────────",
            f"  CVE dirs found            : {self.cve_dirs_total}",
            f"  Skipped (no cache)        : {self.cve_skipped_no_cache}",
            f"  Skipped (date filter)     : {self.cve_skipped_date}",
            f"  Skipped (CVSS score)      : {self.cve_skipped_score}",
            f"  Skipped (CWE filter)      : {self.cve_skipped_cwe}",
            f"  Skipped (tangled commits) : {self.cve_skipped_tangled}",
            f"  Skipped (missing paths)   : {self.cve_skipped_no_paths}",
            f"  Actually processed        : {self.cve_dirs_total - self.cve_skipped_no_cache - self.cve_skipped_date - self.cve_skipped_score - self.cve_skipped_cwe - self.cve_skipped_tangled - self.cve_skipped_no_paths}",
            f"  Date index used           : {self.date_index_used}",
            f"  CVEs pre-filtered         : {self.cve_prefiltered}",
            f"  Workers used              : {self.workers_used}",
            f"  Chunks processed          : {self.chunks_processed}",
            f"  CSV rows written          : {self.csv_rows_written}",
            f"  CSV flush count           : {self.csv_flush_count}",
            "",
            "  ── File-level stats ─────────────────────────────────",
            f"  Source files processed    : {self.files_processed}",
            f"  Source files skipped      : {self.files_skipped}",
            f"  Processing errors         : {self.errors}",
        ]

        if self.files_per_cve:
            avg_files = sum(self.files_per_cve) / len(self.files_per_cve)
            max_files = max(self.files_per_cve)
            lines += [
                f"  Avg files/CVE             : {avg_files:.1f}",
                f"  Max files/CVE             : {max_files}",
            ]

        lines += [
            "",
            "  ── Function-level stats ─────────────────────────────",
            f"  Functions extracted       : {self.functions_found}",
            f"    Changed (label 1+0)     : {self.functions_changed}",
            f"    Unchanged (label 0)     : {self.functions_unchanged}",
            f"    Removed (excluded)      : {self.functions_removed}",
        ]

        if self.func_sizes:
            avg_sz = sum(self.func_sizes) / len(self.func_sizes)
            max_sz = max(self.func_sizes)
            lines += [
                f"  Avg function size (lines) : {avg_sz:.1f}",
                f"  Max function size (lines) : {max_sz}",
            ]

        main_timers = [
            self.io_read,
            self.json_parse,
            self.date_filter,
            self.append,
            self.df_build,
        ]
        lines += [
            "",
            "  ── Main-process phases (% of wall time) ─────────────",
            f"  {'Phase':<25} {'Total (s)':>10}  {'Calls':>8}  {'Avg (ms)':>10}  {'% wall':>8}",
            f"  {'-' * 25} {'-' * 10}  {'-' * 8}  {'-' * 10}  {'-' * 8}",
        ]
        for t in main_timers:
            lines.append(
                f"  {t.name:<25} {t.total:>10.3f}  {t.calls:>8}"
                f"  {t.avg * 1000:>10.3f}  {_pct_wall(t.total)}"
            )

        worker_timers = [self.c_parse_vuln, self.c_parse_patch, self.hashing]
        total_worker_cpu = sum(t.total for t in worker_timers)
        eff = self.parallelism_efficiency
        eff_str = f"{eff * 100:.1f}%" if eff is not None else "n/a"
        budget = self.pool_wall_time * self.workers_used

        lines += [
            "",
            "  ── Worker CPU phases (accumulated across all workers) ─",
            f"  Note: sums exceed 100% wall time by design with {self.workers_used} parallel workers.",
            f"  Pool wall time : {self.pool_wall_time:.3f}s  |  "
            f"CPU budget ({self.workers_used} workers) : {budget:.3f}s  |  "
            f"Parallelism efficiency : {eff_str}",
            f"  {'Phase':<25} {'CPU (s)':>10}  {'Calls':>8}  {'Avg (ms)':>10}  {'% budget':>8}",
            f"  {'-' * 25} {'-' * 10}  {'-' * 8}  {'-' * 10}  {'-' * 8}",
        ]
        for t in worker_timers:
            lines.append(
                f"  {t.name:<25} {t.total:>10.3f}  {t.calls:>8}"
                f"  {t.avg * 1000:>10.3f}  {_pct_pool_cpu(t.total)}"
            )
        lines += [
            f"  {'total worker CPU':<25} {total_worker_cpu:>10.3f}  {'':>8}"
            f"  {'':>10}  {_pct_pool_cpu(total_worker_cpu)}",
        ]

        lines += [
            "",
            "  ── Throughput ───────────────────────────────────────",
            f"  CVEs/sec (wall)           : {self.cve_dirs_total / elapsed:.1f}"
            if elapsed > 0
            else "  CVEs/sec                  : n/a",
            f"  Files/sec (wall)          : {self.files_processed / elapsed:.1f}"
            if elapsed > 0
            else "  Files/sec                 : n/a",
            f"  Functions/sec (wall)      : {self.functions_found / elapsed:.1f}"
            if elapsed > 0
            else "  Functions/sec             : n/a",
        ]
        if self.pool_wall_time > 0:
            lines += [
                f"  Functions/sec (pool only) : {self.functions_found / self.pool_wall_time:.1f}",
            ]
        lines += ["", "=" * 60]

        return "\n".join(lines)


from scripts.c_parser import calculate_ast_hash, extract_functions_from_code
from scripts.utils import strip_c_comments


def calculate_text_hash(code_content: str) -> str:
    clean_code: str = strip_c_comments(code_content)
    normalized_code: str = "".join(clean_code.split())
    return hashlib.sha256(normalized_code.encode("utf-8")).hexdigest()


def calculate_hash(code_content: str, hash_strategy: str = "text") -> str:
    if hash_strategy == "ast":
        return calculate_ast_hash(code_content)
    return calculate_text_hash(code_content)


def calculate_size(code_content: str) -> int:
    return len(code_content.splitlines())


import asyncio
import concurrent.futures
import csv
import json
import multiprocessing
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from itertools import islice
from typing import Any, Dict, List, Optional, Set, Union

import aiofiles
import pandas as pd
import questionary
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from scripts.curator import curate_dataset
from scripts.ui import UI
from scripts.utils import (
    CACHE_DIR,
    RAW_DATA_DIR,
    extract_cve_base_score,
    extract_cve_cwes,
    matches_cve_filters,
    normalize_cwe_filter,
    rebuild_date_index,
)


def _extract_missing_cves(missing_cve_ids: List[str]) -> List[str]:
    from scripts.cve_fetcher import _run_extraction_phase, extract_hash_from_url

    UI.info(f"Scanning cache for {len(missing_cve_ids)} CVEs...")
    cves_with_hashes = []

    for cve_id in missing_cve_ids:
        cache_file = os.path.join(CACHE_DIR, f"{cve_id}.json")
        if not os.path.exists(cache_file):
            continue

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            vuln = data
            if "vulnerabilities" in data and isinstance(data["vulnerabilities"], list):
                vuln = data["vulnerabilities"][0]

            cve_data = vuln.get("cve", {})
            references = cve_data.get("references", [])
            hashes = set()
            for ref in references:
                url = ref.get("url", "")
                h = extract_hash_from_url(url)
                if h:
                    hashes.add(h)

            if hashes:
                cves_with_hashes.append((cve_id, list(hashes)))
        except Exception:
            continue

    if not cves_with_hashes:
        UI.warning("None of the missing CVEs had valid project fixing hashes in cache.")
        return []

    UI.info(f"Found hashes for {len(cves_with_hashes)} CVEs. Starting extraction...")
    _run_extraction_phase(cves_with_hashes)

    successfully_extracted = []
    for cve_id, _ in cves_with_hashes:
        if os.path.exists(os.path.join(RAW_DATA_DIR, cve_id)):
            successfully_extracted.append(cve_id)

    return successfully_extracted


class FunctionsPerSecColumn(ProgressColumn):
    def __init__(self):
        super().__init__()
        self._history = []

    def render(self, task) -> Text:
        now = time.monotonic()
        self._history.append((now, task.completed))
        self._history = [(t, c) for t, c in self._history if now - t <= 5]
        if len(self._history) >= 2:
            dt = self._history[-1][0] - self._history[0][0]
            dc = self._history[-1][1] - self._history[0][1]
            rate = dc / dt if dt > 0 else 0
            return Text(f"{rate:.0f} fn/s", style="bold cyan")
        return Text("-- fn/s", style="dim")


FIELDNAMES = ["cve_id", "filename", "hash", "lines", "label", "function"]
WRITE_BUFFER_SIZE = 500


class BufferedCSVWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._f, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        self._buffer = []
        self.flush_count = 0
        self.rows_written = 0

    def append(self, row: dict):
        self._buffer.append(row)
        if len(self._buffer) >= WRITE_BUFFER_SIZE:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        self._writer.writerows(self._buffer)
        self.rows_written += len(self._buffer)
        self.flush_count += 1
        self._buffer.clear()

    def close(self):
        self._flush()
        self._f.close()


async def _read_file_pair(vuln_path: str, patch_path: str) -> tuple:
    try:
        async with aiofiles.open(vuln_path, encoding="utf-8", errors="ignore") as f:
            vuln = await f.read()
    except Exception:
        vuln = None
    try:
        async with aiofiles.open(patch_path, encoding="utf-8", errors="ignore") as f:
            patch = await f.read()
    except Exception:
        patch = None
    return vuln, patch


async def _read_all_pairs(pairs: list) -> list:
    return await asyncio.gather(*[_read_file_pair(v, p) for v, p in pairs])


def _process_single_cve(args: tuple) -> dict:
    cve_id, root_dir = args[:2]
    paired_only = args[2] if len(args) > 2 else False
    single_function_only = args[3] if len(args) > 3 else False
    max_functions_changed = args[4] if len(args) > 4 else None
    hash_strategy = args[5] if len(args) > 5 else "text"
    result = {
        "cve_id": cve_id,
        "rows": [],
        "files_processed": 0,
        "files_skipped": 0,
        "functions_found": 0,
        "functions_changed": 0,
        "functions_unchanged": 0,
        "functions_removed": 0,
        "func_sizes": [],
        "errors": 0,
        "c_parse_vuln_time": 0.0,
        "c_parse_patch_time": 0.0,
        "hashing_time": 0.0,
        "files_in_cve": 0,
        "is_tangled": False,
    }

    vuln_path = os.path.join(root_dir, cve_id, "vulnerable")
    patched_path = os.path.join(root_dir, cve_id, "patched")
    if not os.path.exists(vuln_path) or not os.path.exists(patched_path):
        return result

    source_files = [
        f
        for f in os.listdir(vuln_path)
        if f.lower().endswith((".c", ".cpp", ".h", ".hpp"))
    ]
    result["files_in_cve"] = len(source_files)

    pairs = []
    valid_files = []
    for filename in source_files:
        patched_file_full = os.path.join(patched_path, filename)
        if not os.path.exists(patched_file_full):
            result["files_skipped"] += 1
            continue
        valid_files.append(filename)
        pairs.append((os.path.join(vuln_path, filename), patched_file_full))

    if not pairs:
        return result

    contents = asyncio.run(_read_all_pairs(pairs))

    for filename, (vuln_content, patched_content) in zip(valid_files, contents):
        if vuln_content is None or patched_content is None:
            result["errors"] += 1
            continue

        try:
            t0 = time.perf_counter()
            vuln_funcs = extract_functions_from_code(vuln_content)
            result["c_parse_vuln_time"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            patched_funcs = extract_functions_from_code(patched_content)
            result["c_parse_patch_time"] += time.perf_counter() - t0

            result["files_processed"] += 1

            for func_name, vuln_code in vuln_funcs.items():
                result["functions_found"] += 1

                t0 = time.perf_counter()
                vuln_hash = calculate_hash(vuln_code, hash_strategy=hash_strategy)
                vuln_size = calculate_size(vuln_code)
                result["hashing_time"] += time.perf_counter() - t0

                result["func_sizes"].append(vuln_size)

                if func_name in patched_funcs:
                    patched_code = patched_funcs[func_name]
                    t0 = time.perf_counter()
                    patched_hash = calculate_hash(
                        patched_code, hash_strategy=hash_strategy
                    )
                    patched_size = calculate_size(patched_code)
                    result["hashing_time"] += time.perf_counter() - t0

                    if vuln_hash != patched_hash:
                        result["functions_changed"] += 1
                        result["rows"].append(
                            {
                                "cve_id": cve_id,
                                "filename": filename,
                                "hash": vuln_hash,
                                "lines": vuln_size,
                                "label": 1,
                                "function": vuln_code,
                            }
                        )
                        result["rows"].append(
                            {
                                "cve_id": cve_id,
                                "filename": filename,
                                "hash": patched_hash,
                                "lines": patched_size,
                                "label": 0,
                                "function": patched_code,
                            }
                        )
                    else:
                        result["functions_unchanged"] += 1
                        if not paired_only:
                            result["rows"].append(
                                {
                                    "cve_id": cve_id,
                                    "filename": filename,
                                    "hash": vuln_hash,
                                    "lines": vuln_size,
                                    "label": 0,
                                    "function": vuln_code,
                                }
                            )
                else:
                    result["functions_removed"] += 1

        except Exception as error:
            result["errors"] += 1

    is_tangled = False
    if single_function_only and result["functions_changed"] != 1:
        is_tangled = True
    elif max_functions_changed is not None and (
        result["functions_changed"] > max_functions_changed
        or result["functions_changed"] == 0
    ):
        is_tangled = True

    if is_tangled:
        result["rows"] = []
        result["is_tangled"] = True

    return result


def chunked(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


def process_cve_dataset(
    root_dir: str,
    output_csv: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    cwes: Optional[Union[str, List[str], Set[str]]] = None,
    single_function_only: bool = False,
    max_functions_changed: Optional[int] = None,
    hash_strategy: str = "text",
    paired_only: bool = False,
    interactive: bool = True,
) -> None:

    m = Metrics()

    cve_dirs: List[str] = [
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    ]
    m.cve_dirs_total = len(cve_dirs)

    pub_start_dt = None
    pub_end_dt = None
    if start_date and end_date:
        pub_start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        pub_end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )

    filter_descs = []
    if start_date and end_date:
        filter_descs.append(f"Published: {start_date} to {end_date}")
    if min_score is not None and max_score is not None:
        filter_descs.append(f"CVSS: {min_score} - {max_score}")
    elif min_score is not None:
        filter_descs.append(f"CVSS >= {min_score}")
    elif max_score is not None:
        filter_descs.append(f"CVSS <= {max_score}")
    if cwes:
        target_cwe_set = normalize_cwe_filter(cwes)
        cwe_display = sorted(list({c for c in target_cwe_set if c.startswith("CWE-")}))
        if not cwe_display:
            cwe_display = sorted(list(target_cwe_set))
        filter_descs.append(f"CWE: {', '.join(cwe_display)}")
    if single_function_only:
        filter_descs.append("Non-tangled (1 function changed only)")
    elif max_functions_changed is not None:
        filter_descs.append(f"Changed functions <= {max_functions_changed}")
    if hash_strategy == "ast":
        filter_descs.append("Hash: AST (tree-sitter)")
    else:
        filter_descs.append("Hash: Text (clean)")

    filter_summary = f" ({', '.join(filter_descs)})" if filter_descs else ""
    mode_str = " (Paired only: 1:1 Vulnerable vs Patched)" if paired_only else ""
    UI.info(f"Building dataset{mode_str}{filter_summary}...")

    has_advanced_filter = (min_score is not None or max_score is not None or bool(cwes))

    filtered_cve_dirs = cve_dirs
    if pub_start_dt and pub_end_dt:
        date_index_path = os.path.join(CACHE_DIR, ".date_index.json")
        date_index = None
        if os.path.exists(date_index_path):
            try:
                with open(date_index_path, "r", encoding="utf-8") as f:
                    date_index = json.load(f)
            except Exception:
                pass

        if not date_index and os.path.exists(CACHE_DIR):
            choice = questionary.select(
                "Date index not found. Building it now allows fast date filtering without reading every JSON file. This is a one-time operation.",
                choices=[
                    "[B] Build index now (recommended)",
                    "[S] Skip and use slow path",
                    "[R] Rebuild index from scratch and continue",
                ],
            ).ask()
            if choice and (choice.startswith("[B]") or choice.startswith("[R]")):
                UI.info("Building date index...")
                rebuild_date_index()
                try:
                    with open(date_index_path, "r", encoding="utf-8") as f:
                        date_index = json.load(f)
                except Exception:
                    pass

        if date_index and os.path.exists(CACHE_DIR):
            current_files = {
                f.replace(".json", "")
                for f in os.listdir(CACHE_DIR)
                if f.startswith("CVE-") and f.endswith(".json")
            }
            indexed_cves = set(date_index.keys())
            new_cves = current_files - indexed_cves

            if new_cves:
                UI.warning(f"Date index is missing {len(new_cves)} CVEs found in cache.")
                choice = questionary.confirm("Do you want to update the index now?").ask()
                if choice:
                    rebuild_date_index()
                    with open(date_index_path, "r", encoding="utf-8") as f:
                        date_index = json.load(f)

        if date_index:
            m.date_index_used = True
            filtered_cve_dirs = []
            missing_raw_data = []

            cve_ids_in_range = []
            for cve_id, date_str in date_index.items():
                try:
                    cve_date = date.fromisoformat(date_str)
                    if pub_start_dt.date() <= cve_date <= pub_end_dt.date():
                        if has_advanced_filter:
                            cache_file = os.path.join(CACHE_DIR, f"{cve_id}.json")
                            if not os.path.exists(cache_file):
                                continue
                            try:
                                with open(cache_file, "r", encoding="utf-8") as f:
                                    cdata = json.load(f)
                                if not matches_cve_filters(cdata, min_score=min_score, max_score=max_score, cwes=cwes):
                                    continue
                            except Exception:
                                continue
                        cve_ids_in_range.append(cve_id)
                except ValueError:
                    continue

            for cve_id in cve_ids_in_range:
                if cve_id in cve_dirs:
                    filtered_cve_dirs.append(cve_id)
                else:
                    missing_raw_data.append(cve_id)

            if missing_raw_data:
                UI.warning(
                    f"Found {len(missing_raw_data)} CVEs matching filter criteria in cache that have no extracted files in '{os.path.basename(root_dir)}/'."
                )
                if UI.ask_boolean(
                    f"Do you want to extract these {len(missing_raw_data)} CVEs from the git repository now?",
                    default=True,
                ):
                    newly_extracted = _extract_missing_cves(missing_raw_data)
                    filtered_cve_dirs.extend(newly_extracted)
                    if len(newly_extracted) < len(missing_raw_data):
                        UI.warning(
                            f"Only {len(newly_extracted)}/{len(missing_raw_data)} CVEs were successfully extracted."
                        )
                else:
                    UI.info(f"Skipping {len(missing_raw_data)} non-extracted CVEs.")
                    if not filtered_cve_dirs:
                        UI.error("No CVEs left to process. Build aborted.")
                        return
                    if not UI.ask_boolean(
                        "Do you want to continue with the already extracted CVEs?",
                        default=True,
                    ):
                        return

            m.cve_prefiltered = len(cve_ids_in_range) - len(filtered_cve_dirs)
            m.cve_skipped_date = m.cve_prefiltered

    elif has_advanced_filter:
        matched = []
        for cve_id in cve_dirs:
            cache_file = os.path.join(CACHE_DIR, f"{cve_id}.json")
            if not os.path.exists(cache_file):
                m.cve_skipped_no_cache += 1
                continue
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
            except Exception:
                m.cve_skipped_no_cache += 1
                continue

            if min_score is not None or max_score is not None:
                score = extract_cve_base_score(cdata)
                if score is None or (min_score is not None and score < min_score) or (max_score is not None and score > max_score):
                    m.cve_skipped_score += 1
                    continue

            if cwes:
                if not matches_cve_filters(cdata, cwes=cwes):
                    m.cve_skipped_cwe += 1
                    continue

            matched.append(cve_id)
        filtered_cve_dirs = matched

    if not filtered_cve_dirs:
        UI.warning("No CVEs matched the specified filter criteria.")
        return

    CHUNK_SIZE = 50
    m.workers_used = os.cpu_count() or 1

    count_label_1 = 0
    count_label_0 = 0

    csv_writer = BufferedCSVWriter(output_csv)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        FunctionsPerSecColumn(),
        console=UI.console if hasattr(UI, "console") else None,
    ) as progress:
        task = progress.add_task("Processing CVEs...", total=len(filtered_cve_dirs))

        _pool_start = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=m.workers_used) as pool:
            futures = []
            for cve_id in filtered_cve_dirs:
                futures.append(
                    pool.submit(
                        _process_single_cve,
                        (
                            cve_id,
                            root_dir,
                            paired_only,
                            single_function_only,
                            max_functions_changed,
                            hash_strategy,
                        ),
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res.get("is_tangled", False):
                    m.cve_skipped_tangled += 1
                    m.chunks_processed += 1
                    progress.update(
                        task,
                        advance=1,
                        description=f"Skipping tangled {res['cve_id']}...",
                    )
                    continue

                m.files_processed += res["files_processed"]
                m.files_skipped += res["files_skipped"]
                m.functions_found += res["functions_found"]
                m.functions_changed += res["functions_changed"]
                m.functions_unchanged += res["functions_unchanged"]
                m.functions_removed += res["functions_removed"]
                m.errors += res["errors"]
                m.func_sizes.extend(res["func_sizes"])
                if res["files_in_cve"] > 0:
                    m.files_per_cve.append(res["files_in_cve"])

                m.c_parse_vuln.total += res["c_parse_vuln_time"]
                m.c_parse_vuln.calls += res["files_processed"]
                m.c_parse_patch.total += res["c_parse_patch_time"]
                m.c_parse_patch.calls += res["files_processed"]
                m.hashing.total += res["hashing_time"]
                m.hashing.calls += res["functions_found"]

                for row in res["rows"]:
                    csv_writer.append(row)
                    if row["label"] == 1:
                        count_label_1 += 1
                    else:
                        count_label_0 += 1

                m.chunks_processed += 1
                progress.update(
                    task, advance=1, description=f"Processing {res['cve_id']}..."
                )

        m.pool_wall_time = time.perf_counter() - _pool_start

    csv_writer.close()

    m.csv_rows_written = csv_writer.rows_written
    m.csv_flush_count = csv_writer.flush_count

    perc_vuln = (count_label_1 / m.csv_rows_written * 100) if m.csv_rows_written > 0 else 0
    perc_clean = (count_label_0 / m.csv_rows_written * 100) if m.csv_rows_written > 0 else 0

    stats_msg = (
        f"[bold white]Label 1 (Vulnerable):[/bold white] [bold red]{count_label_1}[/bold red] ({perc_vuln:.2f}%), "
        f"[bold white]Label 0 (Clean/Patched):[/bold white] [bold green]{count_label_0}[/bold green] ({perc_clean:.2f}%)"
    )
    if paired_only:
        UI.success(
            f"1:1 Paired dataset saved to {output_csv} with {count_label_1} pairs ({m.csv_rows_written} total functions)."
        )
    else:
        UI.success(f"Dataset saved to {output_csv} with {m.csv_rows_written} entries.")
    UI.info(f"Summary: {stats_msg}")

    if interactive and UI.ask_boolean(
        "Do you want to run the Global Curation step? (Removes redundant functions and optional random undersampling)",
        default=True,
    ):
        curate_dataset(output_csv)
