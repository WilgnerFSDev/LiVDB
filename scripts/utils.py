import os
import shutil
import subprocess
from typing import Dict, List, Optional, Set, Union

NVD_API_BASE_URL: str = "https://services.nvd.nist.gov/rest/json/cves/2.0"
from pathlib import Path

CACHE_DIR: str = "cache"
RAW_DATA_DIR: str = "raw_data"
RAW_CODE_DIR: str = "raw_code"
WORKSPACE_DIR_NAME: str = "workspace"
DATASETS_DIR: str = "datasets"
METADATA_FILE: str = os.path.join(CACHE_DIR, ".sync_metadata.json")

import json
from datetime import datetime, timedelta

import functools
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

NVD_API_KEY: Optional[str] = os.environ.get("NVD_API_KEY")
PREPROC_DATASETS_DIR: str = os.environ.get("PREPROC_DATASETS_DIR", "preproc_datasets")


@functools.lru_cache(maxsize=1)
def get_source_code_path() -> str:
    return os.environ.get("SOURCE_CODE_PATH", "")


@functools.lru_cache(maxsize=1)
def get_default_cpe() -> str:
    return os.environ.get("DEFAULT_CPE", "cpe:2.3:o:linux:linux_kernel")


@functools.lru_cache(maxsize=1)
def get_cpe_search_string() -> str:
    prefix = get_default_cpe()
    v = os.environ.get("CPE_VERSION", "*")
    u = os.environ.get("CPE_UPDATE", "*")
    e = os.environ.get("CPE_EDITION", "*")
    l = os.environ.get("CPE_LANGUAGE", "*")
    se = os.environ.get("CPE_SW_EDITION", "*")
    ts = os.environ.get("CPE_TARGET_SW", "*")
    th = os.environ.get("CPE_TARGET_HW", "*")
    o = os.environ.get("CPE_OTHER", "*")

    return f"{prefix}:{v}:{u}:{e}:{l}:{se}:{ts}:{th}:{o}"


import re

_C_COMMENT_RE = re.compile(
    r'//.*?$|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
    re.MULTILINE,
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


def run_git_command(
    command: List[str], cwd: str, silent: bool = False
) -> Optional[str]:
    abs_cwd: str = os.path.abspath(cwd)
    if not os.path.isdir(abs_cwd):
        if not silent:
            print(f"Error: Git repository path not found at '{abs_cwd}'")
        return None
    try:
        result = subprocess.run(
            command,
            cwd=abs_cwd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="ignore",
        )
        return result.stdout
    except FileNotFoundError:
        if not silent:
            print("Error: 'git' command not found. Is Git installed and in your PATH?")
        return None
    except subprocess.CalledProcessError as error:
        if not silent:
            print(f"Error executing Git command: {' '.join(command)}")
            print(f"Git stderr: {error.stderr}")
        return None


def execute_git_pull(repo_path: str) -> bool:
    print(f"Attempting 'git pull' in '{os.path.abspath(repo_path)}'...")
    command: List[str] = ["git", "pull"]
    output: Optional[str] = run_git_command(command, repo_path)
    if output:
        print("Git pull successful:")
        print(output)
        return True
    return False


def clean_workspace(
    clean_cache: bool = True,
    clean_raw: bool = True,
    clean_datasets: bool = True,
    clean_raw_code: bool = True,
) -> None:
    if clean_raw and os.path.exists(RAW_DATA_DIR):
        print(f"Removing raw data directory: {RAW_DATA_DIR}")
        shutil.rmtree(RAW_DATA_DIR)

    if clean_cache and os.path.exists(CACHE_DIR):
        print(f"Removing cache directory: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)

    if clean_datasets and os.path.exists(DATASETS_DIR):
        print(f"Removing datasets directory: {DATASETS_DIR}")
        shutil.rmtree(DATASETS_DIR)

    if clean_raw_code and os.path.exists(RAW_CODE_DIR):
        print(f"Removing raw code directory: {RAW_CODE_DIR}")
        shutil.rmtree(RAW_CODE_DIR)

    print("Cleanup operation complete.")


def get_last_sync_timestamp() -> Optional[str]:
    if not os.path.exists(METADATA_FILE):
        return None

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            return metadata.get("last_sync_date")
    except Exception:
        return None


def save_sync_timestamp(timestamp: str) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    metadata = {"last_sync_date": timestamp}

    try:
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save sync metadata: {e}")


def get_incremental_sync_dates() -> tuple[str, str]:
    last_sync = get_last_sync_timestamp()

    if last_sync:
        mod_start = last_sync
    else:
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        mod_start = thirty_days_ago.strftime("%Y-%m-%dT%H:%M:%S.000")

    mod_end = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000")

    return mod_start, mod_end


def rebuild_date_index() -> int:
    import json
    import os

    from scripts.cve_fetcher import is_project_vulnerable

    index = {}
    if not os.path.exists(CACHE_DIR):
        return 0
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith(".json") and fname.startswith("CVE-"):
            fpath = os.path.join(CACHE_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    vuln = (
                        data.get("vulnerabilities", [{}])[0]
                        if "vulnerabilities" in data
                        else data
                    )

                    if is_project_vulnerable(vuln):
                        pub = vuln.get("cve", {}).get("published")
                        if pub:
                            index[fname.replace(".json", "")] = pub.split("T")[0]
            except Exception:
                pass
    index_path = os.path.join(CACHE_DIR, ".date_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return len(index)


def extract_cve_base_score(cve_data: dict) -> Optional[float]:
    if "cve" in cve_data:
        cve = cve_data["cve"]
    elif "vulnerabilities" in cve_data and cve_data["vulnerabilities"]:
        cve = cve_data["vulnerabilities"][0].get("cve", cve_data["vulnerabilities"][0])
    else:
        cve = cve_data

    metrics = cve.get("metrics", {})
    if not isinstance(metrics, dict):
        return None

    for key in ["cvssMetricV31", "cvssMetricV40", "cvssMetricV30", "cvssMetricV2"]:
        metric_list = metrics.get(key)
        if metric_list and isinstance(metric_list, list):
            primary = [
                m
                for m in metric_list
                if isinstance(m, dict) and m.get("type") == "Primary"
            ]
            target = primary[0] if primary else metric_list[0]
            if isinstance(target, dict):
                cvss_data = target.get("cvssData", {})
                if "baseScore" in cvss_data:
                    try:
                        return float(cvss_data["baseScore"])
                    except (ValueError, TypeError):
                        pass
    return None


def extract_cve_cwes(cve_data: dict) -> List[str]:
    if "cve" in cve_data:
        cve = cve_data["cve"]
    elif "vulnerabilities" in cve_data and cve_data["vulnerabilities"]:
        cve = cve_data["vulnerabilities"][0].get("cve", cve_data["vulnerabilities"][0])
    else:
        cve = cve_data

    cwes = []
    for w in cve.get("weaknesses", []):
        if isinstance(w, dict):
            for desc in w.get("description", []):
                if isinstance(desc, dict):
                    val = desc.get("value", "")
                    if val and val.strip():
                        cwes.append(val.strip())
    return cwes


def normalize_cwe_filter(cwes: Optional[Union[str, List[str], Set[str]]]) -> Set[str]:
    if not cwes:
        return set()

    if isinstance(cwes, str):
        raw_items = [c.strip() for c in cwes.replace(";", ",").split(",") if c.strip()]
    else:
        raw_items = [str(c).strip() for c in cwes if str(c).strip()]

    normalized = set()
    for item in raw_items:
        clean = item.upper().strip()
        if not clean:
            continue
        normalized.add(clean)
        if clean.startswith("CWE-"):
            num = clean.replace("CWE-", "").strip()
            if num:
                normalized.add(num)
        elif clean.isdigit():
            normalized.add(f"CWE-{clean}")
    return normalized


def matches_cve_filters(
    cve_data: dict,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    cwes: Optional[Union[str, List[str], Set[str]]] = None,
) -> bool:
    if min_score is not None or max_score is not None:
        score = extract_cve_base_score(cve_data)
        if score is None:
            return False
        if min_score is not None and score < min_score:
            return False
        if max_score is not None and score > max_score:
            return False

    if cwes:
        target_cwes = normalize_cwe_filter(cwes)
        if target_cwes:
            cve_cwes = extract_cve_cwes(cve_data)
            cve_cwe_set = set()
            for c in cve_cwes:
                c_up = c.upper().strip()
                cve_cwe_set.add(c_up)
                if c_up.startswith("CWE-"):
                    cve_cwe_set.add(c_up.replace("CWE-", "").strip())
                elif c_up.isdigit():
                    cve_cwe_set.add(f"CWE-{c_up}")

            if not (cve_cwe_set & target_cwes):
                return False

    return True
