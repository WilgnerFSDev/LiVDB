import json
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

import requests

from scripts.ui import UI
from scripts.utils import (
    CACHE_DIR,
    NVD_API_BASE_URL,
    RAW_DATA_DIR,
    execute_git_pull,
    get_incremental_sync_dates,
    get_source_code_path,
    get_default_cpe,
    get_cpe_search_string,
    run_git_command,
    save_sync_timestamp,
)


def _get_nvd_api_key() -> Optional[str]:
    return os.environ.get("NVD_API_KEY") or None

_GIT_TIMEOUT = 60


def is_project_vulnerable(vuln_item: Dict[str, Any]) -> bool:
    cve_data = vuln_item.get("cve", {})
    configurations = cve_data.get("configurations", [])
    project_cpe = get_default_cpe()

    for config in configurations:
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                criteria = match.get("criteria", "")
                is_vulnerable = match.get("vulnerable", False)

                if is_vulnerable and project_cpe in criteria:
                    return True
    return False


def _run_git_with_timeout(
    cmd: List[str], cwd: str, timeout: int = _GIT_TIMEOUT
) -> Optional[str]:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def _run_extraction_phase(cves_with_hashes: List) -> int:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    extract_start = time.perf_counter()
    git_errors = 0
    git_calls = 0
    skipped_no_extract = 0
    cache_hits = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("[yellow]ETA:[/yellow]"),
        TimeRemainingColumn(),
        TextColumn("[bold gold1]elapsed:[/bold gold1]"),
        TimeElapsedColumn(),
        transient=False,
        console=UI.console if hasattr(UI, "console") else None,
    ) as progress:
        extract_task = progress.add_task(
            "[magenta]Extracting vulnerable/patched files...",
            total=len(cves_with_hashes),
        )

        extracted_count = 0
        for cve_id, hashes in cves_with_hashes:
            progress.update(
                extract_task, description=f"[magenta]Processing {cve_id}..."
            )

            vulnerable_dir: str = os.path.join(RAW_DATA_DIR, cve_id, "vulnerable")
            patched_dir: str = os.path.join(RAW_DATA_DIR, cve_id, "patched")
            if os.path.exists(vulnerable_dir) and os.path.exists(patched_dir):
                v_files = [f for f in os.listdir(vulnerable_dir) if f.endswith((".c", ".h"))]
                p_files = [f for f in os.listdir(patched_dir) if f.endswith((".c", ".h"))]
                if v_files and p_files:
                    cache_hits += 1
                    extracted_count += 1
                    progress.advance(extract_task)
                    continue

            success = False
            for commit_hash in hashes:
                git_calls += 1
                try:
                    if extract_and_save_files(cve_id, commit_hash, silent=True):
                        success = True
                        break
                except Exception:
                    git_errors += 1
            if success:
                extracted_count += 1
            else:
                skipped_no_extract += 1
            progress.advance(extract_task)

    extract_elapsed = time.perf_counter() - extract_start
    throughput = extracted_count / extract_elapsed if extract_elapsed > 0 else 0

    UI.success(
        f"Extraction complete. Successfully processed {extracted_count}/{len(cves_with_hashes)} CVEs."
    )
    UI.info(
        f"[bold]Extraction metrics:[/bold] "
        f"elapsed=[bold gold1]{extract_elapsed:.1f}s[/bold gold1]  "
        f"throughput=[bold cyan]{throughput:.2f} CVEs/s[/bold cyan]  "
        f"git_calls=[bold]{git_calls}[/bold]  "
        f"cache_hits=[bold green]{cache_hits}[/bold green]  "
        f"errors=[bold red]{git_errors}[/bold red]  "
        f"skipped=[bold yellow]{skipped_no_extract}[/bold yellow]"
    )
    return extracted_count


def extract_and_save_files(cve_id: str, commit_hash: str, silent: bool = False) -> bool:
    vulnerable_dir: str = os.path.join(RAW_DATA_DIR, cve_id, "vulnerable")
    patched_dir: str = os.path.join(RAW_DATA_DIR, cve_id, "patched")

    if os.path.exists(vulnerable_dir) and os.path.exists(patched_dir):
        v_files = [f for f in os.listdir(vulnerable_dir) if f.endswith((".c", ".h"))]
        p_files = [f for f in os.listdir(patched_dir) if f.endswith((".c", ".h"))]
        if v_files and p_files:
            if not silent:
                UI.info(f"-> CVE {cve_id} already extracted. Skipping.")
            return True

    if not silent:
        print(f"\n--- Processing files for commit {commit_hash[:10]}... ---")

    files_changed_cmd: List[str] = [
        "git",
        "show",
        "--pretty=format:",
        "--name-only",
        commit_hash,
    ]
    files_output: Optional[str] = _run_git_with_timeout(
        files_changed_cmd, get_source_code_path()
    )
    if files_output is None:
        return False

    source_files: List[str] = [
        f for f in files_output.strip().split("\n") if f.endswith((".c", ".h"))
    ]
    if not source_files:
        if not silent:
            print("No .c or .h files found in this commit.")
        return False

    vulnerable_dir: str = os.path.join(RAW_DATA_DIR, cve_id, "vulnerable")
    patched_dir: str = os.path.join(RAW_DATA_DIR, cve_id, "patched")
    os.makedirs(vulnerable_dir, exist_ok=True)
    os.makedirs(patched_dir, exist_ok=True)

    extracted_any = False

    for file_path in source_files:
        if not file_path:
            continue

        if not silent:
            print(f"-> Extracting file: {file_path}")

        vulnerable_content_cmd: List[str] = [
            "git",
            "show",
            f"{commit_hash}^:{file_path}",
        ]
        vulnerable_content: Optional[str] = _run_git_with_timeout(
            vulnerable_content_cmd, get_source_code_path()
        )

        patched_content_cmd: List[str] = ["git", "show", f"{commit_hash}:{file_path}"]
        patched_content: Optional[str] = _run_git_with_timeout(
            patched_content_cmd, get_source_code_path()
        )

        base_filename: str = os.path.basename(file_path)
        if vulnerable_content is not None:
            with open(
                os.path.join(vulnerable_dir, base_filename), "w", encoding="utf-8"
            ) as file:
                file.write(vulnerable_content)
                extracted_any = True

        if patched_content is not None:
            with open(
                os.path.join(patched_dir, base_filename), "w", encoding="utf-8"
            ) as file:
                file.write(patched_content)
                extracted_any = True

    return extracted_any


def extract_hash_from_url(url: str) -> Optional[str]:
    patterns: List[str] = [
        r"git\.kernel\.org/.+/c/([0-9a-fA-F]{40})",
        r"github\.com/.+/commit/([0-9a-fA-F]{40})",
    ]

    for pattern in patterns:
        match: Optional[re.Match[str]] = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_commit_hash_from_nvd(
    cve_id: str, use_cache: bool = True
) -> Optional[List[str]]:
    cache_file_path: str = os.path.join(CACHE_DIR, f"{cve_id}.json")
    data: Optional[Dict[str, Any]] = None

    if use_cache and os.path.exists(cache_file_path):
        print(f"-> Cache hit. Loading data from '{cache_file_path}'...")
        try:
            with open(cache_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, IOError) as error:
            print(f"Warning: Could not read cache file. Error: {error}")

    if data is None:
        print(f"-> Cache miss or disabled. Fetching from NVD API for '{cve_id}'...")
        params: Dict[str, str] = {"cveId": cve_id}
        headers: Dict[str, str] = {}
        nvd_key = _get_nvd_api_key()
        if nvd_key:
            headers["apiKey"] = nvd_key
        else:
            print(
                "Warning: No NVD API Key provided. Request might be rate-limited severely."
            )

        time.sleep(6)

        try:
            response: requests.Response = requests.get(
                NVD_API_BASE_URL, params=params, headers=headers, timeout=30
            )

            if response.status_code == 404:
                print(f"Error: CVE '{cve_id}' not found in NVD.")
                return None

            response.raise_for_status()
            data = response.json()

            if use_cache:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(cache_file_path, "w", encoding="utf-8") as file:
                    json.dump(data, file, indent=4)
                print(f"-> Saved API response to '{cache_file_path}'")

        except requests.RequestException as error:
            print(f"An error occurred while calling the NVD API: {error}")
            return None

    if not data:
        return None

    commit_hashes: List[str] = []
    vulnerabilities: List[Dict[str, Any]] = data.get("vulnerabilities", [])

    if not vulnerabilities:
        return None

    cve_item: Dict[str, Any] = vulnerabilities[0].get("cve", {})
    references: List[Dict[str, Any]] = cve_item.get("references", [])

    for reference in references:
        url: str = reference.get("url", "")
        extracted_hash: Optional[str] = extract_hash_from_url(url)
        if extracted_hash:
            commit_hashes.append(extracted_hash)

    return list(set(commit_hashes))


def fetch_and_save_cve(cve_id: str, use_cache: bool, should_pull_git: bool) -> None:
    nvd_key = _get_nvd_api_key()
    if nvd_key:
        masked_key = f"{nvd_key[:8]}****..." if len(nvd_key) > 8 else "****..."
        UI.info(
            f"API Key detected: [bold green]{masked_key}[/bold green] | Limits: 50 req/30sec"
        )
    else:
        UI.warning("API Key not detected! Limits: 5 req/30sec")

    if should_pull_git:
        execute_git_pull(get_source_code_path())

    hashes: Optional[List[str]] = get_commit_hash_from_nvd(cve_id, use_cache=use_cache)

    if hashes:
        print(f"\nSuccessfully found {len(hashes)} fixing commit hash(es):")
        for commit_hash in hashes:
            print(f" -> {commit_hash}")

        for commit_hash in hashes:
            if extract_and_save_files(cve_id, commit_hash):
                print(
                    f"Successfully processed commit {commit_hash[:10]}. Ignoring remaining hashes."
                )
                break

    elif hashes is not None:
        print("\nNo fixing commit hashes were found for this CVE in NVD references.")


def fetch_cves_by_date(start_date: str, end_date: str, should_pull_git: bool) -> None:
    date_index = {}
    date_index_path = os.path.join(CACHE_DIR, ".date_index.json")
    if os.path.exists(date_index_path):
        try:
            with open(date_index_path, "r", encoding="utf-8") as f:
                date_index = json.load(f)
        except Exception:
            pass
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    nvd_key = _get_nvd_api_key()
    if nvd_key:
        masked_key = f"{nvd_key[:8]}****..." if len(nvd_key) > 8 else "****..."
        UI.info(
            f"API Key detected: [bold green]{masked_key}[/bold green] | Limits: 50 req/30sec"
        )
    else:
        UI.warning("API Key not detected! Limits: 5 req/30sec")

    if should_pull_git:
        execute_git_pull(get_source_code_path())

    from datetime import datetime, timedelta

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    chunks = []
    current_start = start_dt
    while current_start <= end_dt:
        current_end = min(current_start + timedelta(days=110), end_dt)
        chunks.append(
            (
                current_start.strftime("%Y-%m-%dT00:00:00.000"),
                current_end.strftime("%Y-%m-%dT23:59:59.999"),
            )
        )
        current_start = current_end + timedelta(days=1)

    all_vulnerabilities = []
    stats_total = 0
    stats_included = 0
    stats_ignored = 0

    UI.info(f"Fetching CVEs from {start_date} to {end_date} (Split into {len(chunks)} chunks)...")

    for i, (chunk_start, chunk_end) in enumerate(chunks):
        if len(chunks) > 1:
            UI.info(f"Processing Chunk {i+1}/{len(chunks)}: {chunk_start[:10]} to {chunk_end[:10]}")

        total_results = 1
        start_index = 0
        results_per_page = 500

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            fetch_task = progress.add_task(f"[cyan]Downloading Chunk {i+1}...", total=None)

            while start_index < total_results:
                params: Dict[str, Any] = {
                    "pubStartDate": chunk_start,
                    "pubEndDate": chunk_end,
                    "startIndex": start_index,
                    "resultsPerPage": results_per_page,
                    "virtualMatchString": f"{get_default_cpe()}:*:*:*:*:*:*:*:*",
                }

                headers = {}
                nvd_key = _get_nvd_api_key()
                if nvd_key:
                    headers["apiKey"] = nvd_key

                time.sleep(6)

                try:
                    response = requests.get(
                        NVD_API_BASE_URL, params=params, headers=headers, timeout=30
                    )
                    response.raise_for_status()
                    data = response.json()

                    if start_index == 0:
                        total_results = data.get("totalResults", 0)
                        progress.update(fetch_task, total=total_results if total_results > 0 else 1)

                    vulns = data.get("vulnerabilities", [])
                    valid_vulns = [v for v in vulns if is_project_vulnerable(v)]

                    stats_total += len(vulns)
                    stats_included += len(valid_vulns)
                    stats_ignored += len(vulns) - len(valid_vulns)

                    all_vulnerabilities.extend(valid_vulns)

                    os.makedirs(CACHE_DIR, exist_ok=True)
                    for vuln_item in valid_vulns:
                        cve_id = vuln_item.get("cve", {}).get("id")
                        if cve_id:
                            cache_file_path = os.path.join(CACHE_DIR, f"{cve_id}.json")
                            with open(cache_file_path, "w", encoding="utf-8") as file:
                                json.dump(vuln_item, file, indent=4)
                            pub = vuln_item.get("cve", {}).get("published")
                            if pub:
                                date_index[cve_id] = pub.split("T")[0]

                    start_index += len(vulns)
                    progress.update(fetch_task, completed=start_index)

                    if len(vulns) == 0:
                        break

                except requests.RequestException as error:
                    UI.error(f"Failed to fetch from NVD for chunk {i+1}: {error}")
                    break

    UI.info(
        f"[bold]NVD Filter Statistics:[/bold] "
        f"Total received: {stats_total} | "
        f"Included (Project): [green]{stats_included}[/green] | "
        f"Ignored: [yellow]{stats_ignored}[/yellow]"
    )

    if date_index:
        try:
            with open(date_index_path, "w", encoding="utf-8") as f:
                json.dump(date_index, f, indent=2)
        except Exception:
            pass

    if not all_vulnerabilities:
        UI.warning("No CVEs found in the specified date range.")
        return

    UI.success(
        f"Matched {len(all_vulnerabilities)} CVEs in the requested date range. Scanning hashes..."
    )

    cves_with_hashes = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        scan_task = progress.add_task(
            "[cyan]Scanning references...", total=len(all_vulnerabilities)
        )

        for vuln_item in all_vulnerabilities:
            cve_data = vuln_item.get("cve", {})
            cve_id = cve_data.get("id")

            if not cve_id:
                progress.advance(scan_task)
                continue

            references = cve_data.get("references", [])
            hashes = set()
            for ref in references:
                url = ref.get("url", "")
                h = extract_hash_from_url(url)
                if h:
                    hashes.add(h)

            if hashes:
                cves_with_hashes.append((cve_id, list(hashes)))

            progress.advance(scan_task)

    if not cves_with_hashes:
        UI.warning("None of the fetched CVEs contained target project fixing commits.")
        return

    UI.success(f"Discovered {len(cves_with_hashes)} CVEs with project patches.")

    extracted_count = _run_extraction_phase(cves_with_hashes)
    UI.success(
        f"{extracted_count} CVEs were successfully saved to {os.path.basename(RAW_DATA_DIR)}."
    )


def fetch_cves_incremental(should_pull_git: bool) -> None:
    date_index = {}
    date_index_path = os.path.join(CACHE_DIR, ".date_index.json")
    if os.path.exists(date_index_path):
        try:
            with open(date_index_path, "r", encoding="utf-8") as f:
                date_index = json.load(f)
        except Exception:
            pass

    from datetime import datetime
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    nvd_key = _get_nvd_api_key()
    if nvd_key:
        masked_key = f"{nvd_key[:8]}****..." if len(nvd_key) > 8 else "****..."
        UI.info(
            f"API Key detected: [bold green]{masked_key}[/bold green] | Limits: 50 req/30sec"
        )
    else:
        UI.warning("API Key not detected! Limits: 5 req/30sec")

    if should_pull_git:
        execute_git_pull(get_source_code_path())

    mod_start, mod_end = get_incremental_sync_dates()

    UI.info(f"Fetching modified CVEs from {mod_start} to {mod_end} from NVD...")
    UI.info("(Following NIST Best Practice: Only recently modified CVEs)")

    total_results = 1
    start_index = 0
    results_per_page = 500
    all_vulnerabilities = []

    stats_total = 0
    stats_included = 0
    stats_ignored = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        fetch_task = progress.add_task("[cyan]Downloading NVD CVE pages...", total=None)

        while start_index < total_results:
            params: Dict[str, Any] = {
                "lastModStartDate": mod_start,
                "lastModEndDate": mod_end,
                "startIndex": start_index,
                "resultsPerPage": results_per_page,
                "virtualMatchString": f"{get_default_cpe()}:*:*:*:*:*:*:*:*",
            }

            headers = {}
            nvd_key = _get_nvd_api_key()
            if nvd_key:
                headers["apiKey"] = nvd_key

            time.sleep(6)

            try:
                response = requests.get(
                    NVD_API_BASE_URL, params=params, headers=headers, timeout=30
                )
                response.raise_for_status()
                data = response.json()

                if start_index == 0:
                    total_results = data.get("totalResults", 0)
                    progress.update(fetch_task, total=total_results)

                vulns = data.get("vulnerabilities", [])

                valid_vulns = [v for v in vulns if is_project_vulnerable(v)]

                stats_total += len(vulns)
                stats_included += len(valid_vulns)
                stats_ignored += len(vulns) - len(valid_vulns)

                all_vulnerabilities.extend(valid_vulns)

                os.makedirs(CACHE_DIR, exist_ok=True)
                for vuln_item in valid_vulns:
                    cve_id = vuln_item.get("cve", {}).get("id")
                    if cve_id:
                        cache_file_path = os.path.join(CACHE_DIR, f"{cve_id}.json")
                        with open(cache_file_path, "w", encoding="utf-8") as file:
                            json.dump(vuln_item, file, indent=4)
                        pub = vuln_item.get("cve", {}).get("published")
                        if pub:
                            date_index[cve_id] = pub.split("T")[0]

                start_index += len(vulns)
                progress.update(fetch_task, completed=start_index)

                if len(vulns) == 0:
                    break

            except requests.RequestException as error:
                UI.error(f"Failed to fetch from NVD: {error}")
                break

    UI.info(
        f"[bold]NVD Filter Statistics:[/bold] "
        f"Total received: {stats_total} | "
        f"Included (Project): [green]{stats_included}[/green] | "
        f"Ignored: [yellow]{stats_ignored}[/yellow]"
    )

    if date_index:
        try:
            with open(date_index_path, "w", encoding="utf-8") as f:
                json.dump(date_index, f, indent=2)
        except Exception:
            pass

    if not all_vulnerabilities:
        UI.warning("No modified CVEs found since last sync.")
        return

    UI.success(f"Matched {len(all_vulnerabilities)} modified CVEs. Scanning hashes...")

    cves_with_hashes = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        scan_task = progress.add_task(
            "[cyan]Scanning references...", total=len(all_vulnerabilities)
        )

        for vuln_item in all_vulnerabilities:
            cve_data = vuln_item.get("cve", {})
            cve_id = cve_data.get("id")

            if not cve_id:
                progress.advance(scan_task)
                continue

            references = cve_data.get("references", [])
            hashes = set()
            for ref in references:
                url = ref.get("url", "")
                h = extract_hash_from_url(url)
                if h:
                    hashes.add(h)

            if hashes:
                cves_with_hashes.append((cve_id, list(hashes)))

            progress.advance(scan_task)

    if not cves_with_hashes:
        UI.warning("None of the modified CVEs contained target project fixing commits.")
        return

    UI.success(f"Discovered {len(cves_with_hashes)} CVEs with project patches.")

    extracted_count = _run_extraction_phase(cves_with_hashes)
    UI.success(
        f"{extracted_count} CVEs were successfully saved to {os.path.basename(RAW_DATA_DIR)}."
    )

    save_sync_timestamp(mod_end)
    UI.success(
        f"Sync timestamp saved. Next incremental update will fetch from {mod_end}."
    )


def process_cached_cves(
    should_pull_git: bool,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    from datetime import datetime, timezone
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    if not os.path.exists(CACHE_DIR) or not os.listdir(CACHE_DIR):
        UI.warning("No cached CVEs found. Please run 'Fetch CVEs' first.")
        return

    if should_pull_git:
        execute_git_pull(get_source_code_path())

    UI.step("Reading cached CVEs...")
    all_vulnerabilities = []

    stats_total = 0

    pub_start_dt = None
    pub_end_dt = None
    if start_date and end_date:
        pub_start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        pub_end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )

    if os.path.exists(CACHE_DIR):
        for filename in os.listdir(CACHE_DIR):
            if not filename.endswith(".json") or filename.startswith("."):
                continue
            try:
                cache_file_path = os.path.join(CACHE_DIR, filename)
                with open(cache_file_path, "r", encoding="utf-8") as f:
                    vuln_item = json.load(f)

                if "vulnerabilities" in vuln_item and isinstance(
                    vuln_item["vulnerabilities"], list
                ):
                    vuln_item = vuln_item["vulnerabilities"][0]

                stats_total += 1

                cve_obj = vuln_item.get("cve", {})
                all_vulnerabilities.append({"cve": cve_obj})

                cve_id = cve_obj.get("id")
                if cve_id and cve_id not in date_index:
                    pub = cve_obj.get("published")
                    if pub:
                        date_index[cve_id] = pub.split("T")[0]
            except Exception:
                pass

    UI.info(
        f"[bold]Cache Statistics:[/bold] "
        f"Total loaded from cache: {stats_total}"
    )

    if date_index:
        try:
            with open(date_index_path, "w", encoding="utf-8") as f:
                json.dump(date_index, f, indent=2)
        except Exception:
            pass

    if not all_vulnerabilities:
        UI.warning("No valid CVE data found in the local cache.")
        return

    UI.success(f"Loaded {len(all_vulnerabilities)} CVEs from cache. Scanning hashes...")

    cves_with_hashes = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        scan_task = progress.add_task(
            "[cyan]Scanning references...", total=len(all_vulnerabilities)
        )

        for vuln_item in all_vulnerabilities:
            cve_data = vuln_item.get("cve", {})
            cve_id = cve_data.get("id")

            if not cve_id:
                progress.advance(scan_task)
                continue

            pub_date_str = cve_data.get("published")
            if pub_date_str and pub_start_dt and pub_end_dt:
                try:
                    pub_dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                    if not (pub_start_dt <= pub_dt <= pub_end_dt):
                        progress.advance(scan_task)
                        continue
                except ValueError:
                    pass

            references = cve_data.get("references", [])
            hashes = set()
            for ref in references:
                url = ref.get("url", "")
                h = extract_hash_from_url(url)
                if h:
                    hashes.add(h)

            if hashes:
                cves_with_hashes.append((cve_id, list(hashes)))

            progress.advance(scan_task)

    if not cves_with_hashes:
        UI.warning(
            "None of the cached CVEs contained target project fixing commits in their references."
        )
        return

    UI.success(f"Discovered {len(cves_with_hashes)} CVEs with project patches.")

    _run_extraction_phase(cves_with_hashes)


def fetch_all_cves(should_pull_git: bool) -> None:
    date_index = {}
    date_index_path = os.path.join(CACHE_DIR, ".date_index.json")
    if os.path.exists(date_index_path):
        try:
            with open(date_index_path, "r", encoding="utf-8") as f:
                date_index = json.load(f)
        except Exception:
            pass
    from datetime import datetime, timezone
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    nvd_key = _get_nvd_api_key()
    if nvd_key:
        masked_key = f"{nvd_key[:8]}****..." if len(nvd_key) > 8 else "****..."
        UI.info(
            f"API Key detected: [bold green]{masked_key}[/bold green] | Limits: 50 req/30sec"
        )
    else:
        UI.warning("API Key not detected! Limits: 5 req/30sec")

    if should_pull_git:
        execute_git_pull(get_source_code_path())

    metadata_path = os.path.join(CACHE_DIR, ".metadata.json")
    last_sync_time = None
    current_dt = datetime.now(timezone.utc)
    current_time_str = current_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                last_sync_time = json.load(f).get("last_sync_time")
            if last_sync_time:
                last_dt = datetime.fromisoformat(
                    last_sync_time.replace("Z", "")
                ).replace(tzinfo=timezone.utc)
                if (current_dt - last_dt).days > 100:
                    last_sync_time = None
        except:
            last_sync_time = None

    total_results = 1
    start_index = 0
    results_per_page = 500

    stats_nvd_total = 0
    stats_nvd_included = 0
    stats_nvd_ignored = 0

    if last_sync_time:
        UI.info(
            f"Cache detected. Syncing updates modified since [bold]{last_sync_time}[/bold]..."
        )
    else:
        UI.info(
            "Initial Data Population: Fetching ALL target project CVEs (this will take a few minutes)..."
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        fetch_task = progress.add_task("[cyan]Downloading NVD CVE pages...", total=None)

        while start_index < total_results:
            params: Dict[str, Any] = {
                "startIndex": start_index,
                "resultsPerPage": results_per_page,
                "virtualMatchString": f"{get_default_cpe()}:*:*:*:*:*:*:*:*",
            }
            if last_sync_time:
                params["lastModStartDate"] = last_sync_time
                params["lastModEndDate"] = current_time_str

            headers = {}
            nvd_key = _get_nvd_api_key()
            if nvd_key:
                headers["apiKey"] = nvd_key

            time.sleep(6)

            try:
                response = requests.get(
                    NVD_API_BASE_URL, params=params, headers=headers, timeout=30
                )
                response.raise_for_status()
                data = response.json()

                if start_index == 0:
                    total_results = data.get("totalResults", 0)
                    progress.update(fetch_task, total=total_results)

                vulns = data.get("vulnerabilities", [])

                valid_vulns = [v for v in vulns if is_project_vulnerable(v)]

                stats_nvd_total += len(vulns)
                stats_nvd_included += len(valid_vulns)
                stats_nvd_ignored += len(vulns) - len(valid_vulns)

                os.makedirs(CACHE_DIR, exist_ok=True)
                for vuln_item in valid_vulns:
                    cve_id = vuln_item.get("cve", {}).get("id")
                    if cve_id:
                        cache_file_path = os.path.join(CACHE_DIR, f"{cve_id}.json")
                        with open(cache_file_path, "w", encoding="utf-8") as file:
                            json.dump(vuln_item, file, indent=4)
                        pub = vuln_item.get("cve", {}).get("published")
                        if pub:
                            date_index[cve_id] = pub.split("T")[0]

                start_index += len(vulns)
                progress.update(fetch_task, completed=start_index)

                if len(vulns) == 0:
                    break

            except requests.RequestException as error:
                UI.error(f"Failed to fetch from NVD: {error}")
                break

    if stats_nvd_total > 0:
        UI.info(
            f"[bold]NVD Filter Statistics:[/bold] "
            f"Total received: {stats_nvd_total} | "
            f"Included (Project): [green]{stats_nvd_included}[/green] | "
            f"Ignored: [yellow]{stats_nvd_ignored}[/yellow]"
        )

    if start_index >= total_results or total_results == 0:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump({"last_sync_time": current_time_str}, f)
        except:
            pass

    UI.step("Filtering local cache for all CVEs...")
    all_vulnerabilities = []

    stats_cache_total = 0

    if os.path.exists(CACHE_DIR):
        for filename in os.listdir(CACHE_DIR):
            if not filename.endswith(".json") or filename == ".metadata.json" or filename == ".date_index.json":
                continue
            try:
                cache_file_path = os.path.join(CACHE_DIR, filename)
                with open(cache_file_path, "r", encoding="utf-8") as f:
                    vuln_item = json.load(f)

                if "vulnerabilities" in vuln_item and isinstance(
                    vuln_item["vulnerabilities"], list
                ):
                    vuln_item = vuln_item["vulnerabilities"][0]

                stats_cache_total += 1

                all_vulnerabilities.append({"cve": vuln_item.get("cve", {})})
            except Exception:
                pass

    UI.info(
        f"[bold]Cache Statistics:[/bold] "
        f"Total loaded from cache: {stats_cache_total}"
    )

    if date_index:
        try:
            with open(date_index_path, "w", encoding="utf-8") as f:
                json.dump(date_index, f, indent=2)
        except Exception:
            pass

    if not all_vulnerabilities:
        UI.warning("No CVEs found in the local cache.")
        return

    UI.success(
        f"Matched {len(all_vulnerabilities)} CVEs in the local cache. Scanning hashes..."
    )

    cves_with_hashes = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        scan_task = progress.add_task(
            "[cyan]Scanning references...", total=len(all_vulnerabilities)
        )

        for vuln_item in all_vulnerabilities:
            cve_data = vuln_item.get("cve", {})
            cve_id = cve_data.get("id")

            if not cve_id:
                progress.advance(scan_task)
                continue

            references = cve_data.get("references", [])
            hashes = set()
            for ref in references:
                url = ref.get("url", "")
                h = extract_hash_from_url(url)
                if h:
                    hashes.add(h)

            if hashes:
                cves_with_hashes.append((cve_id, list(hashes)))

            progress.advance(scan_task)

    if not cves_with_hashes:
        UI.warning("None of the fetched CVEs contained target project fixing commits.")
        return

    UI.success(f"Discovered {len(cves_with_hashes)} CVEs with project patches.")

    extracted_count = _run_extraction_phase(cves_with_hashes)
    UI.success(f"{extracted_count} CVEs were successfully saved to raw_data.")
