import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.ui import UI

RAW_CODE_DIR = os.path.join(_ROOT, "raw_code")

RAM_SAFETY_MARGIN = 0.20
DEFAULT_RAM_MB = 1024
OOM_RAM_BOOST_FACTOR = 1.5
MAX_RETRIES_OOM = 1
MAX_RETRIES_OTHER = 1

FILES_PER_MB_ESTIMATE = 4.0

FALLBACK_BATCH_SIZE = 500
FALLBACK_MAX_RAM_MB = 512
FALLBACK_WORKERS = 2

OOM_KEYWORDS = (
    "OutOfMemoryError",
    "GC overhead limit",
    "java.lang.OutOfMemory",
    "Cannot allocate memory",
    "GC overhead",
)


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def _get_free_ram_mb() -> int:
    if HAS_PSUTIL:
        return int(psutil.virtual_memory().available / 1024 / 1024)
    return FALLBACK_MAX_RAM_MB


def _compute_plan(
    total_files: int,
    ram_per_worker_mb: int,
    manual_workers: Optional[int] = None,
    safe_margin: float = 0.20,
    manual_batch_size: Optional[int] = None,
) -> dict:
    free_mb = _get_free_ram_mb()
    usable_mb = int(free_mb * (1 - safe_margin))

    if not HAS_PSUTIL:
        UI.warning(
            "psutil not installed - using fallback. "
            "Run 'pip install psutil' for automatic RAM detection."
        )
        return {
            "ram_per_worker_mb": ram_per_worker_mb,
            "max_workers": manual_workers if manual_workers else FALLBACK_WORKERS,
            "batch_size": manual_batch_size if manual_batch_size else FALLBACK_BATCH_SIZE,
            "free_mb": FALLBACK_MAX_RAM_MB,
            "usable_mb": FALLBACK_MAX_RAM_MB,
            "is_manual_workers": manual_workers is not None,
            "is_manual_batch": manual_batch_size is not None,
        }

    if manual_workers:
        max_workers = manual_workers
    else:
        max_workers = max(1, usable_mb // ram_per_worker_mb)
        cpu_limit = os.cpu_count() or 1
        max_workers = min(max_workers, cpu_limit)

    ram_per_worker_mb = usable_mb // max_workers

    if manual_batch_size:
        batch_size = manual_batch_size
    else:
        safe_batch_limit = int(ram_per_worker_mb * FILES_PER_MB_ESTIMATE)
        ideal_batch = math.ceil(total_files / max_workers)
        batch_size = max(300, min(ideal_batch, safe_batch_limit))

    return {
        "ram_per_worker_mb": ram_per_worker_mb,
        "max_workers": max_workers,
        "batch_size": batch_size,
        "free_mb": free_mb,
        "usable_mb": usable_mb,
        "is_manual_workers": manual_workers is not None,
        "is_manual_batch": manual_batch_size is not None,
    }


def check_dependencies() -> bool:
    if shutil.which("java") is None:
        UI.error("Java is not installed or not in PATH. Joern requires Java.")
        return False
    if shutil.which("joern-parse") is None:
        UI.error("'joern-parse' command not found in PATH.")
        UI.info("Please ensure Joern is installed and available in your shell.")
        return False
    return True


def _run_batch_once(
    batch_index: int,
    batch_files: list,
    input_dir: str,
    ram_mb: int,
) -> tuple:
    import sys

    current_module = sys.modules[__name__]
    output_dir = getattr(
        current_module, "CURRENT_OUTPUT_DIR", os.path.join(_ROOT, "joern_output")
    )

    temp_dir = os.path.join(output_dir, f"temp_raw_code_batch_{batch_index}")
    output_file = os.path.join(output_dir, f"cpg_batch_{batch_index}.bin")

    env = os.environ.copy()
    xms = max(256, ram_mb // 4)
    jvm_flags = f"-Xms{xms}m -Xmx{ram_mb}m -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
    env["JOERN_JAVA_OPTS"] = f"{jvm_flags} --enable-native-access=ALL-UNNAMED"
    env["JAVA_TOOL_OPTIONS"] = f"-Xms{xms}m -Xmx{ram_mb}m -XX:+UseG1GC"

    try:
        os.makedirs(temp_dir, exist_ok=True)
        for f in batch_files:
            src = os.path.join(input_dir, f)
            dst = os.path.join(temp_dir, f)
            try:
                os.symlink(src, dst)
            except (OSError, NotImplementedError):
                shutil.copy(src, dst)

        with subprocess.Popen(
            [
                "joern-parse",
                f"-J-Xms{xms}m",
                f"-J-Xmx{ram_mb}m",
                temp_dir,
                "--output",
                output_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        ) as proc:
            combined = proc.stdout.read()
            proc.wait()

        is_oom = any(kw in combined for kw in OOM_KEYWORDS)
        return (proc.returncode == 0), combined, is_oom

    except Exception as exc:
        return False, str(exc), False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _run_batch_with_retry(
    batch_index: int,
    batch_files: list,
    input_dir: str,
    ram_mb: int,
    ram_override: list,
    skip_set: set,
    lock: threading.Lock,
) -> tuple:
    start_t = time.time()
    import sys

    current_module = sys.modules[__name__]
    output_dir = getattr(
        current_module, "CURRENT_OUTPUT_DIR", os.path.join(_ROOT, "joern_output")
    )
    output_file = os.path.join(output_dir, f"cpg_batch_{batch_index}.bin")
    if os.path.exists(output_file):
        return batch_index, True, None, False, 0.0

    current_ram = ram_mb
    oom_retries = 0
    other_retries = 0

    while True:
        with lock:
            if batch_index in skip_set:
                return batch_index, False, "skipped", True, time.time() - start_t
            if ram_override[0] is not None:
                current_ram = ram_override[0]

        success, output, is_oom = _run_batch_once(
            batch_index, batch_files, input_dir, current_ram
        )

        if success:
            return batch_index, True, None, False, time.time() - start_t

        if is_oom and oom_retries < MAX_RETRIES_OOM:
            boosted = int(current_ram * OOM_RAM_BOOST_FACTOR)
            UI.warning(
                f"  Batch {batch_index + 1}: OOM - auto-retry with {boosted} MB..."
            )
            current_ram = boosted
            oom_retries += 1
            continue

        if not is_oom and other_retries < MAX_RETRIES_OTHER:
            UI.warning(f"  Batch {batch_index + 1}: failed - auto-retry (1x)...")
            other_retries += 1
            continue

        return batch_index, False, ("OOM" if is_oom else "error"), False, time.time() - start_t


def generate_cpg(input_dir: str = RAW_CODE_DIR) -> bool:
    if not check_dependencies():
        return False

    if not os.path.exists(input_dir):
        UI.error(f"Input directory not found: {input_dir}")
        UI.info("Run 'Export dataset to raw .c files' first.")
        return False

    all_files = [f for f in os.listdir(input_dir) if f.endswith(".c")]
    if not all_files:
        UI.error(f"No .c files found in {input_dir}")
        return False

    total_files = len(all_files)

    import sys
    current_module = sys.modules[__name__]
    output_dir = getattr(
        current_module, "CURRENT_OUTPUT_DIR", os.path.join(_ROOT, "joern_output")
    )
    os.makedirs(output_dir, exist_ok=True)

    existing_cpgs = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith("cpg_batch_") and f.endswith(".bin")
    ])

    if existing_cpgs:
        UI.warning(f"Found {len(existing_cpgs)} already-processed CPG batch file(s) in {output_dir}.")
        UI.warning("All existing CPGs will be deleted if you choose to regenerate them.")
        answer = (
            UI.ask_string(
                "All existing CPGs will be deleted. Do you want to continue? [y/N]",
                default="N",
            )
            .strip()
            .lower()
        )
        if answer in ("y", "yes", "s", "sim"):
            UI.warning("Deleting all existing CPG files...")
            for f in existing_cpgs:
                fp = os.path.join(output_dir, f)
                try:
                    os.remove(fp)
                except OSError:
                    pass
            UI.info("All existing CPGs deleted. Processing will start from scratch.")
        else:
            UI.info("Keeping existing CPGs. Skipping CPG generation.")
            return True

    print()
    raw_ram = input(
        f"  RAM per Joern worker in MB [default: {DEFAULT_RAM_MB}]: "
    ).strip()
    try:
        user_ram_mb = int(raw_ram) if raw_ram else DEFAULT_RAM_MB
        if user_ram_mb < 64:
            UI.warning("Value too low, using 64 MB minimum.")
            user_ram_mb = 64
    except ValueError:
        UI.warning(f"Invalid value, using {DEFAULT_RAM_MB} MB.")
        user_ram_mb = DEFAULT_RAM_MB

    raw_safe = input("  Safety RAM margin % [default: 20]: ").strip()
    try:
        safe_margin = float(raw_safe) / 100.0 if raw_safe else 0.20
        if safe_margin < 0 or safe_margin >= 1.0:
            UI.warning("Invalid margin, using 20%.")
            safe_margin = 0.20
    except ValueError:
        UI.warning("Invalid margin, using 20%.")
        safe_margin = 0.20

    free_mb = _get_free_ram_mb()
    usable_mb = int(free_mb * (1 - safe_margin))
    auto_workers = max(1, usable_mb // user_ram_mb)
    cpu_limit = os.cpu_count() or 1
    auto_workers = min(auto_workers, cpu_limit)

    raw_workers = input(
        f"  Parallel workers [default: {auto_workers} (auto)]: "
    ).strip()
    user_workers = None
    if raw_workers and raw_workers.lower() != "auto":
        try:
            user_workers = int(raw_workers)
            if user_workers < 1:
                UI.warning("Value too low, using at least 1 worker.")
                user_workers = 1
        except ValueError:
            UI.warning("Invalid value for workers, using 'auto'.")
            user_workers = None

    initial_plan = _compute_plan(
        total_files, user_ram_mb, manual_workers=user_workers, safe_margin=safe_margin
    )
    auto_batch = initial_plan["batch_size"]

    raw_batch = input(
        f"  Batch size [default: {auto_batch} (auto)]: "
    ).strip()
    user_batch = None
    if raw_batch and raw_batch.lower() != "auto":
        try:
            user_batch = int(raw_batch)
            if user_batch < 1:
                UI.warning("Value too low, using at least 1 file/batch.")
                user_batch = 1
        except ValueError:
            UI.warning("Invalid value for batch size, using 'auto'.")
            user_batch = None

    plan = _compute_plan(
        total_files,
        user_ram_mb,
        manual_workers=user_workers,
        safe_margin=safe_margin,
        manual_batch_size=user_batch,
    )

    batch_size = plan["batch_size"]
    max_workers = plan["max_workers"]
    ram_per_worker = plan["ram_per_worker_mb"]
    num_batches = math.ceil(total_files / batch_size)

    UI.header("Generating Code Property Graphs (CPG)")
    UI.info(f"Total files       : {total_files}")
    if HAS_PSUTIL:
        UI.info(
            f"Free RAM          : {plan['free_mb']} MB  →  usable {plan['usable_mb']} MB"
        )
    UI.info(f"RAM per worker    : {ram_per_worker} MB  (-J-Xmx{ram_per_worker}m)")

    worker_mode = "(manual)" if plan["is_manual_workers"] else "(auto)"
    UI.info(f"Parallel workers  : {max_workers}  {worker_mode}")
    batch_mode = "(manual)" if plan["is_manual_batch"] else "(auto)"
    UI.info(f"Batch size        : {batch_size} files/batch  {batch_mode}")
    UI.info(f"Total batches     : {num_batches}")

    pending = [
        (i, all_files[i * batch_size : (i + 1) * batch_size])
        for i in range(num_batches)
        if not os.path.exists(os.path.join(output_dir, f"cpg_batch_{i}.bin"))
    ]

    already_done = num_batches - len(pending)

    if not pending:
        UI.success("All batches already processed. Nothing to do.")
        return True

    lock = threading.Lock()
    ram_override = [None]
    skip_set: set = set()

    successful = already_done
    failed_batches = []

    start_time = time.time()
    completed_count = already_done

    def _log_progress(batch_idx: int, ok: bool, duration: float = 0.0, reason: str = "") -> None:
        nonlocal completed_count
        elapsed = time.time() - start_time
        completed_count += 1
        done_now = completed_count
        remaining = num_batches - done_now
        avg = elapsed / max(done_now - already_done, 1)
        eta = avg * remaining

        timing = f"elapsed {_fmt_duration(elapsed)}  ETA {_fmt_duration(eta) if remaining else '-'}"
        batch_timing = f" [{duration:.1f}s]" if duration > 0 else ""
        label = f"Batch {batch_idx + 1}/{num_batches}"

        if ok:
            UI.step(f"{label} ✓{batch_timing}  ({done_now}/{num_batches} done)  [{timing}]")
        else:
            tag = f" [{reason}]" if reason else ""
            UI.error(f"{label} ✗{tag}{batch_timing}  ({done_now}/{num_batches})  [{timing}]")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict = {
            executor.submit(
                _run_batch_with_retry,
                idx,
                files,
                input_dir,
                ram_per_worker,
                ram_override,
                skip_set,
                lock,
            ): (idx, files)
            for idx, files in pending
        }

        for future in as_completed(futures):
            batch_idx, ok, err, skipped, duration = future.result()

            if skipped:
                _log_progress(batch_idx, False, duration, "skipped")
                failed_batches.append((batch_idx, "skipped"))
                continue

            if ok:
                successful += 1
                _log_progress(batch_idx, True, duration)
                continue

            is_oom = err == "OOM"
            _log_progress(batch_idx, False, duration, err)
            failed_batches.append((batch_idx, err))


            if is_oom:
                UI.warning(f"  Batch {batch_idx + 1} failed with OOM after auto-retry.")
                print("    [r] Retry with more RAM (this batch only)")
                print("    [g] Apply new RAM to ALL remaining batches")
                print("    [s] Skip this batch")
                choice = input("  → [r/g/s]: ").strip().lower()

                if choice in ("r", "g"):
                    try:
                        new_ram = int(
                            input(
                                f"  New RAM in MB (current {ram_per_worker}): "
                            ).strip()
                        )
                    except ValueError:
                        new_ram = int(ram_per_worker * OOM_RAM_BOOST_FACTOR)
                        UI.warning(f"  Invalid value, using {new_ram} MB.")

                    if choice == "g":
                        with lock:
                            ram_override[0] = new_ram
                        UI.info(f"  Global RAM updated to {new_ram} MB.")

                    idx_files = futures[future]
                    nf = executor.submit(
                        _run_batch_with_retry,
                        idx_files[0],
                        idx_files[1],
                        input_dir,
                        new_ram,
                        ram_override,
                        skip_set,
                        lock,
                    )
                    futures[nf] = idx_files
                else:
                    with lock:
                        skip_set.add(batch_idx)
                    failed_batches.append((batch_idx, "skipped"))

            else:
                UI.warning(f"  Batch {batch_idx + 1} failed with a generic error.")
                print("    [r] Retry")
                print("    [s] Skip")
                choice = input("  → [r/s]: ").strip().lower()

                if choice == "r":
                    idx_files = futures[future]
                    nf = executor.submit(
                        _run_batch_with_retry,
                        idx_files[0],
                        idx_files[1],
                        input_dir,
                        ram_per_worker,
                        ram_override,
                        skip_set,
                        lock,
                    )
                    futures[nf] = idx_files
                else:
                    with lock:
                        skip_set.add(batch_idx)
                    failed_batches.append((batch_idx, "skipped"))

    total_elapsed = time.time() - start_time
    print()
    UI.info(f"Total time: {_fmt_duration(total_elapsed)}")

    if not failed_batches:
        UI.success(f"All {num_batches} batches processed successfully!")
        return True

    skipped_count = sum(1 for _, r in failed_batches if "skip" in r)
    errors_count = len(failed_batches) - skipped_count
    UI.warning(
        f"Completed {successful}/{num_batches} batches.  "
        f"Errors: {errors_count}  Skipped: {skipped_count}  "
        f"Failed: {[i + 1 for i, _ in sorted(failed_batches)]}"
    )
    return False


if __name__ == "__main__":
    generate_cpg()
