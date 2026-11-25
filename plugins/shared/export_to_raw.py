import csv
import hashlib
import os
import shutil
import sys
import time


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.ui import UI
from scripts.utils import DATASETS_DIR

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)


def calculate_file_hash(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def export_to_raw(output_dir: str = "raw_code", dataset_path: str = None):

    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_ROOT, output_dir)

    resolved_dataset_path = (
        dataset_path if dataset_path else os.path.join(DATASETS_DIR, "dataset.csv")
    )
    if not os.path.isabs(resolved_dataset_path):
        resolved_dataset_path = os.path.join(_ROOT, resolved_dataset_path)

    output_csv_dir = os.path.dirname(resolved_dataset_path)

    if not os.path.exists(output_csv_dir):
        UI.error(f"Output directory not found: {output_csv_dir}")
        return

    csv_files = [f for f in os.listdir(output_csv_dir) if f.endswith(".csv")]

    if not csv_files:
        UI.error(
            f"No CSV datasets found in {output_csv_dir}. Please build the dataset first."
        )
        return

    if dataset_path:
        resolved_dataset_path = (
            dataset_path
            if os.path.isabs(dataset_path)
            else os.path.join(_ROOT, dataset_path)
        )
        UI.info(
            f"Using dataset: [bold cyan]{os.path.basename(resolved_dataset_path)}[/bold cyan]"
        )
    elif len(csv_files) == 1:
        resolved_dataset_path = os.path.join(output_csv_dir, csv_files[0])
        UI.info(f"Found single dataset: [bold cyan]{csv_files[0]}[/bold cyan]")
    else:
        choices = []
        for idx, filename in enumerate(csv_files):
            file_path = os.path.join(output_csv_dir, filename)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            choices.append((str(idx + 1), filename, f"Size: {size_mb:.2f} MB"))

        choices.append(("0", "Cancel", "Return to menu"))

        result = UI.ask_choice("Select dataset to export:", choices)

        if result == "0":
            return

        selected_index = int(result) - 1
        resolved_dataset_path = os.path.join(output_csv_dir, csv_files[selected_index])

    current_hash = calculate_file_hash(resolved_dataset_path)
    hash_file_path = os.path.join(output_dir, ".dataset_hash")
    existing_hash = ""

    if os.path.exists(hash_file_path):
        with open(hash_file_path, "r", encoding="utf-8") as hf:
            existing_hash = hf.read().strip()

    if current_hash == existing_hash:
        UI.step("Dataset hash matches existing raw_code/ folder. No changes detected.")
        force_reexport = UI.ask_boolean(
            "Force re-export of dataset.csv to raw_code/ anyway?", default=False
        )
        if not force_reexport:
            UI.success("Skipping export (using cached raw_code/).")
            return True
    else:
        if existing_hash:
            UI.warning("Dataset has changed (hash mismatch).")
        else:
            UI.info("No previous export metadata found.")

        proceed = UI.ask_boolean("Proceed with fresh export?", default=True)
        if not proceed:
            UI.info("Export cancelled by user.")
            return False

    if os.path.exists(output_dir):
        UI.info(f"Clearing existing files in {output_dir}...")
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                UI.error(f"Failed to delete {file_path}: {e}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(resolved_dataset_path, "r", encoding="utf-8") as f:
            total_rows = sum(1 for _ in csv.DictReader(f))
    except Exception as e:
        UI.error(f"Failed to read dataset: {e}")
        return

    UI.header("Exporting Functions to Raw Source Files")
    UI.info(f"Source: {resolved_dataset_path}")
    UI.info(f"Destination: {output_dir}")
    UI.info(f"Total functions: {total_rows}")

    exported_count = 0
    skipped_count = 0

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    start_time = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Exporting functions...", total=total_rows)

        with open(resolved_dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for index, row in enumerate(reader):
                try:
                    cve_id = row.get("cve_id", "unknown")
                    label = row.get("label", "0")
                    function_code = row.get("function", "")

                    if not function_code:
                        skipped_count += 1
                        progress.update(task, advance=1)
                        continue

                    orig_filename = row.get("filename", "unknown.c")
                    ext = os.path.splitext(orig_filename)[1] or ".c"

                    filename = f"{index}_{cve_id}_{label}{ext}"
                    filepath = os.path.join(output_dir, filename)

                    with open(
                        filepath, "w", encoding="utf-8", buffering=1 << 16
                    ) as out_f:
                        out_f.write(function_code)

                    exported_count += 1
                    progress.update(task, advance=1)

                except Exception:
                    skipped_count += 1
                    progress.update(task, advance=1)

    with open(hash_file_path, "w", encoding="utf-8") as hf:
        hf.write(current_hash)

    elapsed = time.time() - start_time

    UI.success("Export complete!")
    UI.info(f"  - Exported: {exported_count}")
    UI.info(f"  - Skipped:  {skipped_count}")
    UI.info(f"  - Location: {output_dir}")
    UI.info(f"  - Time:     [yellow]{elapsed:.1f}s[/yellow]")
    return True


if __name__ == "__main__":
    export_to_raw()
