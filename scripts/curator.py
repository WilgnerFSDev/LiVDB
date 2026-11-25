import hashlib
import os

import pandas as pd

from scripts.ui import UI


def curate_dataset(input_csv_path: str) -> str | None:
    if not os.path.exists(input_csv_path):
        UI.error(f"Dataset file not found: {input_csv_path}")
        return None

    UI.header("Global Dataset Curation")
    UI.info(f"Loading dataset: {os.path.basename(input_csv_path)}")

    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        UI.error(f"Failed to read CSV: {e}")
        return None

    original_size = len(df)
    if original_size == 0:
        UI.warning("Dataset is empty. Skipping curation.")
        return input_csv_path


    UI.info("Calculating semantic hashes and removing duplicates...")

    from scripts.utils import strip_c_comments

    def generate_hash(code):
        if not isinstance(code, str):
            return ""
        clean_code = strip_c_comments(code)
        normalized = "".join(clean_code.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    df["hash"] = df["function"].apply(generate_hash)
    df["_orig_idx"] = df.index

    df = df.sort_values(by="label", ascending=False)

    df_curated = (
        df.drop_duplicates(subset=["hash"], keep="first")
        .sort_values(by="_orig_idx")
        .drop(columns=["_orig_idx"])
        .reset_index(drop=True)
        .copy()
    )

    n_vuln = len(df_curated[df_curated["label"] == 1])
    n_clean = len(df_curated[df_curated["label"] == 0])
    total_dedup = n_vuln + n_clean
    pct_vuln_dedup = (n_vuln / total_dedup * 100) if total_dedup > 0 else 0
    pct_clean_dedup = (n_clean / total_dedup * 100) if total_dedup > 0 else 0

    UI.info(
        f"Current balance after deduplication: "
        f"[bold red]{n_vuln} vulnerable ({pct_vuln_dedup:.2f}%)[/bold red] vs "
        f"[bold green]{n_clean} clean ({pct_clean_dedup:.2f}%)[/bold green]"
    )

    choices = [
        ("0", "No balancing", "Keep all cleaned functions."),
        ("1", "1:1 Ratio", f"Keep {n_vuln} clean functions (Total: {n_vuln*2})."),
        ("2", "1:2 Ratio", f"Keep {n_vuln*2} clean functions (Total: {n_vuln*3})."),
        ("3", "1:10 Ratio", f"Keep {n_vuln*10} clean functions (Total: {n_vuln*11}).")
    ]

    balance_choice = UI.ask_choice("Do you want to balance the dataset (Undersample clean functions)?", choices)

    undersampling_removed = 0
    if balance_choice != "0":
        ratio = 1 if balance_choice == "1" else (2 if balance_choice == "2" else 10)
        target_clean = n_vuln * ratio

        seed_str = UI.ask_string("Random seed for sampling (press Enter for default 42, or 0 for ISSTA'24 paper):", default="42")
        try:
            seed = int(seed_str.strip())
        except ValueError:
            seed = 42

        if target_clean < n_clean:
            df_v = df_curated[df_curated["label"] == 1]
            df_c = df_curated[df_curated["label"] == 0]

            df_c_sampled = df_c.sample(n=target_clean, random_state=seed)
            undersampling_removed = n_clean - target_clean

            df_curated = pd.concat([df_v, df_c_sampled]).sample(frac=1, random_state=seed).reset_index(drop=True)
            UI.success(f"Undersampling applied: Removed {undersampling_removed} clean functions (seed={seed}).")
        else:
            UI.warning(f"Target clean count ({target_clean}) is greater than available clean functions ({n_clean}). Skipping undersampling.")
    else:
        if UI.ask_boolean("Do you want to shuffle the dataset?", default=False):
            seed_str = UI.ask_string(
                "Random seed for shuffling (press Enter for default 42, or leave blank for random):",
                default="42",
            )
            val = seed_str.strip()
            if not val:
                seed = None
            else:
                try:
                    seed = int(val)
                except ValueError:
                    UI.warning(f"Invalid seed '{val}'. Falling back to random shuffle.")
                    seed = None

            df_curated = df_curated.sample(frac=1, random_state=seed).reset_index(drop=True)
            if seed is not None:
                UI.success(f"Dataset shuffled (seed={seed}).")
            else:
                UI.success("Dataset shuffled randomly (no seed).")
        else:
            UI.info("Maintained original extraction order.")

    final_size = len(df_curated)
    duplicates_removed = original_size - (final_size + undersampling_removed)
    vuln_count = len(df_curated[df_curated["label"] == 1])
    clean_count = len(df_curated[df_curated["label"] == 0])
    vuln_percentage = (vuln_count / final_size * 100) if final_size > 0 else 0
    clean_percentage = (clean_count / final_size * 100) if final_size > 0 else 0

    base, ext = os.path.splitext(input_csv_path)
    output_path = f"{base}_curated{ext}"

    if "hash" in df_curated.columns:
        df_curated = df_curated.drop(columns=["hash"])
    df_curated.to_csv(output_path, index=False)

    UI.success(f"Curation complete! Saved to: {os.path.basename(output_path)}")

    UI.info("── Curation Statistics ──────────────────────────────")
    UI.info(f"Original size            : {original_size}")
    UI.info(f"Duplicates removed       : {duplicates_removed}")
    UI.info(f"Undersampling removed     : {undersampling_removed}")
    UI.info(f"Final size               : {final_size}")
    UI.info(f"Label 1 (Vulnerable)     : [bold red]{vuln_count}[/bold red] ({vuln_percentage:.2f}%)")
    UI.info(f"Label 0 (Clean/Patched)  : [bold green]{clean_count}[/bold green] ({clean_percentage:.2f}%)")
    UI.info("─────────────────────────────────────────────────────")

    return output_path
