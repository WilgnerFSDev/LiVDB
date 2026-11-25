import argparse
import glob
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from main import main


def discover_dataset(cli_dataset=None):
    candidates = [
        os.path.join(current_dir, "..", "preproc_datasets"),
        os.path.join(current_dir, "preproc_datasets"),
    ]
    preproc_dir = None
    for c in candidates:
        if os.path.isdir(c):
            preproc_dir = c
            break

    model_suffix = "_devign_canonical"
    default_livdb = "linux_kernel_2015-01-01_2026-09-10_ast_curated_balanced1to1"

    if cli_dataset:
        if preproc_dir:
            for d in os.listdir(preproc_dir):
                if d == cli_dataset or d.replace(model_suffix, "") == cli_dataset:
                    return cli_dataset, os.path.join(preproc_dir, d)
        return cli_dataset, os.path.join(preproc_dir or "preproc_datasets", f"{cli_dataset}{model_suffix}")

    if not preproc_dir:
        return default_livdb, os.path.join("preproc_datasets", f"{default_livdb}{model_suffix}")

    available = sorted([
        d for d in os.listdir(preproc_dir)
        if os.path.isdir(os.path.join(preproc_dir, d)) and d.endswith(model_suffix)
    ])
    if not available:
        available = sorted([
            d for d in os.listdir(preproc_dir)
            if os.path.isdir(os.path.join(preproc_dir, d)) and os.path.exists(os.path.join(preproc_dir, d, "train_GGNNinput.jsonl"))
        ])

    if not available:
        return default_livdb, os.path.join(preproc_dir, f"{default_livdb}{model_suffix}")

    available.sort(key=lambda x: (
        0 if "linux_kernel" in x.lower() or "livdb" in x.lower() else 1,
        x
    ))

    if len(available) == 1:
        chosen = available[0]
        ds_name = chosen.replace(model_suffix, "")
        print(f"\n[Dataset Auto-Detection] Found single preprocessed dataset: {chosen}")
        return ds_name, os.path.join(preproc_dir, chosen)

    print("\n" + "=" * 65)
    print("  Preprocessed Datasets Available in preproc_datasets/:")
    print("=" * 65)
    for idx, d in enumerate(available, start=1):
        clean_name = d.replace(model_suffix, "")
        tag = " (LiVDB - Default)" if idx == 1 else ""
        print(f"  [{idx}] {clean_name}{tag}")
    print("=" * 65)

    try:
        choice = input(f"Select dataset to train [1-{len(available)}, default: 1 (LiVDB)]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    if choice.isdigit() and 1 <= int(choice) <= len(available):
        chosen = available[int(choice) - 1]
    else:
        matched = [d for d in available if choice and (choice == d or choice == d.replace(model_suffix, ""))]
        if matched:
            chosen = matched[0]
        else:
            chosen = available[0]

    ds_name = chosen.replace(model_suffix, "")
    print(f"✔ Selected dataset: {ds_name}\n")
    return ds_name, os.path.join(preproc_dir, chosen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Devign Canonical Training (Zhou et al. 2019)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g. linux_kernel_2015-01-01_2026-09-10_ast_curated_balanced1to1)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to preproc_datasets folder containing train/valid GGNNinput.jsonl")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: 42 for LiVDB, 1000 for author datasets)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (default: auto based on GPU VRAM)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 0.0001)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience (default: 100)")
    parser.add_argument("--num_steps", type=int, default=6,
                        help="GGNN timesteps (default: 6)")
    parser.add_argument("--feature_size", type=int, default=100,
                        help="Feature dimension d (default: 100)")
    parser.add_argument("--graph_embed_size", type=int, default=200,
                        help="Graph embedding size (default: 200)")

    cli_args, remaining = parser.parse_known_args()

    detected_name, detected_dir = discover_dataset(cli_args.dataset)
    cli_args.dataset = detected_name
    if cli_args.input_dir is None and detected_dir:
        cli_args.input_dir = detected_dir

    if cli_args.seed is None:
        if "linux_kernel" in cli_args.dataset or "livdb" in cli_args.dataset:
            cli_args.seed = 42
        else:
            cli_args.seed = 1000

    print(f"Dataset resolved: {cli_args.dataset}")
    print(f"Seed resolved   : {cli_args.seed}")

    forward_args = ["--config", "devign", "--dataset", cli_args.dataset, "--seed", str(cli_args.seed)]
    if cli_args.input_dir:
        forward_args.extend(["--input_dir", cli_args.input_dir])
    if cli_args.batch_size:
        forward_args.extend(["--batch_size", str(cli_args.batch_size)])
    if cli_args.lr:
        forward_args.extend(["--lr", str(cli_args.lr)])
    if cli_args.patience:
        forward_args.extend(["--patience", str(cli_args.patience)])
    forward_args.extend(remaining)

    main(forward_args)
