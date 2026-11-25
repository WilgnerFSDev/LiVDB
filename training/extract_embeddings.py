import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum
from modules.reveal_mlp import ReVealMLP


def resolve_src(input_dir, split):
    if not os.path.exists(input_dir):
        for candidate in [os.path.join('..', input_dir), os.path.join('..', 'preproc_datasets', input_dir)]:
            if os.path.exists(candidate):
                input_dir = candidate
                break
    for ext in [".jsonl", ".json"]:
        path = os.path.join(input_dir, f"{split}_GGNNinput{ext}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No GGNNinput file found for split: {split} in directory: {input_dir}")


def extract():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["devign", "reveal"], default="devign",
                        help="Hyperparameter preset to align with main.py training.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to load.")
    parser.add_argument(
        "--model_type", type=str, choices=["ggnn", "devign"], default="devign"
    )
    parser.add_argument(
        "--phase", type=str, choices=["ggnn", "reveal"], required=True,
        help="Extract Phase I (ggnn) or Phase II (reveal) embeddings."
    )
    parser.add_argument("--ggnn_model", type=str, default=None,
                        help="Optional explicit path to Phase I GGNN weights (required for phase=reveal).")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "valid", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--feature_size", type=int, default=130)
    parser.add_argument("--graph_embed_size", type=int, default=200)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--node_tag", type=str, default="node_features")
    parser.add_argument("--graph_tag", type=str, default="graph")
    parser.add_argument("--label_tag", type=str, default="targets")
    parser.add_argument("--output", type=str, default="embeddings.npz")
    args = parser.parse_args()

    CONFIGS = {
        "devign": {"num_steps": 6, "batch_size": 128},
        "reveal": {"num_steps": 8, "batch_size": 128},
    }
    cfg = CONFIGS[args.config]
    if args.num_steps is None:
        args.num_steps = cfg["num_steps"]
    if args.batch_size is None:
        args.batch_size = cfg["batch_size"]

    print(f"Using config: {args.config.upper()} (num_steps={args.num_steps})")

    dataset = DataSet(
        train_src=resolve_src(args.input_dir, "train"),
        valid_src=resolve_src(args.input_dir, "valid"),
        test_src=resolve_src(args.input_dir, "test"),
        batch_size=args.batch_size,
        n_ident=args.node_tag,
        g_ident=args.graph_tag,
        l_ident=args.label_tag,
    )

    if args.split == "train":
        num_batches = dataset.initialize_train_batch()
        get_batch = dataset.get_next_train_batch
    elif args.split == "valid":
        num_batches = dataset.initialize_valid_batch()
        get_batch = dataset.get_next_valid_batch
    else:
        num_batches = dataset.initialize_test_batch()
        get_batch = dataset.get_next_test_batch

    if args.model_type == "ggnn":
        ggnn = GGNNSum(
            input_dim=args.feature_size,
            output_dim=args.graph_embed_size,
            num_steps=args.num_steps,
            max_edge_types=dataset.max_edge_type,
        )
    else:
        ggnn = DevignModel(
            input_dim=args.feature_size,
            output_dim=args.graph_embed_size,
            num_steps=args.num_steps,
            max_edge_types=dataset.max_edge_type,
        )

    if args.phase == "reveal":
        ggnn_path = args.ggnn_model
        if not ggnn_path:
            dir_name = os.path.dirname(args.model_path)
            options = ["GGNNSum-model.bin", "GGNNSumModel-model.bin", "DevignModel-model.bin"]
            for opt in options:
                potential = os.path.join(dir_name, opt)
                if os.path.exists(potential):
                    ggnn_path = potential
                    break

        if not ggnn_path or not os.path.exists(ggnn_path):
             print(f"Error: Phase II extraction requires Phase I GGNN weights. "
                   f"Please provide --ggnn_model explicitly.")
             return

        print(f"Loading Phase I GGNN base weights from: {ggnn_path}")
        ggnn.load_state_dict(torch.load(ggnn_path))

        mlp = ReVealMLP(input_dim=args.graph_embed_size)
        print(f"Loading Phase II ReVeal MLP weights from: {args.model_path}")
        mlp.load_state_dict(torch.load(args.model_path))
        mlp.cuda()
        mlp.eval()
    else:
        print(f"Loading GGNN weights from: {args.model_path}")
        ggnn.load_state_dict(torch.load(args.model_path))

    ggnn.cuda()
    ggnn.eval()

    embeddings = []
    labels_list = []

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Extracting"):
            batch, labels = get_batch()
            xg = ggnn.forward_embedding(batch, cuda=True)

            if args.phase == "reveal":
                h_xg = mlp.transform_only(xg)
                embeddings.append(h_xg.cpu().numpy())
            else:
                embeddings.append(xg.cpu().numpy())

            labels_list.append(labels.cpu().numpy())

    emb_arr = np.concatenate(embeddings)
    lab_arr = np.concatenate(labels_list)

    np.savez(args.output, embeddings=emb_arr, labels=lab_arr)
    print(f"Saved {len(emb_arr)} embeddings to {args.output}")


if __name__ == "__main__":
    extract()
