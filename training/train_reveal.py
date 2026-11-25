import argparse
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum
from modules.reveal_loss import ReVealLoss
from modules.reveal_mlp import ReVealMLP
from modules.smote import smote
from trainer import train


def discover_dataset(cli_dataset=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "..", "preproc_datasets"),
        os.path.join(current_dir, "preproc_datasets"),
    ]
    preproc_dir = None
    for c in candidates:
        if os.path.isdir(c):
            preproc_dir = c
            break

    model_suffix = "_reveal_canonical"
    default_livdb = "linux_kernel_2015-01-01_2026-09-10_ast_curated"

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


def resolve_src(input_dir, split):
    for ext in [".jsonl", ".json"]:
        path = os.path.join(input_dir, f"{split}_GGNNinput{ext}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No GGNNinput file found for split '{split}' in: {input_dir}")


def resolve_ggnn_model(path=None, dataset="dataset_reveal"):
    if path and os.path.exists(path):
        return os.path.abspath(path)
    candidates = [
        path,
        os.path.join("models", dataset, "GGNNSum-model.bin"),
        os.path.join("..", "models", dataset, "GGNNSum-model.bin"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", dataset, "GGNNSum-model.bin"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(os.path.join("models", dataset, "GGNNSum-model.bin"))


def train_phase_1(args, dataset, device):
    print("═" * 75)
    print(" [ReVeal Pipeline] Phase I: GGNN Backbone Training (Chakraborty et al. 2020)")
    print(f" Dataset     : {args.dataset}")
    print(f" Model       : GGNNSum (8 layers, tanh, sum pooling)")
    print(f" Node Features: d={dataset.feature_size} (Tv=69 one-hot AST + Cv=100 W2V)")
    print(f" Embed Size  : {args.graph_embed_size}")
    print(f" Learning Rate: 0.0001 (Adam, weight_decay=0.0)")
    print(f" Batch Size  : {args.batch_size}")
    print(f" Early Stop  : patience=50 iterations (validation F1)")
    print("═" * 75)

    model = GGNNSum(
        input_dim=dataset.feature_size,
        output_dim=args.graph_embed_size,
        num_steps=args.num_steps,
        max_edge_types=dataset.max_edge_type,
    ).to(device)

    loss_function = nn.BCELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)

    model_dir = os.path.join("models", args.dataset)
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "GGNNSum")
    dev_every = len(dataset.train_batches)

    train(
        model=model,
        dataset=dataset,
        max_steps=64000,
        dev_every=dev_every,
        loss_function=loss_function,
        optimizer=optimizer,
        save_path=save_path,
        max_patience=50,
        log_every=50,
    )
    checkpoint_file = save_path + "-model.bin"
    print(f"\n✔ Phase I GGNN Backbone trained successfully! Saved to: {checkpoint_file}\n")
    return checkpoint_file


def extract_xg(ggnn, dataset, split="train", cuda=True):
    ggnn.eval()
    if split == "train":
        num_batches = dataset.initialize_train_batch()
        get_batch = dataset.get_next_train_batch
    elif split == "valid":
        num_batches = dataset.initialize_valid_batch()
        get_batch = dataset.get_next_valid_batch
    else:
        num_batches = dataset.initialize_test_batch()
        get_batch = dataset.get_next_test_batch

    all_xg = []
    all_labels = []

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc=f"Extracting xg ({split})"):
            batch, labels = get_batch()
            if cuda:
                labels = labels.cuda()
            xg = ggnn.forward_embedding(batch, cuda=cuda)
            all_xg.append(xg.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_xg), np.concatenate(all_labels)


def train_reveal():
    parser = argparse.ArgumentParser(
        description="Train ReVeal Canonical Pipeline (IEEE TSE 2020) with default parameters."
    )
    parser.add_argument("--config", choices=["devign", "reveal"], default="reveal",
                        help="Hyperparameter preset (default: reveal).")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (default: auto-detected, LiVDB default).")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing preprocessed dataset.")
    parser.add_argument("--ggnn_model", type=str, default=None,
                        help="Path to pre-trained GGNN checkpoint. If not found, Phase I will train automatically.")
    parser.add_argument("--force_retrain_ggnn", action="store_true",
                        help="Force retraining Phase I GGNN backbone even if checkpoint exists.")
    parser.add_argument("--model_type", type=str, choices=["devign", "ggnn"], default="ggnn")
    parser.add_argument("--feature_size", type=int, default=169,
                        help="Size of node feature vector (default: 169 = Tv 69 + Cv 100).")
    parser.add_argument("--graph_embed_size", type=int, default=200,
                        help="Size of graph embedding (default: 200).")
    parser.add_argument("--num_steps", type=int, default=8,
                        help="Number of GGNN timesteps (default: 8).")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (default: 128 for >=6GB VRAM, 64 for 4GB).")
    parser.add_argument("--node_tag", type=str, default="node_features")
    parser.add_argument("--graph_tag", type=str, default="graph")
    parser.add_argument("--label_tag", type=str, default="targets")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: 42 for LiVDB, 1000 for official ReVeal repo).")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs for Phase II MLP (default: 100).")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience for Phase II (default: 5 consecutive epochs).")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Phase II learning rate (default: 0.001).")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for classification loss (default: 0.5).")
    parser.add_argument("--beta", type=float, default=0.001,
                        help="Weight for L2 regularizer (default: 0.001).")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Weight for triplet margin loss (default: 0.5).")
    parser.add_argument("--smote_k", type=int, default=5,
                        help="SMOTE nearest neighbors k (default: 5).")
    args = parser.parse_args()

    detected_name, detected_dir = discover_dataset(args.dataset)
    args.dataset = detected_name
    input_dir = args.input_dir or detected_dir

    if args.seed is None:
        if "linux_kernel" in args.dataset or "livdb" in args.dataset:
            args.seed = 42
        else:
            args.seed = 1000

    print(f"Dataset resolved: {args.dataset}")
    print(f"Seed resolved   : {args.seed}")
    print(f"Dataset directory resolved : {input_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.batch_size is None:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            args.batch_size = 128 if vram_gb >= 6.0 else 64
        else:
            args.batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()

    print(f"Loading dataset from {input_dir}...")
    dataset = DataSet(
        train_src=resolve_src(input_dir, "train"),
        valid_src=resolve_src(input_dir, "valid"),
        test_src=resolve_src(input_dir, "test"),
        batch_size=args.batch_size,
        n_ident=args.node_tag,
        g_ident=args.graph_tag,
        l_ident=args.label_tag,
    )

    ggnn_checkpoint = resolve_ggnn_model(args.ggnn_model, args.dataset)
    if not os.path.exists(ggnn_checkpoint) or args.force_retrain_ggnn:
        print(f"\n[ReVeal Pipeline] Pre-trained GGNN backbone not found at: {ggnn_checkpoint}")
        print("[ReVeal Pipeline] Automatically starting Phase I training first...\n")
        ggnn_checkpoint = train_phase_1(args, dataset, device)
    else:
        print(f"\n✔ [ReVeal Pipeline] Found pre-trained GGNN backbone at: {ggnn_checkpoint}")
        print("[ReVeal Pipeline] Skipping Phase I training, proceeding to Phase II.\n")

    if args.model_type == "ggnn":
        ggnn = GGNNSum(
            input_dim=dataset.feature_size,
            output_dim=args.graph_embed_size,
            num_steps=args.num_steps,
            max_edge_types=dataset.max_edge_type,
        )
    else:
        ggnn = DevignModel(
            input_dim=dataset.feature_size,
            output_dim=args.graph_embed_size,
            num_steps=args.num_steps,
            max_edge_types=dataset.max_edge_type,
        )

    print(f"Loading pre-trained {args.model_type.upper()} weights from {ggnn_checkpoint}...")
    ggnn.load_state_dict(torch.load(ggnn_checkpoint, map_location=device))
    ggnn.to(device)
    ggnn.eval()

    print("\n═" * 75)
    print(" [ReVeal Pipeline] Phase II: Feature Extraction & Representation Learning")
    print("═" * 75)
    train_xg, train_labels = extract_xg(ggnn, dataset, "train", cuda=use_cuda)
    valid_xg, valid_labels = extract_xg(ggnn, dataset, "valid", cuda=use_cuda)
    test_xg, test_labels = extract_xg(ggnn, dataset, "test", cuda=use_cuda)

    print(f"\nApplying SMOTE (k={args.smote_k}) to balance vulnerable samples...")
    bal_xg, bal_labels = smote(
        embeddings=train_xg,
        labels=train_labels,
        k=args.smote_k,
        m=None,
        seed=args.seed,
    )

    bal_xg_t = torch.from_numpy(bal_xg).float()
    bal_labels_t = torch.from_numpy(bal_labels).float()

    vuln_indices = np.where(bal_labels == 1)[0]
    clean_indices = np.where(bal_labels == 0)[0]

    train_loader = DataLoader(
        TensorDataset(bal_xg_t, bal_labels_t, torch.arange(len(bal_labels))),
        batch_size=args.batch_size,
        shuffle=True,
    )

    mlp = ReVealMLP(input_dim=args.graph_embed_size).to(device)
    criterion = ReVealLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    optimizer = optim.Adam(mlp.parameters(), lr=args.lr)

    best_f1 = 0.0
    model_dir = os.path.join("models", args.dataset)
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "ReVealMLP-model.bin")

    patience_counter = 0
    max_patience = args.patience

    print(f"Starting Phase II Training (Triplet Loss + SMOTE) on {device.type.upper()}...")
    print(f"Max Epochs: {args.epochs} | Early Stopping Patience: {max_patience} epochs\n")

    for epoch in range(args.epochs):
        mlp.train()
        epoch_loss = 0.0
        for xg, lg, indices in train_loader:
            xg, lg = xg.to(device), lg.to(device)

            xp_list, xn_list = [], []
            for i, label in enumerate(lg):
                if label == 1:
                    p_idx = random.choice(vuln_indices)
                    n_idx = random.choice(clean_indices)
                else:
                    p_idx = random.choice(clean_indices)
                    n_idx = random.choice(vuln_indices)
                xp_list.append(bal_xg[p_idx])
                xn_list.append(bal_xg[n_idx])

            xp = torch.from_numpy(np.array(xp_list)).to(device)
            xn = torch.from_numpy(np.array(xn_list)).to(device)

            optimizer.zero_grad()
            hg, yg = mlp.predict(xg)
            hp = mlp.transform_only(xp)
            hn = mlp.transform_only(xn)

            loss = criterion(hg, yg, lg, hp, hn)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        mlp.eval()
        with torch.no_grad():
            v_xg = torch.from_numpy(valid_xg).to(device)
            _, v_yg = mlp.predict(v_xg)
            v_yg = v_yg.cpu().numpy()
            v_preds = (v_yg > 0.5).astype(int)
            v_f1 = f1_score(valid_labels, v_preds, zero_division=0)

        if v_f1 > best_f1:
            best_f1 = v_f1
            patience_counter = 0
            torch.save(mlp.state_dict(), save_path)
            print(f"  ● Epoch {epoch+1:03d}/{args.epochs} | Loss: {epoch_loss:.4f} | Valid F1: {v_f1*100:.2f}%  ✓ (New Best)")
        else:
            patience_counter += 1
            if (epoch + 1) % 5 == 0 or patience_counter >= max_patience:
                print(f"  ● Epoch {epoch+1:03d}/{args.epochs} | Loss: {epoch_loss:.4f} | Valid F1: {v_f1*100:.2f}% (Patience: {patience_counter}/{max_patience})")
            if patience_counter >= max_patience:
                print(f"\n[Early Stopping] No improvement in validation F1 for {max_patience} consecutive epochs. Stopping at epoch {epoch+1}.")
                break

    print("\n" + "═" * 75)
    print(" [ReVeal Pipeline] Final Evaluation on Held-Out Test Set (10%)")
    print("═" * 75)
    mlp.load_state_dict(torch.load(save_path, map_location=device))
    mlp.eval()
    with torch.no_grad():
        t_xg = torch.from_numpy(test_xg).to(device)
        _, t_yg = mlp.predict(t_xg)
        t_yg = t_yg.cpu().numpy()
        t_preds = (t_yg > 0.5).astype(int)

        acc = accuracy_score(test_labels, t_preds)
        pr = precision_score(test_labels, t_preds, zero_division=0)
        rc = recall_score(test_labels, t_preds, zero_division=0)
        f1 = f1_score(test_labels, t_preds, zero_division=0)

        print(f"Accuracy  : {acc * 100:.2f}%")
        print(f"Precision : {pr * 100:.2f}%")
        print(f"Recall    : {rc * 100:.2f}%")
        print(f"F1 Score  : {f1 * 100:.2f}%")
        print("═" * 75)
        print(f"✔ Best ReVeal model saved to: {save_path}\n")


if __name__ == "__main__":
    train_reveal()
