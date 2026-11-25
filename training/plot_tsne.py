import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances


def plot_tsne():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--title", type=str, default="t-SNE Visualization")
    parser.add_argument("--output", type=str, default="tsne.png")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples per class")
    args = parser.parse_args()

    data = np.load(args.embeddings)
    X = data["embeddings"]
    y = data["labels"]

    vuln_idx = np.where(y == 1)[0]
    clean_idx = np.where(y == 0)[0]

    n_samples = min(len(vuln_idx), len(clean_idx), args.max_samples)

    np.random.seed(args.seed)
    vuln_idx = np.random.choice(vuln_idx, n_samples, replace=False)
    clean_idx = np.random.choice(clean_idx, n_samples, replace=False)

    indices = np.concatenate([vuln_idx, clean_idx])
    X = X[indices]
    y = y[indices]

    print(f"Running t-SNE on {len(X)} samples ({n_samples} per class)...")
    tsne = TSNE(n_components=2, random_state=args.seed, init="pca", learning_rate="auto")
    X_embedded = tsne.fit_transform(X)

    x_min, x_max = np.min(X_embedded, 0), np.max(X_embedded, 0)
    X_embedded = (X_embedded - x_min) / (x_max - x_min)

    vuln = X_embedded[y == 1]
    clean = X_embedded[y == 0]

    centroid_vuln = np.median(vuln, axis=0)
    centroid_clean = np.median(clean, axis=0)
    dist = np.linalg.norm(centroid_vuln - centroid_clean)

    plt.figure(figsize=(10, 8), dpi=300)

    plt.scatter(
        clean[:, 0],
        clean[:, 1],
        c="green",
        marker="o",
        s=25,
        alpha=0.5,
        edgecolors='none',
        label=f"Clean (n={len(clean)})",
    )

    plt.scatter(
        vuln[:, 0],
        vuln[:, 1],
        c="red",
        marker="+",
        s=50,
        alpha=0.8,
        label=f"Vulnerable (n={len(vuln)})",
    )

    plt.scatter(centroid_clean[0], centroid_clean[1], c="darkgreen", marker="X", s=200, edgecolors='white', label="Centroid Clean")
    plt.scatter(centroid_vuln[0], centroid_vuln[1], c="darkred", marker="X", s=200, edgecolors='white', label="Centroid Vuln")

    plt.title(f"{args.title}\nMedian Centroid Distance: {dist:.4f}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")
    print(f"Median Centroid Distance: {dist:.4f}")


if __name__ == "__main__":
    plot_tsne()
