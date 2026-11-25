import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    k:          int   = 5,
    m:          int   = None,
    seed:       int   = 1000,
) -> tuple:
    rng = np.random.default_rng(seed)

    vuln_mask  = labels == 1
    clean_mask = labels == 0

    vuln_list  = list(embeddings[vuln_mask])
    clean_list = list(embeddings[clean_mask])

    n_vuln  = len(vuln_list)
    n_clean = len(clean_list)

    if m is None:
        m = (n_vuln + n_clean) // 2

    print(f"  SMOTE target m = {m} per class")
    print(f"  Before: {n_vuln} vulnerable + {n_clean} clean = {n_vuln + n_clean} total")

    clean_arr = np.array(clean_list)
    if len(clean_arr) > m:
        idx = rng.choice(len(clean_arr), size=m, replace=False)
        clean_arr = clean_arr[idx]

    vuln_arr = np.array(vuln_list)
    nn_model = NearestNeighbors(
        n_neighbors=min(k + 1, len(vuln_arr)),
        metric='euclidean'
    ).fit(vuln_arr)
    _, neighbor_indices = nn_model.kneighbors(vuln_arr)

    synthetic = []

    while len(vuln_arr) + len(synthetic) < m:
        i = rng.integers(0, len(vuln_arr))
        x = vuln_arr[i]

        nbrs = neighbor_indices[i][1:]

        for j in nbrs:
            if len(vuln_arr) + len(synthetic) >= m:
                break
            n_vec = vuln_arr[j]
            alpha = rng.random()
            xs    = x + alpha * (n_vec - x)
            synthetic.append(xs)

    if synthetic:
        synthetic_arr = np.array(synthetic, dtype=np.float32)
        vuln_final    = np.vstack([vuln_arr, synthetic_arr])
    else:
        vuln_final    = vuln_arr

    vuln_final  = vuln_final[:m]
    clean_final = clean_arr[:m]

    bal_emb    = np.vstack([vuln_final, clean_final]).astype(np.float32)
    bal_labels = np.concatenate([
        np.ones(len(vuln_final)),
        np.zeros(len(clean_final))
    ]).astype(np.float32)

    idx = rng.permutation(len(bal_labels))
    bal_emb    = bal_emb[idx]
    bal_labels = bal_labels[idx]

    n_v = int((bal_labels == 1).sum())
    n_c = int((bal_labels == 0).sum())
    print(f"  After:  {n_v} vulnerable + {n_c} clean = {n_v + n_c} total")

    return bal_emb, bal_labels
