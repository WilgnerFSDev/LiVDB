# Devign Plugin: Canonical 6-Edge Model (Zhou et al., NeurIPS 2019)

This plugin implements the strict, canonical pipeline from the paper **"Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks"** (*NeurIPS 2019*), following the multi-relational graph formalization of *Allamanis et al. (ICLR 2018)*.

---

## 1. The 6 Canonical Edges of the Composite Graph

Devign replaces generic Joern edges with **6 formal semantic edge types**:

| ID | Edge Type | Full Name | Semantic Description |
|---|---|---|---|
| **1** | `AST` | Abstract Syntax Tree | Direct parent-child syntactic hierarchy between AST nodes. |
| **2** | `CFG` | Control Flow Graph | Execution order of statements and conditional branching. |
| **3** | `NCS` | Natural Code Sequence | Connects adjacent leaf tokens ($v_i \to v_{i+1}$) in raw source code sequence. |
| **4** | `DFG_R` | LastRead | Connects a variable read to its immediately preceding read in scope. |
| **5** | `DFG_W` | LastWrite | Connects a variable occurrence to the immediately preceding statement where it was defined/written. |
| **6** | `DFG_C` | ComputedFrom | Connects expression operands on the right-hand side ($rhs$) to the destination variable on the left-hand side ($lhs$). |

> **Methodological Note:** Edge subtypes `DFG_R`, `DFG_W`, and `DFG_C` were specifically designed in Devign to prevent edge combinatorial explosion in Joern, condensing all data flow into 3 canonical definition-use patterns.

---

## 2. Hyperparameters and Paper Specifications

- **Node Threshold Filter:** Functions with $|V| > 500$ nodes in the CPG are discarded (~15% of the original dataset).
- **Initial Node Embedding Dimension ($d$):** $100$ dimensions generated via Word2Vec trained exclusively on the training split (`vector_size=100`, `window=5`, `sg=1`, `negative=5`).
- **GNN Recurrent Hidden State Dimension ($z$):** $200$ dimensions in recurrent layers.
- **Graph Timesteps ($T$):** $6$ timesteps ($T=6$).
- **Edge Aggregation Function:** `SUM` across the 6 relation projection matrices.
- **Convolutional Module (ConvModule):**
  - Track 1 (on final state $H^{(T)}$): Conv1D(kernel=3, ReLU) $\to$ MaxPool1D(kernel=3, stride=2) $\to$ Conv1D(kernel=1) $\to$ MaxPool1D(kernel=2, stride=2).
  - Track 2 (on concatenation $[H^{(T)} \parallel X^{(0)}]$): Conv1D(kernel=3, ReLU) $\to$ MaxPool1D(kernel=3, stride=2) $\to$ Conv1D(kernel=1) $\to$ MaxPool1D(kernel=2, stride=2).
- **Dataset Partition (Split):** 75% Training and 25% Validation/Test (random shuffle with configurable seed, default: `42`).
- **Optimizer & Training:** Adam, Learning Rate = $0.0001$, Batch Size = $128$, $L_2$ regularization (`weight_decay=0.001`), Early Stopping Patience = $100$ epochs.

---

## 3. How to Run

### Step 1: Preprocessing via LiVDB CLI
From the root directory:
```bash
python main.py
```
1. Select **"Run pre-processing plugins..."**.
2. Select **"Devign: Canonical 6-Edge Model (Zhou et al., NeurIPS 2019)"**.
3. Select your dataset (e.g., `dataset_devign.csv`).
4. Choose whether to wipe old workspace artifacts (default: `N`) and specify the random seed (default: `42`, or `0` for auto/random).
5. The plugin builds CPGs using Joern in parallel, extracts the 6 canonical edges, and exports training-ready files to:
   `preproc_datasets/dataset_devign_devign_canonical/`
   - `train_GGNNinput.jsonl` (75%)
   - `valid_GGNNinput.jsonl` (25%)
   - `test_GGNNinput.jsonl` (identical held-out validation set)
   - `split_manifest.json` (nominal audit of splits)
   - `devign_canonical_metadata.json` (reproducibility hyperparameter metadata)

### Step 2: Model Training (Direct Python Execution)
Navigate to the `training/` directory:
```bash
cd training
```

#### Option A: Linux / macOS (Bash)
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python main.py \
  --config devign \
  --model_type devign \
  --dataset dataset_devign \
  --input_dir ../preproc_datasets/dataset_devign_devign_canonical \
  --feature_size 100 \
  --graph_embed_size 200 \
  --batch_size 128 \
  --patience 100 \
  --seed 42
```

#### Option B: Windows Command Prompt (CMD)
```cmd
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py ^
  --config devign ^
  --model_type devign ^
  --dataset dataset_devign ^
  --input_dir ..\preproc_datasets\dataset_devign_devign_canonical ^
  --feature_size 100 ^
  --graph_embed_size 200 ^
  --batch_size 128 ^
  --patience 100 ^
  --seed 42
```

#### Option C: Windows PowerShell
```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python main.py `
  --config devign `
  --model_type devign `
  --dataset dataset_devign `
  --input_dir ..\preproc_datasets\dataset_devign_devign_canonical `
  --feature_size 100 `
  --graph_embed_size 200 `
  --batch_size 128 `
  --patience 100 `
  --seed 42
```

---

## 4. Reproducibility & Cross-Machine Portability

To ensure the experimental results are **100% reproducible by advisors and external researchers**:

### A. Automated Tool Provisioning (JDK 11 & Joern 1.1.260)
On a fresh machine or clean git clone:
```bash
python scripts/setup_devign_tools.py
```
The script downloads and unpacks official binaries into `tools/`:
- **Eclipse Adoptium OpenJDK Temurin 11.0.22+7** (required for Joern 1.1's Scala 2.13.5 compiler).
- **Joern v1.1.260** (exact historical version used in the literature).

If this script has not been run prior to plugin execution, `main_plugin.py` automatically detects missing tools and launches the setup script.

### B. Data & Split Determinism
- **Dataset Conversion (`convert_devign.py`):** Deterministic with verified SHA-256 hash (`d2ca8bf8717f92ff68e7fe4595c9cec9d6b24f60ba672fa7c06a38315caf9883`).
- **Dataset Split (75/25):** Lexicographically sorted prior to shuffling with pseudo-random seed (default: `seed=42`).
- **Audit Manifest:** `split_manifest.json` records the exact file assignment for train and valid sets.
- **Deterministic Word2Vec:** When a seed is specified (e.g. `seed=42`), Word2Vec runs with `workers=1` to guarantee bit-for-bit identical vector embeddings across runs.

