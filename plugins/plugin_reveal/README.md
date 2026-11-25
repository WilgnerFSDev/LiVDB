# ReVeal Plugin: Comprehensive Representation Learning (Chakraborty et al., IEEE TSE 2020)

This plugin implements the strict, canonical pipeline from the paper **"Deep Learning based Vulnerability Detection: Are We There Yet?"** (*IEEE Transactions on Software Engineering 2020*), authored by Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, and Baishakhi Ray.

---

## 1. Multi-Relational Code Property Graph (12 Edge Types)

Unlike vanilla single-relation or standard Devign models, ReVeal leverages a rich multi-relational CPG representation capturing syntactic, control flow, control dependency, and data dependency semantics:

| ID | Edge Type | Source / Semantic Description |
|---|---|---|
| **1** | `AST` | Abstract Syntax Tree parent-child structural edges. |
| **2** | `CFG` | Control Flow Graph transitions between consecutive instructions and branches. |
| **3** | `CDG` | Control Dependency Graph (predicate controlling statement execution). |
| **4** | `REACHING_DEF` | Data Dependency (def-use chains and reaching definitions). |
| **5** | `DOMINATE` | Dominator tree relationship in the control flow graph. |
| **6** | `POST_DOMINATE` | Post-dominator tree relationship. |
| **7** | `ARGUMENT` | Links call sites to their evaluated argument expressions. |
| **8** | `CALL` | Call invocations linking methods and invoked call targets. |
| **9** | `RETURN` | Method return expressions and exit transitions. |
| **10** | `PARAMETER_LINK` | Method formal parameter declarations and bindings. |
| **11** | `CONTAINS` | AST and structural containment relations. |
| **12** | `EVAL_TYPE` | Type evaluation dependencies. |

---

## 2. ReVeal Two-Phase Architecture & Hyperparameters

ReVeal addresses the severe class imbalance problem in vulnerability datasets through a **two-phase learning framework**:

### Phase I: Feature Extraction Backbone (`GGNNSum`)
- **Node Threshold Filter:** Discards functions with $|V| > 500$ nodes in the CPG to match the paper limit.
- **Initial Node Feature Dimension ($d$):** **$169$ dimensions** (Table 2: $T_v + C_v = 169$):
  - $T_v = 69$: One-hot encoding of the 69 AST node types (`REVEAL_TYPE_MAP`).
  - $C_v = 100$: Word2Vec embedding average across code tokens (`vector_size=100`, `window=10`, `sg=1`, `negative=5`).
- **Graph Neural Network:** Gated Graph Neural Network with **8 layers/timesteps** ($T=8$).
- **Hidden State Dimension ($z$):** $200$ dimensions.
- **Graph Activation Function:** `tanh` (applied to node embeddings after recurrent propagation, per Table 2).
- **Pooling (Readout):** Element-wise summation ($\sum_{v \in V} h_v$), producing graph embedding $x_g \in \mathbb{R}^{200}$.
- **Dropout:** $0.2$ probability.
- **Optimizer:** Adam, Learning Rate = $0.0001$, Batch Size = $128$.
- **Stopping Criterion:** Maximum of $500$ iterations with early stopping patience of $50$ consecutive epochs.

### Phase II: Representation Learning (`ReVealMLP` + SMOTE + Triplet Loss)
- **Class Balancing:** SMOTE oversampling ($k=5$ nearest neighbors) applied on the training graph embeddings $x_g$ to balance vulnerable ($y=1$) and clean ($y=0$) functions.
- **Representation MLP Architecture:** 3 hidden layers ($200 \to 256 \to 128 \to 256 \to 1$).
- **Activation & Dropout:** ReLU activation, Dropout = $0.2$.
- **Triplet Loss Objective:**
  $$\mathcal{L} = \alpha \cdot \mathcal{L}_{bce} + \beta \cdot \mathcal{L}_{reg} + \gamma \cdot \mathcal{L}_{triplet}$$
  - $\alpha = 0.5$ (Binary Cross-Entropy classification weight)
  - $\beta = 0.001$ (L2 feature regularizer weight)
  - $\gamma = 0.5$ (Triplet margin loss separation boundary)
- **Phase II Training:** Adam, Learning Rate = $0.001$, Max Iterations = $100$ epochs, Early Stopping Patience = $5$ consecutive epochs.

---

## 3. How to Run

### Step 1: Preprocessing via LiVDB CLI
From the LiVDB root directory:
```bash
python main.py
```
1. Select **"Run pre-processing plugins..."**.
2. Select **"ReVeal: Comprehensive Representation Learning (Chakraborty et al. 2020)"**.
3. Select your dataset: `dataset_reveal.csv` (Chromium + Debian, 22,718 functions).
4. Enter the random seed (default: `42`, or `0` for auto/random).
5. The plugin runs Joern CPG generation, extracts the 12 multi-relational edges via `joern_reveal_extract.sc`, computes the 169d node features ($T_v=69$ AST one-hot + $C_v=100$ Word2Vec, window=10), splits into 80/10/10, and outputs:
   `preproc_datasets/dataset_reveal_reveal_canonical/`
   - `train_GGNNinput.jsonl` (80% training partition)
   - `valid_GGNNinput.jsonl` (10% validation partition)
   - `test_GGNNinput.jsonl` (10% held-out test partition)
   - `split_manifest.json` (audit manifest of splits)
   - `reveal_canonical_metadata.json` (reproducibility hyperparameter metadata)

### Step 2: Phase I Backbone Training (GGNN Backbone)
Navigate to the `training/` directory:
```bash
cd training
```

#### Option A: Linux / macOS (Bash)
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python main.py \
  --config reveal \
  --model_type ggnn \
  --dataset dataset_reveal \
  --input_dir ../preproc_datasets/dataset_reveal_reveal_canonical \
  --feature_size 169 \
  --graph_embed_size 200 \
  --batch_size 128 \
  --seed 42
```

#### Option B: Windows Command Prompt (CMD)
```cmd
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py ^
  --config reveal ^
  --model_type ggnn ^
  --dataset dataset_reveal ^
  --input_dir ..\preproc_datasets\dataset_reveal_reveal_canonical ^
  --feature_size 169 ^
  --graph_embed_size 200 ^
  --batch_size 128 ^
  --seed 42
```

#### Option C: Windows PowerShell
```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python main.py `
  --config reveal `
  --model_type ggnn `
  --dataset dataset_reveal `
  --input_dir ..\preproc_datasets\dataset_reveal_reveal_canonical `
  --feature_size 169 `
  --graph_embed_size 200 `
  --batch_size 128 `
  --seed 42
```

> **Note:** Phase I trains `GGNNSum` (8 layers, tanh activation, sum pooling, feature_size=169) and saves weights to `models/dataset_reveal/GGNNSum-model.bin`.

---

### Step 3: Phase II Representation Learning (SMOTE + Triplet Loss)

#### Option A: Linux / macOS (Bash)
```bash
python train_reveal.py \
  --config reveal \
  --model_type ggnn \
  --dataset dataset_reveal \
  --input_dir ../preproc_datasets/dataset_reveal_reveal_canonical \
  --ggnn_model models/dataset_reveal/GGNNSum-model.bin \
  --feature_size 169 \
  --graph_embed_size 200 \
  --batch_size 128
```

#### Option B: Windows Command Prompt (CMD)
```cmd
python train_reveal.py ^
  --config reveal ^
  --model_type ggnn ^
  --dataset dataset_reveal ^
  --input_dir ..\preproc_datasets\dataset_reveal_reveal_canonical ^
  --ggnn_model models\dataset_reveal\GGNNSum-model.bin ^
  --feature_size 169 ^
  --graph_embed_size 200 ^
  --batch_size 128
```

#### Option C: Windows PowerShell
```powershell
python train_reveal.py `
  --config reveal `
  --model_type ggnn `
  --dataset dataset_reveal `
  --input_dir ..\preproc_datasets\dataset_reveal_reveal_canonical `
  --ggnn_model models\dataset_reveal\GGNNSum-model.bin `
  --feature_size 169 `
  --graph_embed_size 200 `
  --batch_size 128
```

> **Result:** Phase II extracts the $x_g \in \mathbb{R}^{200}$ graph embeddings, balances the training partition with SMOTE ($k=5$), trains `ReVealMLP` with Triplet Loss, and outputs the final Accuracy, Precision, Recall, and F1-Score on the held-out test set.

---

## 4. GPU Environment Setup (CUDA)

In your conda or Python virtual environment:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric orjson gensim scikit-learn tqdm
```
Once installed, execute Step 2 and Step 3 directly via Python as detailed above.
