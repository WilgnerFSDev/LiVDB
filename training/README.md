# Training Scripts

This directory contains scripts for training and evaluating models based on the Devign (Zhou et al. 2019) and ReVeal (Chakraborty et al. 2020) architectures.

## Quick Start (Zero Configuration)

Both architectures have dedicated single-command runners with all canonical hyperparameters pre-configured:

### 1. Devign Canonical Training (Zhou et al. 2019)
Trains the canonical Devign model (6 timesteps, ConvModule, BCELoss, 75/25 split, d=100):
```cmd
python train_devign.py
```

### 2. ReVeal Canonical Training (Chakraborty et al. 2020)
End-to-end automated runner (Phase I GGNN backbone + Phase II SMOTE + Triplet Loss MLP, 80/10/10 split, d=169):
```cmd
python train_reveal.py
```

---

## Detailed Experiment Configurations
Original Devign architecture (Zhou et al. 2019) trained on the full imbalanced dataset.
```powershell
python main.py `
  --config devign `
  --model_type devign `
  --dataset exp1_devign_no_balancing `
  --input_dir ../preproc_datasets/dataset_reveal `
  --feature_size 130 `
  --batch_size 128
```
*Model saved at: `models/exp1_devign_no_balancing/DevignModel-model.bin`*

## Experiment 2 - Devign + Random Undersampling
Devign architecture trained on a balanced dataset (50/50) generated via random undersampling of the majority class.
```powershell
python main.py `
  --config devign `
  --model_type devign `
  --dataset exp2_devign_undersampling `
  --input_dir ../preproc_datasets/dataset_reveal `
  --train_file train_undersampled_GGNNinput.jsonl `
  --feature_size 130 `
  --batch_size 128
```
*Model saved at: `models/exp2_devign_undersampling/DevignModel-model.bin`*

## Experiment 3 - ReVeal (GGNN backbone, Phase I)
Training the GGNN backbone according to ReVeal's Phase I (Chakraborty et al. 2020).
```powershell
python main.py `
  --config reveal `
  --model_type ggnn `
  --dataset exp3_reveal_phase1 `
  --input_dir ../preproc_datasets/dataset_reveal `
  --feature_size 130 `
  --batch_size 128
```
*Model saved at: `models/exp3_reveal_phase1/GGNNSum-model.bin`*

## Experiment 3 - ReVeal (Phase II: MLP + Triplet Loss + SMOTE)
Representation learning phase using Triplet Loss and SMOTE on the embeddings extracted from the Phase I model.
```powershell
python train_reveal.py `
  --ggnn_model models/exp3_reveal_phase1/GGNNSum-model.bin `
  --model_type ggnn `
  --input_dir ../preproc_datasets/dataset_reveal `
  --dataset exp3_reveal_phase2 `
  --alpha 0.5 `
  --beta 0.001 `
  --gamma 0.5 `
  --lr 0.001 `
  --epochs 100 `
  --batch_size 128
```
*Model saved at: `models/exp3_reveal_phase2/ReVealMLP-model.bin`*

## Hyperparameter Reference (ReVeal Table 2 + Section 5.1)

| Component  | Parameter       | Devign | ReVeal |
|------------|----------------|--------|--------|
| Word2Vec   | window_size     | 5      | 10     |
| Word2Vec   | vector_size     | 100    | 100    |
| GGNN       | num_steps       | 6      | 8      |
| GGNN       | graph_embed_size| 200    | 200    |
| GGNN       | lr              | 0.0001 | 0.0001 |
| GGNN       | max_iterations  | -      | 500    |
| GGNN       | activation      | default| tanh   |
| Repr-model | hidden_layers   | -      | 256-128-256 |
| Repr-model | dropout         | -      | 0.2    |
| Repr-model | alpha           | -      | 0.5    |
| Repr-model | beta            | -      | 0.001  |
| Repr-model | gamma           | -      | 0.5    |
| Repr-model | lr              | -      | 0.001  |
| Repr-model | max_iterations  | -      | 100    |
| Repr-model | optimizer       | -      | Adam   |
