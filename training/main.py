import argparse
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import pickle
import random
import sys

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum
from trainer import train
from utils import tally_param, debug


def main(cli_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=['devign', 'reveal'], default='devign',
                        help='Hyperparameter preset:\n'
                             '  devign : Original Devign (Zhou et al. 2019)\n'
                             '           num_steps=6, lr=0.0001, BCELoss\n'
                             '  reveal : ReVeal GGNN backbone (Chakraborty et al. 2020, Table 2)\n'
                             '           num_steps=8, lr=0.0001, tanh activation\n')
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default=None)
    parser.add_argument('--dataset', type=str, default=None,
                        help='Name of the dataset for experiment (default: dataset_devign / dataset_reveal).')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input Directory of preprocessed dataset.')
    parser.add_argument('--train_file', type=str, help='Specific training file to use', default=None)
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='targets')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=None)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=None)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training (default: 128 for >=6GB VRAM, 64 for 4GB)', default=None)
    parser.add_argument('--lr', type=float, help='Learning rate', default=None)
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum training steps. If None, uses config preset.')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience in epochs (default: 100 for Devign, 50 for ReVeal).')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: 1000, Chakraborty et al. 2020)')
    if cli_args is not None:
        args = parser.parse_args(cli_args)
    else:
        args = parser.parse_args()

    if args.seed is None:
        args.seed = 1000

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    CONFIGS = {
        'devign': {
            'num_steps': 6,
            'graph_embed_size': 200,
            'batch_size': 128,
            'lr': 0.0001,
            'weight_decay': 0.001,
            'activation': 'ReLU (Conv module only - no tanh)',
            'max_steps': 1000000,
            'reference': 'Zhou et al. 2019',
            'pooling': 'mean (after Conv module)',
            'note': 'tanh is exclusive to ReVeal (Table 2), not used here'
        },
        'reveal': {
            'num_steps': 8,
            'graph_embed_size': 200,
            'feature_size': 169,
            'batch_size': 128,
            'lr': 0.0001,
            'weight_decay': 0.0,
            'activation': 'tanh (after GGNN - ReVeal Table 2)',
            'max_steps': 64000,
            'patience': 50,
            'reference': 'Chakraborty et al. 2020, IEEE TSE, Table 2',
            'pooling': 'sum (ReVeal Section 4.1: element-wise summation)',
            'note': 'feature_size: 169 (69 AST one-hot + 100 W2V, window=10), patience: 50, max 500 iterations'
        },
    }
    cfg = CONFIGS[args.config]

    if args.config == 'reveal':
        if args.model_type is None:
            args.model_type = 'ggnn'
        if args.dataset is None:
            args.dataset = 'linux_kernel_2015-01-01_2026-09-10_ast_curated'
        if args.input_dir is None:
            args.input_dir = f'preproc_datasets/{args.dataset}_reveal_canonical'
        if args.feature_size is None:
            args.feature_size = cfg['feature_size']
        if args.patience is None:
            args.patience = cfg['patience']
    else:
        if args.model_type is None:
            args.model_type = 'devign'
        if args.dataset is None:
            args.dataset = 'linux_kernel_2015-01-01_2026-09-10_ast_curated_balanced1to1'
        if args.input_dir is None:
            args.input_dir = f'preproc_datasets/{args.dataset}_devign_canonical'
        if args.feature_size is None:
            args.feature_size = 100
        if args.patience is None:
            args.patience = 100

    if args.num_steps is None:
        args.num_steps = cfg['num_steps']
    if args.batch_size is None:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            args.batch_size = cfg['batch_size'] if vram_gb >= 6.0 else 64
        else:
            args.batch_size = cfg['batch_size']
    if args.lr is None:
        args.lr = cfg['lr']
    if args.max_steps is None:
        args.max_steps = cfg['max_steps']

    model_name = 'GGNNSum' if args.model_type == 'ggnn' else 'DevignModel'

    debug('═' * 40)
    debug(f' Model       : {model_name}' + (' (Phase I backbone)' if args.model_type == 'ggnn' and args.config == 'reveal' else ''))
    debug(f' Config      : {args.config.upper()}')
    debug(f' Reference   : {cfg["reference"]}')
    debug(f' num_steps   : {args.num_steps}')
    debug(f' hidden_size : {args.graph_embed_size}')
    debug(f' feature_size: {args.feature_size}')
    debug(f' lr          : {args.lr}')
    debug(f' batch_size  : {args.batch_size}')
    debug(f' max_steps   : {args.max_steps}')
    debug(f' patience    : {args.patience}')
    debug(f' weight_decay: {cfg["weight_decay"]}')
    debug(f' activation  : {cfg["activation"]}')
    debug(f' pooling     : {cfg["pooling"]}')
    debug(f' Seed        : {args.seed}')
    if 'note' in cfg:
        debug(f' Note        : {cfg["note"]}')
    debug('═' * 40)

    model_dir = os.path.join('models', args.dataset)
    os.makedirs(model_dir, exist_ok=True)
    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        candidates = [
            os.path.join('..', input_dir),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', input_dir),
            os.path.join('..', 'preproc_datasets', os.path.basename(input_dir)),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'preproc_datasets', os.path.basename(input_dir)),
        ]
        for c in candidates:
            if os.path.exists(c):
                input_dir = c
                break

    processed_data_path = os.path.join(input_dir, 'processed.bin')

    def resolve_src(input_dir, split):
        if split == 'train' and args.train_file:
            path = os.path.join(input_dir, args.train_file)
            if os.path.exists(path):
                return path
            raise FileNotFoundError(f'Train file not found: {path}')
        for ext in ['.jsonl', '.json']:
            path = os.path.join(input_dir, f'{split}_GGNNinput{ext}')
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f'No GGNNinput file found for split: {split} in directory: {os.path.abspath(input_dir)}')

    dataset = DataSet(train_src=resolve_src(input_dir, 'train'),
                      valid_src=resolve_src(input_dir, 'valid'),
                      test_src=resolve_src(input_dir, 'test'),
                      batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                      l_ident=args.label_tag)

    if args.model_type == 'ggnn':
        model = GGNNSum(
            input_dim=dataset.feature_size,
            output_dim=args.graph_embed_size,
            num_steps=args.num_steps,
            max_edge_types=dataset.max_edge_type
        )
    else:
        model = DevignModel(
            input_dim=dataset.feature_size,
            output_dim=args.graph_embed_size,
            num_steps=args.num_steps,
            max_edge_types=dataset.max_edge_type
        )

    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    model.cuda()

    debug('Testing Forward Pass with PyG...')
    graph, targets = dataset.get_next_train_batch()
    predictions = model(graph, cuda=True)
    debug(f"Predictions shape: {predictions.shape}")
    debug("Forward pass OK")

    loss_function = BCELoss(reduction='sum')
    optim = Adam(model.parameters(), lr=args.lr, weight_decay=cfg['weight_decay'])

    save_path = os.path.join(model_dir, model_name)
    dev_every = len(dataset.train_batches)
    train(model=model, dataset=dataset, max_steps=args.max_steps, dev_every=dev_every,
          loss_function=loss_function, optimizer=optim,
          save_path=save_path, max_patience=args.patience, log_every=50)


if __name__ == '__main__':
    main()
