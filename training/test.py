import argparse
import os
import torch
import numpy as np
from torch.nn import BCELoss
from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum
from trainer import evaluate_metrics
from utils import debug, tally_param

def resolve_src(input_dir, split):
    for ext in ['.jsonl', '.json']:
        path = os.path.join(input_dir, f'{split}_GGNNinput{ext}')
        if os.path.exists(path):
            return path
    for f in os.listdir(input_dir):
        if 'GGNNinput' in f:
            return os.path.join(input_dir, f)
    raise FileNotFoundError(f'No GGNNinput file found in {input_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=['devign', 'reveal'], default='devign',
                        help='Configuration preset used during training.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the .bin model checkpoint.')
    parser.add_argument('--mlp_path', type=str, default=None,
                        help='Optional path to ReVeal MLP checkpoint (for ReVeal Phase II).')
    parser.add_argument('--model_type', choices=['devign', 'ggnn'], default='devign',
                        help='Type of architecture (devign or ggnn).')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing the preprocessed JSON files.')
    parser.add_argument('--split', type=str, choices=['train', 'valid', 'test'], default='test',
                        help='Split to test on (default: test).')
    parser.add_argument('--feature_size', type=int, default=130)
    parser.add_argument('--graph_embed_size', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    CONFIGS = {
        'devign': {'num_steps': 6},
        'reveal': {'num_steps': 8}
    }
    num_steps = CONFIGS[args.config]['num_steps']

    debug('═' * 40)
    debug(f' Testing Model : {args.model_path}')
    debug(f' Dataset Dir   : {args.input_dir}')
    debug(f' Split         : {args.split.upper()}')
    debug(f' Config        : {args.config.upper()} (steps={num_steps})')
    debug('═' * 40)

    test_src = resolve_src(args.input_dir, args.split)

    dataset = DataSet(train_src=test_src, valid_src=test_src, test_src=test_src,
                      batch_size=args.batch_size)

    if args.model_type == 'ggnn':
        model = GGNNSum(
            input_dim=dataset.feature_size,
            output_dim=args.graph_embed_size,
            num_steps=num_steps,
            max_edge_types=dataset.max_edge_type
        )
    else:
        model = DevignModel(
            input_dim=dataset.feature_size,
            output_dim=args.graph_embed_size,
            num_steps=num_steps,
            max_edge_types=dataset.max_edge_type
        )

    debug(f'Loading weights from {args.model_path}...')
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    model.eval()

    if args.mlp_path:
        from modules.reveal_mlp import ReVealMLP
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from tqdm import tqdm

        debug(f'Loading ReVeal MLP weights from {args.mlp_path}...')
        mlp = ReVealMLP(input_dim=args.graph_embed_size)
        mlp.load_state_dict(torch.load(args.mlp_path))
        mlp.cuda()
        mlp.eval()

        debug('Starting Evaluation with ReVeal Phase II (MLP)...')
        num_batches = dataset.initialize_test_batch()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc='Evaluating MLP'):
                batch, targets = dataset.get_next_test_batch()
                xg = model.forward_embedding(batch, cuda=True)
                _, yg = mlp.predict(xg)
                preds = (yg > 0.5).int().cpu().numpy().tolist()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy().tolist())

        acc = accuracy_score(all_targets, all_preds) * 100
        pr  = precision_score(all_targets, all_preds, zero_division=0) * 100
        rc  = recall_score(all_targets, all_preds, zero_division=0) * 100
        f1  = f1_score(all_targets, all_preds, zero_division=0) * 100
    else:
        loss_function = BCELoss(reduction='sum')
        debug('Starting Evaluation...')
        num_batches = dataset.initialize_test_batch()
        acc, pr, rc, f1 = evaluate_metrics(model, loss_function, num_batches, dataset.get_next_test_batch)

    debug('═' * 40)
    debug(f'Results on {args.input_dir} ({args.split.upper()} split):')
    debug(f'Accuracy  : {acc:.2f}%')
    debug(f'Precision : {pr:.2f}%')
    debug(f'Recall    : {rc:.2f}%')
    debug(f'F1 Score  : {f1:.2f}%')
    debug('═' * 40)
