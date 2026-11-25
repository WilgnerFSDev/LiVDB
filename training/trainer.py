import copy
import time
import datetime

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from utils import debug


def _predict(model, loss_function, num_batches, data_iter):
    model.eval()
    with torch.no_grad():
        total_loss, total_samples = 0.0, 0
        all_preds, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            total_loss += loss_function(predictions, targets).detach().cpu().item()
            total_samples += targets.size(0)
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_preds.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_preds.extend(
                    predictions.ge(0.5).to(dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
    model.train()
    avg_loss = total_loss / max(total_samples, 1)
    acc = accuracy_score(all_targets, all_preds) * 100
    pr  = precision_score(all_targets, all_preds, zero_division=0) * 100
    rc  = recall_score(all_targets, all_preds, zero_division=0) * 100
    f1  = f1_score(all_targets, all_preds, zero_division=0) * 100
    return avg_loss, acc, pr, rc, f1


def _fmt(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def _header():
    return (
        f"{'Time':>10}  {'Epoch':<12}  {'T-Loss':>10}  {'V-Loss':>10}  "
        f"{'Accuracy':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  "
        f"{'Best F1':>10}  {'Patience':>9}  {'Elapsed':>10}"
    )


def _row(walltime, epoch, max_epochs, t_loss, v_loss, acc, pr, rc, f1, best_f1, patience, elapsed):
    epoch_str = f"{epoch}/{max_epochs}"
    return (
        f"{walltime:>10}  {epoch_str:<12}  {t_loss:>10.3f}  {v_loss:>10.3f}  "
        f"{acc:>10.2f}  {pr:>10.2f}  {rc:>10.2f}  {f1:>10.2f}  "
        f"{best_f1:>10.2f}  {patience:>9}  {elapsed:>10}"
    )


def evaluate_loss(model, loss_function, num_batches, data_iter, cuda=False):
    loss, acc, _, _, _ = _predict(model, loss_function, num_batches, data_iter)
    return loss, acc


def evaluate_metrics(model, loss_function, num_batches, data_iter):
    _, acc, pr, rc, f1 = _predict(model, loss_function, num_batches, data_iter)
    return acc, pr, rc, f1


def train(model, dataset, max_steps, dev_every, loss_function, optimizer,
          save_path, log_every=50, max_patience=5):

    debug('Start Training')

    max_epochs       = max_steps // dev_every
    train_loss_sum   = 0.0
    train_n_samples  = 0
    best_model       = None
    patience_counter = 0
    best_f1          = 0.0
    best_epoch       = 0
    t_start          = time.time()
    t_epoch_s        = time.time()

    print(_header())

    try:
        for step_count in range(max_steps):
            model.train()
            model.zero_grad()
            graph, targets = dataset.get_next_train_batch()
            targets        = targets.cuda()
            predictions    = model(graph, cuda=True)
            batch_loss     = loss_function(predictions, targets)
            train_loss_sum  += batch_loss.detach().cpu().item()
            train_n_samples += targets.size(0)
            batch_loss.backward()
            optimizer.step()

            step_in_epoch = (step_count % dev_every) + 1
            if log_every and (step_in_epoch % log_every == 0 or step_in_epoch == dev_every):
                cur_epoch = (step_count // dev_every) + 1
                curr_dt = time.time() - t_epoch_s
                speed = step_in_epoch / max(curr_dt, 1e-4)
                eta_s = (dev_every - step_in_epoch) / max(speed, 1e-4)
                cur_avg_loss = train_loss_sum / max(train_n_samples, 1)
                pct = int(100.0 * step_in_epoch / dev_every)
                print(f"  ● Epoch {cur_epoch}/{max_epochs} | Batch {step_in_epoch}/{dev_every} ({pct}%) | Loss: {cur_avg_loss:.4f} | Speed: {speed:.1f} it/s | ETA: {_fmt(eta_s)}", flush=True)

            if step_count % dev_every == (dev_every - 1):
                epoch    = (step_count + 1) // dev_every
                t_loss   = train_loss_sum / max(train_n_samples, 1)
                train_loss_sum  = 0.0
                train_n_samples = 0

                v_loss, acc, pr, rc, f1 = _predict(
                    model, loss_function,
                    dataset.initialize_valid_batch(),
                    dataset.get_next_valid_batch
                )

                elapsed  = _fmt(time.time() - t_epoch_s)
                walltime = datetime.datetime.now().strftime('%H:%M:%S')
                t_epoch_s = time.time()

                if f1 > best_f1:
                    patience_counter = 0
                    best_f1          = f1
                    best_epoch       = epoch
                    best_model       = copy.deepcopy(model.state_dict())
                    _save_file       = open(save_path + '-model.bin', 'wb')
                    torch.save(model.state_dict(), _save_file)
                    _save_file.close()
                    marker = '  ✓'
                else:
                    patience_counter += 1
                    marker = ''

                print(_row(walltime, epoch, max_epochs, t_loss, v_loss,
                           acc, pr, rc, f1, best_f1,
                           patience_counter, elapsed) + marker)

                if patience_counter >= max_patience:
                    debug(
                        f'Early stopping at epoch {epoch}. '
                        f'F1 did not improve for {max_patience} consecutive epochs. '
                        f'Best model was epoch {best_epoch} with validation F1: {best_f1:.2f}%'
                    )
                    break

    except KeyboardInterrupt:
        debug(f'Training interrupted by user. Best model was epoch {best_epoch} with validation F1: {best_f1:.2f}%')

    if best_model is not None:
        model.load_state_dict(best_model)
    _save_file = open(save_path + '-model.bin', 'wb')
    torch.save(model.state_dict(), _save_file)
    _save_file.close()

    debug('Evaluating on test set...')
    _, acc, pr, rc, f1 = _predict(
        model, loss_function,
        dataset.initialize_test_batch(),
        dataset.get_next_test_batch
    )
    total_time = _fmt(time.time() - t_start)
    debug(
        f'Final Results - {save_path} (best epoch: {best_epoch})'
        f'\n  ── Test Set Evaluation (unseen data) ──'
        f'\n  Accuracy  : {acc:.2f}%'
        f'\n  Precision : {pr:.2f}%'
        f'\n  Recall    : {rc:.2f}%'
        f'\n  F1 Score  : {f1:.2f}%'
        f'\n  ── Model Selection ──'
        f'\n  Best Valid F1: {best_f1:.2f}% (epoch {best_epoch})'
        f'\n  Total time: {total_time}'
    )
