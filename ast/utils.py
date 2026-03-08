import os 
import json 
import copy

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, Dataset

import numpy as np
import time
import wandb
from scipy.signal import butter, filtfilt

from sklearn.model_selection import StratifiedKFold
from transformers import get_cosine_schedule_with_warmup
from collections import Counter

from helper.golbal_variables import *


def highpass_filter(
    audio: torch.Tensor, 
    cutoff: int = 40, 
    sample_rate: int = 16000
):

    """
    Applies a Butterworth high-pass filter to remove very low-frequency noise.

    Args:
        audio (torch.Tensor): Input waveform tensor of shape [1, samples].
        cutoff (int): Cutoff frequency in Hz. Default is 40 Hz.
        sample_rate (int): Sampling rate of the waveform. Default is 16000 Hz.

    Returns:
        torch.Tensor: Filtered waveform with same shape and dtype as input.
    """

    b, a = butter(5, cutoff / (sample_rate / 2), btype='high')

    audio_np = audio.squeeze(0).cpu().numpy()

    filtered = filtfilt(b, a, audio_np)
    filtered = np.ascontiguousarray(filtered)
    filtered = torch.from_numpy(filtered).unsqueeze(0)

    return filtered.type_as(audio)



def mixup_data(
    x: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float = 0.4
):
    """Returns mixed inputs, pairs of targets, and lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion, 
    pred, 
    y_a, 
    y_b,
    lam
):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def _print_trainable(model, current_epoch, total_epochs):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"        Trainable: {trainable:,} / {total:,}")
    print(f"{'=' * 70}\n")


def get_llrd_optimizer(
    model: nn.Module, 
    base_lr: float = 1e-5, 
    decay_factor: float = 0.65,
    weight_decay: float = 0.05
):
    """
    Layer-wise Learning Rate Decay (LLRD) with strict parameter tracking.
    """
    param_groups = []
    assigned_params = set()

    # 1. Classifier head — highest LR
    classifier_params = []
    for n, p in model.named_parameters():
        if "classifier" in n and p.requires_grad:
            classifier_params.append(p)
            assigned_params.add(p)
    
    if classifier_params:
        param_groups.append({
            'params': classifier_params,
            'lr': base_lr * 10,
            'name': 'classifier'
        })

    # 2. Transformer layers — decaying LR from top (11) to bottom (0)
    num_layers = 12
    for layer_idx in range(num_layers - 1, -1, -1):
        lr = base_lr * (decay_factor ** (num_layers - 1 - layer_idx))
        layer_params = []
        for n, p in model.named_parameters():
            if f"encoder.layer.{layer_idx}." in n and p.requires_grad and p not in assigned_params:
                layer_params.append(p)
                assigned_params.add(p)
        
        if layer_params:
            param_groups.append({
                'params': layer_params,
                'lr': lr,
                'name': f'encoder_layer_{layer_idx}'
            })

    # 3. Embeddings — lowest LR
    embed_params = []
    for n, p in model.named_parameters():
        if 'embeddings' in n and p.requires_grad and p not in assigned_params:
            embed_params.append(p)
            assigned_params.add(p)
    
    if embed_params:
        param_groups.append({
            'params': embed_params,
            'lr': base_lr * 0.1,
            'name': 'embeddings'
        })

    # 4. Catch-all for unassigned parameters (like AST's final layernorm)
    other_params = []
    for n, p in model.named_parameters():
        if p.requires_grad and p not in assigned_params:
            other_params.append(p)
            assigned_params.add(p)
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'name': 'other_components'
        })

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def train_ast(
    model: nn.Module,
    dataloader: DataLoader, 
    loss_function: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: str,
    use_mixup:bool=True, 
    mixup_alpha:float=0.4, 
    scaler: nn.Module=None, 
    grad_accum_steps: int=1,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None
):
    """
    Training loop with three memory-saving techniques for Stage 3:
      - Mixed-precision (AMP): halves activation memory automatically
      - Gradient accumulation: lets you simulate a larger batch with less VRAM
      - Gradient checkpointing: trades compute for memory in the encoder
        (enabled on the model before calling this function)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for step, (input_values, labels) in enumerate(dataloader):
        input_values = input_values.to(device)
        labels = labels.to(device)

        # forward (with AMP if scaler provided)
        with autocast("cuda", enabled=(scaler is not None)):
            if use_mixup:
                input_values, labels_a, labels_b, lam = mixup_data(input_values, labels, alpha=mixup_alpha)
                logits = model(input_values)
                loss = mixup_criterion(loss_function, logits, labels_a, labels_b, lam)
            else:
                logits = model(input_values)
                loss = loss_function(logits, labels)

            # Scale loss for gradient accumulation
            loss: torch.Tensor = loss / grad_accum_steps

        if torch.isnan(loss):
            print("nan loss detected, skipping batch")
            optimizer.zero_grad()
            continue

        # backward
        if scaler is not None:
            """
            dynamically estimates the scale factor each iteration
            Helps perform the steps of gradient scaling
            conveniently.

            scaler.scale()
            multiplies a given loss by ``scaler``'s current scale factor.
            """
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # optimizer step every grad_accum_steps batches
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):

            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * grad_accum_steps   # un-scale for logging

        _, predicted = torch.max(logits, 1)

        ref_labels = labels if not use_mixup else labels_a

        correct += (predicted == ref_labels).sum().item()
        total += ref_labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy


def test_ast(
    model: nn.Module, 
    dataloader: DataLoader, 
    loss_function: torch.optim.Optimizer, 
    device: str
):
    """Chunk-level evaluation (fast, used every epoch)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_values, labels in dataloader:
            input_values = input_values.to(device)
            labels = labels.to(device)

            logits = model(input_values)
            loss = loss_function(logits, labels)

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0
    return loss, accuracy


def test_ast_song_level(
    model: nn.Module, 
    dataset_subset: Subset, 
    original_dataset: Dataset, 
    device: str
):
    """
    Song-level evaluation using majority voting across all chunks of a song.

    Args:
        dataset_subset : torch Subset wrapping Fast_AST_Dataset (val split)
        original_dataset: the underlying Fast_AST_Dataset (has index_map + labels)
    Returns:
        song_accuracy (float)
    """
    model.eval()

    # Collect per-chunk predictions grouped by song_idx
    song_preds = {}   # song_idx -> list of predicted class ints
    song_labels = {}  # song_idx -> true label

    with torch.no_grad():
        for i, real_idx in enumerate(dataset_subset.indices):
            spec, label = original_dataset[real_idx]
            spec = spec.unsqueeze(0).to(device)

            logits = model(spec)
            pred = torch.argmax(logits, dim=1).item()
            label_val = label.item() if isinstance(label, torch.Tensor) else label

            song_idx = original_dataset.index_map[real_idx][0]

            if song_idx not in song_preds:
                song_preds[song_idx] = []
                song_labels[song_idx] = label_val

            song_preds[song_idx].append(pred)

    # Majority vote
    correct = 0
    for song_idx, preds in song_preds.items():
        voted = max(set(preds), key=preds.count)
        if voted == song_labels[song_idx]:
            correct += 1

    return 100 * correct / len(song_preds) if song_preds else 0


def train_ast_model(
    model: nn.Module, 
    device: str, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    save_path: str, 
    val_dataset_subset: Subset=None, 
    original_dataset: Subset=None,
    epochs: int=30
):
    
    """
    Trains one model with the 3-stage progressive unfreezing strategy.

      - weight_decay 1e-4 → 0.05  (proper transformer regularisation)
      - LR warmup added to every stage  (prevents instability)
      - Stage 2 unfreezes layers 6-11 + layernorm  (was 8-11 only)
      - Stage 3 uses LLRD instead of flat LR
      - Scheduler stepped per-batch (required for warmup)
      - Mixup active from epoch 0 (not just stage 2+)
      - Stage boundaries: 20 / 65 split  (more time for full fine-tune)
    """

    use_cuda = (device == "cuda")
    if use_cuda:
        torch.cuda.empty_cache()

    # AMP scaler — only meaningful on CUDA
    scaler = GradScaler() if use_cuda else None

    model = model.to(device)

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_time = time.time()
    best_val_acc = 0.0
    best_val_loss = float('inf')

    # Stage boundaries — 30 / 50 / 20 split
    stage2_start = int(epochs * 0.20)
    stage3_start = int(epochs * 0.65)

    optimizer = None
    scheduler = None
    grad_accum_steps = 1   
    mixup_alpha = 0.4       

    for epoch in range(epochs):
        epoch_start = time.time()

        if epoch == 0:
            print(f"STAGE 1: Classifier head only  (lr=1e-3, epochs 1-{stage2_start})")

            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True

            grad_accum_steps = 1
            mixup_alpha = 0.4  

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3, weight_decay=0.05
            )

            total_steps = (stage2_start * len(train_loader)) // grad_accum_steps
            warmup_steps = (2 * len(train_loader)) // grad_accum_steps 

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            _print_trainable(model, epoch + 1, epochs)
            wandb.log({"stage": 1}, step=epoch)

        elif epoch == stage2_start:
            print(f"\n{'=' * 70}")
            print(f"STAGE 2: Layers 8-11 unfrozen  (lr=5e-5, epochs {stage2_start+1}-{stage3_start})")

            for name, param in model.named_parameters():
                for layer in ["encoder.layer.6", "encoder.layer.7", "encoder.layer.8", "encoder.layer.9", "encoder.layer.10", "encoder.layer.11","layernorm"]:
                    if layer in name:
                        param.requires_grad = True

            grad_accum_steps = 1
            mixup_alpha = 0.2  

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=5e-5, weight_decay=0.05
            )

            n_steps = ((stage3_start - stage2_start) * len(train_loader)) // grad_accum_steps
            warmup_steps = (1 * len(train_loader)) // grad_accum_steps

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=n_steps
            )

            _print_trainable(model, epoch + 1, epochs)
            wandb.log({"stage": 2}, step=epoch)

        elif epoch == stage3_start:
            print(f"\n{'=' * 70}")
            print(f"STAGE 3: Full fine-tune  (lr=1e-5, epochs {stage3_start+1}-{epochs})")
            print(f"         [AMP + gradient checkpointing + grad accum x4]")

            # Unfreeze everything except embeddings
            for name, param in model.named_parameters():
                if "embeddings" not in name:
                    param.requires_grad = True

            grad_accum_steps = 4
            mixup_alpha = 0.2

            if hasattr(model, 'ast'):
                model.ast.gradient_checkpointing_enable()
            elif hasattr(model, 'model'):
                model.model.gradient_checkpointing_enable()
            elif hasattr(model, 'encoder'):
                model.encoder.gradient_checkpointing_enable()
            elif hasattr(model, 'base_model'):
                model.base_model.gradient_checkpointing_enable()
            else:
                print("⚠️ Warning: Could not find the internal AST model to enable gradient checkpointing. Continuing without it.")

            grad_accum_steps = 4

            optimizer = get_llrd_optimizer(model, base_lr=1e-5, decay_factor=0.65)
            
            n_steps = ((epochs - stage3_start) * len(train_loader)) // grad_accum_steps
            warmup_steps = (1 * len(train_loader)) // grad_accum_steps

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=n_steps
            )

            if use_cuda:
                torch.cuda.empty_cache()

            _print_trainable(model, epoch + 1, epochs)
            wandb.log({"stage": 3}, step=epoch)

        use_mixup = (epoch >= stage2_start)

        train_loss, train_acc = train_ast(
            model=model,
            dataloader=train_loader, 
            loss_function=loss_function, 
            optimizer=optimizer, 
            device=device,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            scaler=scaler, 
            grad_accum_steps=grad_accum_steps,
            scheduler=scheduler
        )
        val_loss, val_acc_chunk = test_ast(
            model=model, 
            dataloader=val_loader, 
            loss_function=loss_function, 
            device=device
        )

        # Song-level majority vote (only if caller provides the subset + dataset)
        val_acc_song = None
        # if val_dataset_subset is not None and original_dataset is not None:
        val_acc_song = test_ast_song_level(
            model=model,
            dataset_subset=val_dataset_subset, original_dataset=original_dataset, 
            device=device
        )


        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy_chunk": val_acc_chunk,
            "learning_rate": current_lr,
            "epoch_time": epoch_time,
        }
        if val_acc_song is not None:
            log_dict["val_accuracy_song"] = val_acc_song

        wandb.log(log_dict)

        song_str = f" | Song Acc: {val_acc_song:6.2f}%" if val_acc_song is not None else ""
        print(
            f"Epoch [{epoch+1:3d}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Chunk Acc: {val_acc_chunk:6.2f}%"
            f"{song_str} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Save on best chunk-level val acc (or song-level if available)
        primary_acc = val_acc_song if val_acc_song is not None else val_acc_chunk

        if primary_acc > best_val_acc:
            best_val_acc = primary_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            metric_name = "song" if val_acc_song is not None else "chunk"
            print(f"    ✓ Saved [{metric_name}: {primary_acc:.2f}%]  →  {save_path}")

    total_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"AST TRAINING COMPLETE  (v2)")
    print(f"{'=' * 70}")
    print(f"Total Training Time    : {total_time / 60:.1f} minutes")
    print(f"Best Validation Acc    : {best_val_acc:.2f}%")
    print(f"Best Validation Loss   : {best_val_loss:.4f}")
    print(f"Model saved as         : best_ast_genre.pth")
    print(f"{'=' * 70}\n")

    wandb.finish()
    return best_val_acc

def train_single_corss_validation(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    val_dataset_subset: Subset = None,
    original_dataset: Subset = None,
    epochs: int = 30,
    save_dir: str = "."
):
    print("\n" + "=" * 70)
    print("AST GENRE CLASSIFICATION — SINGLE RUN")
    print("=" * 70 + "\n")

    use_cuda = device == "cuda"
    if use_cuda:
        torch.cuda.empty_cache()

    run = wandb.init(
        project="ast-genre-classification",
        name="single-run",
        config={
            "epochs": epochs, 
            "optimizer": "AdamW",
            "label_smoothing": 0.1, 
            "amp": device == "cuda",
            "scheduler": "cosine-with-warmup",
            "architecture": model.__class__.__name__,
        }
    )
    wandb.watch(model, log="all", log_freq=50)

    model = model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_ast_genre.pth")

    best_acc = train_ast_model(
        model=model, 
        device=device,
        train_loader=train_loader, 
        val_loader=val_loader,
        save_path=save_path,
        val_dataset_subset=val_dataset_subset,
        original_dataset=original_dataset,
        epochs=epochs, 
    )

    wandb.finish()
    return best_acc


def train_ast_cross_validation(
    model_fn,
    device: str,
    full_dataset: Subset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_folds: int = 5, 
    epochs: int = 30,
    save_dir: str = "cv_models_ast",
    wandb_project: str = "ast-genre-classification-cross-validation",
):
    """
    Stratified K-Fold cross-validation.

    Each fold trains a completely independent model, saved to:
        {save_dir}/fold_{k}/best_ast_genre_fold{k}.pth

    A metadata JSON is also saved:
        {save_dir}/cv_results.json

    These saved models can later be loaded together for ensemble inference
    """

    os.makedirs(save_dir, exist_ok=True)

    all_labels = np.array(full_dataset.labels)
    all_indices = np.arange(len(full_dataset))

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=2
    )

    fold_results = []

    print("\n" + "=" * 70)
    print(f"AST GENRE CLASSIFICATION — {n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 70)
    print(f"  Dataset size : {len(full_dataset)} chunks")
    print(f"  Folds        : {n_folds}")
    print(f"  Epochs/fold  : {epochs}")
    print(f"  Save dir     : {save_dir}")
    print("=" * 70 + "\n")

    cv_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(all_indices, all_labels)
    ):
        fold_num = fold_idx + 1 

        print(f"\n{'═' * 70}")
        print(f"  FOLD {fold_num}/{n_folds}  |  train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'═' * 70}\n")

        fold_dir = os.path.join(save_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)

        save_path = os.path.join(fold_dir, f"best_ast_genre_fold{fold_num}.pth")

        train_subset = Subset(full_dataset, train_idx)
        val_subset   = Subset(full_dataset, val_idx)

        model = model_fn().to(device)

        run = wandb.init(
            project=wandb_project,
            name=f"fold-{fold_num}-of-{n_folds}",
            group="cross-validation",
            config={
                "fold": fold_num, 
                "n_folds": n_folds,
                "epochs": epochs, 
                "optimizer": "AdamW",
                "label_smoothing": 0.1,
                "scheduler": "cosine-with-warmup",
                "architecture": model.__class__.__name__,
            },
            reinit=True,
        )

        fold_start = time.time()

        best_acc = train_ast_model(
            model=model, 
            device=device,
            train_loader=train_loader, 
            val_loader=val_loader,
            save_path=save_path,
            val_dataset_subset=val_subset,
            original_dataset=full_dataset,
            epochs=epochs, 
        )

        fold_time = time.time() - fold_start

        result = {
            "fold": fold_num,
            "best_val_acc": round(best_acc, 4),
            "model_path": save_path,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "epochs": epochs,
            "time_minutes": round(fold_time / 60, 2),
        }
        fold_results.append(result)

        print(f"\n  Fold {fold_num} done  |  Best acc: {best_acc:.2f}%  |  {fold_time/60:.1f} min")
        wandb.finish()

    accs = [r["best_val_acc"] for r in fold_results]
    cv_summary = {
        "mean_acc": round(float(np.mean(accs)), 4),
        "std_acc": round(float(np.std(accs)), 4),
        "max_acc": round(float(np.max(accs)), 4),
        "min_acc": round(float(np.min(accs)), 4),
        "fold_results": fold_results,
        "total_time_minutes": round((time.time() - cv_start) / 60, 2),
    }

    results_path = os.path.join(save_dir, "cv_results.json")
    with open(results_path, "w") as f:
        json.dump(cv_summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"CROSS-VALIDATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Mean acc : {cv_summary['mean_acc']:.2f}% ± {cv_summary['std_acc']:.2f}%")
    print(f"  Best fold: {cv_summary['max_acc']:.2f}%")
    print(f"  Results  : {results_path}")
    for r in fold_results:
        print(f"    Fold {r['fold']}: {r['best_val_acc']:.2f}%  →  {r['model_path']}")
    print(f"{'=' * 70}\n")

    return fold_results


