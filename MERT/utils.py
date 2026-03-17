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
    Layer-wise Learning Rate Decay (LLRD) for MERT.
    
    MERT has:
    - 24 transformer layers (mert.encoder.layer.0 to mert.encoder.layer.23)
    - Embeddings (mert.embeddings)
    - Layer aggregator (layer_aggregator)
    - Classifier head (classifier)
    
    Strategy:
    - Classifier & layer_aggregator: highest LR (base_lr * 10)
    - Top transformer layers (23 → 12): decaying LR
    - Bottom transformer layers (11 → 0): lower LR
    - Embeddings: lowest LR (base_lr * 0.1)
    """
    param_groups = []
    assigned_params = set()

    # 1. Classifier head + layer aggregator — highest LR
    head_params = []
    for n, p in model.named_parameters():
        if ("classifier" in n or "layer_aggregator" in n) and p.requires_grad:
            head_params.append(p)
            assigned_params.add(id(p))
    
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': base_lr * 10,
            'name': 'classification_head'
        })

    # 2. MERT transformer layers — decaying LR from top (23) to bottom (0)
    num_layers = 24
    for layer_idx in range(num_layers - 1, -1, -1):
        lr = base_lr * (decay_factor ** (num_layers - 1 - layer_idx))
        layer_params = []
        
        for n, p in model.named_parameters():
            if f"mert.encoder.layer.{layer_idx}." in n and p.requires_grad and id(p) not in assigned_params:
                layer_params.append(p)
                assigned_params.add(id(p))
        
        if layer_params:
            param_groups.append({
                'params': layer_params,
                'lr': lr,
                'name': f'mert_layer_{layer_idx}'
            })

    # 3. Embeddings — lowest LR
    embed_params = []
    for n, p in model.named_parameters():
        if 'mert.embeddings' in n and p.requires_grad and id(p) not in assigned_params:
            embed_params.append(p)
            assigned_params.add(id(p))
    
    if embed_params:
        param_groups.append({
            'params': embed_params,
            'lr': base_lr * 0.1,
            'name': 'embeddings'
        })

    # 4. Catch-all for unassigned parameters
    other_params = []
    for n, p in model.named_parameters():
        if p.requires_grad and id(p) not in assigned_params:
            other_params.append(p)
            assigned_params.add(id(p))
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'name': 'other_components'
        })

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def train_mert(
    model: nn.Module,
    dataloader: DataLoader, 
    loss_function: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: str,
    use_mixup: bool = True, 
    mixup_alpha: float = 0.4, 
    scaler: GradScaler = None, 
    grad_accum_steps: int = 1,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None
):
    """
    Training loop for MERT with:
    - Mixed-precision (AMP)
    - Gradient accumulation
    - Mixup augmentation
    - Gradient clipping
    
    Note: No chunking needed - all songs are 30 seconds.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for step, (input_values, labels) in enumerate(dataloader):
        input_values = input_values.to(device)
        labels = labels.to(device)

        # Forward pass (with AMP if scaler provided)
        with autocast("cuda", enabled=(scaler is not None)):
            if use_mixup:
                input_values, labels_a, labels_b, lam = mixup_data(
                    input_values, labels, alpha=mixup_alpha
                )
                logits = model(input_values)
                loss = mixup_criterion(loss_function, logits, labels_a, labels_b, lam)
            else:
                logits = model(input_values)
                loss = loss_function(logits, labels)

            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps

        if torch.isnan(loss):
            print("⚠ NaN loss detected, skipping batch")
            optimizer.zero_grad()
            continue

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step every grad_accum_steps batches
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

        running_loss += loss.item() * grad_accum_steps

        _, predicted = torch.max(logits, 1)
        ref_labels = labels if not use_mixup else labels_a
        correct += (predicted == ref_labels).sum().item()
        total += ref_labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy


def test_mert(
    model: nn.Module, 
    dataloader: DataLoader, 
    loss_function: nn.Module, 
    device: str
):
    """
    Song-level evaluation (each sample is a full 30-second song).
    No chunking or majority voting needed.
    """
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



def train_mert_model(
    model: nn.Module,
    device: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_path: str,
    epochs: int = 30
):
    """
    3-Stage Progressive Fine-tuning for MERT-v1-330M:
    
    Stage 1 (0-30% of epochs): Classifier head only
        - Freeze MERT backbone
        - Train only classifier head + layer aggregator
        - LR: 1e-3
        - No mixup
    
    Stage 2 (30-80% of epochs): Partial unfreezing
        - Unfreeze top 12 layers (12-23)
        - LR: 5e-5
        - Enable mixup (alpha=0.4)
    
    Stage 3 (80-100% of epochs): Full fine-tuning
        - Unfreeze all layers (except embeddings)
        - Use LLRD optimizer
        - LR: 1e-5 (base, with layer-wise decay)
        - Gradient accumulation: 4
        - Reduced mixup (alpha=0.2)
    """
    use_cuda = device == "cuda"
    if use_cuda:
        torch.cuda.empty_cache()

    scaler = GradScaler() if use_cuda else None
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_time = time.time()
    best_val_acc = 0.0
    best_val_loss = float("inf")

    # Stage boundaries
    stage2_start = int(epochs * 0.30)
    stage3_start = int(epochs * 0.80)

    optimizer = None
    scheduler = None
    grad_accum_steps = 1
    mixup_alpha = 0.4

    for epoch in range(epochs):
        epoch_start = time.time()

        # STAGE 1: Classifier Head Only 
        if epoch == 0:
            print(f"\n{'='*70}")
            print(f"STAGE 1: Classifier head + layer aggregator only (lr=1e-3)")
            print(f"{'='*70}")

            # Freeze MERT backbone
            for param in model.mert.parameters():
                param.requires_grad = False

            # Unfreeze classifier and layer aggregator
            if hasattr(model, 'layer_aggregator') and model.layer_aggregator is not None:
                for param in model.layer_aggregator.parameters():
                    param.requires_grad = True
            
            for param in model.classifier.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3,
                weight_decay=1e-4
            )

            total_steps = stage2_start * len(train_loader)
            warmup_steps = 2 * len(train_loader)

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            _print_trainable(model, epoch + 1, epochs)


        # STAGE 2: Partial Unfreezing 
        elif epoch == stage2_start:
            print(f"\n{'='*70}")
            print("STAGE 2: Unfreeze top 12 MERT layers (12-23)")
            print(f"{'='*70}")

            # First, freeze everything
            for param in model.parameters():
                param.requires_grad = False

            # 1. Keep classifier trainable (from Stage 1)
            for param in model.classifier.parameters():
                param.requires_grad = True
            
            # 2. Keep layer aggregator trainable (from Stage 1)
            if hasattr(model, 'layer_aggregator') and model.layer_aggregator is not None:
                for param in model.layer_aggregator.parameters():
                    param.requires_grad = True
            
            # 3. Unfreeze top 12 transformer layers (12-23)
            for layer_idx in range(12, 24):
                layer_name = f"mert.encoder.layer.{layer_idx}"
                for name, param in model.named_parameters():
                    if layer_name in name:
                        param.requires_grad = True

            # 4. Unfreeze layer norm if it exists
            for name, param in model.named_parameters():
                if "mert.layernorm" in name or "mert.encoder.layernorm" in name:
                    param.requires_grad = True

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=5e-5,
                weight_decay=1e-4
            )

            n_steps = (stage3_start - stage2_start) * len(train_loader)
            warmup_steps = len(train_loader)

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=n_steps
            )

            _print_trainable(model, epoch + 1, epochs)

        # STAGE 3: Full Fine-tuning 
        elif epoch == stage3_start:
            print(f"\n{'='*70}")
            print("STAGE 3: Full fine-tuning with LLRD")
            print(f"{'='*70}")

            # Unfreeze all except embeddings
            for name, param in model.named_parameters():
                if "mert.embeddings" not in name:
                    param.requires_grad = True

            # Enable gradient accumulation and reduce mixup
            grad_accum_steps = 4
            mixup_alpha = 0.2

            # Use Layer-wise Learning Rate Decay
            optimizer = get_llrd_optimizer(model, base_lr=1e-5)

            n_steps = (epochs - stage3_start) * len(train_loader) // grad_accum_steps
            warmup_steps = len(train_loader) // grad_accum_steps

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=n_steps
            )

            if use_cuda:
                torch.cuda.empty_cache()

            _print_trainable(model, epoch + 1, epochs)

        # Enable mixup from Stage 2 onwards
        use_mixup = epoch >= stage2_start

    
        train_loss, train_acc = train_mert(
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

    
        val_loss, val_acc = test_mert(
            model=model,
            dataloader=val_loader,
            loss_function=loss_function,
            device=device
        )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr,
            "epoch_time": epoch_time
        }

        wandb.log(log_dict)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss

            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved best model [val_acc: {val_acc:.2f}%] → {save_path}")
 
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("MERT TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total Training Time : {total_time/60:.1f} minutes")
    print(f"Best Validation Acc : {best_val_acc:.2f}%")
    print("=" * 70)

    return best_val_acc



def train_single_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 30,
    save_dir: str = "."
):
    """Single training run with WandB logging."""
    
    print("\n" + "=" * 70)
    print("MERT GENRE CLASSIFICATION — SINGLE RUN")
    print("=" * 70 + "\n")

    use_cuda = device == "cuda"
    if use_cuda:
        torch.cuda.empty_cache()

    wandb.init(
        project="mert-genre-classification-single-run",
        name="3-stage-mert-330m-llrd-mixup-amp",
        config={
            "epochs": epochs,
            "optimizer": "AdamW + LLRD",
            "mixup_alpha": 0.4,
            "label_smoothing": 0.1,
            "amp": use_cuda,
            "grad_accum_stage3": 4,
            "scheduler": "CosineAnnealingLR",
            "dataset": "Genre",
            "architecture": model.__class__.__name__,
            "base_model": "m-a-p/MERT-v1-330M",
            "sample_rate": 24000,
            "chunk_duration": 30,
        }
    )
    
    wandb.watch(model, log="all", log_freq=50)

    model = model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_mert_genre.pth")

    best_acc = train_mert_model(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=save_path,
        epochs=epochs,
    )

    wandb.finish()
    return best_acc


def train_mert_cross_validation(
    model_fn,
    device: str,
    full_dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    n_folds: int = 5, 
    epochs: int = 30,
    save_dir: str = "cv_models_mert",
    wandb_project: str = "mert-genre-classification-cross-validation",
):
    """
    Stratified K-Fold cross-validation for MERT.

    Each fold trains a completely independent model, saved to:
        {save_dir}/fold_{k}/best_mert_genre_fold{k}.pth

    A metadata JSON is also saved:
        {save_dir}/cv_results.json
    """
    from mert_dataset import mert_collate_fn
    
    os.makedirs(save_dir, exist_ok=True)

    all_labels = np.array(full_dataset.labels)
    all_indices = np.arange(len(full_dataset))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    print("\n" + "=" * 70)
    print(f"MERT GENRE CLASSIFICATION — {n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 70)
    print(f"  Dataset size : {len(full_dataset)} songs")
    print(f"  Folds        : {n_folds}")
    print(f"  Epochs/fold  : {epochs}")
    print(f"  Save dir     : {save_dir}")
    print("=" * 70 + "\n")

    cv_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
        fold_num = fold_idx + 1

        print(f"\n{'═' * 70}")
        print(f"  FOLD {fold_num}/{n_folds}  |  train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'═' * 70}\n")

        fold_dir = os.path.join(save_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        save_path = os.path.join(fold_dir, f"best_mert_genre_fold{fold_num}.pth")

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        model = model_fn().to(device)

        run = wandb.init(
            project=wandb_project,
            name=f"fold-{fold_num}-of-{n_folds}",
            group="cross-validation",
            config={
                "fold": fold_num,
                "n_folds": n_folds,
                "epochs": epochs,
                "optimizer": "AdamW + LLRD",
                "architecture": model.__class__.__name__,
                "base_model": "m-a-p/MERT-v1-330M",
                "sample_rate": 24000,
                "chunk_duration": 30,
            },
            reinit=True,
        )

        fold_start = time.time()

        fold_train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=mert_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

        fold_val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=mert_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

        best_acc = train_mert_model(
            model=model,
            device=device,
            train_loader=fold_train_loader,
            val_loader=fold_val_loader,
            save_path=save_path,
            epochs=epochs
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

