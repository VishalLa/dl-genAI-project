import os
import json

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, Dataset

import numpy as np
import time
import wandb

from sklearn.model_selection import StratifiedKFold
from transformers import get_cosine_schedule_with_warmup
from collections import Counter
from scipy.signal import butter, filtfilt

from helper.golbal_variables import *


def highpass_filter(audio: torch.Tensor, cutoff: int = 40, sample_rate: int = 16000):
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
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_criterion(
    criterion, 
    pred, 
    y_a, 
    y_b, 
    lam
):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_llrd_optimizer_wav2vec2(
    model: nn.Module, 
    base_lr: float = 1e-5,
    decay_factor: float = 0.65,
    weight_decay: float = 0.05
):
    """
    Layer-wise LR Decay for Wav2Vec2.
    Wav2Vec2 encoder layers are named  wav2vec.encoder.layers.N
    (vs AST's  ast.encoder.layer.N)
    """
    param_groups = []
    assigned     = set()

    # 1. Classifier head — highest LR
    clf = [p for n, p in model.named_parameters()
           if "classifier" in n and p.requires_grad]
    for p in clf:
        assigned.add(id(p))
    if clf:
        param_groups.append({'params': clf, 'lr': base_lr * 10, 'name': 'classifier'})

    # 2. Transformer encoder layers — decay top (11) → bottom (0)
    for layer_idx in range(11, -1, -1):
        lr = base_lr * (decay_factor ** (11 - layer_idx))
        lp = [p for n, p in model.named_parameters()
              if f"encoder.layers.{layer_idx}." in n        # Wav2Vec2 naming
              and p.requires_grad and id(p) not in assigned]
        for p in lp:
            assigned.add(id(p))
        if lp:
            param_groups.append({'params': lp, 'lr': lr, 'name': f'enc_{layer_idx}'})

    # 3. Feature projection (between CNN and transformer)
    fp = [p for n, p in model.named_parameters()
          if "feature_projection" in n and p.requires_grad and id(p) not in assigned]
    for p in fp:
        assigned.add(id(p))
    if fp:
        param_groups.append({'params': fp, 'lr': base_lr * 0.1, 'name': 'feature_projection'})

    # 4. Catch-all
    other = [p for n, p in model.named_parameters()
             if p.requires_grad and id(p) not in assigned]
    if other:
        param_groups.append({'params': other, 'lr': base_lr, 'name': 'other'})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def _print_trainable(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.1f}%)")


def train_wav2vec2(
    model: nn.Module, 
    dataloader: DataLoader,
    loss_function: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: str,
    use_mixup: bool=True, 
    mixup_alpha: float=0.4, 
    scaler: nn.Module=None,
    grad_accum_steps: int=1, 
    scheduler: torch.optim.lr_scheduler._LRScheduler=None
):
    """
    Training loop with three memory-saving techniques for Stage 3:
      - Mixed-precision (AMP): halves activation memory automatically
      - Gradient accumulation: lets you simulate a larger batch with less VRAM
      - Gradient checkpointing: trades compute for memory in the encoder
        (enabled on the model before calling this function)
    """
    model.train()
    running_loss = correct = total = 0
    optimizer.zero_grad()

    for step, (input_values, attention_mask, labels) in enumerate(dataloader):
        
        input_values = input_values.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with autocast("cuda", enabled=(scaler is not None)):
            
            if use_mixup:
                # mix waveforms; attention mask is all-ones so no need to mix it
                input_values, la, lb, lam = mixup_data(input_values, labels, mixup_alpha)
                logits = model(input_values, attention_mask)
                loss = mixup_criterion(loss_function, logits, la, lb, lam)
            
            else:
                logits = model(input_values, attention_mask)
                loss = loss_function(logits, labels)
            loss = loss / grad_accum_steps

        if torch.isnan(loss):
            print(f"  NaN loss at step {step}, skipping")
            optimizer.zero_grad()
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):

            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * grad_accum_steps
        _, predicted = torch.max(logits, 1)

        ref_labels = la if use_mixup else labels

        correct += (predicted == ref_labels).sum().item()
        total += ref_labels.size(0)

    return running_loss / len(dataloader), 100 * correct / total


def test_wav2vec2(
    model: nn.Module,
    dataloader: DataLoader, 
    loss_function: nn.Module, 
    device: str
):
    model.eval()
    running_loss = correct = total = 0

    with torch.no_grad():
        for input_values, attention_mask, labels in dataloader:

            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_values, attention_mask)
            loss = loss_function(logits, labels)

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return (running_loss / len(dataloader) if dataloader else 0, 100 * correct / total if total else 0)


def test_wav2vec2_song_level(
    model: nn.Module, 
    dataset_subset: Dataset, 
    original_dataset: Dataset, 
    device: str
):
    """Song-level majority-vote evaluation."""
    model.eval()
    song_preds  = {}
    song_labels = {}

    with torch.no_grad():
        for real_idx in dataset_subset.indices:

            waveform, label = original_dataset[real_idx]
            waveform = waveform.unsqueeze(0).to(device)
            attn_mask = torch.ones(waveform.shape, dtype=torch.long, device=device)

            pred = torch.argmax(model(waveform, attn_mask), dim=1).item()

            label_val = label.item() if isinstance(label, torch.Tensor) else label

            song_idx = original_dataset.index_map[real_idx][0]

            if song_idx not in song_preds:
                song_preds[song_idx]  = []
                song_labels[song_idx] = label_val
            song_preds[song_idx].append(pred)

    correct = sum(
        1 for sid, preds in song_preds.items()
        if max(set(preds), key=preds.count) == song_labels[sid]
    )
    return 100 * correct / len(song_preds) if song_preds else 0



def train_wav2vec2_model(
    model: nn.Module, 
    device: str, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    save_path: Path, 
    val_dataset_subset: Subset=None,
    original_dataset: Subset=None, 
    epochs: int=30,
    wandb_run = None
):
    """
    3-stage progressive unfreezing for Wav2Vec2 — mirrors AST training exactly.

    Stage 1 (0–20%)  : classifier head only,        lr=1e-3
    Stage 2 (20–65%) : top 6 encoder layers unfrozen, lr=5e-5
    Stage 3 (65–100%): full fine-tune with LLRD,    lr=1e-5

    Improvements vs original:
      - AdamW instead of Adam
      - Cosine warmup scheduler instead of ReduceLROnPlateau
      - AMP (mixed precision)
      - Gradient accumulation in Stage 3
      - Gradient checkpointing in Stage 3
      - Mixup from epoch 0
      - Label smoothing
      - LLRD in Stage 3
      - weight_decay 0.05 (was 1e-4)
      - Stage boundaries 20/65 (was epochs//3)
    """
    use_cuda = device == "cuda"
    if use_cuda:
        torch.cuda.empty_cache()

    scaler = GradScaler() if use_cuda else None
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    model = model.to(device)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    start_time = time.time()

    stage2_start = int(epochs * 0.20)
    stage3_start = int(epochs * 0.65)

    optimizer = None
    scheduler = None
    grad_accum_steps = 1
    mixup_alpha = 0.4

    for epoch in range(epochs):
        epoch_start = time.time()

        # STAGE 1: classifier head only
        if epoch == 0:
            print(f"STAGE 1 — head only  (epochs 1-{stage2_start})")
            for p in model.parameters():
                p.requires_grad = False
            for p in model.classifier.parameters():
                p.requires_grad = True

            grad_accum_steps = 1
            mixup_alpha = 0.4

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3, weight_decay=0.05
            )

            total_steps = (stage2_start * len(train_loader)) // grad_accum_steps
            warmup_steps = (2 * len(train_loader)) // grad_accum_steps

            scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_steps, total_steps
            )
            _print_trainable(model)

        # STAGE 2: unfreeze top 6 encoder layers 
        elif epoch == stage2_start:
            print(f"\n{'='*70}")
            print(f"STAGE 2 — layers 6-11 unfrozen  (epochs {stage2_start+1}-{stage3_start})")

            # Wav2Vec2 layers are named encoder.layers.N (not encoder.layer.N)
            for name, param in model.named_parameters():
                for layer_idx in range(6, 12):
                    if f"encoder.layers.{layer_idx}" in name:
                        param.requires_grad = True
                if "final_layer_norm" in name:
                    param.requires_grad = True

            grad_accum_steps = 2
            mixup_alpha = 0.3

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=5e-5, weight_decay=0.05
            )

            n_steps = ((stage3_start - stage2_start) * len(train_loader)) // grad_accum_steps
            warmup_steps = len(train_loader) // grad_accum_steps

            scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_steps, n_steps
            )
            _print_trainable(model)

        # STAGE 3: full fine-tune with LLRD 
        elif epoch == stage3_start:
            print(f"\n{'='*70}")
            print(f"STAGE 3 — full fine-tune LLRD  (epochs {stage3_start+1}-{epochs})")

            # Unfreeze everything except CNN feature extractor (always frozen)
            for name, param in model.named_parameters():
                if "feature_extractor" not in name:
                    param.requires_grad = True

            grad_accum_steps = 4
            mixup_alpha = 0.2

            model.wav2vec.gradient_checkpointing_enable()

            optimizer = get_llrd_optimizer_wav2vec2(model, base_lr=1e-5, decay_factor=0.65)

            n_steps = ((epochs - stage3_start) * len(train_loader)) // grad_accum_steps
            warmup_steps = len(train_loader) // grad_accum_steps

            scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_steps, n_steps
            )
            if use_cuda:
                torch.cuda.empty_cache()
            _print_trainable(model)

        train_loss, train_acc = train_wav2vec2(
            model=model, 
            dataloader=train_loader, 
            loss_function=loss_function,
            optimizer= optimizer, 
            device=device,
            use_mixup=True, 
            mixup_alpha=mixup_alpha,
            scaler=scaler,
            grad_accum_steps=grad_accum_steps,
            scheduler=scheduler
        )

        val_loss, val_acc_chunk = test_wav2vec2(
            model=model,
            dataloader=val_loader,
            loss_function= loss_function, 
            device=device
        )

        val_acc_song = None
        if val_dataset_subset is not None and original_dataset is not None:
            val_acc_song = test_wav2vec2_song_level(
                model=model,
                dataset_subset=val_dataset_subset,
                original_dataset=original_dataset,
                device=device,
            )

        epoch_time  = time.time() - epoch_start
        current_lr  = optimizer.param_groups[0]["lr"]
        primary_acc = val_acc_song if val_acc_song is not None else val_acc_chunk

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

        if wandb_run is not None:
            wandb_run.log(log_dict)

        song_str = f" | Song: {val_acc_song:6.2f}%" if val_acc_song else ""
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"Train {train_acc:6.2f}% | Chunk {val_acc_chunk:6.2f}%"
              f"{song_str} | Loss {val_loss:.4f} | LR {current_lr:.2e} | {epoch_time:.1f}s")

        if primary_acc > best_val_acc:
            best_val_acc  = primary_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            metric = "song" if val_acc_song else "chunk"
            print(f"    ✓ Saved [{metric}: {primary_acc:.2f}%]  →  {save_path}")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Wav2Vec2 training complete | Best: {best_val_acc:.2f}% | {total_time/60:.1f} min")
    print(f"{'='*70}\n")

    return best_val_acc


def train_wav2vec2_single_run(
    model: nn.Module, 
    device: str, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    val_dataset_subset: Subset=None, 
    original_dataset: DataLoader=None,
    epochs: int=30, 
    save_dir: str="."
):
    
    print("\n" + "="*70)
    print("WAV2VEC2 GENRE CLASSIFICATION — SINGLE RUN")
    print("="*70 + "\n")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_wav2vec2_genre.pth")

    run = wandb.init(
        project="wav2vec2-genre-classification",
        name="single-run",
        config={
            "epochs": epochs,
            "optimizer": "AdamW",
            "label_smoothing": 0.1, 
            "amp": device == "cuda",
            "scheduler": "cosine-with-warmup",
            "architecture": model.__class__.__name__,
            "base_model": "facebook/wav2vec2-base",
        }
    )
    wandb.watch(model, log="all", log_freq=50)

    best_acc = train_wav2vec2_model(
        model=model, 
        device=device,
        train_loader=train_loader, 
        val_loader=val_loader,
        save_path=save_path,
        val_dataset_subset=val_dataset_subset,
        original_dataset=original_dataset,
        epochs=epochs, 
        wandb_run=run,
    )

    wandb.finish()
    return best_acc


def train_wav2vec2_cross_validation(
    model_fn,
    device: str,
    full_dataset: Subset,
    fold_data_fn,
    n_folds: int = 5, 
    epochs: int = 30,
    save_dir="cv_models_wav2vec2",
    wandb_project="wav2vec2-genre-classification-cv",
):
    """
    Stratified K-Fold CV for Wav2Vec2 — mirrors AST CV exactly.
    Collate function must return (input_values, attention_mask, labels).
    """
    os.makedirs(save_dir, exist_ok=True)

    all_labels  = np.array([full_dataset.labels[full_dataset.index_map[i][0]] for i in range(len(full_dataset))])
    all_indices = np.arange(len(full_dataset))

    skf = StratifiedKFold(
        n_splits=n_folds, 
        shuffle=True, 
        random_state=2
    )

    fold_results = []

    print("\n" + "="*70)
    print(f"WAV2VEC2 GENRE CLASSIFICATION — {n_folds}-FOLD CROSS-VALIDATION")
    print("="*70)
    print(f"  Dataset : {len(full_dataset)} chunks | Folds: {n_folds} | Epochs: {epochs}")
    print("="*70 + "\n")

    cv_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
        fold_num = fold_idx + 1
        print(f"\n{'═'*70}")
        print(f"  FOLD {fold_num}/{n_folds}  |  train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'═'*70}\n")

        fold_dir  = os.path.join(save_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        save_path = os.path.join(fold_dir, f"best_wav2vec2_fold{fold_num}.pth")

        train_loader, val_loader, val_subset = fold_data_fn(train_idx, val_idx)

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
                "architecture": model.__class__.__name__,
            },
            reinit=True,
        )

        fold_start = time.time()

        best_acc = train_wav2vec2_model(
            model=model, device=device,
            train_loader=train_loader, 
            val_loader=val_loader,
            save_path=save_path,
            val_dataset_subset=val_subset,
            original_dataset=full_dataset,
            epochs=epochs, 
            wandb_run=run,
        )

        fold_time = time.time() - fold_start
        fold_results.append({
            "fold": fold_num,
            "best_val_acc": round(best_acc, 4),
            "model_path": save_path,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "epochs": epochs,
            "time_minutes": round(fold_time / 60, 2),
        })

        print(f"\n  Fold {fold_num} done | Best: {best_acc:.2f}% | {fold_time/60:.1f} min")
        wandb.finish()

    accs = [r["best_val_acc"] for r in fold_results]
    cv_summary = {
        "mean_acc": round(float(np.mean(accs)), 4),
        "std_acc":  round(float(np.std(accs)),  4),
        "max_acc":  round(float(np.max(accs)),  4),
        "min_acc":  round(float(np.min(accs)),  4),
        "fold_results": fold_results,
        "total_time_minutes": round((time.time() - cv_start) / 60, 2),
    }

    with open(os.path.join(save_dir, "cv_results.json"), "w") as f:
        json.dump(cv_summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION COMPLETE")
    print(f"  Mean: {cv_summary['mean_acc']:.2f}% ± {cv_summary['std_acc']:.2f}%  "
          f"| Best: {cv_summary['max_acc']:.2f}%")
    for r in fold_results:
        print(f"    Fold {r['fold']}: {r['best_val_acc']:.2f}%  →  {r['model_path']}")
    print(f"{'='*70}\n")

    return fold_results

