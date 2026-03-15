import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, Dataset
from collections import Counter
import numpy as np
import time

import wandb
WANDB_AVAILABLE = False


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def single_train_preload(
    model: nn.Module, 
    dataloader: DataLoader,
    loss_function: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: str,
    scaler: nn.modules=None, 
    use_mixup: bool=False, 
    mixup_alpha: float=0.4
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for step, (img, label) in enumerate(dataloader):
        img   = img.to(device)
        label = label.to(device)

        with autocast("cuda", enabled=(scaler is not None)):
            
            if use_mixup:
                img, label_a, label_b, lam = mixup_data(img, label, alpha=mixup_alpha)
                logits = model(img)
                loss = mixup_criterion(loss_function, logits, label_a, label_b, lam)
            
            else:
                logits = model(img)
                loss = loss_function(logits, label)

        if torch.isnan(loss):
            print("nan loss detected, skipping batch")
            optimizer.zero_grad()
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
       
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        optimizer.zero_grad()

        running_loss += loss.item()
        _, predicted  = torch.max(logits, 1)
        ref_labels = label if not use_mixup else label_a
        correct += (predicted == ref_labels).sum().item()
        total += ref_labels.size(0)

    return running_loss / len(dataloader), 100 * correct / total


def test_preload(
    model: nn.Module , 
    dataloader: DataLoader, 
    loss_function: nn.Module, 
    device: str
):
    """Chunk-level evaluation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for spectrum, label in dataloader:
            spectrum = spectrum.to(device)
            label = label.to(device)

            logits = model(spectrum)
            loss = loss_function(logits, label)

            running_loss += loss.item()
            _, predicted  = torch.max(logits, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

    loss  = running_loss / len(dataloader) if len(dataloader) > 0 else 0
    
    accuracy = 100 * correct / total if total > 0 else 0
    
    return loss, accuracy


def test_song_level(
    model: nn.Module, 
    dataset_subset: Subset, 
    original_dataset: Dataset, 
    device: str
):
    """
    Song-level majority vote evaluation.
    Requires original_dataset to have an index_map: list of (song_idx, ...).
    """
    model.eval()
    song_preds  = {}
    song_labels = {}

    with torch.no_grad():
        for real_idx in dataset_subset.indices:
            img, label = original_dataset[real_idx]
            img = img.unsqueeze(0).to(device)

            logits = model(img)
            pred = torch.argmax(logits, dim=1).item()
            lv = label.item() if isinstance(label, torch.Tensor) else label

            song_idx = original_dataset.index_map[real_idx][0]
            song_preds.setdefault(song_idx, []).append(pred)
            song_labels[song_idx] = lv

    correct = sum(
        max(set(p), key=p.count) == song_labels[sid]
        for sid, p in song_preds.items()
    )
    return 100 * correct / len(song_preds) if song_preds else 0.0


def train_on_pre_loaded_images(
    model: nn.Module,
    device: str, 
    train_dataloader: DataLoader, 
    test_dataloader: DataLoader,
    training_type: str, 
    epochs: int=30,
    val_dataset_subset: Subset=None, 
    original_dataset: Dataset=None,
):
    print(f"\n{'=' * 70}")
    print(f"CRNN IMAGE CLASSIFICATION TRAINING  ({training_type})")
    print(f"{'=' * 70}\n")

    use_cuda = (device == "cuda")
    if use_cuda:
        torch.cuda.empty_cache()

    scaler = GradScaler() if use_cuda else None

    if WANDB_AVAILABLE:
        wandb.init(
            project="crnn-image-classification",
            name=f"training-{training_type}",
            config={
                "epochs":        epochs,
                "optimizer":     "AdamW",
                "mixup_alpha":   0.4,
                "label_smooth":  0.1,
                "amp":           use_cuda,
                "scheduler":     "CosineAnnealingLR",
                "dataset":       "preloaded-spectrograms",
                "architecture":  model.__class__.__name__,
                "training_type": training_type,
            }
        )
        wandb.watch(model, log="all", log_freq=50)

    model = model.to(device)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Mixup on after 20% of training — let model learn basics first
        use_mixup = (epoch >= max(1, int(epochs * 0.2)))

        train_loss, train_acc = single_train_preload(
            model, train_dataloader, loss_function, optimizer, device,
            scaler=scaler, use_mixup=use_mixup, mixup_alpha=0.4
        )

        val_loss, val_acc_chunk = test_preload(
            model, test_dataloader, loss_function, device
        )

        # Song-level accuracy (optional — requires val_dataset_subset)
        val_acc_song = None
        if val_dataset_subset is not None and original_dataset is not None:
            val_acc_song = test_song_level(
                model, val_dataset_subset, original_dataset, device
            )

        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        song_str = f" | Song Acc: {val_acc_song:6.2f}%" if val_acc_song is not None else ""
        print(
            f"Epoch [{epoch+1:3d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Chunk Acc: {val_acc_chunk:6.2f}%"
            f"{song_str} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        if WANDB_AVAILABLE:
            log_dict = {
                "epoch":          epoch + 1,
                "train_loss":     train_loss,
                "train_accuracy": train_acc,
                "val_loss":       val_loss,
                "val_acc_chunk":  val_acc_chunk,
                "learning_rate":  current_lr,
                "epoch_time":     epoch_time,
            }
            if val_acc_song is not None:
                log_dict["val_acc_song"] = val_acc_song
            wandb.log(log_dict)

        # Save on best song-level acc (falls back to chunk-level if not available)
        primary_acc = val_acc_chunk
        if primary_acc > best_val_acc:
            best_val_acc  = primary_acc
            best_val_loss = val_loss

            checkpoint = {
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc':         best_val_acc,
                'best_val_loss':        best_val_loss,
            }

            torch.save(checkpoint, "best_crnn_preload_images.pth")
            metric = "song_acc" if val_acc_song is not None else "chunk_acc"
            print(f"         ✓ Saved best model ({metric}: {primary_acc:.2f}%, "
                  f"val_loss: {val_loss:.4f})")
            
            if WANDB_AVAILABLE:
                wandb.save("best_crnn_preload_images.pth")

    total_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total Training Time    : {total_time / 60:.1f} minutes")
    print(f"Total Epochs Trained   : {epoch + 1} / {epochs}")
    print(f"Best Validation Acc    : {best_val_acc:.2f}%")
    print(f"Best Validation Loss   : {best_val_loss:.4f}")
    print(f"Model saved as         : best_crnn_preload_images.pth")
    print(f"{'=' * 70}\n")

    if WANDB_AVAILABLE:
        wandb.finish()

    return best_val_acc
