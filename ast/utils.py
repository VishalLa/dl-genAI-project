import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
import numpy as np
import time
import wandb
from scipy.signal import butter, filtfilt


from helper.golbal_variables import *


def highpass_filter(audio: torch.Tensor, cutoff: int = 40, sample_rate: int = 16000):

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



def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Returns mixed inputs, pairs of targets, and lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_ast(model, dataloader, loss_function, optimizer, device,
              use_mixup=True, mixup_alpha=0.4, scaler=None, grad_accum_steps=1):
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
            loss = loss / grad_accum_steps

        if torch.isnan(loss):
            print("nan loss detected, skipping batch")
            optimizer.zero_grad()
            continue

        # backward
        if scaler is not None:
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

        running_loss += loss.item() * grad_accum_steps   # un-scale for logging
        _, predicted = torch.max(logits, 1)
        ref_labels = labels if not use_mixup else labels_a
        correct += (predicted == ref_labels).sum().item()
        total += ref_labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy


def test_ast(model, dataloader, loss_function, device):
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


def test_ast_song_level(model, dataset_subset, original_dataset, device):
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


def train_ast_model(model, device, train_loader, val_loader,
                    val_dataset_subset=None, original_dataset=None,
                    epochs=30):

    print("\n" + "=" * 70)
    print("AST GENRE CLASSIFICATION TRAINING  (v4)")
    print("=" * 70 + "\n")

    use_cuda = (device == "cuda")
    if use_cuda:
        torch.cuda.empty_cache()

    # AMP scaler — only meaningful on CUDA
    scaler = GradScaler() if use_cuda else None

    wandb.init(
        project="ast-genre-classification",
        name="3-stage-v3-mixup-specaug-amp",
        config={
            "epochs": epochs,
            "optimizer": "AdamW",
            "mixup_alpha": 0.4,
            "label_smoothing": 0.1,
            "spec_augment": True,
            "amp": use_cuda,
            "grad_checkpointing_stage3": True,
            "grad_accum_stage3": 4,
            "scheduler": "CosineAnnealingLR",
            "dataset": "Genre",
            "architecture": model.__class__.__name__,
            "base_model": "ast-finetuned-audioset-10-10-0.4593",
        }
    )
    wandb.watch(model, log="all", log_freq=50)

    model = model.to(device)

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_time = time.time()
    best_val_acc = 0.0
    best_val_loss = float('inf')

    # Stage boundaries — 30 / 50 / 20 split
    stage2_start = int(epochs * 0.30)
    stage3_start = int(epochs * 0.80)

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
                lr=1e-3, weight_decay=1e-4
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=stage2_start, eta_min=1e-5)

            _print_trainable(model, epoch + 1, epochs)
            wandb.log({"stage": 1}, step=epoch)

        elif epoch == stage2_start:
            print(f"\n{'=' * 70}")
            print(f"STAGE 2: Layers 8-11 unfrozen  (lr=5e-5, epochs {stage2_start+1}-{stage3_start})")

            for name, param in model.named_parameters():
                for layer in ["encoder.layer.8", "encoder.layer.9",
                               "encoder.layer.10", "encoder.layer.11"]:
                    if layer in name:
                        param.requires_grad = True

            grad_accum_steps = 1
            mixup_alpha = 0.2   # reduced from 0.4 — cleaner gradients for encoder fine-tuning

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=5e-5, weight_decay=1e-4
            )
            n_epochs_stage2 = stage3_start - stage2_start
            scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs_stage2, eta_min=5e-6)  # was 1e-6

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

            mixup_alpha = 0.2   # keep same as Stage 2

            model.ast.audio_spectrogram_transformer.encoder.gradient_checkpointing = True

            grad_accum_steps = 4

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-5, weight_decay=5e-3
            )
            n_epochs_stage3 = epochs - stage3_start
            scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs_stage3, eta_min=1e-7)

            if use_cuda:
                torch.cuda.empty_cache()

            _print_trainable(model, epoch + 1, epochs)
            wandb.log({"stage": 3}, step=epoch)

        use_mixup = (epoch >= stage2_start)

        train_loss, train_acc = train_ast(
            model, train_loader, loss_function, optimizer, device,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha,
            scaler=scaler, grad_accum_steps=grad_accum_steps
        )
        val_loss, val_acc_chunk = test_ast(
            model, val_loader, loss_function, device
        )

        # Song-level majority vote (only if caller provides the subset + dataset)
        val_acc_song = None
        if val_dataset_subset is not None and original_dataset is not None:
            val_acc_song = test_ast_song_level(
                model, val_dataset_subset, original_dataset, device
            )

        scheduler.step()

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
            torch.save(model.state_dict(), "best_ast_genre.pth")
            wandb.save("best_ast_genre.pth")
            metric_name = "song_acc" if val_acc_song is not None else "chunk_acc"
            print(f"        ✓ Saved best model ({metric_name}: {primary_acc:.2f}%, val_loss: {val_loss:.4f})")

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


def _print_trainable(model, current_epoch, total_epochs):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"        Trainable: {trainable:,} / {total:,}")
    print(f"{'=' * 70}\n")



def resume_stage3(model, checkpoint_path, device, train_loader, val_loader,
                  val_dataset_subset=None, original_dataset=None,
                  epochs=8):
    """
    Continue fine-tuning from a checkpoint with layers 8-11 + classifier.
    """
    print("\n" + "=" * 70)
    print("STAGE 2 EXTENSION — Continuing from checkpoint")
    print("=" * 70)

    use_cuda = (device == "cuda")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    print(f"Loaded: {checkpoint_path}")

    # Freeze everything first, then selectively unfreeze
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        for layer in ["encoder.layer.8", "encoder.layer.9",
                      "encoder.layer.10", "encoder.layer.11"]:
            if layer in name:
                param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    _print_trainable(model=model)

    optimizer     = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5, weight_decay=1e-4
    )
    scheduler     = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler        = GradScaler() if use_cuda else None
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc  = 0.0
    best_val_loss = float('inf')
    start_time    = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        running_loss, correct, total_samples = 0.0, 0, 0
        optimizer.zero_grad()

        for step, (specs, labels) in enumerate(train_loader):
            specs, labels = specs.to(device), labels.to(device)

            with autocast("cuda", enabled=use_cuda):
                logits = model(specs)
                loss   = loss_function(logits, labels)

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

            running_loss   += loss.item()
            _, preds        = torch.max(logits, 1)
            correct        += (preds == labels).sum().item()
            total_samples  += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc  = 100 * correct / total_samples

        # val (chunk level)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                logits  = model(specs)
                val_loss += loss_function(logits, labels).item()
                _, preds  = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
        val_loss /= len(val_loader)
        val_acc_chunk = 100 * val_correct / val_total


        # val (song level)
        val_acc_song = None
        if val_dataset_subset is not None and original_dataset is not None:
            song_preds, song_labels_map = {}, {}
            with torch.no_grad():
                for real_idx in val_dataset_subset.indices:
                    spec, label = original_dataset[real_idx]
                    logits = model(spec.unsqueeze(0).to(device))
                    pred   = torch.argmax(logits, 1).item()
                    lv     = label.item() if isinstance(label, torch.Tensor) else label
                    sid    = original_dataset.index_map[real_idx][0]
                    song_preds.setdefault(sid, []).append(pred)
                    song_labels_map[sid] = lv
            song_correct = sum(
                max(set(p), key=p.count) == song_labels_map[sid]
                for sid, p in song_preds.items()
            )
            val_acc_song = 100 * song_correct / len(song_preds)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        song_str = f" | Song Acc: {val_acc_song:6.2f}%" if val_acc_song is not None else ""
        print(
            f"Epoch [{epoch+1:3d}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Chunk Acc: {val_acc_chunk:6.2f}%"
            f"{song_str} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        primary_acc = val_acc_song if val_acc_song is not None else val_acc_chunk
        if primary_acc > best_val_acc:
            best_val_acc  = primary_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_ast_genre_s3.pth")
            metric = "song_acc" if val_acc_song is not None else "chunk_acc"
            print(f"        ✓ Saved best_ast_genre_s3.pth  ({metric}: {primary_acc:.2f}%)")

    print(f"\n{'='*70}")
    print(f"DONE | Best Val Acc: {best_val_acc:.2f}% | "
          f"Time: {(time.time()-start_time)/60:.1f} min")
    print(f"{'='*70}\n")
    return best_val_acc
