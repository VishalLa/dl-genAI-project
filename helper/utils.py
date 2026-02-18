import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
import time

import wandb


class EarlyStopping:
    def __init__(self, patience=7):
        self.patience  = patience
        self.best_loss = float("inf")
        self.counter   = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        
        self.counter += 1
        return self.counter >= self.patience


def train_esc50(model, dataloader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for waveform, label in dataloader:  
        waveform = waveform.to(device)
        label = label.to(device)

        # Forward
        logits = model(waveform) 

        # Loss
        loss = loss_function(logits, label)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    return epoch_loss, accuracy


def test_esc50(model, dataloader, loss_function, device):
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

            _, predicted = torch.max(logits, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

    loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0

    return loss, accuracy


def train_esc50_model(model, device, train_dataloader, test_dataloader, training_type, epochs=50):
    print(f"\n{'='*70}")
    print(f"ESC-50 TRAINING")
    print(f"{'='*70}\n")
    
    # Clear GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()

    wandb.init(
        project=f"esc50-classification",
        name=f"Training {training_type}",
        config={
            "epochs": epochs,
            "optimizer": "Adam",
            "dataset": "ESC-50",
            "architecture": model.__class__.__name__,
            "training_type": training_type,
        }
    )
    wandb.watch(model, log="all", log_freq=50)

    print("\n3. Setting up training...")
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    
    # Track training time
    start_time = time.time()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    change_stage = epochs // 3
    best_val_acc = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_start = time.time()

        if training_type == "FINETUNE":
            if epoch == 0:
                # STAGE 1: Only train final head
                for name, param in model.named_parameters():
                    if 'head' not in name:
                        param.requires_grad = False
                
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=0.001,
                    weight_decay=1e-4
                )

                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())

                print(f"STAGE 1: Training only final head (lr=0.001)")
                print(f"         Trainable params: {trainable:,} / {total_params:,}")
                print(f"{'='*70}\n")
            
            elif epoch == change_stage:
                # STAGE 2: Unfreeze LSTM + attention + head
                for name, param in model.named_parameters():
                    if 'lstm' in name or 'attention' in name or 'head' in name:
                        param.requires_grad = True
                
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=0.0001,
                    weight_decay=1e-4
                )
                
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())

                print(f"\n{'='*70}")
                print(f"STAGE 2: Unfroze LSTM + Attention + Head (lr=0.0001)")
                print(f"         Trainable params: {trainable:,} / {total_params:,}")
                print(f"{'='*70}\n")
            
            elif epoch == change_stage * 2:
                # STAGE 3: Unfreeze entire model
                for param in model.parameters():
                    param.requires_grad = True
                
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=0.00001,
                    weight_decay=1e-4
                )
                
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())

                print(f"\n{'='*70}")
                print(f"STAGE 3: Unfroze entire model (lr=0.00001)")
                print(f"         Trainable params: {trainable:,} / {total_params:,}")
                print(f"{'='*70}\n")

        elif training_type == "SCRATCH":
            if epoch == 0:
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"Training from SCRATCH (lr=0.001)")
                print(f"Trainable params: {trainable:,} / {total_params:,}")
                print(f"{'='*70}\n")

        train_loss, train_acc = train_esc50(
            model=model,
            dataloader=train_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device
        )
        
        val_loss, val_acc = test_esc50(
            model=model,
            dataloader=test_dataloader,
            loss_function=loss_function,
            device=device
        )
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch [{epoch+1:3d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr,
            "epoch_time": epoch_time,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_esc50_model.pth")
            print(f"         ✓ Saved best model (val_acc: {val_acc:.2f}%, val_loss: {val_loss:.4f})")
            wandb.save("best_esc50_model.pth")


    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"ESC-50 TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total Training Time    : {total_time/60:.1f} minutes")
    print(f"Best Validation Acc    : {best_val_acc:.2f}%")
    print(f"Best Validation Loss   : {best_val_loss:.4f}")
    print(f"Model saved as         : best_esc50_model.pth")
    print(f"{'='*70}\n")
    
    return best_val_acc


def train_genre(model, dataloader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for chunks, labels, chunk_counts in dataloader:
        chunks = chunks.to(device)
        labels = labels.to(device)

        logits = model(chunks)

        song_logits = []
        start_idx = 0
        true_labels = []

        for count in chunk_counts:
            end_idx = start_idx + count
            song_chunk_logits = logits[start_idx:end_idx]
            avg_logits = song_chunk_logits.mean(dim=0, keepdim=True)
            song_logits.append(avg_logits)
            true_labels.append(labels[start_idx])
            start_idx = end_idx

        song_logits = torch.cat(song_logits, dim=0)
        true_labels = torch.stack(true_labels)
        
        loss = loss_function(song_logits, true_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(song_logits, 1)
        correct += (predicted == true_labels).sum().item()
        total += true_labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    return epoch_loss, accuracy


def test_genre(model, dataloader, loss_function, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for chunks, labels, chunk_counts in dataloader:
            chunks = chunks.to(device)
            labels = labels.to(device)

            logits = model(chunks) 
            
            song_logits = []
            start_idx = 0
            true_labels = []
            
            for count in chunk_counts:
                end_idx = start_idx + count
                song_chunk_logits = logits[start_idx:end_idx]
                avg_logits = song_chunk_logits.mean(dim=0, keepdim=True)
                song_logits.append(avg_logits)
                true_labels.append(labels[start_idx])
                start_idx = end_idx
            
            song_logits = torch.cat(song_logits, dim=0)
            true_labels = torch.stack(true_labels)
            
            loss = loss_function(song_logits, true_labels)
            running_loss += loss.item()

            _, predicted = torch.max(song_logits, 1)
            correct += (predicted == true_labels).sum().item()
            total += true_labels.size(0)

    loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0

    return loss, accuracy


def train_genre_model(model, device, train_dataloader, test_dataloader, epochs=50):
    print(f"\n{'='*70}")
    print(f"GENRE CLASSIFICATION TRAINING")
    print(f"{'='*70}\n")
    
    if device == "cuda":
        torch.cuda.empty_cache()

    wandb.init(
        project=f"genre-classification",
        name=f"Training transfer learning",
        config={
            "epochs": epochs,
            "optimizer": "Adam",
            "dataset": "Genre",
            "architecture": model.__class__.__name__,
        }
    )
    wandb.watch(model, log="all", log_freq=50)

    print("\n3. Setting up training...")
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()

    start_time = time.time()
    best_val_acc = 0
    best_val_loss = float('inf')
    
    print(f"\n4. Starting training for {epochs} epochs...")
    print(f"{'='*70}\n")

    change_stage = epochs // 3

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    for epoch in range(epochs):
        epoch_start = time.time()

        if epoch == 0:
            # STAGE 1: Only train final head
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.001,
                weight_decay=1e-4
            )
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())

            print(f"STAGE 1: Training only final head (lr=0.001)")
            print(f"         Trainable params: {trainable:,} / {total_params:,}")
            print(f"{'='*70}\n")
      
        elif epoch == change_stage:
            # STAGE 2: Unfreeze LSTM + attention + head
            for name, param in model.named_parameters():
                if 'lstm' in name or 'attention' in name or 'head' in name:
                    param.requires_grad = True
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.0001,
                weight_decay=1e-4
            )
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())

            print(f"\n{'='*70}")
            print(f"STAGE 2: Unfroze LSTM + Attention + Head (lr=0.0001)")
            print(f"         Trainable params: {trainable:,} / {total_params:,}")
            print(f"{'='*70}\n")
            
        elif epoch == change_stage * 2:
            # STAGE 3: Unfreeze entire model
            for param in model.parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.00005,
                weight_decay=1e-4
            )
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())

            print(f"\n{'='*70}")
            print(f"STAGE 3: Unfroze entire model (lr=0.00005)")
            print(f"         Trainable params: {trainable:,} / {total_params:,}")
            print(f"{'='*70}\n")

        train_loss, train_acc = train_genre(
            model=model,
            dataloader=train_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device
        )
        
        val_loss, val_acc = test_genre(
            model=model,
            dataloader=test_dataloader,
            loss_function=loss_function,
            device=device
        )

        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch [{epoch+1:3d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr,
            "epoch_time": epoch_time,
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_genre_model.pth")
            print(f"         ✓ Saved best model (val_acc: {val_acc:.2f}%, val_loss: {val_loss:.4f})")
            wandb.save("best_genre_model.pth")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"GENRE CLASSIFICATION TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total Training Time    : {total_time/60:.1f} minutes")
    print(f"Best Validation Acc    : {best_val_acc:.2f}%")
    print(f"Best Validation Loss   : {best_val_loss:.4f}")
    print(f"Model saved as         : best_genre_model.pth")
    print(f"{'='*70}\n")
    
    return best_val_acc
