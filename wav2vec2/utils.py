import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import wandb


def train_wav2vec2(model, dataloader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for input_values, attention_mask, labels in dataloader:
        input_values = input_values.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_values, attention_mask)
        
        loss = loss_function(logits, labels)
        
        if torch.isnan(loss):
            print(f"⚠ NaN loss detected, skipping batch")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    return epoch_loss, accuracy


def test_wav2vec2(model, dataloader, loss_function, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

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

    loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0

    return loss, accuracy


def train_wav2vec2_model(model, device, train_loader, val_loader, epochs=30):
    
    print("\n" + "="*70)
    print("WAV2VEC2 GENRE CLASSIFICATION TRAINING")
    print("="*70 + "\n")
    
    if device == "cuda":
        torch.cuda.empty_cache()

    wandb.init(
        project="wav2vec2-genre-classification",
        name="3-stage-transfer-learning",
        config={
            "epochs": epochs,
            "optimizer": "Adam",
            "dataset": "Genre",
            "architecture": model.__class__.__name__,
            "base_model": "wav2vec2-base",
        }
    )
    wandb.watch(model, log="all", log_freq=50)

    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    
    start_time = time.time()
    best_val_acc = 0
    best_val_loss = float('inf')
    
    change_stage = epochs // 3  

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    for epoch in range(epochs):
        epoch_start = time.time()
        
        if epoch == 0:
            print(f"STAGE 1: Training only classifier head (lr=1e-3)")
            
            # Freeze entire wav2vec encoder
            for param in model.wav2vec.parameters():
                param.requires_grad = False
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3,
                weight_decay=1e-4
            )
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
            
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"         Trainable: {trainable:,} / {total:,}")
            print(f"{'='*70}\n")
            
            wandb.log({"stage": "classifier_only"}, step=epoch)

        elif epoch == change_stage:
            print(f"\n{'='*70}")
            print(f"STAGE 2: Unfroze top 2 encoder layers (lr=5e-5)")
            
            for name, param in model.wav2vec.named_parameters():
                if "encoder.layers.10" in name or "encoder.layers.11" in name:
                    param.requires_grad = True
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=5e-5,
                weight_decay=1e-4
            )
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
            
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"         Trainable: {trainable:,} / {total:,}")
            print(f"{'='*70}\n")
            
            wandb.log({"stage": "partial_unfreeze"}, step=epoch)

        elif epoch == change_stage * 2:
            print(f"\n{'='*70}")
            print(f"STAGE 3: Full model fine-tuning (lr=1e-5)")
            for name, param in model.wav2vec.named_parameters():
                if "feature_extractor" not in name:
                    param.requires_grad = True
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=1e-5,
                weight_decay=1e-4
            )
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
            
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"         Trainable: {trainable:,} / {total:,}")
            print(f"{'='*70}\n")
            
            wandb.log({"stage": "full_unfreeze"}, step=epoch)

        train_loss, train_acc = train_wav2vec2(
            model, train_loader, loss_function, optimizer, device
        )
        
        val_loss, val_acc = test_wav2vec2(
            model, val_loader, loss_function, device
        )
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr,
            "epoch_time": epoch_time,
        })

        print(
            f"Epoch [{epoch+1:3d}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_wav2vec2_genre.pth")
            wandb.save("best_wav2vec2_genre.pth")
            print(f"         ✓ Saved best model (val_acc: {val_acc:.2f}%, val_loss: {val_loss:.4f})")

    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"WAV2VEC2 TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total Training Time    : {total_time/60:.1f} minutes")
    print(f"Best Validation Acc    : {best_val_acc:.2f}%")
    print(f"Best Validation Loss   : {best_val_loss:.4f}")
    print(f"Model saved as         : best_wav2vec2_genre.pth")
    print(f"{'='*70}\n")

    wandb.finish()
    
    return best_val_acc
