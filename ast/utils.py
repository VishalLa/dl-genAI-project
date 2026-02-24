import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import wandb

import torch
from torch.cuda.amp import autocast, GradScaler


def train_ast(model, dataloader, loss_function, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for input_values, labels in dataloader:
        input_values = input_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 1. Cast the forward pass to 16-bit precision
        with autocast():
            logits = model(input_values)
            loss = loss_function(logits, labels)
        
        if torch.isnan(loss):
            print(f"⚠ NaN loss detected, skipping batch")
            continue

        # 2. Scale the loss and do the backward pass
        scaler.scale(loss).backward()
        
        # 3. Unscale gradients before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 4. Step the optimizer and update the scaler
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    return epoch_loss, accuracy

def test_ast(model, dataloader, loss_function, device):
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

def train_ast_model(model, device, train_loader, val_loader, epochs=20):
    
    print("\n" + "="*70)
    print("AST GENRE CLASSIFICATION TRAINING (2-STAGE)")
    print("="*70 + "\n")
    
    if device == "cuda":
        torch.cuda.empty_cache()

    wandb.init(
        project="ast-genre-classification",
        name="2-stage-golden-run",
        config={
            "epochs": epochs,
            "optimizer": "AdamW",
            "dataset": "Genre",
            "architecture": model.__class__.__name__,
            "base_model": "ast-finetuned-audioset-10-10-0.4593",
        }
    )
    wandb.watch(model, log="all", log_freq=50)

    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    
    start_time = time.time()
    best_val_acc = 0
    best_val_loss = float('inf')
    
    # Split the epochs exactly in half for the 2 stages
    change_stage = epochs // 2

    optimizer = None
    scheduler = None

    # Initialize the scaler outside your training loop in your main script
    scaler = GradScaler() 

    for epoch in range(epochs):
        epoch_start = time.time()
        
        if epoch == 0:
            print(f"STAGE 1: Training only classifier head (lr=1e-3)")
            
            # Freeze everything
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze only the classification head
            for param in model.classifier.parameters():
                param.requires_grad = True
            
            # Upgraded to AdamW with higher weight decay for regularization
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3,
                weight_decay=0.01 
            )
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)
            
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"        Trainable: {trainable:,} / {total_params:,}")
            print(f"{'='*70}\n")
            
            wandb.log({"stage": "classifier_only"}, step=epoch)

        elif epoch == change_stage:
            print(f"\n{'='*70}")
            print(f"STAGE 2: Unfreezing top 4 AST encoder layers (lr=5e-5)")
            
            # Unfreeze layers 8, 9, 10, and 11 to capture deeper audio features
            for name, param in model.named_parameters():
                if any(f"encoder.layer.{i}" in name for i in [8, 9, 10, 11]):
                    param.requires_grad = True
            
            # Upgraded to AdamW
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=5e-5,
                weight_decay=0.01
            )
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
            
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"        Trainable: {trainable:,} / {total_params:,}")
            print(f"{'='*70}\n")
            
            wandb.log({"stage": "partial_unfreeze"}, step=epoch)

        train_loss, train_acc = train_ast(
            model, train_loader, loss_function, optimizer, device, scaler
        )
        
        val_loss, val_acc = test_ast(
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
            torch.save(model.state_dict(), "best_ast_genre.pth")
            wandb.save("best_ast_genre.pth")
            print(f"        ✓ Saved best model (val_acc: {val_acc:.2f}%, val_loss: {val_loss:.4f})")

    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"AST TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total Training Time    : {total_time/60:.1f} minutes")
    print(f"Best Validation Acc    : {best_val_acc:.2f}%")
    print(f"Best Validation Loss   : {best_val_loss:.4f}")
    print(f"Model saved as         : best_ast_genre.pth")
    print(f"{'='*70}\n")

    wandb.finish()
    
    return best_val_acc
