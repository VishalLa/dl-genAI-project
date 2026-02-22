import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
import time

import wandb


class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Don't stop
        
        self.counter += 1
        return self.counter >= self.patience 


def single_train_preload(model, dataloader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for img, label in dataloader:
        img, lable = img.to(device), label.to(device)

        logits = model(img)

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



def test_preload(model, dataloader, loss_function, device):
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


def train_on_pre_loaded_images(
    model, device, train_dataloader, test_dataloader, 
    training_type, epochs=30, patience=5
):
    print(f"\n{'='*70}")
    print(f"CRNN IMAGE CLASSIFICATION TRAINING")
    print(f"{'='*70}\n")

    early_stopping = EarlyStopping(patience=patience)
    
    if device == "cuda":
        torch.cuda.empty_cache()

    wandb.init(
        project="crnn-image-classification",
        name=f"training-{training_type}",
        config={
            "epochs": epochs,
            "optimizer": "Adam",
            "patience": patience,
            "dataset": "preloaded-spectrograms",
            "architecture": model.__class__.__name__,
            "training_type": training_type,
        }
    )
    wandb.watch(model, log="all", log_freq=50)

    print("Setting up training...")
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    
    start_time = time.time()

    optimizer = torch.optim.Adam(
        model.parameters(),
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

    best_val_acc = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss, train_acc = single_train_preload(
            model=model,
            dataloader=train_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device
        )
        
        val_loss, val_acc = test_preload(
            model=model,
            dataloader=test_dataloader,
            loss_function=loss_function,
            device=device
        )

        scheduler.step(val_loss)

        if early_stopping(val_loss):
            print(f"\n Early stopping triggered at epoch {epoch+1}")
            print(f"   Val loss hasn't improved for {patience} epochs")
            # torch.save(checkpoint, "best_crnn_preload_images.pth")
            wandb.finish()
            break
        
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
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
            }
            
            torch.save(checkpoint, "best_crnn_preload_images.pth")
            print(f"         ✓ Saved best model (val_acc: {val_acc:.2f}%, val_loss: {val_loss:.4f})")
            wandb.save("best_crnn_preload_images.pth")

    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total Training Time    : {total_time/60:.1f} minutes")
    print(f"Total Epochs Trained   : {epoch + 1} / {epochs}")
    print(f"Best Validation Acc    : {best_val_acc:.2f}%")
    print(f"Best Validation Loss   : {best_val_loss:.4f}")
    print(f"Model saved as         : best_crnn_preload_images.pth")
    print(f"{'='*70}\n")
    
    wandb.finish()
    
    return best_val_acc
