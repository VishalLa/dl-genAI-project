import torch 
import copy
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from helper.custom_dataloader import ESC50_DataLoader, GenreDataLoader
import copy 
from joblib import Parallel, delayed

import wandb



def train(model, dataloader, loss_function, optimizer, device):

    model.to(device)

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for spectrum, label in dataloader:

        spectrum = spectrum.to(device)
        label = label.to(device)

        # forward
        outputs = model(spectrum)
        loss = loss_function(outputs, label)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # stats
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    return epoch_loss, accuracy


def test(model, dataloader, loss_function, device):

    model.to(device)
    
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for spectrum, label in dataloader:

            spectrum = spectrum.to(device)
            label = label.to(device)

            # forward
            outputs = model(spectrum)
            loss = loss_function(outputs, label)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

    loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0

    return loss, accuracy


def cross_validation(model, esc50_dir, meta_csv, device, epochs=25, batch_size=32, fold=5):
    fold_accuracies = []
    fold_loss = []

    for test_fold in range(1, fold + 1):
        print(f"\================= FOLD {test_fold} =================")

        # clear GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()

        wandb.init(
            project="esc50-audio-classification",
            name=f"fold_{test_fold}",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": 0.001,
                "optimizer": "Adam",
                "dataset": "ESC-50",
                "architecture": model.__class__.__name__,
            }
        )

        wandb.run.define_metric("val_accuracy", summary="max")
        wandb.run.define_metric("val_loss", summary="min")

        train_fold = [f for f in range(1, fold + 1) if f != test_fold]

        train_dataset = ESC50_DataLoader(esc50_dir, allowed_folds=train_fold)
        train_dataset.load_meta_data(meta_csv)

        test_dataset = ESC50_DataLoader(esc50_dir, allowed_folds=[test_fold])
        test_dataset.load_meta_data(meta_csv)

        # dataloader 
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        new_model = model.to(device)
        wandb.watch(new_model, log="all", log_freq=50)

        optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        loss_function = torch.nn.CrossEntropyLoss()

        best_acc = 0

        best_acc = 0
        for epoch in range(epochs):

            train_loss, train_accuracy = train(
                model=new_model,
                dataloader=train_loader,
                loss_function=loss_function,
                optimizer=optimizer,
                device=device, 
            )

            val_loss, val_accuracy = test(
                model=new_model,
                dataloader=test_loader,
                loss_function=loss_function,
                device=device
            )

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% "
                f"| Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
            )  

        
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "lr": optimizer.param_groups[0]["lr"]
            })

            # ---- Save best model ----
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                torch.save(new_model.state_dict(), f"best_model_fold{test_fold}.pth")
                wandb.save(f"best_model_fold{test_fold}.pth")


        all_preds = []
        all_labels = []

        new_model.eval()
        with torch.no_grad():
            for spectrum, label in test_loader:
                spectrum = spectrum.to(device)
                label = label.to(device)
                
                output = new_model(spectrum)

                _, preds = torch.max(output, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        class_names = [str(i) for i in range(50)]

        wandb.log({
            "confusion_matrix":
            wandb.plot.confusion_matrix(
                y_true=all_labels,
                preds=all_preds,
                class_names=class_names
            )
        })


        fold_accuracies.append(val_accuracy)
        fold_loss.append(val_loss)

        wandb.finish()

    print("\n========== FINAL RESULT ==========")
    print("Fold Accuracies:", fold_accuracies)
    print("Mean Accuracy:", np.mean(fold_accuracies))
    print("Std:", np.std(fold_accuracies))
