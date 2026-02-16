import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from collections import Counter

import numpy as np

from helper.custom_dataloader import ESC50_DataLoader_Waveform, GenreDataLoader_Waveform_Chunked, collate_fn_chunks
# import wandb


def train_esc50(model, dataloader, loss_function, optimizer, device):

    model.to(device)

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for waveform, label in dataloader:  
        waveform = waveform.to(device)
        label = label.to(device)

        # Forward
        output_dict = model(waveform)
        logits = output_dict["clipwise_output"]

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
            # Remove the None parameter - model expects only input
            output_dict = model(spectrum)
            logits = output_dict["clipwise_output"]

            loss = loss_function(logits, label)

            running_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

    loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0

    return loss, accuracy



def train_esc50_model(model, device, train_dataloader, test_dataloader, epochs=100):
    print(f"\n{'='*70}")
    print(f"ESC-50 TRAINING")
    print(f"{'='*70}\n")
    
    # Clear GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()


    # wandb.init(
    #     project=f"esc50-classification",
    #     name=f"Traning finetune",
    #     config={
    #         "epochs": epochs,
    #         "batch_size": batch_size,
    #         "optimizer": "Adam",
    #         "dataset": "ESC-50",
    #         "architecture": model.__class__.__name__,
    #     }
    # )

    # wandb.watch(model, log="all", log_freq=50)

    print("\n3. Setting up training...")
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    
    # Initial optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    change_stage = epochs // 3

    best_val_acc = 0
    for epoch in range(epochs):
        if epoch == 0:
            # STAGE 1: Only train final layer
            for name, param in model.named_parameters():
                if 'fc_audioset' not in name:
                    param.requires_grad = False
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.001
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"STAGE 1: Training only final layer (lr=0.001)")
            print(f"         Trainable params: {trainable:,}")
            print(f"{'='*70}\n")
        
        elif epoch == change_stage:
            # STAGE 2: Unfreeze last 2 conv blocks
            for name, param in model.named_parameters():
                if 'conv_block5' in name or 'conv_block6' in name or 'fc' in name:
                    param.requires_grad = True
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.0001
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"\n{'='*70}")
            print(f"STAGE 2: Unfroze conv_block5, conv_block6 (lr=0.0001)")
            print(f"         Trainable params: {trainable:,}")
            print(f"{'='*70}\n")
        
        elif epoch == change_stage * 2:
            # STAGE 3: Unfreeze all conv blocks
            for name, param in model.named_parameters():
                if 'spectrogram' not in name and 'logmel' not in name:
                    param.requires_grad = True
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.00001
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"\n{'='*70}")
            print(f"STAGE 3: Unfroze all conv blocks (lr=0.00001)")
            print(f"         Trainable params: {trainable:,}")
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
        
        print(
            f"Epoch [{epoch+1:3d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}%"
        )

        # wandb.log({
        #     "epoch": epoch + 1,
        #     "train_loss": train_loss,
        #     "train_accuracy": train_acc,
        #     "val_loss": val_loss,
        #     "val_accuracy": val_acc,
        #     "lr": optimizer.param_groups[0]["lr"]
        # })

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_esc50_model.pth")
            print(f"         ✓ Saved best model (val_acc: {val_acc:.2f}%)")
            # wandb.save("best_esc50_model.pth")

    print(f"\n{'='*70}")
    print(f"ESC-50 TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved as: best_esc50_model.pth")
    print(f"{'='*70}\n")
    
    return best_val_acc





def train_genre(model, dataloader, loss_function, optimizer, device):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for chunks, labels, chunk_counts in dataloader:
        # chunks: [total_chunks_in_batch, 1, 160000]
        # labels: [total_chunks_in_batch] - repeated for each chunk
        # chunk_counts: [batch_size] - number of chunks per song
        
        chunks = chunks.to(device)
        labels = labels.to(device)

        output_dict = model(chunks)
        logits = output_dict["clipwise_output"]  # [total_chunks, num_genres]

        song_logits = []
        start_idx = 0
        true_labels = []

        for count in chunk_counts:
            end_idx = start_idx + count
            
            # Get this song's chunk predictions
            song_chunk_logits = logits[start_idx:end_idx]  # [6, num_genres]
            
            # Average across the 6 chunks
            avg_logits = song_chunk_logits.mean(dim=0, keepdim=True)  # [1, num_genres]
            song_logits.append(avg_logits)
            
            # Get true label (same for all chunks)
            true_labels.append(labels[start_idx])
            
            start_idx = end_idx

        song_logits = torch.cat(song_logits, dim=0)  # [batch_size, num_genres]
        true_labels = torch.stack(true_labels)        # [batch_size]
        
        # Loss on averaged predictions
        loss = loss_function(song_logits, true_labels)

        # Backward (same as ESC-50)
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

            output_dict = model(chunks)
            logits = output_dict["clipwise_output"]
            
            # Average predictions per song
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
    
    # Clear GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()


    # wandb.init(
    #     project=f"genre-classification",
    #     name=f"Traning finetune",
    #     config={
    #         "epochs": epochs,
    #         "batch_size": batch_size,
    #         "optimizer": "Adam",
    #         "dataset": "ESC-50",
    #         "architecture": model.__class__.__name__,
    #     }
    # )

    # wandb.watch(model, log="all", log_freq=50)

    print("\n3. Setting up training...")
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    best_val_acc = 0
    
    print(f"\n4. Starting training for {epochs} epochs...")
    print(f"{'='*70}\n")

    change_stage = epochs // 3

    for epoch in range(epochs):
        if epoch == 0:
            # STAGE 1: Only train final layer
            for name, param in model.named_parameters():
                if 'fc_audioset' not in name:
                    param.requires_grad = False
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.001
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"STAGE 1: Training only final layer (lr=0.001)")
            print(f"         Trainable params: {trainable:,}")
            print(f"{'='*70}\n")
        
        elif epoch == change_stage:
            # STAGE 2: Unfreeze last 2 conv blocks
            for name, param in model.named_parameters():
                if 'conv_block5' in name or 'conv_block6' in name or 'fc' in name:
                    param.requires_grad = True
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.0001
            )

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n{'='*70}")
            print(f"STAGE 2: Unfroze conv_block5, conv_block6 (lr=0.0001)")
            print(f"         Trainable params: {trainable:,}")
            print(f"{'='*70}\n")
        
        elif epoch == change_stage * 2:
            # STAGE 3: Unfreeze all conv blocks
            for name, param in model.named_parameters():
                if 'spectrogram' not in name and 'logmel' not in name:
                    param.requires_grad = True
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.00005
            )

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n{'='*70}")
            print(f"STAGE 3: Unfroze all conv blocks (lr=0.00005)")
            print(f"         Trainable params: {trainable:,}")
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

        print(
            f"Epoch [{epoch+1:3d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}%"
        )

        # wandb.log({
        #     "epoch": epoch + 1,
        #     "train_loss": train_loss,
        #     "train_accuracy": train_acc,
        #     "val_loss": val_loss,
        #     "val_accuracy": val_acc,
        #     "lr": optimizer.param_groups[0]["lr"]
        # })
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_genre_model.pth")
            print(f"         ✓ Saved best model (val_acc: {val_acc:.2f}%)")
            # wandb.save("best_genre_model.pth")

    print(f"\n{'='*70}")
    print(f"GENRE CLASSIFICATION TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved as: best_genre_model.pth")
    print(f"{'='*70}\n")
    
    return best_val_acc

