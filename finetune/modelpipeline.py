from collections import Counter

from pathlib import Path
import torch 
import torch.nn as nn 
from torch.utils.data import WeightedRandomSampler, DataLoader, random_split

from helper.utils import train_esc50_model, train_genre_model
from finetune.cnn14 import Cnn14

from helper.DataLoaders.custom_dataloader import (
    GenreDataLoader_Waveform_Chunked,
    ESC50_DataLoader_Waveform,
    collate_fn_chunks
)


"""
first train cnn14 on esc50 data 
then train on genure data
"""

class ModelPipeline:
    def __init__(self, esc50_dir: Path, genre_dir: Path, esc50_epochs:int=30, genre_epochs:int=30, esc50_batch_size: int=64, genre_batch_size: int=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.esc50_dir = esc50_dir
        self.genre_dir = genre_dir
        self.esc50_epochs = esc50_epochs
        self.genre_epochs = genre_epochs
        self.esc50_batch_size = esc50_batch_size
        self.genre_batch_size = genre_batch_size

        self.esc50_num_classes = 50
        self.num_genres = 10

    def get_esc50_dataloader(self):
        print("Loading ESC-50 dataset...")
        train_dataset = ESC50_DataLoader_Waveform(esc50_dir=self.esc50_dir, allowed_folds=[1, 2, 3, 4])
        train_dataset.load_meta_data(self.esc50_dir / "meta" / "esc50.csv")
        test_dataset = ESC50_DataLoader_Waveform(esc50_dir=self.esc50_dir, allowed_folds=[5])
        test_dataset.load_meta_data(self.esc50_dir / "meta" / "esc50.csv")

        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.esc50_batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.esc50_batch_size, 
            shuffle=False
        )

        return train_loader, test_loader
    

    def get_genre_dataloader(self):
        print("Loading genre dataset...")
        genre_dataset = GenreDataLoader_Waveform_Chunked(root_dir=self.genre_dir)

        train_size = int(0.9 * len(genre_dataset))
        val_size = len(genre_dataset) - train_size
        train_dataset, val_dataset = random_split(
            genre_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_indices = train_dataset.indices
        train_labels = [genre_dataset.labels[i] for i in train_indices]

        class_counts = Counter(train_labels)

        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
        print("Class distribution in training:")
        genre_names = {v: k for k, v in genre_dataset.class_map.items()}
        for cls, count in sorted(class_counts.items()):
            print(f"  {genre_names[cls]}: {count} songs (weight: {class_weights[cls]:.3f})")

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.genre_batch_size,
            sampler=sampler,  # NO shuffle when using sampler!
            collate_fn=collate_fn_chunks,
            num_workers=4
        )

        test_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.genre_batch_size,
            shuffle=False,
            collate_fn=collate_fn_chunks,
            num_workers=4
        )

        return train_dataloader, test_dataloader


    def load_pretrained_cnn14(self):

        print("\n" + "="*70)
        print("LOADING PRETRAINED CNN14 FOR ESC-50")
        print("="*70)

        model = Cnn14(classes_num=527)
        
        checkpoint = torch.load("pretrained_models/Cnn14.pth", map_location=self.device)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ Loaded pretrained CNN14 weights")

        # Replace final layer
        model.fc_audioset = nn.Linear(2048, self.esc50_num_classes)
        print(f"✓ Replaced final layer: {self.esc50_num_classes} classes")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ All layers trainable initially")
        print(f"✓ Trainable: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.1f}%)")
        print("="*70 + "\n")

        return model.to(self.device)

    def load_esc50_trained_model(self):

        print("\n" + "="*70)
        print("LOADING ESC-50 TRAINED MODEL FOR GENRES")
        print("="*70)

        # Create model with ESC-50 classes
        model = Cnn14(classes_num=50)
        
        # Load ESC-50 weights
        checkpoint = torch.load("best_esc50_model.pth", map_location=self.device)
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded ESC-50 model (trained on 5-second clips)")
        
        # Replace for genres
        model.fc_audioset = nn.Linear(2048, self.num_genres)
        print(f"✓ Replaced final layer: 2048 -> {self.num_genres} genres")
        
        # Freeze all except classifier
        for name, param in model.named_parameters():
            if 'fc_audioset' not in name:
                param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ All layers trainable initially")
        print(f"✓ Trainable: {trainable_params:,} / {total_params:,}")
        print("="*70 + "\n")
        
        return model.to(self.device)
    

    def complete_training(self):
        
        model_esc50 = self.load_pretrained_cnn14()
        
        esc50_train_dataloader, esc50_test_dataloader = self.get_esc50_dataloader()

        _ = train_esc50_model(
            model=model_esc50,
            device=self.device,
            epochs=self.esc50_epochs,
            train_dataloader=esc50_train_dataloader,
            test_dataloader=esc50_test_dataloader
        )

        del model_esc50, esc50_train_dataloader, esc50_test_dataloader
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()

        model_genre = self.load_esc50_trained_model()

        genre_train_dataloader, genre_test_datalaoder = self.get_genre_dataloader()

        _ = train_genre_model(
            model=model_genre,
            device=self.device,
            epochs=self.genre_epochs,
            train_dataloader=genre_train_dataloader,
            test_dataloader=genre_test_datalaoder
        )

    
