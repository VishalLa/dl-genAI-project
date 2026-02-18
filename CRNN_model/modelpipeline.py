from collections import Counter

from pathlib import Path 
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader, random_split


from CRNN_model.crnn import CRNN
from helper.DataLoaders.dataloader_with_preprocessed import GenreDataLoader, ESC50_DataLoader, collate_fn_chunks
from helper.utils import train_esc50_model, train_genre_model


class ModelPipeline:
    def __init__(
            self, esc50_dir: Path, genre_dir: Path, esc50_epochs: int=50, genre_epochs=30, esc50_batch_size: int=64, genre_batch_size: int = 32
    ):
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
        train_dataset = ESC50_DataLoader(esc50_dir=self.esc50_dir, allowed_folds=[1, 2, 3, 4], augment=True)
        train_dataset.load_meta_data(self.esc50_dir / "meta" / "esc50.csv")
        test_dataset = ESC50_DataLoader(esc50_dir=self.esc50_dir, allowed_folds=[5])
        test_dataset.load_meta_data(self.esc50_dir / "meta" / "esc50.csv")

        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.esc50_batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False  
        )
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.esc50_batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )

        return train_loader, test_loader
    

    def get_genre_dataloader(self):
        print("Loading genre dataset...")
        genre_dataset = GenreDataLoader(root_dir=self.genre_dir, augment=True)

        train_size = int(0.8 * len(genre_dataset))
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
            sampler=sampler,
            collate_fn=collate_fn_chunks,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )

        test_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.genre_batch_size,
            shuffle=False,
            collate_fn=collate_fn_chunks,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )

        return train_dataloader, test_dataloader
    

    def load_esc50_trained_model(self):
        print("\n" + "="*70)
        print("LOADING ESC-50 TRAINED MODEL FOR GENRES")
        print("="*70)

        model = CRNN(
            num_class=50,
            channel=1,
            height=128,
            width=216,
            linear_layer_out=64,
            rnn_hidden=128
        )
        checkpoint = torch.load("best_esc50_model.pth", map_location=self.device)
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded ESC-50 trained CRNN model")
        
        in_features = model.head.in_features  # Should be 256 (rnn_hidden * 2)
        model.head = nn.Linear(in_features, self.num_genres)
        print(f"✓ Replaced final head: {in_features} -> {self.num_genres} genres")
        
        # Freeze CNN + conv_to_fc (only LSTM + attention + head train initially)
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ Initially frozen: CNN + conv_to_fc")
        print(f"✓ Trainable: {trainable_params:,} / {total_params:,} (head only)")
        print("="*70 + "\n")
        
        return model.to(self.device)


    def train(
            self, channel: int, height: int, width: int, linear_layer_out: int, rnn_hidden: int
        ):

        """
        Two-stage training:
        1. Train CRNN from scratch on ESC-50 (50 classes, 5s clips)
        2. Transfer to genre classification (10 classes, 30s songs → 6 chunks)
        """

        # train on esc50 data
        esc50_model = CRNN(
            num_class=self.esc50_num_classes,
            channel=channel,
            height=height,
            width=width,
            linear_layer_out=linear_layer_out,
            rnn_hidden=rnn_hidden
        )

        esc50_train_dataloader, esc50_test_dataloader = self.get_esc50_dataloader()

        _ = train_esc50_model(
            model=esc50_model,
            device=self.device,
            train_dataloader=esc50_train_dataloader,
            test_dataloader=esc50_test_dataloader,
            training_type="SCRATCH",
            epochs=self.esc50_epochs
        )

        del esc50_model, esc50_test_dataloader, esc50_train_dataloader 

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()

        genre_model = self.load_esc50_trained_model()

        genre_train_dataloader, genre_test_datalaoder = self.get_genre_dataloader()

        best_genre_acc = train_genre_model(
            model=genre_model,
            device=self.device,
            epochs=self.genre_epochs,
            train_dataloader=genre_train_dataloader,
            test_dataloader=genre_test_datalaoder
        )


        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"ESC-50 model saved    : best_esc50_model.pth")
        print(f"Genre model saved     : best_genre_model.pth")
        print(f"Best genre accuracy   : {best_genre_acc:.2f}%")
        print("="*70 + "\n")

