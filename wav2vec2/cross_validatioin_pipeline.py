import random
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Subset, WeightedRandomSampler, DataLoader, Dataset

from wav2vec2_model import Wav2Vec2ForGenre
from utils import train_wav2vec2_cross_validation, train_wav2vec2_single_run
from wave2vec2_dataset import Wav2Vec2_Audio_Dataset, collate_fn 

from helper.golbal_variables import *


class Wav2Vec2Pipeline:
    def __init__(
        self,
        genre_dir: Path,
        esc50_dir: Path,
        batch_size: int = 16,
        num_workers: int = 4,
        num_classes: int = 10,
        epochs: int = 30,
        n_folds: int = 5,
        val_split: float = 0.1,
        seed: int = 2,
    ):
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.n_folds = n_folds
        self.num_classes = num_classes
        self.seed = seed

        self.model_fn = lambda: Wav2Vec2ForGenre(num_classes=num_classes)

        self.train_loader, self.val_loader, \
        self.train_dataset, self.val_dataset, \
        self.full_dataset = self._load_data(genre_dir, esc50_dir, val_split)

    def _load_data(self, genre_dir: Path, esc50_dir: Path, val_split: float):
        """
        One dataset instance split by song ID — no song leaks between
        train and val. Songs are shuffled before splitting.
        Noise injection is disabled for val (train=False).
        """
        # Full dataset with augmentation ON (train split)
        full_dataset = Wav2Vec2_Audio_Dataset(
            genre_dir=genre_dir, esc50_dir=esc50_dir, train=True
        )

        # Unique song IDs
        unique_song_ids = list({song_idx for song_idx, _ in full_dataset.index_map})

        random.seed(self.seed)
        random.shuffle(unique_song_ids)

        split = int((1 - val_split) * len(unique_song_ids))
        train_songs = set(unique_song_ids[:split])
        val_songs = set(unique_song_ids[split:])

        print(f"\nSong split : {len(train_songs)} train  |  {len(val_songs)} val")

        train_indices = [
            i for i, (song_idx, _) in enumerate(full_dataset.index_map)
            if song_idx in train_songs
        ]
        val_indices = [
            i for i, (song_idx, _) in enumerate(full_dataset.index_map)
            if song_idx in val_songs
        ]

        # Separate dataset instance for val with noise injection OFF
        val_dataset_no_aug = Wav2Vec2_Audio_Dataset(
            genre_dir=genre_dir, esc50_dir=esc50_dir, train=False
        )

        train_subset = Subset(full_dataset, train_indices)
        val_subset   = Subset(val_dataset_no_aug, val_indices)

        print(f"Train chunks: {len(train_subset)}")
        print(f"Val chunks  : {len(val_subset)}")

        # Weighted sampler for class balance
        train_labels = [full_dataset.labels[full_dataset.index_map[i][0]] for i in train_indices]
        class_counts = Counter(train_labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[lbl] for lbl in train_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        print("\nClass distribution (train chunks):")
        genre_names = {v: k for k, v in full_dataset.class_map.items()}
        for cls, count in sorted(class_counts.items()):
            print(f"  {genre_names[cls]:20s}: {count:5d} chunks  "
                  f"(weight: {class_weights[cls]:.5f})")

        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda"),
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda"),
        )

        return train_loader, val_loader, train_subset, val_subset, full_dataset
    

    def _make_fold_loaders(self, train_idx, val_idx):
        """
        Called once per CV fold. Builds train/val DataLoaders from fold indices.
        All DataLoader config lives here in the pipeline.
        Returns (train_loader, val_loader, val_subset).
        """
        train_subset = Subset(self.full_dataset, train_idx)
        val_subset = Subset(self.full_dataset, val_idx)

        fold_labels = [self.full_dataset.labels[self.full_dataset.index_map[i][0]] for i in train_idx]
        class_counts = Counter(fold_labels)
        class_weights  = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[lbl] for lbl in fold_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda"),
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda"),
        )
        return train_loader, val_loader, val_subset
    

    def single_run(self, save_dir: str = "wav2vec2_single_run"):
        """Train one model on the full train split."""
        model = self.model_fn().to(self.device)
        return train_wav2vec2_single_run(
            model=model,
            device=self.device,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            val_dataset_subset=self.val_dataset,
            original_dataset=self.full_dataset,
            epochs=self.epochs,
            save_dir=save_dir,
        )

    def cross_validation(self, save_dir: str = "cv_models_wav2vec2"):
        """
        Stratified K-Fold CV — each fold trains a fresh model.
        Saved models can be loaded for ensemble inference.
        """
        fold_results = train_wav2vec2_cross_validation(
            model_fn=self.model_fn,
            device=self.device,
            full_dataset=self.full_dataset,
            fold_data_fn=self._make_fold_loaders,
            n_folds=self.n_folds,
            epochs=self.epochs,
            save_dir=save_dir,
        )
        print(fold_results)
        return fold_results
