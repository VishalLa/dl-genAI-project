from ast_model import ASTGenreClassifier 
from utils import train_ast_cross_validation 
from ast_dataset import Fast_AST_Dataset, fast_collate_fn

import torch
from torch.utils.data import Subset, WeightedRandomSampler, DataLoader, Dataset
from pathlib import Path
from collections import Counter


class Pipeline:
    def __init__(
            self, 
            dataset_path: Path,         
            batch_size: int = 16, 
            num_workers: int = 4,
            num_classes: int = 10,
            epochs: int = 30,
            n_fold: int = 5
        ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size= batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.n_fold = n_fold

        self.base_model = ASTGenreClassifier(num_classes=num_classes)

        self.train_dataloader, self.val_dataloader, self.train_full_dataset = self.load_data(dataset_path=dataset_path)

    def load_data(self, dataset_path: Path) -> tuple[DataLoader, DataLoader, Dataset]:

        train_full_dataset = Fast_AST_Dataset(data_dir=dataset_path, train=True)
        val_full_dataset = Fast_AST_Dataset(data_dir=dataset_path, train=False) 

        unique_song_ids = list(set([item[0] for item in train_full_dataset.index_map]))
        num_songs = len(unique_song_ids)

        train_song_count = int(0.8 * num_songs)
        train_songs = set(unique_song_ids[:train_song_count])
        val_songs = set(unique_song_ids[train_song_count:])

        print(f'\nSongs split: {len(train_songs)} train, {len(val_songs)} val')

        train_indices = [
            i for i, (song_idx, _) in enumerate(train_full_dataset.index_map) 
            if song_idx in train_songs
        ]

        val_indices = [
            i for i, (song_idx, _) in enumerate(val_full_dataset.index_map) 
            if song_idx in val_songs
        ]

        train_dataset = Subset(train_full_dataset, train_indices)
        val_dataset = Subset(val_full_dataset,   val_indices)

        print(f'Train chunks: {len(train_dataset)}')
        print(f'Val chunks:   {len(val_dataset)}')

        train_labels = [train_full_dataset.labels[i] for i in train_indices]
        class_counts = Counter(train_labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        print('\nClass distribution in training chunks:')
        genre_names = {v: k for k, v in train_full_dataset.class_map.items()}
        for cls, count in sorted(class_counts.items()):
            print(f'  {genre_names[cls]}: {count} chunks  (weight: {class_weights[cls]:.5f})')

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=fast_collate_fn,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda')
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=fast_collate_fn,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda')
        )

        return train_dataloader, val_dataloader, train_full_dataset

    def cross_validation(self):
        fold_results = train_ast_cross_validation(
            model_fn=self.base_model,
            device=self.device,
            full_dataset=self.train_full_dataset,
            train_loader=self.train_dataloader,
            val_loader=self.val_dataloader,
            epochs=self.epochs,
        )

        print(fold_results)
        return fold_results
    