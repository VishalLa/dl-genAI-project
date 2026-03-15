from collections import Counter

from pathlib import Path 
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader, random_split, Subset, Dataset
import numpy as np

from CRNN_model.crnn import CRNN
from CRNN_model.dataset import ImageDataset_for_CRNN 
from CRNN_model.utils import train_on_pre_loaded_images


class ModelPipeline:
    def __init__(
        self, 
        path: Path, 
        epochs:int=30, 
        num_class: int = 10, 
        target_size: tuple = (400, 600), 
        batch_size:int=64
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path = path
        self.epochs = epochs
        self.num_class = num_class
        self.target_size = target_size
        self.batch_size = batch_size

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader, Dataset, Dataset]:

        """
        Song-level train/val split — no data leakage.

        Uses index_map (song_idx, chunk_idx) from ImageDataset_for_CRNN
        to ensure all chunks of a song land in the same split, exactly
        like the AST pipeline.

        Returns:
            train_dataloader, val_dataloader,
            val_dataset (Subset),  val_full_dataset (ImageDataset_for_CRNN)
        """

        print("Loading genre dataset")
        train_full = ImageDataset_for_CRNN(
            path=self.path, target_size=self.target_size, train=True
        )
        val_full = ImageDataset_for_CRNN(
            path=self.path, target_size=self.target_size, train=False
        )

        unique_song_ids = list(set(item[0] for item in train_full.index_map))
        np.random.seed(42)
        np.random.shuffle(unique_song_ids)

        train_song_count = int(0.8 * len(unique_song_ids))
        train_songs = set(unique_song_ids[:train_song_count])
        val_songs = set(unique_song_ids[train_song_count:])

        print(f"Songs split: {len(train_songs)} train, {len(val_songs)} val")

        train_indices = [i for i, (sid, _) in enumerate(train_full.index_map) if sid in train_songs]
        val_indices   = [i for i, (sid, _) in enumerate(val_full.index_map) if sid in val_songs]

        train_dataset = Subset(train_full, train_indices)
        val_dataset   = Subset(val_full,   val_indices)

        print(f"Train chunks: {len(train_dataset)}")
        print(f"Val chunks  : {len(val_dataset)}")

        # Safety check
        train_song_set = set(train_full.index_map[i][0] for i in train_indices)
        val_song_set = set(val_full.index_map[i][0]   for i in val_indices)
        overlap = train_song_set & val_song_set
        assert len(overlap) == 0, f"⚠ Data leakage! {len(overlap)} songs in both splits"
        print("✅ No data leakage\n")

        train_labels = [train_full.labels[i] for i in train_indices]
        class_counts = Counter(train_labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[lbl] for lbl in train_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        print("Class distribution in training chunks:")
        genre_names = {v: k for k, v in train_full.class_map.items()}
        for cls, count in sorted(class_counts.items()):
            print(f"  {genre_names[cls]:<12} {count:5d} chunks "
                  f"(weight: {class_weights[cls]:.5f})")

        pin = (self.device == "cuda")

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=pin
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=pin
        )

        print(f"\n{'=' * 70}")
        print(f"DATALOADERS READY")
        print(f"{'=' * 70}")
        print(f"Train batches : {len(train_dataloader)}")
        print(f"Val batches   : {len(val_dataloader)}")
        print(f"{'=' * 70}\n")

        return train_dataloader, val_dataloader, val_dataset, val_full
    

    def train(self, channel: int, linear_layer_out: int, rnn_hidden: int):

        model = CRNN(
            num_class=self.num_class,
            channel=channel,
            height=self.target_size[0],
            width=self.target_size[1],
            linear_layer_out=linear_layer_out,
            rnn_hidden=rnn_hidden
        )

        train_dataloader, val_dataloader, val_dataset, val_full = self.get_dataloaders()

        best_acc = train_on_pre_loaded_images(
            model=model,
            device=self.device,
            train_dataloader=train_dataloader,
            test_dataloader=val_dataloader,
            training_type="CRNN-v2",
            epochs=self.epochs,
            val_dataset_subset=val_dataset,
            original_dataset=val_full
        )

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Best genre accuracy : {best_acc:.2f}%")
        print("=" * 70 + "\n")
        