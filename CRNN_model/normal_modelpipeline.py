from collections import Counter

from pathlib import Path 
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader, random_split


from CRNN_model.crnn import CRNN
from CRNN_model.crnn_dataloader_for_images import ImageDataset_for_CRNN 
from CRNN_model.utils import train_on_pre_loaded_images


class ModelPipeline:
    def __init__(self, path: Path, epochs:int=30, num_class: int = 10, target_size: tuple = (400, 600), batch_size:int=64):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path = path
        self.epochs = epochs
        self.num_class = num_class
        self.target_size = target_size
        self.batch_size = batch_size

    def get_dataloaders(self):
        print("Loading genre dataset")
        dataset = ImageDataset_for_CRNN(path=self.path, target_size=self.target_size)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_indices = train_dataset.indices
        train_labels = [dataset.labels[i] for i in train_indices]

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
        genre_names = {v: k for k, v in dataset.class_map.items()}
        for cls, count in sorted(class_counts.items()):
            print(f"  {genre_names[cls]}: {count} songs (weight: {class_weights[cls]:.3f})")

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )

        test_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )

        return train_dataloader, test_dataloader
    

    def train(self, channel: int, linear_layer_out: int, rnn_hidden: int):

        model = CRNN(
            num_class=self.num_class,
            channel=channel,
            height=self.target_size[0],
            width=self.target_size[1],
            linear_layer_out=linear_layer_out,
            rnn_hidden=rnn_hidden
        )

        train_dataloader, val_dataloader = self.get_dataloaders()

        best_acc = train_on_pre_loaded_images(
            model=model,
            device=self.device,
            train_dataloader=train_dataloader,
            test_dataloader=val_dataloader,
            training_type="STRACH",
            epochs=self.epochs
        )

        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Best genre accuracy   : {best_acc:.2f}%")
        print("="*70 + "\n")
        