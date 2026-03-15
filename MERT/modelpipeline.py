from mert_model import MERTGenreClassifier 
from utils import train_mert_cross_validation, train_single_run 
from mert_dataset import MERTAudioDataset, mert_collate_fn

import random
import torch
from torch.utils.data import Subset, WeightedRandomSampler, DataLoader, Dataset
from pathlib import Path
from collections import Counter


class MERTPipeline:
    def __init__(
        self,
        genre_dir: Path, 
        esc50_dir: Path,
        batch_size: int = 8,
        num_workers: int = 4,
        num_classes: int = 10,
        epochs: int = 30,
        n_folds: int = 5,
        sample_rate: int = 24000,
        chunk_duration: int = 30,
        use_weighted_layers: bool = True,
        seed: int = 42,
    ):
        """
        MERT Training Pipeline
        
        Args:
            genre_dir: Path to genre dataset directory
            esc50_dir: Path to ESC-50 noise dataset
            batch_size: Batch size for training (default 8, lower than AST due to larger model)
            num_workers: Number of data loading workers
            num_classes: Number of genre classes
            epochs: Number of training epochs
            n_folds: Number of folds for cross-validation
            sample_rate: Audio sample rate (must be 24000 for MERT)
            chunk_duration: Audio chunk duration in seconds (default 30)
            use_weighted_layers: Whether to use weighted layer aggregation
            seed: Random seed for reproducibility
        """

        self.genre_dir = genre_dir
        self.esc50_dir = esc50_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.n_folds = n_folds
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.use_weighted_layers = use_weighted_layers
        self.seed = seed

        # Set random seeds for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Model factory function
        self.model_fn = lambda: MERTGenreClassifier(
            num_classes=num_classes,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            use_weighted_layers=use_weighted_layers
        )

        # Load data
        self.train_dataloader, self.val_dataloader, self.full_dataset = self.load_data()

    def load_data(self) -> tuple[DataLoader, DataLoader, Dataset]:
        """
        Load and prepare MERT dataset with train/val split.
        
        Returns:
            tuple: (train_dataloader, val_dataloader, full_dataset)
        """
        print("\n" + "=" * 70)
        print("LOADING MERT DATASET")
        print("=" * 70)

        # Load train and val datasets separately
        train_full_dataset = MERTAudioDataset(
            genre_dir=self.genre_dir,
            esc50_dir=self.esc50_dir,
            train=True
        )
        
        val_full_dataset = MERTAudioDataset(
            genre_dir=self.genre_dir,
            esc50_dir=self.esc50_dir,
            train=False
        )

        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Audio duration: {self.chunk_duration} seconds")
        print(f"Expected audio length: {self.chunk_duration * self.sample_rate} samples")

        # Get unique songs from train dataset
        num_songs = len(train_full_dataset.samples)
        all_indices = list(range(num_songs))
        all_labels = train_full_dataset.labels

        # 80-20 train-val split (stratified by genre)
        from sklearn.model_selection import train_test_split
        
        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=0.2,
            stratify=all_labels,
            random_state=self.seed
        )

        print(f'\nSongs split: {len(train_indices)} train, {len(val_indices)} val')

        # Create subsets
        train_dataset = Subset(train_full_dataset, train_indices)
        val_dataset = Subset(val_full_dataset, val_indices)

        print(f'Train songs: {len(train_dataset)}')
        print(f'Val songs:   {len(val_dataset)}')

        # Calculate class weights for balanced sampling
        train_labels = [train_full_dataset.labels[i] for i in train_indices]
        class_counts = Counter(train_labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]

        # Create weighted sampler for balanced training
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        print('\nClass distribution in training set:')
        genre_names = {v: k for k, v in train_full_dataset.class_map.items()}
        for cls, count in sorted(class_counts.items()):
            print(f'  {genre_names[cls]}: {count} songs  (weight: {class_weights[cls]:.5f})')

        # Create dataloaders
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=mert_collate_fn,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda'),
            persistent_workers=True if self.num_workers > 0 else False
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=mert_collate_fn,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda'),
            persistent_workers=True if self.num_workers > 0 else False
        )

        print(f'\nDataLoader created:')
        print(f'  Train batches: {len(train_dataloader)}')
        print(f'  Val batches:   {len(val_dataloader)}')
        print("=" * 70 + "\n")

        return train_dataloader, val_dataloader, train_full_dataset


    def single_run(self, save_dir: str = "./mert_models"):
        """
        Perform a single training run (no cross-validation).
        
        Args:
            save_dir: Directory to save the best model
            
        Returns:
            float: Best validation accuracy achieved
        """
        print("\n" + "=" * 70)
        print("STARTING SINGLE TRAINING RUN")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Model: MERT-v1-330M")
        print(f"Weighted layers: {self.use_weighted_layers}")
        print("=" * 70 + "\n")

        model = self.model_fn()
        
        best_acc = train_single_run(
            model=model,
            train_loader=self.train_dataloader,
            val_loader=self.val_dataloader,
            device=self.device,
            epochs=self.epochs,
            save_dir=save_dir
        )

        print(f"\n{'=' * 70}")
        print(f"SINGLE RUN COMPLETE")
        print(f"{'=' * 70}")
        print(f"Best Validation Accuracy: {best_acc:.2f}%")
        print(f"Model saved to: {save_dir}/best_mert_genre.pth")
        print(f"{'=' * 70}\n")

        return best_acc
    

    def cross_validation(self, save_dir: str = "cv_models_mert"):
        """
        Perform k-fold cross-validation.
        
        Args:
            save_dir: Directory to save cross-validation models and results
            
        Returns:
            list: Results from each fold
        """
        print("\n" + "=" * 70)
        print("STARTING CROSS-VALIDATION")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Folds: {self.n_folds}")
        print(f"Epochs per fold: {self.epochs}")
        print(f"Model: MERT-v1-330M")
        print(f"Weighted layers: {self.use_weighted_layers}")
        print("=" * 70 + "\n")

        fold_results = train_mert_cross_validation(
            model_fn=self.model_fn,
            device=self.device,
            full_dataset=self.full_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            epochs=self.epochs,
            n_folds=self.n_folds,
            save_dir=save_dir,
            wandb_project="mert-genre-classification-cross-validation"
        )

        print("\n" + "=" * 70)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 70)
        for result in fold_results:
            print(f"Fold {result['fold']}: {result['best_val_acc']:.2f}% "
                  f"({result['time_minutes']:.1f} min)")
        print("=" * 70 + "\n")

        return fold_results
    

    def get_model_info(self):
        """Print model architecture and parameter information."""
        model = self.model_fn()
        params = model.count_parameters()
        
        print("\n" + "=" * 70)
        print("MERT MODEL INFORMATION")
        print("=" * 70)
        print(f"Model: {model.__class__.__name__}")
        print(f"Backbone: {model.model_name}")
        print(f"Number of classes: {model.num_classes}")
        print(f"Sample rate: {model.sample_rate} Hz")
        print(f"Chunk duration: {model.chunk_duration} seconds")
        print(f"Expected input shape: [batch, {model.expected_length}]")
        print(f"Weighted layer aggregation: {model.use_weighted_layers}")
        print(f"\nParameters:")
        print(f"  Total:     {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,} ({params['trainable_pct']:.2f}%)")
        print(f"  Frozen:    {params['frozen']:,}")
        print("=" * 70 + "\n")

        
