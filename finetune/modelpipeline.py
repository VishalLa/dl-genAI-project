import os 
import sys 

from pathlib import Path
import torch 
import torch.nn as nn 

from finetune.utils import train_esc50_model, train_genre_model
from finetune.cnn14 import Cnn14


"""
first train cnn14 on esc50 data 
then train on genure data
"""

class ModelPipeline:
    def __init__(self, esc50_dir: Path, genre_dir: Path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.esc50_dir = esc50_dir
        self.genre_dir = genre_dir

        self.esc50_num_classes = 50
        self.num_genres = 10


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
    

    def complete_training(self, esc50_epochs=60, genre_epochs=60, esc50_batch_size=64, genre_batch_size=32):
        
        model_esc50 = self.load_pretrained_cnn14()
        
        _ = train_esc50_model(
            model=model_esc50,
            esc50_dir=self.esc50_dir,
            meta_csv=self.esc50_dir / "meta" / "esc50.csv",
            device=self.device,
            epochs=esc50_epochs,
            batch_size=esc50_batch_size
        )

        del model_esc50
        if self.device == "cuda":
            torch.cuda.empty_cache()

        model_genre = self.load_esc50_trained_model()

        _ = train_genre_model(
            model=model_genre,
            genre_dir=self.genre_dir,
            device=self.device,
            epochs=genre_epochs,
            batch_size=genre_batch_size
        )

    
