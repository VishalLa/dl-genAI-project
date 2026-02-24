from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset_for_CRNN(Dataset):
    def __init__(self, path: Path, target_size: tuple=(400, 600), in_channel:int=1):
        self.target_size = target_size
        self.in_channel = in_channel
        self.path = path

        self.samples = []
        self.labels = []
        self.class_map = {}

        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.Grayscale(num_output_channels=self.in_channel),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * self.in_channel, std=[0.5] * self.in_channel)
        ])

        self._load_files()

    
    def _load_files(self):
        for label_id, genre_dir in enumerate(sorted(self.path.iterdir())):

            if not genre_dir.is_dir():
                continue

            genre_name = genre_dir.name
            self.class_map[genre_name] = label_id

            all_images = sorted(list(genre_dir.glob("*.png")))
            
            if len(all_images) == 0:
                print(f"⚠ No images found in {genre_dir}")
                continue

            for img_path in all_images:
                self.samples.append(img_path)
                self.labels.append(label_id)

        print(f"Classes: {self.class_map}")
        print(f"Total images: {len(self.samples)}")

        from collections import Counter
        label_counts = Counter(self.labels)
        print("\nImages per genre:")
        for genre, class_id in sorted(self.class_map.items(), key=lambda x: x[1]):
            count = label_counts[class_id]
            print(f"  {genre:<12} {count:4d} images")

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path = self.samples[index]
        label = self.labels[index]

        try:
            # Load image
            img = Image.open(img_path).convert('L' if self.in_channel == 1 else 'RGB')
            
            # Apply transforms
            img_tensor = self.transform(img)  # (1, 400, 600)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img_tensor = torch.zeros(self.in_channel, *self.target_size)
        
        return img_tensor, torch.tensor(label, dtype=torch.long)
