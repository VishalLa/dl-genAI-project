from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset_for_CRNN(Dataset):
    """
    ImageDataset_for_CRNN

    Custom PyTorch Dataset for loading spectrogram or image-based
    genre classification data for CRNN models.

    Expected directory structure:

        root_path/
            genre_1/
                image1.png
                image2.png
            genre_2/
                image3.png
                image4.png

    This dataset:
    - Scans folders as class labels
    - Loads .png images
    - Resizes images to a fixed size
    - Converts to grayscale or RGB
    - Normalizes pixel values to [-1, 1]
    - Returns (image_tensor, label)
    """

    def __init__(self, path: Path, target_size: tuple = (400, 600), in_channel: int = 1):
        """
        Initializes the dataset.

        Args:
            path (Path): Root directory containing genre folders.
            target_size (tuple): Desired image size (height, width).
                                 Default = (400, 600)
            in_channel (int): Number of image channels.
                              1 → Grayscale
                              3 → RGB
        """
        self.target_size = target_size
        self.in_channel = in_channel
        self.path = path

        self.samples = []
        self.labels = []
        self.class_map = {}

        """
        Transform sequence:
            1. Resize to fixed dimensions
            2. Convert to grayscale (if needed)
            3. Convert to tensor (range [0, 1])
            4. Normalize to [-1, 1]
        """
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.Grayscale(num_output_channels=self.in_channel),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5] * self.in_channel,
                std=[0.5] * self.in_channel
            )
        ])

        self._load_files()

    def _load_files(self):
        """
        Scans the root directory and builds:

            self.samples → list of image paths
            self.labels → numeric labels
            self.class_map → {genre_name: label_id}

        Also prints:
            - Class mapping
            - Total image count
            - Images per genre
        """
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
        """
        Returns total number of images in dataset.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Loads and returns one sample.

        Args:
            index (int): Index of image

        Returns:
            Tuple:
                - image_tensor (torch.Tensor)
                    Shape: [C, H, W]
                    Example: [1, 400, 600]
                - label (torch.LongTensor)
        """
        img_path = self.samples[index]
        label = self.labels[index]

        try:
            # Load image
            img = Image.open(img_path).convert(
                'L' if self.in_channel == 1 else 'RGB'
            )

            # Apply transforms
            img_tensor = self.transform(img)

        except Exception as e:
            print(f"Error loading {img_path}: {e}")

            # Fallback in case of corrupted image
            img_tensor = torch.zeros(self.in_channel, *self.target_size)

        return img_tensor, torch.tensor(label, dtype=torch.long)
    