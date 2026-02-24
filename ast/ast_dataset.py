import torch
from torch.utils.data import Dataset
from pathlib import Path

class Fast_AST_Dataset(Dataset):
    def __init__(self, data_dir: Path):

        self.data_dir = data_dir
        
        self.samples = []
        self.labels = []
        self.class_map = {}
        self.index_map = [] 
        self._load_files()
        
        print(f"Loaded {len(self.samples)} pre-computed spectrogram chunks.")

    def _load_files(self):
        for label_id, genre_dir in enumerate(sorted(self.data_dir.iterdir())):
            if not genre_dir.is_dir():
                continue

            genre_name = genre_dir.name 
            self.class_map[genre_name] = label_id

            # Get .pt files and sort them numerically by chunk number
            chunk_paths = sorted(
                list(genre_dir.glob("*.pt")), 
                key=lambda p: int(p.stem.split('_')[1])
            )

            if len(chunk_paths) == 0:
                continue 
            
            # Append the path AND the label together so they map 1-to-1
            for chunk_path in chunk_paths:
                self.samples.append(chunk_path)
                self.labels.append(label_id)
                
                # Reconstruct song_idx for your data leakage split
                # 3 chunks per song (0,1,2 -> song 0 | 3,4,5 -> song 1)
                chunk_num = int(chunk_path.stem.split('_')[1])
                song_idx = chunk_num // 3
                
                # Append to index_map (song_idx, dummy_start, dummy_aug)
                self.index_map.append((song_idx, 0, 0))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # The .pt file already contains the (spectrogram, label) tuple
        spectrogram, label = torch.load(self.samples[idx])
        
        return spectrogram, label
    


def fast_collate_fn(batch):
    """
    Collate function for pre-computed AST data.
    Takes a list of tuples (spectrogram, label) and stacks them into batch tensors.
    """
    # 1. Unpack the batch list of tuples
    spectrograms = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 2. Stack the 2D spectrograms into a 3D batch tensor
    # Shape becomes: (Batch_Size, 1024, 128)
    batched_spectrograms = torch.stack(spectrograms, dim=0)
    
    # 3. Convert labels into a 1D tensor safely
    # This handles both raw integers and 0D tensors depending on how they saved
    clean_labels = [l.item() if isinstance(l, torch.Tensor) else l for l in labels]
    batched_labels = torch.tensor(clean_labels, dtype=torch.long)
    
    return batched_spectrograms, batched_labels

