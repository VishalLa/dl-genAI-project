import torch
import random
from torch.utils.data import Dataset
from pathlib import Path


class Fast_AST_Dataset(Dataset):

    def __init__(self, data_dir: Path, train: bool = True):
        self.data_dir = data_dir
        self.train = train

        self.samples = []
        self.labels = []
        self.class_map = {}
        self.index_map = []   # list of (global_song_idx, chunk_num)

        self._load_files()

        mode = "train" if train else "val"
        print(f"[{mode}] Loaded {len(self.samples)} pre-computed spectrogram chunks.")

    def _load_files(self):
        global_song_counter = 0

        for label_id, genre_dir in enumerate(sorted(self.data_dir.iterdir())):
            if not genre_dir.is_dir():
                continue

            genre_name = genre_dir.name
            self.class_map[genre_name] = label_id

            # Sort .pt files numerically by chunk number
            chunk_paths = sorted(
                list(genre_dir.glob("*.pt")),
                key=lambda p: int(p.stem.split('_')[1])
            )

            if len(chunk_paths) == 0:
                continue

            chunks_per_song = 3  # matches pre_compute_spec.py (10.24s * 3 ~ 30s song)

            for chunk_path in chunk_paths:
                chunk_num = int(chunk_path.stem.split('_')[1])
                # local song index within this genre
                local_song_idx = chunk_num // chunks_per_song
                # global unique song index across all genres
                global_song_idx = global_song_counter + local_song_idx

                self.samples.append(chunk_path)
                self.labels.append(label_id)
                self.index_map.append((global_song_idx, chunk_num))

            max_chunk = int(chunk_paths[-1].stem.split('_')[1])
            n_songs_this_genre = (max_chunk // chunks_per_song) + 1
            global_song_counter += n_songs_this_genre

    @staticmethod
    def spec_augment(spec: torch.Tensor) -> torch.Tensor:
        """
        spec: [time_frames, mel_bins] = [1024, 128]
        Applies random time and frequency masking in-place on a copy.
        """
        spec = spec.clone()
        T, F = spec.shape  # 1024, 128

        # Frequency masking (up to 20 mel bins masked, 2 masks) 
        for _ in range(2):
            f = random.randint(0, 20)
            if f == 0:
                continue
            f0 = random.randint(0, F - f)
            spec[:, f0:f0 + f] = 0.0

        # Time masking (up to 80 frames masked, 2 masks)
        for _ in range(2):
            t = random.randint(0, 80)
            if t == 0:
                continue
            t0 = random.randint(0, T - t)
            spec[t0:t0 + t, :] = 0.0

        return spec


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spectrogram, label = torch.load(self.samples[idx], weights_only=True)

        if self.train:
            spectrogram = self.spec_augment(spectrogram)

        return spectrogram, label


def fast_collate_fn(batch):
    """
    Collate pre-computed spectrograms into a batched tensor.
    Input:  list of (spectrogram [1024, 128], label)
    Output: (batch [B, 1024, 128], labels [B])
    """
    spectrograms = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    batched_spectrograms = torch.stack(spectrograms, dim=0)

    clean_labels = [l.item() if isinstance(l, torch.Tensor) else l for l in labels]
    batched_labels = torch.tensor(clean_labels, dtype=torch.long)

    return batched_spectrograms, batched_labels
