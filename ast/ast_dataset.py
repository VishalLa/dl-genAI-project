import torch
import random
from torch.utils.data import Dataset
from pathlib import Path


class Fast_AST_Dataset(Dataset):

    """
    Dataset class for ast model which loads pre computed spectrum from genre_pt folder

    arguments:
        data_dir: path for genre_pt folder 
        train: bool value if true data will be agumented 
    """

    def __init__(self, data_dir: Path, train: bool = True):
        self.data_dir = data_dir
        self.train = train

        self.samples = []       # list of all chunks from data 
        self.labels = []
        self.class_map = {}     # dict for storing index and name of genre
        self.index_map = []   # list of (global_song_idx, chunk_num)

        self._load_files()

        mode = "train" if train else "val"
        n_songs = len({s for s, _ in self.index_map})
        print(f"[{mode}] {len(self.samples)} chunks from {n_songs} songs | {len(self.class_map)} classes")

    def _load_files(self):
        """
        load all files in samples list
        """

        global_song_counter = 0
        self.song_chunk_counts: dict[int, int] = {}

        # iterate over upper dir like blue, classical etc
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

            prev_chunk_num = None
            local_song_idx = 0
            chunk_idx_within = 0

            for chunk_path in chunk_paths:
                chunk_num = int(chunk_path.stem.split("_")[1])

                # Gap in numbering or first chunk -> new song
                if prev_chunk_num is None or chunk_num != prev_chunk_num + 1:
                    if prev_chunk_num is not None:
                        local_song_idx += 1
                    chunk_idx_within = 0

                global_song_idx = global_song_counter + local_song_idx

                self.samples.append(chunk_path)
                self.labels.append(label_id)
                self.index_map.append((global_song_idx, chunk_idx_within))

                # Track how many chunks this song has
                self.song_chunk_counts[global_song_idx] = (
                    self.song_chunk_counts.get(global_song_idx, 0) + 1
                )

                prev_chunk_num   = chunk_num
                chunk_idx_within += 1

            global_song_counter += local_song_idx + 1

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

    Input  : list of (spectrogram [1024, 128], label)
    Output : (batch_spectrograms [B, 1024, 128], batch_labels [B])
    """
    spectrograms = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor(
        [l.item() if isinstance(l, torch.Tensor) else l for l in [item[1] for item in batch]],
        dtype=torch.long,
    )
    return spectrograms, labels
