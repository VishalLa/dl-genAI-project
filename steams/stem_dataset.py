from __future__ import annotations

import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

import re
import random
import warnings
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


class AST_Dataset(Dataset):
    """
    Dataset over the precomputed AST spectrograms (.pt files).

    Every .pt file holds (spectrogram, genre_label, stem_id):
        flattened_* : spectrogram [1024, 128]    -> one stem of one 10.24 s chunk
        grouped_*   : spectrogram [4, 1024, 128] -> all 4 stems of one chunk

    Train / val split
        The split is done per SONG (never per chunk), stratified per genre, and is
        deterministic for a given `seed`. So `AST_Dataset(train=True)` and
        `AST_Dataset(train=False)` never share a song, even if one of them is built
        from the clean variant and the other from the augmented one.

        The file names only contain the global item index, not a song id, so the song
        is reconstructed from it. This relies on `chunks_per_song` (3 for 30 s songs:
        ceil(30 s / 10.24 s)) and on the order in which precompute wrote the files:
        song-major, then chunk, then stem. Check the printed songs-per-genre numbers
        (about 100 per genre for your data).

    Args:
        genre_dir        : root folder that contains the 4 variant folders (GENRES_PT).
        train            : True -> train songs, False -> validation songs.
        flatten_stems    : True -> flattened_* variants, False -> grouped_* variants.
        augment          : True -> *_augmented variants, False -> *_clean variants.
        val_ratio        : fraction of songs per genre held out for validation.
        seed             : split seed (use the same value for train and val datasets).
        chunks_per_song  : number of 10.24 s chunks per song in the precompute step.
        spec_augment     : SpecAugment-style frequency / time masking, train only.
        freq_mask        : max number of mel bins masked.
        time_mask        : max number of frames masked (10 ms per frame).
        return_stem_id   : flattened only, also return the stem id (0 bass .. 3 other).
    """
    
    STEM_KEYS = ["bass", "drums", "vocals", "other"]
    
    VARIANTS = {
        (True, True): "flattened_augmented",
        (True, False): "flattened_clean",
        (False, True): "grouped_augmented",
        (False, False): "grouped_clean",
    }
    
    _NAME_RE = re.compile(r"chunk_(\d+)(?:_c(\d+))?(?:_s(-?\d+))?")
    
    def __init__(
        self,
        genre_dir: Path | str,
        train: bool = True,
        flatten_stems: bool = True,
        augment: bool = True,
        val_ratio: float = 0.2,
        seed: int = 42,
        chunks_per_song: int = 3,
        spec_augment: bool = False,
        freq_mask: int = 24,
        time_mask: int = 96,
        return_stem_id: bool = False,
    ) -> None:
        
        self.train = train
        self.flatten_stems = flatten_stems
        self.augment = augment
        self.val_ratio = val_ratio
        self.seed = seed
        self.chunks_per_song = chunks_per_song
        self.return_stem_id = return_stem_id and flatten_stems

        # select which of the 4 precomputed variants to load
        self.data_dir = Path(genre_dir) / self.VARIANTS[(flatten_stems, augment)]
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"variant folder not found: {self.data_dir}")
        
        # optional SpecAugment on the cached spectrograms (train only)
        self.spec_augment = spec_augment and train
        if self.spec_augment:
            self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask)
            self.time_masking = torchaudio.transforms.TimeMasking(time_mask)
            
        
        self.samples = []    # paths of all chunks of the selected split
        self.labels = []     # genre label per sample
        self.stem_ids = []   # stem id per sample (-1 for grouped files)
        self.class_map = {}  # genre name -> label id
        self.index_map = []  # (song_id, position in self.samples)
        
        self._load_files()
        
        self.idx2genre = {v: k for k, v in self.class_map.items()}

        mode = "train" if train else "val"
        n_songs = len({s for s, _ in self.index_map})
        print(f"[{mode}] {len(self.samples)} chunks from {n_songs} songs | "
              f"{len(self.class_map)} classes | variant: {self.data_dir.name}")

        
    def _load_files(self) -> None:
        global_song_counter = 0
        self.song_chunk_counts: dict[int, int] = {}
        
        n_stems = len(self.STEM_KEYS) if self.flatten_stems else 1
        items_per_song = self.chunks_per_song * n_stems
        
        for label_id, genre_dir in enumerate(
            sorted(p for p in self.data_dir.iterdir() if p.is_dir())
        ):
            genre = genre_dir.name
            self.class_map[genre] = label_id
            
            entries = []
            for f in genre_dir.glob("*.pt"):
                m = self._NAME_RE.match(f.stem)
                if m is None:
                    continue
                
                i = int(m.group(1))
                c = int(m.group(2)) if m.group(2) is not None else 0
                s = int(m.group(3)) if m.group(3) is not None else -1
                entries.append((i, c, s, f))
                
            if not entries:
                continue
            entries.sort()
            
            item_ids = sorted({e[0] for e in entries})
            first=  item_ids[0]
            if item_ids[-1] - first + 1 != len(item_ids) or len(item_ids) % items_per_song != 0:
                warnings.warn(
                    f"[{genre}] {len(item_ids)} items is not a clean multiple of {items_per_song} "
                    f"(chunks_per_song={self.chunks_per_song}). Song boundaries may be wrong; "
                    f"check chunks_per_song or store a song id in the precompute step."
                )
                
            songs: dict[int, list] = {}
            for i, c, s, f in entries:
                songs.setdefault((i - first) // items_per_song, []).append((s, f))
            song_keys = sorted(songs)
            
            # deterministic, stratified split: same songs in val for every dataset instance
            order = song_keys[:]
            random.Random(f"{self.seed}-{genre}").shuffle(order)
            n_val = max(1, round(len(order) * self.val_ratio)) if len(order) > 1 else 0
            val_songs = set(order[:n_val])
            
            for key in song_keys:
                is_val = key in val_songs 
                if is_val == self.train:
                    continue
                song_id = global_song_counter + key
                for s, f in songs[key]:
                    self.samples.append(f)
                    self.labels.append(label_id)
                    self.stem_ids.append(s)
                    self.index_map.append((song_id, len(self.samples) -1))
                    self.song_chunk_counts[song_id] = self.song_chunk_counts.get(song_id, 0) + 1
                    
            global_song_counter += max(song_keys) + 1
            print(f"  {genre:10s} songs: {len(song_keys):3d} | val: {n_val:3d}")
            
    
    @property
    def song_ids(self) -> list[int]:
        return [s for s, _ in self.index_map]
    
    
    def _mask(self, spec: torch.Tensor) -> torch.Tensor:
        x = spec.transpose(0, 1)
        x = self.freq_masking(x)
        x = self.time_masking(x)
        return x.transpose(0, 1).contiguous()
    
    
    def _apply_spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.dim() == 2:
            return self._mask(spec)
        return torch.stack([self._mask(s) for s in spec])
    
    
    def __len__(self) -> int:
        return len(self.samples)
    
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spec, _, _ = torch.load(self.samples[idx], map_location="cpu")
        spec = spec.float()
        
        if self.spec_augment:
            spec = self._apply_spec_augment(spec)
            
        label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        
        if self.return_stem_id:
            return spec, label, torch.as_tensor(self.stem_ids[idx], dtype=torch.long)
        return spec, label
