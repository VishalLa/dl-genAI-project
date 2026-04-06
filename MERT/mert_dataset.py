import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

import random
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
import torch

from utils import highpass_filter


SAMPLE_RATE = 24000
DURATION = 5
EXPECTED_LENGTH = int(SAMPLE_RATE * DURATION)  # 120,000 samples


class MERTAudioDataset(Dataset):
    """
    Dataset for MERT genre classification.
    Each song is split into 5-second chunks (MERT's native pre-training context),
    indexed via a flat index_map — a (song_idx, chunk_start_sample) tuple per chunk.

    A 30-second song produces 6 non-overlapping chunks of 120,000 samples each.
    The final chunk is zero-padded if the song length is not an exact multiple of 5s.

    ESC-50 noise injection during training:
        Skipped 20% of the time. Otherwise 1-2 noise events are mixed in,
        each with SNR uniformly sampled from [8, 18] dB.
    """



    def __init__(self, genre_dir: Path, esc50_dir: Path, train: bool = True):
        """
        Args:
            genre_dir: Path to genre dataset directory
            esc50_dir: Path to ESC-50 noise dataset
            train: Whether this is training set (enables noise injection)
        """
        self.train = train
        self.genre_dir = genre_dir
        self.esc50_paths = list(esc50_dir.rglob("*.wav")) if esc50_dir else []

        self.SAMPLE_RATE = SAMPLE_RATE
        self.DURATION = DURATION
        self.EXPECTED_LENGTH = EXPECTED_LENGTH  # 5s * 24000 = 120,000 samples

        self.samples = []
        self.labels = []
        self.class_map = {}

        self._load_files()
        self._build_index_map()

        mode = "train" if train else "val"
        n_chunks = len(self.index_map)
        print(f"[{mode}] {n_chunks} chunks | {len(self.samples)} songs | {len(self.class_map)} classes")


    def _load_files(self):
        for label_id, genre_dir in enumerate(sorted(self.genre_dir.iterdir())):
            if not genre_dir.is_dir():
                continue

            genre_name = genre_dir.name
            self.class_map[genre_name] = label_id

            for song_dir in genre_dir.iterdir():
                if not song_dir.is_dir():
                    continue

                # Get all stems for this song
                wav_parts = sorted(list(song_dir.glob("*.wav")))
                if len(wav_parts) == 0:
                    continue

                self.samples.append(wav_parts)
                self.labels.append(label_id)

        print(f"Total songs  : {len(self.samples)}")
        print(f"Classes      : {list(self.class_map.keys())}")


    def _build_index_map(self):
        """
        Builds self.index_map — a flat list of (song_idx, chunk_start_sample) tuples.
        Only the first stem of each song is loaded to measure length (avoids
        loading all stems just to count chunks).

        Note:
            Songs may be longer or shorter than exactly 30 seconds. The index_map
            makes __getitem__ O(1) regardless of variable song lengths.
        """
        self.index_map = []

        for song_idx in range(len(self.samples)):
            waveform, sr = torchaudio.load(self.samples[song_idx][0])
            waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
                waveform = resampler(waveform)

            total_len = waveform.shape[1]
            n_chunks = (total_len + self.EXPECTED_LENGTH - 1) // self.EXPECTED_LENGTH

            for c in range(n_chunks):
                start = c * self.EXPECTED_LENGTH
                self.index_map.append((song_idx, start))

        print(f"Total chunks : {len(self.index_map)}")


    def __len__(self):
        return len(self.index_map)


    def merge_song(self, wave_list):
        """
        Load all stems, resample, highpass-filter, then mix them by
        stacking + mean (stems are simultaneous, not sequential).

        Args:
            wave_list: List of paths to audio stems

        Returns:
            torch.Tensor: Mixed waveform of shape [1, samples]
        """
        waveforms = []

        for wav_path in wave_list:
            waveform, sr = torchaudio.load(wav_path)

            # Convert to mono
            waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 24kHz if needed
            if sr != self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
                waveform = resampler(waveform)

            # Apply highpass filter
            waveform = highpass_filter(waveform, sample_rate=self.SAMPLE_RATE)

            waveforms.append(waveform)

        # Find max length
        max_len = max(w.shape[1] for w in waveforms)

        # Pad all to max length
        padded = [
            torch.nn.functional.pad(w, (0, max_len - w.shape[1]))
            if w.shape[1] < max_len else w
            for w in waveforms
        ]

        # Mix stems (average)
        return torch.stack(padded, dim=0).mean(dim=0)


    def get_chunk(self, waveform: torch.Tensor, start: int) -> torch.Tensor:
        """
        Extracts a fixed-length window from a waveform.
        Zero-pads the last chunk if it is shorter than EXPECTED_LENGTH.

        Args:
            waveform: Full waveform of shape [1, T]
            start: Start sample index

        Returns:
            torch.Tensor: Waveform chunk of shape [1, EXPECTED_LENGTH]
        """
        end = start + self.EXPECTED_LENGTH

        if waveform.shape[1] < end:
            padding = end - waveform.shape[1]
            chunk = torch.nn.functional.pad(waveform[:, start:], (0, padding))
        else:
            chunk = waveform[:, start:end]

        return chunk



    def inject_esc50_noise(self, music_waveform: torch.Tensor) -> torch.Tensor:
        """
        Randomly mixes ESC-50 environmental noise into a music chunk.
        Skipped with 20% probability so the model also sees clean examples.

        SNR is sampled from [8, 18] dB per noise event:
            18 dB → noise is barely audible.
            8 dB → noise is clearly audible but music still dominates.

        Scaling formula:
            scale = sqrt(music_power / (noise_power * 10 ^ (SNR_dB / 10)))

        Args:
            music_waveform (torch.Tensor): Music chunk of shape [1, T].

        Returns:
            torch.Tensor: Waveform with noise mixed in, shape [1, T].
                          Returned unchanged if augmentation is skipped.
        """
        if not self.esc50_paths or random.random() > 0.8:
            return music_waveform

        samples = music_waveform.shape[1]
        n_events = random.randint(1, 2)

        for _ in range(n_events):
            noise_path = random.choice(self.esc50_paths)
            noise_wave, sr = torchaudio.load(noise_path)
            noise_wave = noise_wave.mean(dim=0, keepdim=True)

            if sr != self.SAMPLE_RATE:
                noise_wave = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(noise_wave)

            noise_len = noise_wave.shape[1]
            if noise_len >= samples:
                noise_wave = noise_wave[:, :samples]
                start = 0
            else:
                start = random.randint(0, samples - noise_len)

            snr_db = random.uniform(8, 18)
            music_power = music_waveform.pow(2).mean()
            noise_power = noise_wave.pow(2).mean() + 1e-9

            scale = torch.sqrt(music_power / (noise_power * (10 ** (snr_db / 10))))
            music_waveform[:, start:start + noise_len] += scale * noise_wave

        return music_waveform


    def __getitem__(self, index: int):
        """
        Returns one augmented waveform chunk and its label.

        Pipeline:
            1. Look up (song_idx, start) from index_map.
            2. Merge all stems for the song via merge_song().
            3. Slice a EXPECTED_LENGTH window via get_chunk().
            4. Inject ESC-50 noise via inject_esc50_noise() (training only).
            5. RMS-normalise: waveform / (rms * 3.0).
            6. Clamp to [-1.0, 1.0] to prevent clipping.

        Args:
            index: Flat index into self.index_map

        Returns:
            waveform: Audio tensor of shape [120000] (5s at 24kHz)
            label: Integer class label
        """
        song_idx, start = self.index_map[index]
        wave_list = self.samples[song_idx]
        label = self.labels[song_idx]

        # Merge all stems for this song
        waveform = self.merge_song(wave_list)

        # Slice the chunk window (with zero-padding if needed)
        waveform = self.get_chunk(waveform, start)

        # Inject noise during training only
        if self.train:
            waveform = self.inject_esc50_noise(waveform)

        # Normalize (RMS normalization)
        rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-9)
        waveform = waveform / (rms * 3.0)
        waveform = waveform.clamp(-1.0, 1.0)

        # Remove channel dimension [1, samples] -> [samples]
        waveform = waveform.squeeze(0)

        return waveform, torch.tensor(label, dtype=torch.long)


processor = Wav2Vec2FeatureExtractor.from_pretrained(
    "m-a-p/MERT-v1-330M",
    trust_remote_code=True
)


def mert_collate_fn(batch):
    """
    Collate function for MERT DataLoader.
    Converts list of (waveform, label) → batched tensors.

    Args:
        batch: List of (waveform, label) tuples

    Returns:
        input_values: Tensor [batch_size, 720000]
        labels: Tensor [batch_size]
    """
    waveforms, labels = [], []

    for waveform, label in batch:
        waveforms.append(waveform.cpu().numpy())
        labels.append(label.item() if isinstance(label, torch.Tensor) else label)

    # Process with MERT feature extractor
    inputs = processor(
        waveforms,
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_tensors="pt",
    )

    return (
        inputs.input_values,
        torch.tensor(labels, dtype=torch.long),
    )

