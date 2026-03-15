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
DURATION = 30
EXPECTED_LENGTH = int(SAMPLE_RATE * DURATION)


class MERTAudioDataset(Dataset):
    """
    Dataset for MERT genre classification.
    Each song is exactly 30 seconds (no chunking needed).
    
    ESC-50 noise injection during training:
    - Background noise: 65% probability, SNR 18-35 dB
    - Foreground events: 35% probability, SNR 2-10 dB
    """

    ESC50_INJECT_PROB = 0.65
    ESC50_FG_PROB = 0.35
    BG_SNR_RANGE = (18, 35)
    FG_SNR_RANGE = (2, 10)

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
        self.EXPECTED_LENGTH = EXPECTED_LENGTH

        self.samples = []
        self.labels = []
        self.class_map = {}

        self._load_files()

        mode = "train" if train else "val"
        n_songs = len(self.samples)
        print(f"[{mode}] {n_songs} songs | {len(self.class_map)} classes")


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



    def __len__(self):
        return len(self.samples)


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


    def ensure_duration(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Ensure waveform is exactly 30 seconds (720,000 samples at 24kHz).
        Pad if shorter, trim if longer.
        
        Args:
            waveform: Input waveform [1, samples]
            
        Returns:
            torch.Tensor: Waveform of shape [1, 720000]
        """
        current_len = waveform.shape[1]

        if current_len < self.EXPECTED_LENGTH:
            # Pad with zeros
            padding = self.EXPECTED_LENGTH - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_len > self.EXPECTED_LENGTH:
            # Take center crop
            start = (current_len - self.EXPECTED_LENGTH) // 2
            waveform = waveform[:, start:start + self.EXPECTED_LENGTH]

        return waveform
    

    def _load_noise(self, target_len: int) -> torch.Tensor:
        """
        Load a random ESC-50 clip, tile if short, crop to target_len.
        
        Args:
            target_len: Desired length in samples
            
        Returns:
            torch.Tensor: Noise waveform [1, target_len]
        """
        noise, sr = torchaudio.load(random.choice(self.esc50_paths))
        noise = noise.mean(dim=0, keepdim=True)

        if sr != self.SAMPLE_RATE:
            noise = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(noise)

        if noise.shape[1] < target_len:
            repeats = (target_len // noise.shape[1]) + 1
            noise = noise.repeat(1, repeats)

        start = random.randint(0, noise.shape[1] - target_len)
        return noise[:, start:start + target_len]


    def inject_esc50_noise(self, music_waveform: torch.Tensor) -> torch.Tensor:
        """
        Inject ESC-50 noise with two modes per event:
          BACKGROUND (65%) — SNR 18-35 dB, quiet ambience across full song
          FOREGROUND (35%) — SNR 2-10 dB, loud event at random onset/duration
          
        Args:
            music_waveform: Clean music waveform [1, samples]
            
        Returns:
            torch.Tensor: Music with injected noise [1, samples]
        """
        if not self.esc50_paths or random.random() > self.ESC50_INJECT_PROB:
            return music_waveform

        song_len = music_waveform.shape[1]

        # Apply 1-3 noise events
        for _ in range(random.randint(1, 3)):
            noise = self._load_noise(song_len)
            is_foreground = random.random() < self.ESC50_FG_PROB
            snr_db = random.uniform(
                *(self.FG_SNR_RANGE if is_foreground else self.BG_SNR_RANGE)
            )

            # Calculate noise scaling factor
            music_power = music_waveform.pow(2).mean().clamp(min=1e-9)
            noise_power = noise.pow(2).mean().clamp(min=1e-9)
            scale = torch.sqrt(music_power / (noise_power * (10 ** (snr_db / 10))))

            if is_foreground:
                # Foreground: random onset and duration
                event_len = int(song_len * random.uniform(0.3, 1.0))
                offset = random.randint(0, song_len - event_len)
                music_waveform[:, offset:offset + event_len] += scale * noise[:, :event_len]
            else:
                # Background: full song
                music_waveform = music_waveform + scale * noise

        return music_waveform


    def __getitem__(self, index: int):
        """
        Get a single 30-second song.
        
        Args:
            index: Sample index
            
        Returns:
            waveform: Audio tensor of shape [720000] (30s at 24kHz)
            label: Integer class label
        """
        label = self.labels[index]

        # Merge all stems for this song
        waveform = self.merge_song(self.samples[index])

        # Ensure exactly 30 seconds
        waveform = self.ensure_duration(waveform)

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

