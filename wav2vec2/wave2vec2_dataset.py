import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

import math
import random
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio
from transformers import Wav2Vec2Processor
import torch
import numpy as np

from utils import highpass_filter


SAMPLE_RATE    = 16000
CHUNK_DURATION = 10.24
CHUNK_SAMPLES  = int(SAMPLE_RATE * CHUNK_DURATION)


class Wav2Vec2_Audio_Dataset(Dataset):

    ESC50_INJECT_PROB = 0.65
    ESC50_FG_PROB = 0.35
    BG_SNR_RANGE = (18, 35)
    FG_SNR_RANGE = (2,  10)


    def __init__(self, genre_dir: Path, esc50_dir: Path, train: bool = True):
        self.train = train
        self.genre_dir = genre_dir
        self.esc50_paths = list(esc50_dir.rglob("*.wav"))

        self.SAMPLE_RATE = SAMPLE_RATE
        self.CHUNK_DURATION = CHUNK_DURATION
        self.CHUNK_SAMPLES = CHUNK_SAMPLES

        self.samples   = []
        self.labels    = []
        self.class_map = {}
        self.index_map = []

        self._load_files()
        self._build_index_map()


        mode = "train" if train else "val"
        n_songs = len(self.samples)
        print(f"[{mode}] {len(self.index_map)} chunks from {n_songs} songs "
              f"| {len(self.class_map)} classes")

    def _load_files(self):

        for label_id, genre_dir in enumerate(sorted(self.genre_dir.iterdir())):

            if not genre_dir.is_dir():
                continue

            genre_name = genre_dir.name
            self.class_map[genre_name] = label_id

            for song_dir in genre_dir.iterdir():

                if not song_dir.is_dir():
                    continue

                wav_parts = sorted(list(song_dir.glob("*.wav")))
                if len(wav_parts) == 0:
                    continue

                self.samples.append(wav_parts)
                self.labels.append(label_id)

    def _build_index_map(self):
        print("Building chunk index map ...")

        for song_idx, wav_parts in enumerate(self.samples):

            waveform, sr = torchaudio.load(wav_parts[0])
            waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(waveform)

            n_chunks = max(1, math.ceil(waveform.shape[1] / self.CHUNK_SAMPLES))

            for c in range(n_chunks):
                self.index_map.append((song_idx, c * self.CHUNK_SAMPLES))

        print(f"Total chunks : {len(self.index_map)}")
        print(f"Total songs  : {len(self.samples)}")
        print(f"Classes      : {list(self.class_map.keys())}")

    def __len__(self):
        return len(self.index_map)

    def merge_song(self, wave_list):

        """
        Load all stems, resample, highpass-filter, then mix them by
        stacking + mean (stems are simultaneous, not sequential).
        Returns [1, total_samples].
        """

        waveforms = []

        for wav_path in wave_list:
            waveform, sr = torchaudio.load(wav_path)

            waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
                waveform = resampler(waveform)

            waveform = highpass_filter(waveform, sample_rate=self.SAMPLE_RATE)

            waveforms.append(waveform)

        max_len = max(w.shape[1] for w in waveforms)

        padded  = [
            torch.nn.functional.pad(w, (0, max_len - w.shape[1]))
            if w.shape[1] < max_len else w
            for w in waveforms
        ]

        return torch.stack(padded, dim=0).mean(dim=0)

    def get_chunk(self, waveform: torch.Tensor, start: int):

        end = start + self.CHUNK_SAMPLES

        if waveform.shape[1] < end:
            padding = end - waveform.shape[1]
            chunk = torch.nn.functional.pad(waveform[:, start:], (0, padding))
        else:
            chunk = waveform[:, start:end]

        return chunk
    
    def _load_noise(self, target_len: int) -> torch.Tensor:
        """Load a random ESC-50 clip, tile if short, crop to target_len."""

        noise, sr = torchaudio.load(random.choice(self.esc50_paths))
        noise = noise.mean(dim=0, keepdim=True)

        if sr != self.SAMPLE_RATE:
            noise = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(noise)

        if noise.shape[1] < target_len:
            repeats = (target_len // noise.shape[1]) + 1
            noise   = noise.repeat(1, repeats)

        start = random.randint(0, noise.shape[1] - target_len)
        return noise[:, start : start + target_len]
    

    def inject_esc50_noise(self, music_waveform: torch.Tensor) -> torch.Tensor:
        """
        Inject ESC-50 noise with two modes per event:
          BACKGROUND (65%) — SNR 18-35 dB, quiet ambience across full chunk
          FOREGROUND (35%) — SNR 2-10 dB,  loud event at random onset/duration
        """
        if not self.esc50_paths or random.random() > self.ESC50_INJECT_PROB:
            return music_waveform

        chunk_len = music_waveform.shape[1]

        for _ in range(random.randint(1, 3)):
            noise = self._load_noise(chunk_len)
            is_foreground = random.random() < self.ESC50_FG_PROB
            snr_db = random.uniform(*(self.FG_SNR_RANGE if is_foreground else self.BG_SNR_RANGE))

            music_power = music_waveform.pow(2).mean().clamp(min=1e-9)
            noise_power = noise.pow(2).mean().clamp(min=1e-9)
            scale = torch.sqrt(music_power / (noise_power * (10 ** (snr_db / 10))))

            if is_foreground:
                event_len = int(chunk_len * random.uniform(0.3, 1.0))
                offset    = random.randint(0, chunk_len - event_len)
                music_waveform[:, offset : offset + event_len] += scale * noise[:, :event_len]
            else:
                music_waveform = music_waveform + scale * noise

        return music_waveform

    def __getitem__(self, index: int):
        song_idx, start = self.index_map[index]
        label = self.labels[song_idx]

        waveform = self.merge_song(self.samples[song_idx])
        waveform = self.get_chunk(waveform, start)

        # Only inject noise during training
        if self.train:
            waveform = self.inject_esc50_noise(waveform)

        rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-9)
        waveform = waveform / (rms * 3.0)
        waveform = waveform.clamp(-1.0, 1.0)
        waveform = waveform.squeeze(0)   # [163 840]

        return waveform, torch.tensor(label, dtype=torch.long)


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def collate_fn(batch):
    """
    Collate function for Wav2Vec2
    Converts list of (waveform, label) → batched tensors
    """
    waveforms, labels = [], []
    for waveform, label in batch:
        waveforms.append(waveform.cpu().numpy())
        labels.append(label.item() if isinstance(label, torch.Tensor) else label)

    inputs = processor(
        waveforms,
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    return (
        inputs.input_values,
        inputs.attention_mask,
        torch.tensor(labels, dtype=torch.long),
    )
    