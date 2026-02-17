# TODO

import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

from pathlib import Path
from torch.utils.data import Dataset
import torchaudio

import pandas as pd
import torch

from scipy.signal import butter, filtfilt

def highpass_filter(audio, cutoff=80, sr=22050):
    b, a = butter(5, cutoff / (sr / 2), btype='high')
    return filtfilt(b, a, audio)

class ESC50_DataLoader_Waveform(Dataset):
    def __init__(self, esc50_dir: Path, allowed_folds=None):
        self.esc50_dir = esc50_dir
        self.audio_dir = esc50_dir / "audio"
        
        self.SAMPLE_RATE = 32000
        self.DURATION = 5
        
        self.SAMPLES = self.SAMPLE_RATE * self.DURATION
        
        self.meta_data = None
        self.allowed_folds = allowed_folds

    def load_meta_data(self, meta_data_path: str):
        print(f"Loading metadata from file: {meta_data_path}")
        df = pd.read_csv(meta_data_path)

        if self.allowed_folds is not None:
            df = df[df["fold"].isin(self.allowed_folds)].reset_index(drop=True)

        self.meta_data = df
    
    def __len__(self):
        return self.meta_data.shape[0]

    def _fix_length(self, signal: torch.Tensor):
        """Fix signal length to match CNN14 requirements (10 seconds at 32kHz)"""
        length = signal.shape[1]

        if length > self.SAMPLES:
            signal = signal[:, :self.SAMPLES]

        elif length < self.SAMPLES:
            padding = self.SAMPLES - length
            signal = torch.nn.functional.pad(signal, (0, padding))

        return signal
    
    def __getitem__(self, index: int):
        row = self.meta_data.iloc[index]
        filename = row["filename"]
        label = int(row["target"])
        file_path = self.audio_dir / filename

        waveform, sr = torchaudio.load(file_path)

        waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            waveform = resampler(waveform)

        # pass through highpass filter to reomve audio with small freq
        waveform = highpass_filter(audio=waveform)

        # normalize amplitued 
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        waveform = self._fix_length(signal=waveform)

        return waveform, torch.tensor(label, dtype=torch.long)


class GenreDataLoader_Waveform_Chunked(Dataset):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir 
        
        # CNN14 requirements
        self.SAMPLE_RATE = 32000
        self.DURATION = 5
        self.CHUNK_SAMPLES = self.SAMPLE_RATE * self.DURATION

        self.samples = []
        self.labels = []
        self.class_map = {}

        self._load_files()

    def _load_files(self):
        for label_id, genre_dir in enumerate(sorted(self.root_dir.iterdir())):

            if not genre_dir.is_dir():
                continue

            genre_name = genre_dir.name 
            self.class_map[genre_name] = label_id

            # each subfolder = One song
            for song_dir in genre_dir.iterdir():

                if not song_dir.is_dir():
                    continue
                
                # collect all wav parts of that song
                wav_parts = sorted(list(song_dir.glob("*.wav")))
                if len(wav_parts) == 0:
                    continue
                
                self.samples.append(wav_parts)
                self.labels.append(label_id)

        print("Classes:", self.class_map)
        print("Total songs:", len(self.samples))

    def __len__(self):
        return len(self.samples)
    
    def _fix_length(self, signal: torch.Tensor) -> torch.Tensor:
        length = signal.shape[1]

        # too long → crop
        if length > self.CHUNK_SAMPLES:
            signal = signal[:, :self.CHUNK_SAMPLES]

        # too short → pad with zeros
        elif length < self.CHUNK_SAMPLES:
            padding = self.CHUNK_SAMPLES - length
            signal = torch.nn.functional.pad(signal, (0, padding))

        return signal
    
    def __getitem__(self, index):
        wave_list = self.samples[index]
        label = self.labels[index]

        waveforms = []

        for wav_path in wave_list:
            waveform, sr = torchaudio.load(wav_path)

            waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sr != self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
                waveform = resampler(waveform)

            waveform = highpass_filter(audio=waveform)

            waveforms.append(waveform)

        full_waveform = torch.cat(waveforms, dim=1)
        full_waveform = self._fix_length(full_waveform, duration=30)

        total_samples = full_waveform.shape[1]
        num_chunks = total_samples // self.CHUNK_SAMPLES

        if num_chunks == 0:
            # Song is shorter than 5 seconds, pad it
            full_waveform = self._fix_length(full_waveform, self.CHUNK_SAMPLES)
            chunks = full_waveform.unsqueeze(0)  # [1, 1, 160000]

        else:
            # Extract complete chunks (30 sec → 6 chunks of 5 sec)
            chunks = []
            for i in range(num_chunks):
                start_idx = i * self.CHUNK_SAMPLES
                end_idx = start_idx + self.CHUNK_SAMPLES

                chunk = full_waveform[:, start_idx: end_idx]
                chunks.append(chunk)
            
            chunks = torch.stack(chunks, dim=0)

        return chunks, torch.tensor(label, dtype=torch.long)
    


def collate_fn_chunks(batch):
    """
    Custom collate function for chunked genre data
    Handles variable number of chunks per song
    """
    all_chunks = []
    labels = []
    chunk_counts = []
    
    for chunks, label in batch:
        num_chunks = chunks.shape[0]
        all_chunks.append(chunks)
        labels.extend([label] * num_chunks)
        chunk_counts.append(num_chunks)
    
    all_chunks = torch.cat(all_chunks, dim=0)  # [total_chunks, 1, 160000]
    labels = torch.tensor(labels, dtype=torch.long)
    
    return all_chunks, labels, chunk_counts
