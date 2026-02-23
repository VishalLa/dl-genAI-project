import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

import random
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio
from transformers import Wav2Vec2Processor
import torch
import numpy as np
from scipy.signal import butter, filtfilt


def highpass_filter(audio: torch.Tensor, cutoff: int = 40, sample_rate: int = 16000):
    b, a = butter(5, cutoff / (sample_rate / 2), btype='high')

    audio_np = audio.squeeze(0).cpu().numpy()
    filtered = filtfilt(b, a, audio_np)

    filtered = np.ascontiguousarray(filtered)

    filtered = torch.from_numpy(filtered).unsqueeze(0)
    return filtered.type_as(audio)


class Audio_DataLoader(Dataset):

    def __init__(self, genre_dir: Path, esc50_dir: Path):

        self.genre_dir = genre_dir
        self.esc50_paths = list(esc50_dir.rglob("*.wav"))

        self.SAMPLE_RATE = 16000
        self.CHUNK_DURATION = 8
        self.CHUNK_SAMPLES = self.SAMPLE_RATE * self.CHUNK_DURATION

        self.samples = []
        self.labels = []
        self.class_map = {}

        self._load_files()
        self._build_index_map()

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

        self.index_map = []

        for song_idx in range(len(self.samples)):

            # load only first stem to know duration
            waveform, sr = torchaudio.load(self.samples[song_idx][0])
            waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
                waveform = resampler(waveform)

            total_len = waveform.shape[1]

            n_chunks = (total_len + self.CHUNK_SAMPLES - 1) // self.CHUNK_SAMPLES

            for c in range(n_chunks):
                start = c * self.CHUNK_SAMPLES

                for augment_version in range(4):
                    self.index_map.append((song_idx, start, augment_version))

        print(f"Total training segments: {len(self.index_map)}")
        print(f"  - Clean segments (v0): {len(self.index_map) // 4}")
        print(f"  - 2-noise segments (v1): {len(self.index_map) // 4}")
        print(f"  - 3-noise segments (v2): {len(self.index_map) // 4}")
        print(f"  - 4-noise segments (v3): {len(self.index_map) // 4}")

    def __len__(self):
        return len(self.index_map)

    def merge_song(self, wave_list):

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

        padded = []
        for w in waveforms:
            if w.shape[1] < max_len:
                w = torch.nn.functional.pad(w, (0, max_len - w.shape[1]))
            padded.append(w)

        full_waveform = torch.stack(padded, dim=0).mean(dim=0)

        return full_waveform

    def get_chunk(self, waveform: torch.Tensor, start: int):

        end = start + self.CHUNK_SAMPLES

        if waveform.shape[1] < end:
            padding = end - waveform.shape[1]
            chunk = torch.nn.functional.pad(waveform[:, start:], (0, padding))
        else:
            chunk = waveform[:, start:end]

        return chunk

    # def inject_esc50_noise(self, music_waveform):

    #     if random.random() > 0.8:
    #         return music_waveform

    #     samples = music_waveform.shape[1]
    #     n_events = random.randint(1, 2)

    #     for _ in range(n_events):

    #         noise_path = random.choice(self.esc50_paths)
    #         noise_wave, sr = torchaudio.load(noise_path)

    #         noise_wave = noise_wave.mean(dim=0, keepdim=True)

    #         if sr != self.SAMPLE_RATE:
    #             resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
    #             noise_wave = resampler(noise_wave)

    #         noise_len = noise_wave.shape[1]

    #         if noise_len >= samples:
    #             noise_wave = noise_wave[:, :samples]
    #             start = 0
    #         else:
    #             start = random.randint(0, samples - noise_len)

    #         snr_db = random.uniform(8, 18)

    #         music_power = music_waveform.pow(2).mean()
    #         noise_power = noise_wave.pow(2).mean() + 1e-9

    #         scale = torch.sqrt(music_power / (noise_power * (10 ** (snr_db / 10))))
    #         music_waveform[:, start:start+noise_len] += scale * noise_wave

    #     return music_waveform


    def inject_esc50_noise(self, music_waveform: torch.Tensor, n_events: int):
        if n_events == 0:
            return music_waveform
        
        samples = music_waveform.shape[1]

        for _ in range(n_events):
            
            noise_path = random.choice(self.esc50_paths)
            noise_wave, sr = torchaudio.load(noise_path)

            noise_wave = noise_wave.mean(dim=0, keepdim=True)
            if sr != self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
                noise_wave = resampler(noise_wave)

            noise_len = noise_wave.shape[1]

            if noise_len >= samples:
                noise_wave = noise_wave[:, :samples]
                start = 0 
            else: 
                start = random.randint(0, samples - noise_len)

            snr_db = random.uniform(10, 18)

            music_power = music_waveform.pow(2).mean()
            noise_power = noise_wave.pow(2).mean() + 1e-9

            scale = torch.sqrt(music_power / (noise_power * (10 ** (snr_db / 10))))
            music_waveform[:, start:start+noise_len] += scale * noise_wave

        return music_waveform


    def __getitem__(self, index):
        # song_idx, start = self.index_map[index]
        song_idx, start, augment_version = self.index_map[index]
        
        wave_list = self.samples[song_idx]
        label = self.labels[song_idx]

        full_waveform = self.merge_song(wave_list)
        full_waveform = self.get_chunk(full_waveform, start)
        # full_waveform = self.inject_esc50_noise(full_waveform)

        n_noise_events = [0, 2, 3, 4][augment_version]
        full_waveform = self.inject_esc50_noise(full_waveform, n_noise_events)

        rms = torch.sqrt(torch.mean(full_waveform**2) + 1e-9)
        full_waveform = full_waveform / (rms * 3.0)
        full_waveform = full_waveform.clamp(-1.0, 1.0)
        full_waveform = full_waveform.squeeze(0)

        return full_waveform, torch.tensor(label, dtype=torch.long)


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def collate_fn(batch):
    """
    Collate function for Wav2Vec2
    Converts list of (waveform, label) → batched tensors
    """
    waveforms = []
    labels = []

    for waveform, label in batch:
        waveform = waveform.cpu().numpy()
        waveforms.append(waveform)
        labels.append(label.item() if isinstance(label, torch.Tensor) else label)

    # Wav2Vec2Processor handles padding and attention mask
    inputs = processor(
        waveforms,
        sampling_rate=16000,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    labels = torch.tensor(labels, dtype=torch.long)

    return inputs.input_values, inputs.attention_mask, labels
