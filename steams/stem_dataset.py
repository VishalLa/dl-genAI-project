from __future__ import annotations
from typing import Optional, Tuple

import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
from pathlib import Path
from torch.utils.data import Dataset

from tqdm import tqdm
from transformers import ASTFeatureExtractor

import random
import torch
import torchaudio
import numpy as np
import soundfile as sf

from scipy.signal import butter, filtfilt

AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
STEM_KEYS = ["bass", "drums", "vocals", "other"]

_feature_extractor: Optional[ASTFeatureExtractor] = None

def get_feature_extractor() -> ASTFeatureExtractor:
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL)
    return _feature_extractor
    

def highpass_filter(
    audio: torch.Tensor,
    cutoff: int = 40,
    sample_rate: int = 16_000
) -> torch.Tensor:
    """
    Applies a Butterworth high-pass filter to remove very low-frequency noise.

    Args:
        audio (torch.Tensor): Input waveform tensor of shape [1, samples].
        cutoff (int): Cutoff frequency in Hz. Default is 40 Hz.
        sample_rate (int): Sampling rate of the waveform. Default is 16000 Hz.

    Returns:
        torch.Tensor: Filtered waveform with same shape and dtype as input.
    """
    b, a = butter(5, cutoff / (sample_rate / 2), btype='high')
    
    audio_np = audio.squeeze(0).cpu().numpy()
    
    filtered = filtfilt(b, a, audio_np)
    filtered = np.ascontiguousarray(filtered)
    filtered = torch.from_numpy(filtered).unsqueeze(0)
    
    return filtered.type_as(audio)


class AST_Stem_Dataset(Dataset):
    
    """
    Loads each stem of each song SEPARATELY, mixes ESC-50 noise into each stem
    independently, and returns 10.24 s chunks ready for ASTFeatureExtractor.

    Expected directory structure:
        genre_dir/
            blues/
                blues.00000/
                    bass.wav, drums.wav, vocals.wav, others.wav
                blues.00001/
                    ...

    Args:
        genre_dir        : root folder with one sub-folder per genre.
        esc50_dir        : folder with ESC-50 .wav files.
        flatten_stems    : True  -> every (song, chunk, stem) is its own item
                                    returns (wave [T], genre_label, stem_id)
                           False -> every (song, chunk) is one item holding all 4 stems
                                    returns (waves [4, T], genre_label, -1)
        augment          : turn ESC-50 mixing on/off (set False for validation).
        noise_prob       : probability that a given stem gets noise.
        snr_range        : (low, high) SNR in dB, sampled per noise event.
        max_noise_events : each noisy stem gets 1..max_noise_events ESC-50 clips.
    """
    
    SAMPLE_RATE = 16_000
    CHUNK_DURATION = 10.24
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
    SILENCE_POWER = 1e-8
    
    def __init__(
        self,
        genre_dir: Path | str,
        esc50_dir: Path | str,
        flatten_stems: bool = True,
        augment: bool = True,
        noise_prob: float = 0.8,
        snr_range = (8, 18),
        max_noise_events: int = 2,
        highpass_hz: int = 40
    ) -> None:
        self.genre_dir = genre_dir
        self.esc50_paths = list(esc50_dir.rglob("*.wav"))
        self.flatten_stems = flatten_stems
        self.augment = augment
        self.noise_prob = noise_prob
        self.snr_range = snr_range
        self.max_noise_events = max_noise_events
        self.highpass_hz = highpass_hz
        
        self.samples = []       # one entry per song: list of 4 stems path (STEM_KEYS order)
        self.labels = []
        self.class_map = {}
        self._resamplers = {}   # cache, one per source smaple rate
        
        self._load_files()
        self._build_index_map()
        
    
    def _load_files(self) -> None:
        for label_id, genre in enumerate(
            sorted(p for p in self.genre_dir.iterdir() if p.is_dir())
        ):
            self.class_map[genre.name] = label_id
            
            for song_dir in sorted(
                p for p in genre.iterdir() if p.is_dir()
            ):
                wavs = list(song_dir.glob("*.wav"))
                stem_paths = []
                for key in STEM_KEYS:
                    match_ = [w for w in wavs if key in w.stem.lower()]
                    if len(match_) != 1:
                        break
                    stem_paths.append(match_[0])
                    
                if len(stem_paths) != len(STEM_KEYS):
                    continue
                
                self.samples.append(stem_paths)
                self.labels.append(label_id)
                
                
    def _build_index_map(self) -> None:
        """
        Flat list of (song_idx, chunk_start_sample[, stem_idx]).
        Uses sf.info to read only the file header, so it is fast.
        """
        self.index_map = []
        
        for song_idx, stem_paths in enumerate(self.samples):
            info = sf.info(str(stem_paths[0]))
            total_len = int(round(info.frames * self.SAMPLE_RATE / info.samplerate))
            n_chunks = max(1, (total_len + self.CHUNK_SAMPLES - 1) // self.CHUNK_SAMPLES)
        
            for c in range(n_chunks):
                start = c * self.CHUNK_SAMPLES
                if self.flatten_stems:
                    for stem_idx in range(len(STEM_KEYS)):
                        self.index_map.append((song_idx, start, stem_idx))
                else:
                    self.index_map.append((song_idx, start))
        
        print(f"Songs: {len(self.samples)} | items: {len(self.index_map)} "
              f"({'per-stem' if self.flatten_stems else 'all 4 stems per item'})")

    
    def __len__(self) -> int:
        return len(self.index_map)
    
    
    def _to_mono_16k(self, path: Path | str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.SAMPLE_RATE:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            waveform = self._resamplers[sr](waveform)
        return waveform
    
    def _load_stem(self, path: Path | str) -> torch.Tensor:
        waveform = self._to_mono_16k(path=path)
        return highpass_filter(waveform, cutoff=self.highpass_hz, sample_rate=self.SAMPLE_RATE)
    
    
    def _get_chunk(self, waveform: torch.Tensor, start: int) -> torch.Tensor:
        end = start + self.CHUNK_SAMPLES
        if waveform.shape[1] < end:
            return torch.nn.functional.pad(waveform[:, start:], (0, end - waveform.shape[1]))
        return waveform[:, start:end]
    
    
    def _normalise(self, chunk: torch.Tensor) -> torch.Tensor:
        power = chunk.pow(2).mean()
        if power < self.SILENCE_POWER:
            return chunk
        rms = torch.sqrt(power + 1e-9)
        return (chunk / (rms * 3.0)).clamp(-1.0, 1.0)
    
    
    def _inject_esc50_noise(self, stem_chunk: torch.Tensor) -> torch.Tensor:
        if not self.augment or random.random() > self.noise_prob:
            return stem_chunk
        
        stem_power = stem_chunk.pow(2).mean()
        if stem_power < self.SILENCE_POWER:
            return stem_chunk
        
        out = stem_chunk.clone()
        n_sample = out.shape[1]
        
        for _ in range(random.randint(1, self.max_noise_events)):
            noise = self._to_mono_16k(random.choice(self.esc50_paths))
            noise_len = noise.shape[1]
            
            if noise_len >= n_sample:
                crop = random.randint(0, noise_len - n_sample)
                noise = noise[:, crop:crop + n_sample]
                place = 0
            else:
                place = random.randint(0, n_sample - noise_len)
                
            active = noise[noise.abs() > 1e-4]
            if active.numel() == 0:
                continue
            noise_power = active.pow(2).mean() + 1e-9
            
            snr_db = random.uniform(*self.snr_range)
            scale = torch.sqrt(stem_power / (self.noise_prob * 10 ** (snr_db / 10)))
            out[:, place:place + noise.shape[1]] += scale * noise
    
        return out
    
    
    def _process_stem(self, path: Path | str, start: int) -> torch.Tensor:
        wave = self._load_stem(path=path)
        chunk = self._get_chunk(wave, start)
        chunk = self._inject_esc50_noise(chunk)
        chunk = self._normalise(chunk)
        return chunk.squeeze(0)

    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.flatten_stems:
            song_idx, start, stem_idx = self.index_map[index]
            label = self.labels[song_idx]
            wave = self._process_stem(self.samples[song_idx][stem_idx], start)
            return wave, torch.tensor(label, dtype=torch.long), torch.tensor(stem_idx, dtype=torch.long)

        song_idx, start = self.index_map[index]
        label = self.labels[song_idx]
        waves = torch.stack([self._process_stem(p, start) for p in self.samples[song_idx]])
        return waves, torch.tensor(label, dtype=torch.long), torch.tensor(-1, dtype=torch.long)
    

def collate_fn(batch):
    """
    flatten_stems=True  -> input_values [B, 1024, 128], labels [B], stem_ids [B]
    flatten_stems=False -> input_values [B, 4, 1024, 128], labels [B], stem_ids [B] (all -1)
    """
    waves, labels, stem_ids = zip(*batch)
    extractor = get_feature_extractor()
    stacked = waves[0].dim() == 2

    if stacked:
        flat = [w.numpy() for item in waves for w in item]   # B*4 waveforms
    else:
        flat = [w.numpy() for w in waves]

    feats = extractor(flat, sampling_rate=16000, padding="max_length", return_tensors="pt").input_values

    if stacked:
        feats = feats.view(len(waves), len(STEM_KEYS), *feats.shape[1:])

    return feats, torch.stack(labels), torch.stack(stem_ids)


def precompute_stem_dataset(
    genre_dir: Path | str, 
    esc50_dir: Path | str, 
    output_dir: Path | str, 
    flatten_stems: bool,
    augment: bool = True, 
    copies: int = 1, 
    dtype=torch.float16
) -> None:
    dataset = AST_Stem_Dataset(
        genre_dir=genre_dir, 
        esc50_dir=esc50_dir, 
        flatten_stems=flatten_stems, 
        augment=augment
    )
    extractor = get_feature_extractor()
    output_dir = Path(output_dir)
    
    id2genre = {v: k for k, v in dataset.class_map.items()}
    for genre in dataset.class_map:
        (output_dir / genre).mkdir(parents=True, exist_ok=True)

    for c in range(copies):
        for i in tqdm(range(len(dataset)), desc=f"copy {c + 1}/{copies}"):
            wave, label, stem_id = dataset[i]
            spec = extractor(
                wave.numpy(), sampling_rate=16000, padding="max_length", return_tensors="pt"
            ).input_values.squeeze(0).to(dtype)

            save_path = output_dir / id2genre[label.item()] / f"chunk_{i}_c{c}_s{stem_id.item()}.pt"
            torch.save((spec, label, stem_id), save_path)

    print(f"Done. Saved to {output_dir.absolute()}")
    
if __name__ == "__main__":
    import sys

    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from helper.golbal_variables import GENRES_STEMS, ESC_50_MASTER, GENRES_PT

    dataset_variants = (
        ("flattened_augmented", True, True),
        ("flattened_clean", True, False),
        ("grouped_augmented", False, True),
        ("grouped_clean", False, False),
    )

    for variant, flatten_stems, augment in dataset_variants:
        print(f"data variation: {variant}")
        precompute_stem_dataset(
            genre_dir=GENRES_STEMS,
            esc50_dir=ESC_50_MASTER,
            output_dir=GENRES_PT / variant,
            flatten_stems=flatten_stems,
            augment=augment,
        )
