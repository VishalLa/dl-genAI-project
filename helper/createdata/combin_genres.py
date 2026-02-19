import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

import random 
import re

from pathlib import Path
import torchaudio

import pandas as pd
import torch
import matplotlib.pyplot as plt
import librosa

from scipy.signal import butter, filtfilt


class Variables:
    SAMPLE_RATE = 22050
    N_MELS = 128
    DURATION = 30   # seconds
    SAMPLES = SAMPLE_RATE * DURATION

    def __init__(self):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft = 1024,
            hop_length = 256,
            n_mels = self.N_MELS,
            f_min = 20,
            f_max = self.SAMPLE_RATE // 2,
            power = 2.0
        )

        # dB=10⋅log10​ (Power) convert to power signal
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80, stype="power")

    def _fix_length(self, signal: torch.Tensor, duration: int = 30) -> torch.Tensor:
        length = signal.shape[1]
        
        samples = duration * self.SAMPLE_RATE

        # too long → crop
        if length > samples:
            signal = signal[:, : samples]

        # too short → pad
        elif length < samples:
            padding = samples - length
            signal = torch.nn.functional.pad(signal, (0, padding))

        return signal
    

    def highpass_filter(self, audio: torch.Tensor, cutoff: int = 40) -> torch.Tensor:
        b, a = butter(5, cutoff / (self.SAMPLE_RATE / 2), btype='high')
        
        audio_np = audio.numpy()
        filtered = filtfilt(b, a, audio_np)
        
        return torch.tensor(filtered.copy(), dtype=audio.dtype)
    
    
    def trim_audio(self, audio: torch.Tensor, top_db: int=20):
        clean_audio, _ = librosa.effects.trim(audio, top_db=20)
        return clean_audio
    

    def _spec_augment(self, spec):
        """
        spec: torch.Tensor of shape (1, n_mels, time)
        """
        import random
        spec = spec.clone()
        n_mels, time_steps = spec.shape[1], spec.shape[2]
        
        # Frequency masking (mask mel bins)
        f_mask = random.randint(2, 5)
        f0 = random.randint(0, max(1, n_mels - f_mask))
        spec[:, f0:f0+f_mask, :] = spec.mean()
        
        # Time masking (mask time frames)  
        t_mask = random.randint(10, 30)
        t0 = random.randint(0, max(1, time_steps - t_mask))
        spec[:, :, t0:t0+t_mask] = spec.mean()
        
        return spec
    

class GenreDataset:
    def __init__(self, genre_dir: Path, esc50_dir: Path, output_dir: Path, augment: bool=False):
        self.genre_dir = genre_dir
        self.esc50_paths = list(Path(esc50_dir).rglob("*.wav"))
        print("ESC50 samples found:", len(self.esc50_paths))

        self.output_dir = output_dir
        self.variables = Variables()
        self.augment = augment


    def merge_song(self, wave_list) -> torch.Tensor:
        waveforms = []

        for wave_path in wave_list:
            waveform, sr = torchaudio.load(wave_path)

            waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.variables.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.variables.SAMPLE_RATE)
                waveform = resampler(waveform)

            waveform = self.variables.highpass_filter(waveform)

            waveforms.append(waveform)

        max_len = max(w.shape[1] for w in waveforms)

        padded = []
        for w in waveforms:
            if w.shape[1] < max_len:
                pad = max_len - w.shape[1]
                w = torch.nn.functional.pad(w, (0, pad))
            padded.append(w)

        full_waveform = torch.stack(padded, dim=0).mean(dim=0)

        max_val = full_waveform.abs().max()
        if max_val > 0:
            full_waveform = full_waveform / max_val

        full_waveform = self.variables._fix_length(full_waveform, duration=30)

        return full_waveform
    

    def save_spectrogram(self, waveform: torch.Tensor, save_path: Path):
        mel = self.variables.mel(waveform)
        mel = self.variables.db(mel)

        # normalize [0,1]
        mel = (mel + 80) / 80

        mel_np = mel.squeeze(0).cpu().numpy()

        plt.figure(figsize=(6,4))
        librosa.display.specshow(
            mel_np,
            sr=self.variables.SAMPLE_RATE,
            hop_length=256,
            x_axis=None,
            y_axis=None,
            cmap="magma"
        )
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close()

    

    def inject_esc50_noise(self, music_waveform: torch.Tensor) -> torch.Tensor:

        if random.random() > 0.8:
            return music_waveform 
        
        samples = music_waveform.shape[1]

        n_events = random.randint(1, 3)

        for _ in range(n_events):
            noise_path = random.choice(self.esc50_paths)
            noise_wave, sr = torchaudio.load(noise_path)

            noise_wave = noise_wave.mean(dim=0, keepdim=True)

            if sr != self.variables.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.variables.SAMPLE_RATE)
                noise_wave = resampler(noise_wave)

            # choose random position in 30 sec song
            noise_len = noise_wave.shape[1]

            if noise_len >= samples:
                noise_wave = noise_wave[:, :samples]
                start = 0
            else:
                start = random.randint(0, samples - noise_len)

            # choose noise loudness (SNR control)
            alpha = random.uniform(0.05, 0.25)

            # inject
            music_waveform[:, start:start+noise_len] += alpha * noise_wave

        max_val = music_waveform.abs().max()
        if max_val > 0:
            music_waveform = music_waveform / max_val 

        return music_waveform


    def process_song(self, song_dir: Path, genre_name: str):

        match = re.search(r"\.(\d+)", song_dir.name)
        if match is None:
            return 
        
        song_id = match.group(1)

        out_folder = self.output_dir / genre_name 
        out_folder.mkdir(parents=True, exist_ok=True)

        wav_paths = sorted(song_dir.glob("*.wav"))
        if len(wav_paths) == 0:
            return 

        # ---------------- CLEAN SONG ----------------
        clean_waveform = self.merge_song(wav_paths)

        clean_save = out_folder / f"{song_id}_clean.png"
        self.save_spectrogram(clean_waveform, clean_save)

        print(f"Saved CLEAN {genre_name}/{song_id}_clean.png")

        # ---------------- NOISY SONG (70%) ----------------
        if random.random() < 0.8:

            noisy_waveform = clean_waveform.clone()
            noisy_waveform = self.inject_esc50_noise(noisy_waveform)

            noise_save = out_folder / f"{song_id}_noise.png"
            self.save_spectrogram(noisy_waveform, noise_save)

            print(f"Saved NOISE {genre_name}/{song_id}_noise.png")

    
    def process_all(self):
        for genre_dir in sorted(self.genre_dir.iterdir()):
            if not genre_dir.is_dir():
                continue 

            genre_name = genre_dir.name 
            print(f"\nProcessing genre: {genre_name}")

            for song_dir in sorted(genre_dir.iterdir()):

                if not song_dir.is_dir():
                    continue

                self.process_song(song_dir, genre_name)

        print("\nALL SONGS CONVERTED SUCCESSFULLY")


if __name__ == "__main__":
    genredataset = GenreDataset(
        genre_dir=Path("Z:\\dlgen-AI-project\\data\\genres_stems"),
        output_dir=Path("Z:\\dlgen-AI-project\\data\\genres_mel"),
        esc50_dir=Path("Z:\\dlgen-AI-project\\data\\ESC-50-master")
    )

    genredataset.process_all()

