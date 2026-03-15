import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

import random
import re
import math

from pathlib import Path
import torchaudio

import pandas as pd
import torch
import matplotlib.pyplot as plt
import librosa

from scipy.signal import butter, filtfilt


class Variables:
    SAMPLE_RATE  = 22050
    N_MELS       = 128
    CHUNK_DURATION = 10.24
    CHUNK_SAMPLES  = int(SAMPLE_RATE * CHUNK_DURATION)  # 225,792 samples

    def __init__(self):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=1024,
            hop_length=256,
            n_mels=self.N_MELS,
            f_min=20,
            f_max=self.SAMPLE_RATE // 2,
            power=2.0
        )

        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80, stype="power")

    def _pad_or_crop(self, signal: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Crop if too long, zero-pad if too short."""
        
        length = signal.shape[1]
        
        if length > n_samples:
            signal = signal[:, :n_samples]
        elif length < n_samples:
            signal = torch.nn.functional.pad(signal, (0, n_samples - length))
        
        return signal


    def highpass_filter(self, audio: torch.Tensor, cutoff: int = 40) -> torch.Tensor:
        b, a = butter(5, cutoff / (self.SAMPLE_RATE / 2), btype='high')

        audio_np = audio.numpy()
        filtered = filtfilt(b, a, audio_np)

        return torch.tensor(filtered.copy(), dtype=audio.dtype)

    def trim_audio(self, audio: torch.Tensor, top_db: int = 20):
        clean_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return clean_audio

    def _spec_augment(self, spec):
        spec = spec.clone()
        n_mels, time_steps = spec.shape[1], spec.shape[2]

        f_mask = random.randint(2, 5)
        f0 = random.randint(0, max(1, n_mels - f_mask))
        spec[:, f0:f0 + f_mask, :] = spec.mean()

        t_mask = random.randint(10, 30)
        t0 = random.randint(0, max(1, time_steps - t_mask))
        spec[:, :, t0:t0 + t_mask] = spec.mean()

        return spec


class GenreDataset:
    def __init__(
        self, 
        genre_dir: Path, 
        esc50_dir: Path, 
        output_dir: Path, 
        augment: bool = False
    ):
        self.genre_dir = genre_dir
        self.esc50_paths = list(Path(esc50_dir).rglob("*.wav"))
        print("ESC50 samples found:", len(self.esc50_paths))

        self.output_dir = output_dir
        self.variables  = Variables()
        self.augment  = augment


    def merge_song(self, wave_list) -> torch.Tensor:
        """Merge stem wavs, highpass filter, normalise — returns full waveform."""
        waveforms = []
        
        for wave_path in wave_list:
            waveform, sr = torchaudio.load(wave_path)
            waveform = waveform.mean(dim=0, keepdim=True)
            
            if sr != self.variables.SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, self.variables.SAMPLE_RATE)(waveform)
            
            waveform = self.variables.highpass_filter(waveform)
            waveforms.append(waveform)

        max_len = max(w.shape[1] for w in waveforms)
        padded  = [torch.nn.functional.pad(w, (0, max_len - w.shape[1]))
                   if w.shape[1] < max_len else w for w in waveforms]

        full_waveform = torch.stack(padded, dim=0).mean(dim=0)

        max_val = full_waveform.abs().max()
        if max_val > 0:
            full_waveform = full_waveform / max_val

        return full_waveform   # shape: [1, total_samples] 
    

    def _load_esc50_clip(self, target_samples: int) -> torch.Tensor:
        """Load a random ESC50 clip, resample, and fit to target_samples."""
        path = random.choice(self.esc50_paths)
        noise, sr  = torchaudio.load(path)
        noise = noise.mean(dim=0, keepdim=True)
        if sr != self.variables.SAMPLE_RATE:
            noise  = torchaudio.transforms.Resample(sr, self.variables.SAMPLE_RATE)(noise)
        return self.variables._pad_or_crop(noise, target_samples)


    def inject_esc50_background(self, music: torch.Tensor) -> torch.Tensor:
        """
        BACKGROUND mixing — ESC50 is quiet and spread across the whole chunk.
        SNR range: 15–25 dB  →  noise is clearly below the music.
        """
        n_samples = music.shape[1]
        n_events = random.randint(1, 2)

        for _ in range(n_events):
            noise = self._load_esc50_clip(n_samples)
            snr_db  = random.uniform(15, 25)

            music_power = music.pow(2).mean() + 1e-9
            noise_power = noise.pow(2).mean() + 1e-9
            scale = torch.sqrt(music_power / (noise_power * 10 ** (snr_db / 10)))

            music = music + scale * noise

        max_val = music.abs().max()
        if max_val > 0:
            music = music / max_val
        return music


    def inject_esc50_foreground(self, music: torch.Tensor) -> torch.Tensor:
        """
        FOREGROUND mixing — ESC50 event is loud and placed at a random position.
        SNR range: 0–8 dB  →  noise is close to or louder than the music locally.
        Simulates a prominent real-world sound (siren, bark, crowd) during recording.
        """
        n_samples = music.shape[1]
        n_events = random.randint(1, 2)

        for _ in range(n_events):
            # Use a shorter clip (1–3 sec) so it feels like a discrete event
            event_dur = random.uniform(1.0, 3.0)
            event_samples = int(event_dur * self.variables.SAMPLE_RATE)
            event_samples = min(event_samples, n_samples)

            noise = self._load_esc50_clip(event_samples)
            # Place the event at a random position in the chunk
            max_start = n_samples - event_samples
            start = random.randint(0, max(0, max_start))

            snr_db = random.uniform(0, 8)
            music_power = music[:, start:start + event_samples].pow(2).mean() + 1e-9
            noise_power = noise.pow(2).mean() + 1e-9
            scale = torch.sqrt(music_power / (noise_power * 10 ** (snr_db / 10)))

            music[:, start:start + event_samples] += scale * noise

        max_val = music.abs().max()
        if max_val > 0:
            music = music / max_val
        return music


    def inject_esc50_noise(self, music: torch.Tensor) -> torch.Tensor:
        """
        Combined foreground + background mixing.
        Randomly applies one or both based on dice rolls.
          - 80% chance of background (subtle ambient noise)
          - 40% chance of foreground (prominent event noise)
        Both can apply in the same chunk.
        """
        if random.random() < 0.80:
            music = self.inject_esc50_background(music)
        if random.random() < 0.40:
            music = self.inject_esc50_foreground(music)
        return music


    def save_spectrogram(self, waveform: torch.Tensor, save_path: Path):
        """Compute mel-spectrogram and save as a PNG image."""
        mel = self.variables.mel(waveform)
        mel = self.variables.db(mel)
        mel = (mel + 80) / 80
        mel_np = mel.squeeze(0).cpu().numpy()

        plt.figure(figsize=(6, 4))
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

        #  Merge stems into one clean waveform (variable length)
        clean_waveform = self.merge_song(wav_paths)          # [1, T]
        total_samples = clean_waveform.shape[1]
        chunk_samples = self.variables.CHUNK_SAMPLES        # 225,792

        #  Split into 10.24s chunks
        n_chunks = max(1, math.ceil(total_samples / chunk_samples))

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_samples
            end = start + chunk_samples

            # Slice + zero-pad last chunk if short
            chunk = clean_waveform[:, start:end]
            if chunk.shape[1] < chunk_samples:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.shape[1]))

            clean_save = out_folder / f"{song_id}_chunk{chunk_idx:02d}_clean.png"
            self.save_spectrogram(chunk, clean_save)

            noisy_chunk = self.inject_esc50_noise(chunk.clone())
            noise_save = out_folder / f"{song_id}_chunk{chunk_idx:02d}_noise.png"
            self.save_spectrogram(noisy_chunk, noise_save)

            print(f"  [{genre_name}] song {song_id} chunk {chunk_idx:02d} → clean + noisy")

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


class MashUP:
    def __init__(
            self, 
            mashup_dir: 
            Path, 
            output_dir: Path
        ):
        self.mashup_dir = mashup_dir
        self.output_dir = output_dir
        
        self.variable = Variables()
        
        os.makedirs(self.output_dir, exist_ok=True)

    def save_spectrogram(self, waveform: torch.Tensor, save_path: Path) -> Path:
        save_path = save_path.with_suffix(".png")
        
        mel = self.variable.mel(waveform)
        mel = self.variable.db(mel)
        
        mel = (mel + 80) / 80
        mel_np = mel.squeeze(0).cpu().numpy()

        plt.figure(figsize=(6, 4))
        librosa.display.specshow(
            mel_np,
            sr=self.variable.SAMPLE_RATE,
            hop_length=256,
            x_axis=None,
            y_axis=None,
            cmap="magma"
        )
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(str(save_path), format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close()
        return save_path


    def process_all(self):
        chunk_samples = self.variable.CHUNK_SAMPLES

        for audio_path in sorted(self.mashup_dir.iterdir()):
            if audio_path.is_dir():
                continue

            waveform, sr = torchaudio.load(audio_path)
            waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sr != self.variable.SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, self.variable.SAMPLE_RATE)(waveform)

            waveform  = self.variable.highpass_filter(audio=waveform)
            song_id = audio_path.stem

            n_chunks = max(1, math.ceil(waveform.shape[1] / self.variable.CHUNK_SAMPLES))

            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_samples
                chunk = waveform[:, start:start + chunk_samples]
                if chunk.shape[1] < chunk_samples:
                    chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.shape[1]))

                save_path = self.output_dir / f"{song_id}_chunk{chunk_idx:02d}.png"
                save_path = self.save_spectrogram(waveform=chunk, save_path=save_path)
                print(f"saved: {save_path}")


if __name__ == "__main__":
    genredataset = GenreDataset(
        genre_dir=Path("Z:\\dlgen-AI-project\\data\\genres_stems"),
        output_dir=Path("Z:\\dlgen-AI-project\\data\\genres_mel"),
        esc50_dir=Path("Z:\\dlgen-AI-project\\data\\ESC-50-master")
    )

    genredataset.process_all()

    # mashup = MashUP(
    #     mashup_dir=Path("Z:\\dlgen-AI-project\\data\\mashups"),
    #     output_dir=Path("Z:\\dlgen-AI-project\\data\\mashups_image")
    # )
    # mashup.process_all()
