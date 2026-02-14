import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

from pathlib import Path
from torch.utils.data import Dataset
import torchaudio

import pandas as pd
import torch


# Took entire day of making the dataloder and debuging it :|

'''
Nyquist theorem:

Max frequency representable = sample_rate / 2
I learned it in sns class

our sample rate is  44.1 kHz, mono

so sample max frequency representable = 22050


currently we have time domain signal 
converts it in frequency domain and it can be done using Fourier Transform (STFT)
'''

class Variables:
    SAMPLE_RATE = 22050
    N_MELS = 256
    DURATION = 5   # seconds
    SAMPLES = SAMPLE_RATE * DURATION

    def __init__(self):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=2048,
            hop_length=512,
            n_mels=self.N_MELS
        )

        # dB=10⋅log10​(Power)
        self.db = torchaudio.transforms.AmplitudeToDB()

    def _fix_length(self, signal: torch.Tensor):
        length = signal.shape[1]

        # too long → crop
        if length > self.SAMPLES:
            signal = signal[:, :self.SAMPLES]

        # too short → pad
        elif length < self.SAMPLES:
            padding = self.SAMPLES - length
            signal = torch.nn.functional.pad(signal, (0, padding))

        return signal


class ESC50_DataLoader(Dataset):
    def __init__(self, esc50_dir: Path):
        self.esc50_dir = esc50_dir
        self.audio_dir = esc50_dir / "audio"
        self.variables = Variables()

        self.meta_data = None

    def load_meta_data(self, meta_data_path: str):
        print(f"Loading metadata from file: {meta_data_path}")
        self.meta_data = pd.read_csv(meta_data_path)
    
    
    def __len__(self):
        return self.meta_data.shape[0]

    
    def __getitem__(self, index: int):
        row = self.meta_data.iloc[index]

        filename = row["filename"]
        label = int(row["target"])

        file_path = self.audio_dir / filename

        num_frames = self.variables.SAMPLES 

        waveform, sr = torchaudio.load(
            file_path,
            frame_offset=0,
            num_frames=num_frames
        )

        # mono as esc50 is has mono channel
        waveform = torch.mean(waveform, dim=0, keepdim=True)

        # resample
        if sr != self.variables.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.variables.SAMPLE_RATE)
            waveform = resampler(waveform)

        waveform = self.variables._fix_length(signal=waveform)

        # mel spectrogram
        mel = self.variables.mel(waveform)
        mel = self.variables.db(mel)

        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        return mel, torch.tensor(label, dtype=torch.long)


class GenreDataLoader(Dataset):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir 
        self.variables = Variables()

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
                # print(wav_parts)
                if len(wav_parts) == 0:
                    continue
                
                self.samples.append(wav_parts)
                self.labels.append(label_id)

        print("Classes:", self.class_map)
        print("Total songs:", len(self.samples))


    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, index):
        '''
        Docstring for __getitem__
        
        I am combining each .wav file for songs to get a final waveform for training for 5 sec window
        '''

        wave_list = self.samples[index]
        label = self.labels[index]

        waveforms = []

        for wav_path in wave_list:
            waveform, sr = torchaudio.load(wav_path)

            # mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)

            # resample
            if sr != self.variables.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.variables.SAMPLE_RATE)
                waveform = resampler(waveform)

            waveform = self.variables._fix_length(signal=waveform)

            waveforms.append(waveform)

        final_waveform = torch.stack(waveforms, dim=0).mean(dim=0)

        # final_waveform = self.variables._fix_length(signal=final_waveform)
        # print(final_waveform.shape)
        
        mel = self.variables.mel(final_waveform)
        mel = self.variables.db(mel)

        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        return mel, torch.tensor(label, dtype=torch.long)
