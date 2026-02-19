import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

from pathlib import Path
from torch.utils.data import Dataset
import torchaudio

import pandas as pd
import torch

import librosa

from scipy.signal import butter, filtfilt


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
    N_MELS = 128
    DURATION = 5   # seconds
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

    def _fix_length(self, signal: torch.Tensor, duration: int = 5) -> torch.Tensor:
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
    


class ESC50_DataLoader(Dataset):

    def __init__(self, esc50_dir: Path, allowed_folds=None, augment: bool=False):
        self.esc50_dir = esc50_dir
        self.audio_dir = esc50_dir / "audio"
        self.variables = Variables()
        self.augment = augment

        self.meta_data = None
        self.allowed_folds = allowed_folds

    def load_meta_data(self, meta_data_path: str):
        print(f"Loading metadata from file: {meta_data_path}")
        df = pd.read_csv(meta_data_path)

        # filter folds
        if self.allowed_folds is not None:
            df = df[df["fold"].isin(self.allowed_folds)].reset_index(drop=True)

        self.meta_data = df
    
    
    def __len__(self):
        return self.meta_data.shape[0]

    
    def __getitem__(self, index: int):
        row = self.meta_data.iloc[index]

        filename = row["filename"]
        label = int(row["target"])

        file_path = self.audio_dir / filename

        waveform, sr = torchaudio.load(file_path)

        # mono as esc50 is has mono channel
        waveform = torch.mean(waveform, dim=0, keepdim=True)

        # resample
        if sr != self.variables.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.variables.SAMPLE_RATE)
            waveform = resampler(waveform)

        # pass through highpass filter to reomve audio with small freq
        waveform = self.variables.highpass_filter(audio=waveform)

        # normalize amplitued 
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        waveform = self.variables._fix_length(signal=waveform, duration=5)

        # mel spectrogram
        mel = self.variables.mel(waveform)
        mel = self.variables.db(mel)

        # augment data for only training 
        if self.augment:
            mel = self.variables._spec_augment(spec=mel)

        # mean = mel.mean()
        # std = mel.std() + 1e-6

        # mel = (mel - mean) / std

        return mel, torch.tensor(label, dtype=torch.long)



class GenreDataLoader(Dataset):
    def __init__(self, root_dir: Path, augment: bool=False):
        self.root_dir = root_dir 
        self.variables = Variables()
        self.augment = augment

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

        print(self.samples)

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

            waveform = self.variables.highpass_filter(audio=waveform)

            # normalize amplitued 
            # max_val = waveform.abs().max()
            # if max_val > 0:
            #     waveform = waveform / max_val

            waveforms.append(waveform)
        

        full_waveform = waveforms[0].clone()

        for wave in waveforms[1:]:
            full_waveform = full_waveform + wave

        full_waveform = full_waveform / len(waveforms)

        full_waveform = self.variables._fix_length(full_waveform, duration=30)

        # get complete length of 30 sec data
        total_samples = full_waveform.shape[1]
        # divide into chunks
        num_chunks = total_samples // self.variables.SAMPLES

        # create chunks for 5 sec
        chunks = []
        # if full_waveform has less than 5 sec data
        if num_chunks == 0:
            full_waveform = self.variables._fix_length(full_waveform, self.variables.SAMPLES, duration=5)
            # chunks = full_waveform.unsqueeze(0)

            chunk_mel = self.variables.mel(full_waveform)
            chunk_mel = self.variables.db(chunk_mel)

            if self.augment:
                chunk_mel = self.variables._spec_augment(spec=chunk_mel)

            # chunk_mean = chunk_mel.mean()
            # chunk_std = chunk_mel.std() + 1e-6
            
            # mel = (chunk_mel - chunk_mean) / chunk_std

            chunks.append(chunk_mel)
        
        else:
            for i in range(num_chunks):
                # get 5 sec data from entire data 
                start_idx = i * self.variables.SAMPLES
                end_idx = start_idx + self.variables.SAMPLES

                chunk = full_waveform[:, start_idx: end_idx]

                chunk_mel = self.variables.mel(chunk)
                chunk_mel = self.variables.db(chunk_mel)

                if self.augment:
                    chunk_mel = self.variables._spec_augment(spec=chunk_mel)

                # chunk_mean = chunk_mel.mean()
                # chunk_std = chunk_mel.std() + 1e-6
                
                # mel = (chunk_mel - chunk_mean) / chunk_std

                chunks.append(chunk_mel)

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
