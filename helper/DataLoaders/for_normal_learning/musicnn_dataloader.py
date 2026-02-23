import random
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio
import torch
import torch.nn.functional as F
from scipy.signal import butter, filtfilt


# ----------------------------
# Audio preprocessing helpers
# ----------------------------

def highpass_filter(audio: torch.Tensor, cutoff=40, sample_rate=22050):
    b, a = butter(5, cutoff / (sample_rate / 2), btype='high')
    audio_np = audio.squeeze(0).cpu().numpy()
    filtered = filtfilt(b, a, audio_np)
    return torch.from_numpy(filtered.copy()).unsqueeze(0).type_as(audio)


def rms_normalize(waveform):
    rms = torch.sqrt(torch.mean(waveform**2) + 1e-9)
    waveform = waveform / (rms + 1e-9)
    return waveform.clamp(-1.0, 1.0)


# ----------------------------
# MUSICNN 3-SEC DATASET
# ----------------------------

class MusicNNDataset(Dataset):

    def __init__(self, genre_dir: Path, esc50_dir: Path):

        self.genre_dir = genre_dir
        self.esc50_paths = list(esc50_dir.rglob("*.wav"))

        self.SAMPLE_RATE = 22050
        self.CHUNK_DURATION = 3
        self.CHUNK_SAMPLES = self.SAMPLE_RATE * self.CHUNK_DURATION

        self.samples = []
        self.labels = []
        self.class_map = {}

        # 96 mel bins (MusicNN requirement)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=96
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        self._load_files()

    def _load_files(self):

        for label_id, genre_dir in enumerate(sorted(self.genre_dir.iterdir())):

            if not genre_dir.is_dir():
                continue

            self.class_map[genre_dir.name] = label_id

            for wav in genre_dir.rglob("*.wav"):
                self.samples.append(wav)
                self.labels.append(label_id)

        print("Classes:", self.class_map)
        print("Total audio files:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def random_crop(self, waveform):

        if waveform.shape[1] <= self.CHUNK_SAMPLES:
            pad = self.CHUNK_SAMPLES - waveform.shape[1]
            return F.pad(waveform, (0, pad))

        start = random.randint(0, waveform.shape[1] - self.CHUNK_SAMPLES)
        return waveform[:, start:start+self.CHUNK_SAMPLES]

    def inject_esc50_noise(self, waveform):

        if random.random() > 0.8:
            return waveform

        noise_path = random.choice(self.esc50_paths)
        noise_wave, sr = torchaudio.load(noise_path)

        noise_wave = noise_wave.mean(dim=0, keepdim=True)

        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            noise_wave = resampler(noise_wave)

        noise_wave = self.random_crop(noise_wave)

        snr_db = random.uniform(8, 18)

        music_power = waveform.pow(2).mean()
        noise_power = noise_wave.pow(2).mean() + 1e-9

        scale = torch.sqrt(music_power / (noise_power * (10 ** (snr_db / 10))))
        waveform = waveform + scale * noise_wave

        return waveform

    def __getitem__(self, index):

        path = self.samples[index]
        label = self.labels[index]

        waveform, sr = torchaudio.load(path)

        waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            waveform = resampler(waveform)

        # -------- PREPROCESSING --------
        waveform = highpass_filter(waveform, sample_rate=self.SAMPLE_RATE)
        waveform = self.random_crop(waveform)
        waveform = self.inject_esc50_noise(waveform)
        waveform = rms_normalize(waveform)

        # -------- MEL SPECTROGRAM --------
        mel = self.mel_transform(waveform)
        mel = self.db_transform(mel)

        # Normalize spectrogram
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        return mel, torch.tensor(label, dtype=torch.long)
    
def musicnn_collate_fn(batch):
    """
    batch: list of (mel, label)

    Returns:
        mels   → (B, 1, 96, T)
        labels → (B,)
    """

    mels = []
    labels = []

    for mel, label in batch:
        mels.append(mel)
        labels.append(label)

    mels = torch.stack(mels, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return mels, labels