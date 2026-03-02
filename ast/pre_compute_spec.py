import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
import random
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio

from tqdm import tqdm

from transformers import ASTFeatureExtractor 
import torch

from utils import highpass_filter

from helper.golbal_variables import *

class AST_Audio_DataLoader(Dataset):

    """
    Custom PyTorch Dataset for preparing audio data for Audio Spectrogram Transformer (AST).

    This dataset:
    - Loads multi-stem music (.wav) files
    - Converts to mono and resamples to 16kHz
    - Applies high-pass filtering
    - Merges stems into a single waveform
    - Splits audio into 10.24-second chunks (1024 spectrogram frames)
    - Optionally injects ESC-50 environmental noise
    - Normalizes waveform
    - Returns 1D waveform + label

    Each song may generate multiple training samples (chunks).
    """

    def __init__(self, genre_dir: Path, esc50_dir: Path):

        """
        Args:
            genre_dir (Path): Root directory containing genre folders.
            esc50_dir (Path): Root directory containing ESC-50 noise files.
        """

        self.genre_dir = genre_dir
        self.esc50_paths = list(esc50_dir.rglob("*.wav"))

        self.SAMPLE_RATE = 16000
        
        # AST is optimized for 10.24 seconds (exactly 1024 spectrogram frames)
        self.CHUNK_DURATION = 10.24 
        self.CHUNK_SAMPLES = int(self.SAMPLE_RATE * self.CHUNK_DURATION)

        self.samples = []
        self.labels = []
        self.class_map = {}

        self._load_files()
        self._build_index_map()


    def _load_files(self):

        """
        Loads all song stems from genre directory.

        Expected structure:
            genre/
                song_1/
                    bass.wav
                    drums.wav
                    vocals.wav
                    other.wav

        Builds:
            self.samples → list of list of wav paths
            self.labels → genre label per song
            self.class_map → {genre_name: label_id}
        """

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

        """
        Builds an index mapping from dataset index → (song_index, chunk_start_sample).

        Each song is divided into multiple 10.24-second chunks.
        This allows each chunk to act as an independent training sample.
        """

        self.index_map = []
        for song_idx in range(len(self.samples)):
            waveform, sr = torchaudio.load(self.samples[song_idx][0])
            waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
                waveform = resampler(waveform)

            total_len = waveform.shape[1]
            n_chunks = (total_len + self.CHUNK_SAMPLES - 1) // self.CHUNK_SAMPLES

            for c in range(n_chunks):
                start = c * self.CHUNK_SAMPLES
                self.index_map.append((song_idx, start))

        print("Total training segments:", len(self.index_map))

    def __len__(self):
        """Returns total number of audio chunks in dataset."""
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

        """
        Extracts a 10.24-second chunk from waveform.

        Pads if the remaining samples are shorter than chunk size.

        Args:
            waveform (torch.Tensor): Input waveform [1, samples]
            start (int): Starting sample index

        Returns:
            torch.Tensor: Chunked waveform [1, CHUNK_SAMPLES]
        """

        end = start + self.CHUNK_SAMPLES

        if waveform.shape[1] < end:
            padding = end - waveform.shape[1]
            chunk = torch.nn.functional.pad(waveform[:, start:], (0, padding))
        else:
            chunk = waveform[:, start:end]

        return chunk

    def inject_esc50_noise(self, music_waveform):

        """
        Randomly injects environmental noise from ESC-50 dataset.

        - 20% probability of skipping injection
        - Random 1–2 noise events
        - Random SNR between 8–18 dB

        Args:
            music_waveform (torch.Tensor): Clean music waveform

        Returns:
            torch.Tensor: Augmented waveform
        """

        if random.random() > 0.8:
            return music_waveform
        
        samples = music_waveform.shape[1]
        n_events = random.randint(1, 3)

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

            snr_db = random.uniform(8, 18)
            music_power = music_waveform.pow(2).mean()
            noise_power = noise_wave.pow(2).mean() + 1e-9

            scale = torch.sqrt(music_power / (noise_power * (10 ** (snr_db / 10))))
            music_waveform[:, start:start+noise_len] += scale * noise_wave

        return music_waveform

    def __getitem__(self, index):

        """
        Returns:
            Tuple:
                - 1D normalized waveform tensor [samples]
                - genre label tensor (long)
        """

        song_idx, start = self.index_map[index]
        wave_list = self.samples[song_idx]
        label = self.labels[song_idx]

        full_waveform = self.merge_song(wave_list)
        full_waveform = self.get_chunk(full_waveform, start)
        full_waveform = self.inject_esc50_noise(full_waveform)

        rms = torch.sqrt(torch.mean(full_waveform**2) + 1e-9)
        full_waveform = full_waveform / (rms * 3.0)
        full_waveform = full_waveform.clamp(-1.0, 1.0)
        
        full_waveform = full_waveform.squeeze(0) 

        return full_waveform, torch.tensor(label, dtype=torch.long)

feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

def collate_fn(batch):    
    """    
    Collate function for AST    
    Converts list of (1D waveform, label) -> batched 2D spectrogram tensors    
    """
    waveforms = []
    labels = []

    for waveform, label in batch:
        # Convert torch tensor to numpy array for the HF feature extractor
        waveform = waveform.cpu().numpy()
        waveforms.append(waveform)
        labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    
    
    # This automatically calculates the STFT (short time foriur transformer) and builds the Mel-spectrograms
    inputs = feature_extractor(
        waveforms,
        sampling_rate=16000,
        padding="max_length", # Ensures all spectrograms are exactly the same size
        return_tensors="pt"
    )

    labels = torch.tensor(labels, dtype=torch.long)

    # Vision Transformers (like AST) generally do not use attention masks 
    # for classification the way Wav2Vec2 does.
    return inputs.input_values, labels


def precompute_ast_dataset(genre_path: Path, esc50_path: Path, output_path: Path):
    print("Initializing original dataset (this might take a second)...")
    original_dataset = AST_Audio_DataLoader(genre_dir=genre_path, esc50_dir=esc50_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create genre subfolders
    id2genre = {v: k for k, v in original_dataset.class_map.items()}
    for genre in original_dataset.class_map.keys():
        (output_path / genre).mkdir(exist_ok=True)

    # Load the AST Extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    print(f"\nProcessing and saving {len(original_dataset)} spectrograms...")
    
    for i in tqdm(range(len(original_dataset))):
        
        # 1. Get the complex merged/filtered/noisy 1D waveform from your class
        waveform, label = original_dataset[i]
        genre_name = id2genre[label.item()]
        
        # 2. Convert to AST 2D Spectrogram
        inputs = feature_extractor(
            waveform.numpy(),
            sampling_rate=16000,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Remove the batch dimension so shape is exactly [1024, 128]
        spectrogram = inputs.input_values.squeeze(0) 
        
        # 3. Save the tensor and label directly to disk
        save_file = output_path / genre_name / f"chunk_{i}.pt"
        torch.save((spectrogram, label), save_file)

    print(f"\nDone! All data saved to: {output_path.absolute()}")


precompute_ast_dataset(
    genre_dir=GENRES_STEMS, 
    esc50_dir=ESC_50_MASTER, 
    output_dir=GENRES_PT
)
