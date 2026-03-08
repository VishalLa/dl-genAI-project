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


SAMPLE_RATE = 16000
CHUNK_DURATION = 10.24


class AST_Audio_DataLoader(Dataset):

    """
    Custom PyTorch Dataset for preparing audio data for Audio Spectrogram Transformer (AST).

    This dataset:
    - Loads multi-stem music (.wav) files
    - Converts to mono and resamples to 16kHz
    - Applies high-pass filtering
    - Merges stems into a single waveform
    - 8-second chunks  (128 000 samples @ 16 kHz)
      - ESC-50 noise injection with two modes:
          * BACKGROUND  — noise sits quietly behind the music  (SNR 18–35 dB)
          * FOREGROUND  — noise is clearly audible / overpowers music (SNR 2–10 dB)
        35% of injected events are foreground, 65% are background.

    Each song may generate multiple training samples (chunks).
    """

    # ESC-50 noise injection probabilities
    ESC50_INJECT_PROB = 0.65   # chance any chunk gets noise at all
    ESC50_FG_PROB = 0.35   # of injected chunks: chance noise is FOREGROUND

    # SNR ranges (dB)  — higher SNR = quieter noise relative to music
    BG_SNR_RANGE = (18, 35)    # background: barely audible
    FG_SNR_RANGE = (2,  10)    # foreground: clearly audible / overpowers music

    def __init__(self, genre_dir: Path, esc50_dir: Path):

        """
        Args:
            genre_dir (Path): Root directory containing genre folders.
            esc50_dir (Path): Root directory containing ESC-50 noise files.
        """

        self.genre_dir = genre_dir
        self.esc50_paths = list(esc50_dir.rglob("*.wav"))
        
        self.SAMPLE_RATE = SAMPLE_RATE 
        self.CHUNK_DURATION = CHUNK_DURATION
        self.CHUNK_SAMPLES  = int(self.SAMPLE_RATE * self.CHUNK_DURATION)

        self.samples = []
        self.labels = []
        self.class_map = {}
        self.index_map = []

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
        Pre-compute (song_idx, start_sample) for every 8-second chunk
        across all songs. Overlapping is NOT used — clean non-overlapping grid.
        """
        for song_idx, wave_parts in enumerate(tqdm(self.samples)):
            waveform, sr = torchaudio.load(wave_parts[0])
            waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(waveform)

            # for multi-part songs, estimate totlat length from part count 
            part_len = waveform.shape[1]
            est_total = part_len * len(wave_parts)

            n_chunks = max(1, est_total // self.CHUNK_SAMPLES)
            for c in range(n_chunks):
                self.index_map.append((song_idx, c * self.CHUNK_SAMPLES))

        print(f"Total chunks : {len(self.index_map)}")
        print(f"Total songs  : {len(self.samples)}")
        print(f"Classes      : {list(self.class_map.keys())}")

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

        return torch.cat(waveforms, dim=1)

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
    

    def _load_noise(self, target_len: int) -> torch.Tensor:
        """
        Load a random ESC-50 clip, resample, and fit it to target_len samples.
        Short clips are tiled; long clips are randomly cropped.
        """

        noise_path = random.choice(self.esc50_paths)
        noise, sr = torchaudio.load(noise_path)

        if sr != self.SAMPLE_RATE:
            noise = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(noise)

        noise_len = noise.shape[1]

        if noise_len < target_len:
            # Tile the short clip to fill the required length
            repeats = (target_len // noise_len) + 1
            noise = noise.repeat(1, repeats)

        # Random crop to exactly target_len
        max_start = noise.shape[1] - target_len
        start = random.randint(0, max_start)
        return noise[:, start : start + target_len]


    def inject_esc50_noise(self, music_waveform: torch.Tensor) -> torch.Tensor:

        """
        Inject ESC-50 environmental noise into the music chunk.

        Two modes chosen randomly per event:
          BACKGROUND (65% of events) — low volume, noise barely noticeable
              SNR range: 18–35 dB  → noise is ~8–56× quieter than music
          FOREGROUND (35% of events) — noise clearly audible, can overpower
              SNR range: 2–10 dB   → noise is ~0.4–3× the music level

        Overall injection probability: ESC50_INJECT_PROB (0.65)
        Number of simultaneous events: 1–3
        """

        if not self.esc50_paths or random.random() > self.ESC50_INJECT_PROB:
            return music_waveform

        chunk_len = music_waveform.shape[1]
        n_events = random.randint(1, 3)

        for _ in range(n_events):
            noise = self._load_noise(chunk_len)   # [1, chunk_len]

            # Decide foreground vs background for this individual event
            is_foreground = random.random() < self.ESC50_FG_PROB
            snr_range = self.FG_SNR_RANGE if is_foreground else self.BG_SNR_RANGE
            snr_db = random.uniform(*snr_range)

            # Compute power-based scale factor
            music_power = music_waveform.pow(2).mean().clamp(min=1e-9)
            noise_power = noise.pow(2).mean().clamp(min=1e-9)
            scale = torch.sqrt(music_power / (noise_power * (10 ** (snr_db / 10))))

            # For foreground events, optionally place noise at a random time offset
            # so it sounds like something "happening" rather than constant background
            if is_foreground:
                event_len = int(chunk_len * random.uniform(0.3, 1.0))
                offset = random.randint(0, chunk_len - event_len)
                music_waveform[:, offset : offset + event_len] += (
                    scale * noise[:, :event_len]
                )
            else:
                # Background: mix across entire chunk at low volume
                music_waveform = music_waveform + scale * noise

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


def precompute_ast_dataset(genre_dir: Path, esc50_dir: Path, output_dir: Path, n_augment_passes: int = 1):
    """
    Pre-compute and save AST spectrograms to disk.

    Each chunk is saved as a .pt file containing (spectrogram [1024,128], label).

    Args:
        genre_path      : Root directory of genre folders.
        esc50_path      : Root directory of ESC-50 wav files.
        output_path     : Where to save .pt spectrogram files.
        n_augment_passes: How many times to iterate the dataset with different
                          random noise seeds (data multiplication).
                          1 = no duplication, 2 = 2× data, etc.
    """
    print("Initializing dataset ...")
    dataset = AST_Audio_DataLoader(genre_dir=genre_dir, esc50_dir=esc50_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create per-genre subfolders
    id2genre = {v: k for k, v in dataset.class_map.items()}
    for genre in dataset.class_map:
        (output_dir / genre).mkdir(exist_ok=True)

    total = len(dataset) * n_augment_passes
    print(f"\nSaving {total} spectrograms  "
          f"({len(dataset)} chunks × {n_augment_passes} augment pass(es)) ...")
    print(f"Chunk duration : {CHUNK_DURATION:.2f} s  ({int(SAMPLE_RATE * CHUNK_DURATION)} samples)")

    global_idx = 0
    for pass_idx in range(n_augment_passes):
        desc = f"Pass {pass_idx + 1}/{n_augment_passes}"
        for i in tqdm(range(len(dataset)), desc=desc):
            waveform, label = dataset[i]
            genre_name = id2genre[label.item()]

            inputs = feature_extractor(
                waveform.numpy(),
                sampling_rate=SAMPLE_RATE,
                padding="max_length",
                return_tensors="pt",
            )
            spectrogram = inputs.input_values.squeeze(0)  # [1024, 128]

            save_path = output_dir / genre_name / f"chunk_{global_idx:06d}.pt"
            torch.save((spectrogram, label), save_path)
            global_idx += 1

    print(f"\nDone! {global_idx} files saved to: {output_dir.absolute()}")
    print(f"Class map: {dataset.class_map}")


precompute_ast_dataset(
    genre_dir=GENRES_STEMS, 
    esc50_dir=ESC_50_MASTER, 
    output_dir=GENRES_PT,
    n_augment_passes=1
)
