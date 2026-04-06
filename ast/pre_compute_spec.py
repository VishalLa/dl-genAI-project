import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
import random
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio

from tqdm import tqdm

from transformers import ASTFeatureExtractor 
import torch
import numpy as np
from scipy.signal import butter, filtfilt

from helper.golbal_variables import *


def highpass_filter(
    audio: torch.Tensor,
    cutoff: int = 40,
    sample_rate: int = 16000
):

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


class AST_Audio_DataLoader(Dataset):

    """
    Loads raw multi-stem audio, merges stems, augments, and returns
    10.24-second waveform chunks for pre-computation.

    Note:
        This class is NOT used during model training. It is only called by
        precompute_ast_dataset() to generate the .pt spectrogram cache.
        Fast_AST_Dataset is used for actual training (much faster).

    Expected directory structure:
        genre_dir/
            blues/
                blues.00000/
                    blues.00000.wav
                blues.00001/
                    ...
            classical/
                ...

    Args:
        genre_dir (Path): Root directory containing one subfolder per genre.
        esc50_dir (Path): Root directory of ESC-50 dataset (.wav files)
                          used for noise augmentation.
    """

    def __init__(self, genre_dir: Path, esc50_dir: Path):
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
        Builds self.index_map — a flat list of (song_idx, chunk_start_sample) tuples.
        Only the first stem of each song is loaded to measure length (avoids
        loading all stems just to count chunks).

        Note:
            This is needed because songs have variable lengths, so a flat integer
            index cannot map directly to a (song, chunk) pair without scanning all
            songs first. The index_map makes __getitem__ O(1).
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
        return len(self.index_map)


    def merge_song(self, wave_list):
        
        """
        Loads, resamples, filters, and averages all stems for one song.
        Shorter stems are zero-padded to the longest before averaging.

        Args:
            wave_list (list[Path]): Stem .wav files for one song.

        Returns:
            torch.Tensor: Merged mono waveform of shape [1, T].
        """

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
        Extracts a fixed-length window from a waveform.
        Zero-pads the last chunk if it is shorter than CHUNK_SAMPLES.

        Args:
            waveform (torch.Tensor): Full waveform of shape [1, T].
            start         (int)    : Start sample index.

        Returns:
            torch.Tensor: Waveform chunk of shape [1, CHUNK_SAMPLES].
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
        Randomly mixes ESC-50 environmental noise into a music chunk.
        Skipped with 20% probability so the model also sees clean examples.

        SNR is sampled from [8, 18] dB per noise event:
            18 dB → noise is barely audible.
            8 dB → noise is clearly audible but music still dominates.

        Scaling formula:
            scale = sqrt(music_power / (noise_power * 10 ^ (SNR_dB / 10)))

        Args:
            music_waveform (torch.Tensor): Music chunk of shape [1, T].

        Returns:
            torch.Tensor: Waveform with noise mixed in, shape [1, T].
                          Returned unchanged if augmentation is skipped.
        """

        if random.random() > 0.8:
            return music_waveform
        
        samples = music_waveform.shape[1]
        n_events = random.randint(1, 2)

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
        Returns one augmented waveform chunk and its label.

        Pipeline:
            1. Look up (song_idx, start) from index_map.
            2. Merge all stems for the song via merge_song().
            3. Slice a CHUNK_SAMPLES window via get_chunk().
            4. Inject ESC-50 noise via inject_esc50_noise() (skipped 20% of the time).
            5. RMS-normalise: waveform / (rms * 3.0).
            6. Clamp to [-1.0, 1.0] to prevent clipping.

        Args:
            index (int): Flat index into self.index_map.

        Returns:
            waveform (torch.Tensor): 1D waveform of shape [CHUNK_SAMPLES],
                                     ready for ASTFeatureExtractor.
            label    (torch.Tensor): Scalar int64 genre label.
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
