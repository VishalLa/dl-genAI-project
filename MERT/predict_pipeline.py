import os 
import json
import copy
import math
import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
import torchaudio

from collections import Counter
from typing import Callable, Dict, List, Optional
from tqdm import tqdm
from pathlib import Path

from transformers import Wav2Vec2FeatureExtractor
from scipy.signal import butter, filtfilt


def highpass_filter(
    audio: torch.Tensor, 
    cutoff: int = 40, 
    sample_rate: int = 24000
):
    """
    Applies a Butterworth high-pass filter to remove very low-frequency noise.

    Args:
        audio (torch.Tensor): Input waveform tensor of shape [1, samples].
        cutoff (int): Cutoff frequency in Hz. Default is 40 Hz.
        sample_rate (int): Sampling rate of the waveform. Default is 24000 Hz.

    Returns:
        torch.Tensor: Filtered waveform with same shape and dtype as input.
    """
    b, a = butter(5, cutoff / (sample_rate / 2), btype='high')
    audio_np = audio.squeeze(0).cpu().numpy()
    filtered = filtfilt(b, a, audio_np)
    filtered = np.ascontiguousarray(filtered)
    filtered = torch.from_numpy(filtered).unsqueeze(0)
    return filtered.type_as(audio)


class MERTPredictionPipeline:
    """
    Single MERT model prediction pipeline for genre classification.
    
    Args:
        model: Single MERT model
        genre_map: {genre_name: label_id}
        batch_size: chunks per forward pass
    
    Usage:
        model = MERTGenreClassifier(num_classes=10)
        model.load_state_dict(torch.load("best_model.pth"))
        pipeline = MERTPredictionPipeline(model, genre_map)
        genre, conf, probs = pipeline.predict("song.mp3")
    """
    
    MERT_SAMPLE_RATE = 24000
    MERT_CHUNK_SIZE = 5
    MERT_CHUNK_SAMPLES = int(MERT_SAMPLE_RATE * MERT_CHUNK_SIZE)

    def __init__(
        self, 
        model: nn.Module,
        genre_map: dict,
        batch_size: int = 8
    ):
        self.model = model
        self.genre_map = genre_map
        self.index_to_genre = {v: k for k, v in self.genre_map.items()}
        self.batch_size = batch_size 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", 
            trust_remote_code=True
        )

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        print(f"\n{'='*70}")
        print(f"MERT Pipeline Initialized")
        print(f"{'='*70}")
        print(f"Device:      {self.device}")
        print(f"Batch size:  {self.batch_size}")
        print(f"Genres:      {list(self.genre_map.keys())}")
        print(f"{'='*70}\n")

    def _load_and_preprocess_mert(self, audio_path: Path) -> torch.Tensor:
        """Load audio for MERT: 24kHz, highpass filtered."""
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.MERT_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.MERT_SAMPLE_RATE)
            waveform = resampler(waveform)

        waveform = highpass_filter(
            audio=waveform, 
            cutoff=40, 
            sample_rate=self.MERT_SAMPLE_RATE
        )
        return waveform

    def _chunk_waveform_mert(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Split waveform into 30s chunks for MERT."""
        total_samples = int(waveform.shape[1])
        
        # If song is shorter than 30s, pad it and return 1 chunk
        if total_samples < self.MERT_CHUNK_SAMPLES:
            chunk = waveform[:, :]
            padding = self.MERT_CHUNK_SAMPLES - total_samples
            chunk = torch.nn.functional.pad(chunk, (0, padding))
            return [chunk]
        
        # Calculate number of complete 30s chunks (drops incomplete)
        n_chunks = total_samples // self.MERT_CHUNK_SAMPLES        
        chunks = []

        for c in range(n_chunks):
            start = c * self.MERT_CHUNK_SAMPLES
            end = start + self.MERT_CHUNK_SAMPLES
            chunk = waveform[:, start:end]
            chunks.append(chunk)

        return chunks

    def _chunks_to_mert_input(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        """Process waveform chunks for MERT."""
        normalized = []
        for chunk in chunks:
            rms = torch.sqrt(torch.mean(chunk ** 2) + 1e-9)
            chunk = chunk / (rms * 3.0)
            chunk = chunk.clamp(-1.0, 1.0)
            normalized.append(chunk.squeeze(0).cpu().numpy())

        inputs = self.mert_processor(
            normalized,
            sampling_rate=self.MERT_SAMPLE_RATE,
            padding=True,
            return_tensors="pt"
        )
        return inputs.input_values

    @torch.no_grad()
    def _get_mert_chunk_probs_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Get predictions from MERT model (batch processing).
        
        Args:
            waveforms: [N, 720000]
        
        Returns:
            chunk_probs: [N, num_classes]
        """
        n_chunks = len(waveforms)
        all_probs = []

        for i in range(0, n_chunks, self.batch_size):
            batch = waveforms[i:i + self.batch_size].to(self.device)
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1).cpu()
            all_probs.append(probs)

        # Concatenate all batches
        chunk_probs = torch.cat(all_probs, dim=0)
        return chunk_probs

    @torch.no_grad()
    def _get_mert_chunk_probs_single(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Get predictions from MERT model (single processing, lower memory).
        
        Args:
            waveforms: [N, 720000]
        
        Returns:
            chunk_probs: [N, num_classes]
        """
        n_chunks = len(waveforms)
        all_probs = []

        for i in range(n_chunks):
            chunk = waveforms[i].unsqueeze(0).to(self.device)
            logits = self.model(chunk)
            probs = torch.softmax(logits, dim=1).cpu()
            all_probs.append(probs)

        # Concatenate all chunks
        chunk_probs = torch.cat(all_probs, dim=0)
        return chunk_probs

    def _vote(
        self,
        mert_probs: torch.Tensor,
        n_mert_chunks: int,
        verbose: bool = False,
        audio_path: Path = None,
        return_json: bool = False,
    ) -> tuple | dict:
        """
        Vote on MERT probabilities.
        
        Returns:
            If return_json=False: (predicted_genre, confidence, all_probs_sorted)
            If return_json=True: dict with detailed prediction info
        """
        # Average across chunks
        avg_probs = mert_probs.mean(dim=0)  # [C]

        # Soft vote (primary prediction)
        soft_idx = avg_probs.argmax().item()
        soft_confidence = avg_probs[soft_idx].item()
        predicted_genre = self.index_to_genre[soft_idx]

        # Hard vote (diagnostic)
        chunk_preds = mert_probs.argmax(dim=1).tolist()
        hard_idx = Counter(chunk_preds).most_common(1)[0][0]
        hard_votes = chunk_preds.count(hard_idx)
        hard_genre = self.index_to_genre[hard_idx]

        all_probs_sorted = dict(sorted(
            {self.index_to_genre[i]: avg_probs[i].item() for i in range(len(avg_probs))}.items(),
            key=lambda x: x[1], 
            reverse=True,
        ))

        if verbose:
            agreement = (
                "✅ agree" if soft_idx == hard_idx
                else f"⚠️ disagree (hard → {hard_genre} {hard_votes}/{len(chunk_preds)} chunks)"
            )
            print(f"\n[MERT Pipeline] {Path(audio_path).name if audio_path else 'audio'}")
            print(f"  MERT chunks:  {n_mert_chunks}")
            print(f"  Soft vote:    {predicted_genre} ({soft_confidence:.1%})")
            print(f"  Hard vote:    {hard_genre} ({hard_votes}/{len(chunk_preds)} chunks) {agreement}")
            print(f"  Top-3:")
            for genre, prob in list(all_probs_sorted.items())[:3]:
                print(f"      {genre:20s}: {prob:.1%}")

        if return_json:
            return {
                "genre": predicted_genre,
                "confidence": round(soft_confidence, 4),
                "all_probs": {k: round(v, 4) for k, v in all_probs_sorted.items()},
                "hard_vote": {
                    "genre": hard_genre,
                    "votes": hard_votes,
                    "chunks": len(chunk_preds),
                    "agree": soft_idx == hard_idx,
                },
                "chunks_info": {
                    "mert_chunks": n_mert_chunks,
                }
            }

        return predicted_genre, soft_confidence, all_probs_sorted
    

    def predict_batch(
        self, 
        audio_path: Path, 
        verbose: bool = False,
        return_json: bool = False
    ) -> tuple | dict:
        """Batch prediction (faster)."""
        audio_path = Path(audio_path)

        mert_waveform = self._load_and_preprocess_mert(audio_path)
        mert_chunks = self._chunk_waveform_mert(mert_waveform)
        mert_inputs = self._chunks_to_mert_input(mert_chunks)
        mert_probs = self._get_mert_chunk_probs_batch(mert_inputs)

        return self._vote(
            mert_probs=mert_probs,
            n_mert_chunks=len(mert_chunks),
            verbose=verbose,
            audio_path=audio_path,
            return_json=return_json
        )
    

    def predict_single(
        self, 
        audio_path: Path, 
        verbose: bool = False,
        return_json: bool = False
    ) -> tuple | dict:
        """Single prediction (lower memory)."""
        audio_path = Path(audio_path)

        mert_waveform = self._load_and_preprocess_mert(audio_path)
        mert_chunks = self._chunk_waveform_mert(mert_waveform)
        mert_inputs = self._chunks_to_mert_input(mert_chunks)
        mert_probs = self._get_mert_chunk_probs_single(mert_inputs)

        return self._vote(
            mert_probs=mert_probs,
            n_mert_chunks=len(mert_chunks),
            verbose=verbose,
            audio_path=audio_path,
            return_json=return_json
        )


    def predict_dataset(
        self, 
        test_df: pd.DataFrame, 
        data_dir: Path, 
        mode: str = "batch"
    ) -> pd.DataFrame:
        """
        Run inference on entire dataset.

        Args:
            test_df: DataFrame with columns ["id", "filename"]
            data_dir: Root directory containing the audio files
            mode: "batch" (faster) or "single" (lower memory)

        Returns:
            DataFrame with columns ["id", "genre"]
        """
        assert mode in ("batch", "single"), "mode must be 'batch' or 'single'"
        
        predict_fn = self.predict_batch if mode == "batch" else self.predict_single

        print(f"\n{'='*70}")
        print(f"MERT PREDICTION")
        print(f"{'='*70}")
        print(f"Dataset size: {len(test_df)}")
        print(f"Mode:         {mode}")
        print(f"{'='*70}\n")

        results = {}
        errors = []

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying"):
            song_id = row["id"]
            audio_path = data_dir / row["filename"]

            try:
                predicted_genre, _, _ = predict_fn(
                    audio_path=audio_path,
                    verbose=True,
                    return_json=False
                )
                results[song_id] = predicted_genre
            except Exception as e:
                print(f"\n⚠️ Error on {row['filename']} (id={song_id}): {e}")
                errors.append(song_id)
                results[song_id] = "blues"  # Default fallback

        results_df = pd.DataFrame(
            list(results.items()), columns=["id", "genre"]
        ).set_index("id")

        print(f"\n{'='*70}")
        print(f"PREDICTION COMPLETE")
        print(f"{'='*70}")
        print(f"Successful: {len(results) - len(errors)}/{len(test_df)}")
        print(f"Errors:     {len(errors)}")
        
        # Genre distribution
        print(f"\nGenre Distribution:")
        genre_counts = Counter(results.values())
        for genre, count in genre_counts.most_common():
            print(f"  {genre:12} {count:4d} ({count/len(results)*100:.1f}%)")
        print(f"{'='*70}\n")

        return results_df

