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

from transformers import ASTFeatureExtractor, Wav2Vec2FeatureExtractor
from scipy.signal import butter, filtfilt


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



def load_ensemble_(model_fn: Callable, model_paths: List[str], device: str) -> List[nn.Module]:
    """
    Load multiple saved fold models into a list, ready for ensemble inference.

    Args:
        model_fn    : Zero-argument callable returning a fresh model architecture.
        model_paths : List of .pth file paths (one per fold).
        device      : 'cuda' or 'cpu'.

    Returns:
        models : list[nn.Module]  (all in eval mode, on device)
    """

    models = []
    for path in model_paths: 
        # Create a deep copy so each fold gets its own instance in memory
        m: nn.Module = copy.deepcopy(model_fn()).to(device) 
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)
        print(f" loaded {path}")

    return models



def load_ensemble(
    ast_model_fn: Callable, 
    mert_model_fn: Callable, 
    json_path: str, 
    device: str
) -> Dict[str, List[nn.Module]]:
 
    """
    Auto-discovers fold model paths from the provided JSON and loads them into an ensemble dictionary.

    Returns:
        models: Dictionary containing lists of initialized models for both AST and Wav2Vec2.
    """

    with open(json_path) as f:
        paths = json.load(f)

    model_paths_dict = paths["model_paths"]

    ast_paths = list(model_paths_dict["ast-model"].values())
    mert_paths = list(model_paths_dict["mert-model"].values())

    print(f"Loading {len(ast_paths)} AST models and {len(mert_paths)} Wav2Vec2 models from {json_path} ...")
    

    ast_models = load_ensemble_(
        model_fn=ast_model_fn, 
        model_paths=ast_paths, 
        device=device
    )

    mert_models = load_ensemble_(
        model_fn=mert_model_fn, 
        model_paths=mert_paths, 
        device=device
    )

    return {
        "ast-models": ast_models,
        "mert-models": mert_models
    }


class PredictionPipeline:
    """
    End-to-end inference pipeline for AST + MERT ensemble genre classification.

    Supports:
        - Multi-model ensemble prediction for both AST and MERT
        - Song-level prediction via chunk aggregation
        - Soft voting: weighted average of AST and MERT probabilities
        - Hard voting: majority vote over per-chunk argmaxes (diagnostic)
        - Temperature scaling for calibration
        - Batch or single-sample processing

    Args:
        models: {"ast-models": [...], "mert-models": [...]}
        genre_map: {genre_name: label_id}
        batch_size: chunks per forward pass
        ratio: weight of MERT in final blend (0 = AST only, 1 = MERT only)
               None = auto-weight by model count
        temperature: softmax temperature for probability calibration

    Usage:
        models = load_ensemble(ast_fn, mert_fn, "paths.json", device)
        pipeline = HybridPredictionPipeline(models, genre_map)
        genre, confidence, probs = pipeline.predict("song.mp3", verbose=True)
    """

    AST_SAMPLE_RATE: int = 16000
    AST_CHUNK_SIZE = 10.24
    AST_CHUNK_SAMPLES = int(AST_SAMPLE_RATE * AST_CHUNK_SIZE)

    MERT_SAMPLE_RATE = 24000
    MERT_CHUNK_SIZE = 30
    MERT_CHUNK_SAMPLES = int(MERT_SAMPLE_RATE * MERT_CHUNK_SIZE)

    def __init__(
        self, 
        models: Dict[str, List[nn.Module]],
        genre_map: dict,
        batch_size: int = 32,
        ratio: float = None,
        temperature: float = 1.0,
    ):
        
        self.models = models
        self.genre_map = genre_map
        self.index_to_genre = {v: k for k, v in self.genre_map.items()}
        self.batch_size = batch_size 
        self.temperature = temperature
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        n_ast = len(self.models["ast-models"])
        n_mert = len(self.models["mert-models"])
        if ratio is None:
            self.ast_weight = n_ast / (n_ast + n_mert)
            self.mert_weight = n_mert / (n_ast + n_mert)
        else:
            self.ast_weight = 1 - ratio
            self.mert_weight = ratio

        self.ast_processor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", 
            trust_remote_code=True
        )

        for key in self.models.keys():
            for m in self.models[key]:
                m.to(self.device)
                m.eval()


    def _load_and_preprocess_ast(self, audio_path: Path) -> torch.Tensor:
        """Load audio for AST: 16kHz, highpass filtered."""

        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.AST_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.AST_SAMPLE_RATE)
            waveform = resampler(waveform)

        waveform = highpass_filter(
            audio=waveform, 
            cutoff=40, 
            sample_rate=self.AST_SAMPLE_RATE
        )

        return waveform
    

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

    
    def _chunk_wavefrom_ast(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Split waveform into fixed-length chunks; pad the last one if needed."""

        total_samples = waveform.shape[1]
        n_chunks = max(1, math.ceil(total_samples / self.AST_CHUNK_SAMPLES))

        chunks = []
        for c in range(n_chunks):
            start = c * self.AST_CHUNK_SAMPLES 
            end = start + self.AST_CHUNK_SAMPLES 

            if end > total_samples:
                chunk = waveform[:, start:]
                padding = self.AST_CHUNK_SAMPLES - chunk.shape[1]

                if padding > 0:
                    chunk = nn.functional.pad(chunk, (0, padding))

            else:
                chunk = waveform[:, start:end]

            chunks.append(chunk)

        return chunks
    

    def _chunk_waveform_mert(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Split waveform into 30s chunks for MERT."""
        total_samples = waveform.shape[1]
        n_chunks = max(1, total_samples // self.MERT_CHUNK_SAMPLES)
        chunks = []

        for c in range(n_chunks):
            start = c * self.MERT_CHUNK_SAMPLES
            end = start + self.MERT_CHUNK_SAMPLES

            if end > total_samples:
                chunk = waveform[:, start:]
                padding = self.MERT_CHUNK_SAMPLES - chunk.shape[1]
                if padding > 0:
                    chunk = torch.nn.functional.pad(chunk, (0, padding))
            else:
                chunk = waveform[:, start:end]

            chunks.append(chunk)

        return chunks
    

    @staticmethod
    def _normalize_chunk(chunk: torch.Tensor) -> torch.Tensor:
        """RMS normalization — same as training pipeline."""
        rms = torch.sqrt(torch.mean(chunk ** 2) + 1e-9)
        chunk = chunk / (rms * 3.0)
        return chunk.clamp(-1.0, 1.0)
    

    def _chunks_to_spectrograms(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        """
        Convert a list of 1-D waveform chunks to a batched spectrogram tensor.
        Returns shape: [N, time_steps, freq_bins]
        Processing the whole song in one feature_extractor call is faster and
        ensures identical padding behaviour across all chunks.
        """

        numpy_chunks = [c.squeeze(0).cpu().numpy() for c in chunks]

        inputs = self.ast_processor(
            numpy_chunks,
            sampling_rate=self.AST_SAMPLE_RATE,
            return_tensors="pt"
        )

        return inputs.input_values.to(self.device)   # [N, 1024, 128]
    

    def _chunks_to_mert_input(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        """Process waveform chunks for MERT."""
        # Normalize chunks
        normalized = []

        for chunk in chunks:
            rms = torch.sqrt(torch.mean(chunk ** 2) + 1e-9)
            chunk = chunk / (rms * 3.0)
            chunk = chunk.clamp(-1.0, 1.0)
            normalized.append(chunk.squeeze(0).cpu().numpy())

        # Process with MERT processor
        inputs = self.mert_processor(
            normalized,
            sampling_rate=self.MERT_SAMPLE_RATE,
            padding=True,
            return_tensors="pt"
        )

        return inputs.input_values

    
    @torch.no_grad()
    def _get_chunk_probs_for_ast_for_batch(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Run all chunks through all models in batches.

        Returns:
            chunk_probs : [N, num_classes]  — averaged over ensemble models.
        """

        n_chunks = len(spectrograms)
        models_chunk_probs = []

        for model in self.models["ast-models"]:
            model_probs = []

            for i in range(0, n_chunks, self.batch_size):
                batch = spectrograms[i: i+self.batch_size].to(self.device)
                logits = model(batch)

                probs = torch.softmax(logits, dim=1).cpu()
                model_probs.append(probs)

            models_chunk_probs.append(torch.cat(model_probs, dim=0))

        # Average across ensemble models → [N, C]
        ensemble_probs = torch.stack(models_chunk_probs).mean(dim=0)
        return ensemble_probs
    

    @torch.no_grad()
    def _get_mert_chunk_probs_for_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Get averaged predictions from MERT ensemble for all chunks.
        
        Args:
            waveforms: [N, 720000]
        
        Returns:
            chunk_probs: [N, num_classes]
        """
        n_chunks = len(waveforms)
        all_model_probs = []

        for model in self.models["mert-models"]:
            model_probs = []

            for i in range(0, n_chunks, self.batch_size):
                batch = waveforms[i:i + self.batch_size].to(self.device)
                logits = model(batch)
                probs = torch.softmax(logits, dim=1).cpu()
                model_probs.append(probs)

            all_model_probs.append(torch.cat(model_probs, dim=0))

        # Average across all MERT models
        ensemble_probs = torch.stack(all_model_probs).mean(dim=0)
        return ensemble_probs
    

    @torch.no_grad()
    def _get_chunk_probs_for_ast_for_single(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Run N individual spectrogram chunks through all AST fold models one at a time.
        spectrograms : [N, 1024, 128]
        
        Returns: 
            [N, C] averaged over ensemble.
        """

        n_chunks = spectrograms.shape[0]
        models_chunks_probs = []

        for model in self.models["ast-models"]:
            model_probs = []

            for i in range(0, n_chunks):
                chunk  = spectrograms[i].unsqueeze(0).to(self.device)
                logits = model(chunk)

                probs = torch.softmax(logits, dim=1).cpu()
                model_probs.append(probs)

            models_chunks_probs.append(torch.cat(model_probs, dim=0))

        ensemble_probs = torch.stack(models_chunks_probs).mean(dim=0)
        return ensemble_probs
    
    
    @torch.no_grad()
    def _get_mert_chunk_probs_single(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Run N individual waveform chunks through all MERT models one at a time.
        Slower but uses less memory than batch processing.
        
        Args:
            waveforms: [N, 720000]
        
        Returns:
            chunk_probs: [N, num_classes] averaged over ensemble
        """
        
        n_chunks = len(waveforms)
        all_model_probs = []

        for model in self.models["mert-models"]:
            model_probs = []

            for i in range(n_chunks):
                chunk = waveforms[i].unsqueeze(0).to(self.device)
                logits = model(chunk)
                probs = torch.softmax(logits, dim=1).cpu()
                model_probs.append(probs)

            all_model_probs.append(torch.cat(model_probs, dim=0))

        # Average across all MERT models → [N, C]
        ensemble_probs = torch.stack(all_model_probs).mean(dim=0)
        return ensemble_probs


    def _apply_temperature(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Soften overconfident probability distributions.
        T > 1 flattens distribution, T < 1 sharpens it.
        Applied before blending so neither model dominates unfairly.
        """
        if self.temperature == 1.0:
            return probs  # no-op, skip computation
        log_probs = torch.log(probs.clamp(min=1e-9)) / self.temperature
        return torch.softmax(log_probs, dim=-1)
    

    def _blend_and_vote(
        self,
        ast_probs: torch.Tensor,
        mert_probs: torch.Tensor,
        n_ast_chunks: int,
        n_mert_chunks: int,
        verbose: bool = False,
        audio_path: Path = None,
        return_json: bool = False,
    ) -> tuple | dict:
        """
        Blend AST and MERT probabilities and perform voting.
        
        Args:
            ast_probs: [N_ast, num_classes] probabilities from AST ensemble
            mert_probs: [N_mert, num_classes] probabilities from MERT ensemble
            n_ast_chunks: Number of AST chunks processed
            n_mert_chunks: Number of MERT chunks processed
            verbose: Print detailed prediction info
            audio_path: Path to audio file (for display)
            return_json: Return result as dict instead of tuple
        
        Returns:
            If return_json=False: (predicted_genre, confidence, all_probs_sorted)
            If return_json=True: dict with detailed prediction info
        """
        # Apply temperature scaling
        ast_probs = self._apply_temperature(ast_probs)
        mert_probs = self._apply_temperature(mert_probs)

        # Average across chunks for each model type
        ast_avg = ast_probs.mean(dim=0)   # [C]
        mert_avg = mert_probs.mean(dim=0)  # [C]

        # Weighted blend
        blended_avg = self.ast_weight * ast_avg + self.mert_weight * mert_avg  # [C]

        # Soft vote (primary prediction)
        soft_idx = blended_avg.argmax().item()
        soft_confidence = blended_avg[soft_idx].item()
        predicted_genre = self.index_to_genre[soft_idx]

        # Hard vote (diagnostic - per-chunk majority)
        # Use minimum number of chunks or interpolate
        min_chunks = min(len(ast_probs), len(mert_probs))
        
        # Take first min_chunks from each
        ast_probs_aligned = ast_probs[:min_chunks]
        mert_probs_aligned = mert_probs[:min_chunks]
        
        # Blend per-chunk probabilities
        blended_chunks = (self.ast_weight * ast_probs_aligned + self.mert_weight * mert_probs_aligned)  # [min_chunks, C]
        
        chunk_preds = blended_chunks.argmax(dim=1).tolist()
        hard_idx = Counter(chunk_preds).most_common(1)[0][0]
        hard_votes = chunk_preds.count(hard_idx)
        hard_genre = self.index_to_genre[hard_idx]

        # Sort all probabilities
        all_probs_sorted = dict(sorted(
            {self.index_to_genre[i]: blended_avg[i].item() for i in range(len(blended_avg))}.items(),
            key=lambda x: x[1], 
            reverse=True,
        ))

        if verbose:
            agreement = (
                "✅ agree" if soft_idx == hard_idx
                else f"⚠️ disagree (hard → {hard_genre} {hard_votes}/{len(chunk_preds)} chunks)"
            )
            print(f"\n[HybridPredictionPipeline]  {Path(audio_path).name if audio_path else 'audio'}")
            print(f"  AST chunks   : {n_ast_chunks}")
            print(f"  MERT chunks  : {n_mert_chunks}")
            print(f"  AST weight   : {self.ast_weight:.0%}  |  MERT weight: {self.mert_weight:.0%}")
            print(f"  Temperature  : {self.temperature}")
            print(f"  Soft vote    : {predicted_genre}  ({soft_confidence:.1%})")
            print(f"  Hard vote    : {hard_genre}  ({hard_votes}/{len(chunk_preds)} chunks)  {agreement}")
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
                "weights": {
                    "ast": round(self.ast_weight, 3),
                    "mert": round(self.mert_weight, 3),
                },
                "chunks_info": {
                    "ast_chunks": n_ast_chunks,
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
        """
        Full song-level prediction using batch processing (faster).

        Blends AST and MERT chunk probabilities with soft voting:
            final = ast_weight * ast_avg + mert_weight * mert_avg

        Args:
            audio_path: Path to audio file
            verbose: Print detailed prediction info
            return_json: Return result as dict instead of tuple

        Returns:
            If return_json=False: (predicted_genre, confidence, all_probs_sorted)
            If return_json=True: dict with detailed prediction info
        """
        audio_path = Path(audio_path)

        # AST pipeline
        ast_waveform = self._load_and_preprocess_ast(audio_path)
        ast_chunks = self._chunk_wavefrom_ast(ast_waveform)
        ast_spectrograms = self._chunks_to_spectrograms(ast_chunks)
        
        # MERT pipeline
        mert_waveform = self._load_and_preprocess_mert(audio_path)
        mert_chunks = self._chunk_waveform_mert(mert_waveform)
        mert_inputs = self._chunks_to_mert_input(mert_chunks)

        # Batch processing
        ast_probs = self._get_chunk_probs_for_ast_for_batch(ast_spectrograms)
        mert_probs = self._get_mert_chunk_probs_for_batch(mert_inputs)

        # Blend and vote
        return self._blend_and_vote(
            ast_probs=ast_probs,
            mert_probs=mert_probs,
            n_ast_chunks=len(ast_chunks),
            n_mert_chunks=len(mert_chunks),
            verbose=verbose,
            audio_path=audio_path,
            return_json=return_json
        )
    

    def single_prediction(self, audio_path: Path, verbose: bool = False, return_json: bool = False) -> tuple | dict:

        """
        Full song-level prediction for single song.

        Blends AST and Wav2Vec2 chunk probabilities with soft voting:
            final = (1 - ratio) * ast_avg + ratio * wav2vec2_avg

        Returns:
            predicted_genre : str
            confidence      : float   (winning class probability)
            all_probs       : dict    {genre: prob} sorted descending
        """

        waveform = self._load_and_preprocess_ast(audio_path=audio_path)
        ast_chunks = self._chunk_wavefrom_ast(waveform=waveform)
        ast_input = self._chunks_to_spectrograms(chunks=ast_chunks)

        mert_waveform = self._load_and_preprocess_mert(audio_path)
        mert_chunks = self._chunk_waveform_mert(mert_waveform)
        mert_inputs = self._chunks_to_mert_input(mert_chunks)

        ast_probs = self._get_chunk_probs_for_ast_for_single(ast_input)
        mert_probs = self._get_mert_chunk_probs_single(mert_inputs)

        return self._blend_and_vote(
            ast_probs=ast_probs,
            mert_probs=mert_probs,
            n_ast_chunks=len(ast_chunks),
            n_mert_chunks=len(mert_chunks),
            verbose=verbose,
            audio_path=audio_path,
            return_json=return_json
        )
        

    def predict_dataset(self, test_df: pd.DataFrame, data_dir: Path, mode:str="batch") -> pd.DataFrame:
        """
        Run inference on every row in test_df.

        Args:
            test_df  : DataFrame with columns ["id", "filename"]
            data_dir : Root directory containing the audio files

        Returns:
            DataFrame with columns ["id", "genre"]
        """

        assert mode in ("batch", "single"), "mode must be 'batch' or 'single'"
        
        predict_fn = self.predict_batch if mode == "batch" else self.single_prediction

        print(f"\nStarting inference on {len(test_df)} files")
        print(f"  AST models:   {len(self.models['ast-models'])}")
        print(f"  MERT models:  {len(self.models['mert-models'])}")
        print(f"  Mode:         {mode}")
        print()


        results = {}

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying"):
            song_id = row["id"]
            audio_path = data_dir / row["filename"]

            try :
                predicted_genre, _, _ = predict_fn(
                    audio_path=audio_path,
                    verbose=False,
                    return_json=False
                )
                results[song_id] = predicted_genre
            except Exception as e:
                print(f"\n  Error on {row['filename']} (id={song_id}): {e}")
                results[song_id] = "blues"

        
        results_df = pd.DataFrame(
            list(results.items()), columns=["id", "genre"]
        ).set_index("id")

        return results_df
