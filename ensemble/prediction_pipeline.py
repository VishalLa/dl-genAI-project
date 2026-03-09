import os 
import json
import copy
import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
import torchaudio

from collections import Counter
from typing import Callable, Dict, List
from tqdm import tqdm
from pathlib import Path

from transformers import ASTFeatureExtractor, Wav2Vec2Processor
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
    wav2vec2_model_fn: Callable, 
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
    wav2vec2_paths = list(model_paths_dict["wav2vec2-model"].values())

    print(f"Loading {len(ast_paths)} AST models and {len(wav2vec2_paths)} Wav2Vec2 models from {json_path} ...")
    

    ast_models = load_ensemble_(
        model_fn=ast_model_fn, 
        model_paths=ast_paths, 
        device=device
    )

    wav2vec2_models = load_ensemble_(
        model_fn=wav2vec2_model_fn, 
        model_paths=wav2vec2_paths, 
        device=device
    )

    return {
        "ast-models": ast_models,
        "wav2vec2-models": wav2vec2_models
    }


class PredictionPipeline:
    """
    End-to-end inference pipeline for AST + Wav2Vec2 ensemble genre classification.

    Supports:
        - Single-model or multi-fold ensemble prediction
        - Song-level prediction via chunk aggregation
        - Soft voting: weighted average of AST and Wav2Vec2 probabilities
        - Hard voting: majority vote over per-chunk argmaxes (diagnostic)

    Args:
        models     : {"ast-models": [...], "wav2vec2-models": [...]}
        genre_map  : {genre_name: label_id}
        batch_size : chunks per forward pass
        ratio      : weight of Wav2Vec2 in final blend  (0 = AST only, 1 = Wav2Vec2 only)
                     default 0.25 → 75% AST + 25% Wav2Vec2

    Usage:
        models = load_ensemble(ast_fn, w2v_fn, "paths.json", device)
        pipeline = PredictionPipeline(models, genre_map)
        genre, confidence, probs = pipeline.predict("song.mp3")
    """

    SAMPLE_RATE: int = 16000
    CHUNK_SIZE = 10.24
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SIZE)
    AST_FEATURE_EXTRACTION = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    WAV2VEC2_FEATURE_EXTRACTION = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    def __init__(
            self, 
            models: Dict[str, List[nn.Module]],
            genre_map: dict,
            batch_size: int = 32,
            ratio: float = 0.25
    ):
        
        self.models = models

        self.genre_map = genre_map
        self.index_to_genre = {v: k for k, v in self.genre_map.items()}

        self.ratio = ratio
        self.batch_size = batch_size 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        for key in self.models.keys():
            for m in self.models[key]:
                m.to(self.device)
                m.eval()

    
    def _load_and_preprocess(self, audio_path: str) -> torch.Tensor:
        """Load audio, convert to mono, resample, and apply highpass filter."""
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            waveform = resampler(waveform)

        waveform = highpass_filter(
            audio=waveform, 
            cutoff=40, 
            sample_rate=self.SAMPLE_RATE
        )

        return waveform
    

    def _chunk_wavefrom(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Split waveform into fixed-length chunks; pad the last one if needed."""

        total_samples = waveform.shape[1]
        n_chunks = max(1, total_samples // self.CHUNK_SAMPLES)

        chunks = []
        for c in range(n_chunks):
            start = c * self.CHUNK_SAMPLES 
            end = start + self.CHUNK_SAMPLES 

            if end > total_samples:
                chunk = waveform[:, start:]
                padding = self.CHUNK_SAMPLES - chunk.shape[1]

                if padding > 0:
                    chunk = nn.functional.pad(chunk, (0, padding))

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
    

    def _chunks_to_spectrograms_for_ast(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        """
        Convert a list of 1-D waveform chunks to a batched spectrogram tensor.
        Returns shape: [N, time_steps, freq_bins]
        Processing the whole song in one feature_extractor call is faster and
        ensures identical padding behaviour across all chunks.
        """

        numpy_chunks = [c.squeeze(0).cpu().numpy() for c in chunks]

        inputs = self.AST_FEATURE_EXTRACTION(
            numpy_chunks,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt"
        )

        return inputs.input_values.to(self.device)   # [N, 1024, 128]
    

    def _chunk_processor_for_wav2vec2(self, chunks: List[torch.Tensor]) -> dict:
        """
        Processes a list of 1-D waveform chunks for Wav2Vec2.
        Returns a dictionary containing:
            - input_values: Raw waveform arrays, shape [N, sequence_length]
            - attention_mask: Mask for padded regions, shape [N, sequence_length]
        """    
        numpy_chunks = [c.squeeze(0).cpu().numpy() for c in chunks]
        
        inputs = self.WAV2VEC2_FEATURE_EXTRACTION(
            numpy_chunks, 
            sampling_rate=self.SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_values": inputs.input_values.to(self.device),
            "attention_mask": inputs.attention_mask.to(self.device)
        }

    
    @torch.no_grad()
    def _get_chunk_probs_for_ast(self, spectrograms: torch.Tensor) -> torch.Tensor:
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
    def _get_chunk_probs_for_wav2vec2(self, waveform: Dict[str, torch.Tensor]) -> torch.Tensor:

        """
        Run all waveform chunks through all Wav2Vec2 models in batches.

        Returns:
            ensemble_probs : [N, num_classes] — averaged over ensemble models.
        """

        n_chunks = len(waveform["input_values"])
        models_chunk_probs = []

        for model in self.models["wav2vec2-models"]:
            model_probs = []

            for i in range(0, n_chunks, self.batch_size):
                batch_inputs = waveform["input_values"][i: i + self.batch_size].to(self.device)
                batch_mask = waveform["attention_mask"][i: i + self.batch_size].to(self.device)

                logits = model(input_values=batch_inputs, attention_mask=batch_mask)

                probs = torch.softmax(logits, dim=1).cpu()
                model_probs.append(probs)
            
            models_chunk_probs.append(torch.cat(model_probs, dim=0))
        
        # Average across ensemble models → [N, C]
        ensemble_probs = torch.stack(models_chunk_probs).mean(dim=0)
        return ensemble_probs


    def predict(self, audio_path: Path, verbose: bool = False) -> tuple:

        """
        Full song-level prediction.

        Blends AST and Wav2Vec2 chunk probabilities with soft voting:
            final = (1 - ratio) * ast_avg + ratio * wav2vec2_avg

        Returns:
            predicted_genre : str
            confidence      : float   (winning class probability)
            all_probs       : dict    {genre: prob} sorted descending
        """

        waveform = self._load_and_preprocess(audio_path=audio_path)
        chunks = self._chunk_wavefrom(waveform=waveform)

        chunks = [self._normalize_chunk(chunk=chunk) for chunk in chunks]

        ast_input = self._chunks_to_spectrograms_for_ast(chunks=chunks)
        wav2vec2_input = self._chunk_processor_for_wav2vec2(chunks=chunks)

        ast_probs = self._get_chunk_probs_for_ast(spectrograms=ast_input)
        wav2vec2_probs = self._get_chunk_probs_for_wav2vec2(waveform=wav2vec2_input)

        blended_probs = (1 - self.ratio) * ast_probs + self.ratio * wav2vec2_probs  # [N, C]
        avg_probs = blended_probs.mean(dim=0)      # [C]

        # Soft vote 
        soft_idx = avg_probs.argmax().item()
        soft_confidence = avg_probs[soft_idx].item()
        predicted_genre = self.index_to_genre[soft_idx]

        # Hart vote 
        chunk_preds = blended_probs.argmax(dim=1).tolist()
        hard_idx = Counter(chunk_preds).most_common(1)[0][0]
        hard_votes = chunk_preds.count(hard_idx)
        hard_genre = self.index_to_genre[hard_idx]


        all_probs_sorted = dict(sorted(
            {self.index_to_genre[i]: avg_probs[i].item() for i in range(len(avg_probs))}.items(),
            key=lambda x: x[1], reverse=True
        ))

        if verbose:
            agreement = (
                "✅ agree" if soft_idx == hard_idx
                else f"⚠️ disagree (hard → {hard_genre} {hard_votes}/{len(chunks)} chunks)"
            )
            print(f"\n[PredictionPipeline]  {Path(audio_path).name}")
            print(f"  Chunks     : {len(chunks)}")
            print(f"  AST weight : {(1-self.ratio):.0%}  |  Wav2Vec2 weight: {self.ratio:.0%}")
            print(f"  Soft vote  : {predicted_genre}  ({soft_confidence:.1%})")
            print(f"  Hard vote  : {hard_genre}  ({hard_votes}/{len(chunks)} chunks)  {agreement}")
            print(f"  Top-3      :")
            for genre, prob in list(all_probs_sorted.items())[:3]:
                print(f"      {genre:20s}: {prob:.1%}")

        return predicted_genre, soft_confidence, all_probs_sorted
    

    # def predict_dataset(self, audio_paths: list[Path]) -> list[dict]:
    #     """
    #     Process a large list of audio files and return the results.
    #     Uses tqdm for a progress bar.
    #     """
    #     results = []
    #     print(f"Starting inference on {len(audio_paths)} files using {len(self.models)} models...")
        
    #     for path in tqdm(audio_paths, desc="Classifying Songs"):
    #         try:
    #             genre, conf, all_probs = self.predict(path, verbose=False)
    #             results.append({
    #                 "id": path.name,
    #                 "predicted_genre": genre,
    #                 "confidence": conf,
    #                 "all_probs": all_probs
    #             })
    #         except Exception as e:
    #             print(f"\nError processing {path.name}: {e}")
    #             results.append({
    #                 "file_name": path.name,
    #                 "predicted_genre": "ERROR",
    #                 "confidence": 0.0,
    #                 "all_probs": {}
    #             })
                
    #     return results
    

    def predict_dataset(self, test_df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
        """
        Run inference on every row in test_df.

        Args:
            test_df  : DataFrame with columns ["id", "filename"]
            data_dir : Root directory containing the audio files

        Returns:
            DataFrame with columns ["id", "genre"]
        """
        results = {}
        print(f"Inference on {len(test_df)} files | "
              f"{len(self.models['ast-models'])} AST + "
              f"{len(self.models['wav2vec2-models'])} Wav2Vec2 models")

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying"):
            song_id = row["id"]
            audio_path = data_dir / row["filename"]
            try:
                genre, _, _ = self.predict(audio_path=audio_path, verbose=False)
                results[song_id] = genre
            except Exception as e:
                print(f"\n  Error on {row['filename']} (id={song_id}): {e}")
                results[song_id] = "blues"   # fallback

        results_df = pd.DataFrame(
            list(results.items()), columns=["id", "genre"]
        ).set_index("id")
        return results_df
