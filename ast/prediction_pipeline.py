import os 
import json
import copy
from collections import Counter
import pandas as pd
from tqdm import tqdm
import torch 
import torch.nn as nn
import torchaudio
from pathlib import Path
from transformers import ASTFeatureExtractor

from utils import highpass_filter


def load_ensemble(model_fn: function, model_paths: list, device: str) -> list[nn.Module]:
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


def load_ensemble_from_cv_dir(model_fn: function, results_path: str, device: str) -> tuple[list[nn.Module], dict]:
    """
    Auto-discovers fold model paths from modelspaths.json and loads ensemble.

    Returns:
        models (list), cv_summary (dict)
    """
    # results_path = os.path.join(save_dir, "modelspaths.json")
    with open(results_path) as f:
        cv_summary = json.load(f)

    model_paths = list(cv_summary["models_paths"].values())
    
    print(f"Loading {len(model_paths)}-model ensemble from {results_path} ...")

    models = load_ensemble(model_fn, model_paths, device)
    return models, cv_summary


def ensemble_predict_chunk(
    models: list[nn.Module],
    spec: torch.Tensor, 
    device: str, 
    mode: str = "avg_prob"
):
    """
    Ensemble prediction for a single spectrogram chunk.

    Args:
        models : list of loaded nn.Module models.
        spec   : Tensor of shape [1, time, freq] or [time, freq].
        device : 'cuda' or 'cpu'.
        mode   : 'avg_prob'  — average softmax probabilities  (recommended)
                 'majority'  — majority vote over argmax predictions

    Returns:
        predicted_class (int)
    """

    if spec.dim() == 2: 
        spec = spec.unsqueeze(0)
    spec = spec.to(device)

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for model in models:
            logits = model(spec)
            probs = torch.softmax(logits, dim=-1)

            all_probs.append(probs)
            all_preds.append(torch.argmax(probs, dim=-1).item())
    
    if mode == "avg_prob":
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return torch.argmax(avg_probs, dim=-1).item()
    elif mode == "majority":
        return Counter(all_preds).most_common(1)[0][0]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'avg_prob' or 'majority'.")


class PredictionPipeline:

    """
    End-to-end inference pipeline for AST-based music genre classification.

    Supports:
      • Single-model prediction
      • Multi-model ensemble prediction (avg_prob or majority)
      • Song-level prediction via chunk aggregation + soft/hard voting

    Usage:
        # Single model
        pipeline = PredictionPipeline("song.mp3", models=[model])
        genre, confidence, probs = pipeline.predict()

        # Ensemble (e.g. from cross-validation)
        models, _ = load_ensemble_from_cv_dir(model_fn, "cv_models", device)
        pipeline = PredictionPipeline("song.mp3", models=models)
        genre, confidence, probs = pipeline.predict()
    """

    SAMPLE_RATE: int = 16000
    CHUNK_SIZE = 10.24
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SIZE)
    FEATURE_EXTRACTION = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    def __init__(
            self,
            models: list[nn.Module], 
            genre_map: dict,
            chunk_mode: str = "avg_prob",
            batch_size: int = 8
        ):
        self.models = models

        self.genre_map = genre_map
        self.index_to_genre = {v: k for k, v in genre_map.items()}

        self.chunk_mode = chunk_mode
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        for m in self.models:
            m.to(self.device)
            m.eval()


    def _load_and_preprocess(self, audio_path: Path):
        """Load audio, convert to mono, resample, and apply highpass filter."""
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            waveform = resampler(waveform)

        waveform = highpass_filter(audio=waveform, cutoff=40, sample_rate=self.SAMPLE_RATE)
        return waveform 
    

    def _chunk_waveform(self, waveform: torch.Tensor) -> list[torch.Tensor]:
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
    

    def _chunks_to_spectrograms(self, chunks: list[torch.Tensor]) -> torch.Tensor:
        """
        Convert a list of 1-D waveform chunks to a batched spectrogram tensor.
        Returns shape: [N, 1024, 128]
        Processing the whole song in one feature_extractor call is faster and
        ensures identical padding behaviour across all chunks.
        """

        numpy_chunks = [c.squeeze(0).cpu().numpy() for c in chunks]

        inputs = self.FEATURE_EXTRACTION(
            numpy_chunks,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt"
        )
        return inputs.input_values   # [N, 1024, 128]
    
    @torch.no_grad()
    def _get_chunk_probs(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Run all chunks through all models in batches.

        Returns:
            chunk_probs : [N, num_classes]  — averaged over ensemble models.
        """

        n_chunks = len(spectrograms)
        model_chunk_probs = []

        for model in self.models:
            model_probs = []

            for i in range(0, n_chunks, self.batch_size):
                batch = spectrograms[i : i + self.batch_size].to(self.device)
                logits = model(batch)

                probs = torch.softmax(logits, dim=1).cpu()
                model_probs.append(probs)

            model_chunk_probs.append(torch.cat(model_probs, dim=0))
        
        # Average across ensemble models → [N, C]
        ensemble_probs = torch.stack(model_chunk_probs).mean(dim=0)
        return ensemble_probs

    
    def predict(self, audio_path: Path, verbose: bool = False) -> tuple:
        """
        Full song-level prediction.

        Combines:
          1. Soft vote  — average chunk probabilities → argmax  (primary result)
          2. Hard vote  — majority vote over per-chunk argmaxes (diagnostic)

        Returns:
            predicted_genre : str
            confidence      : float   (soft-vote probability of top class)
            all_probs       : dict    genre → probability, sorted descending
        """

        waveform = self._load_and_preprocess(audio_path)
        chunks = self._chunk_waveform(waveform)
        chunks = [self._normalize_chunk(chunk) for chunk in chunks]
        spectrograms = self._chunks_to_spectrograms(chunks)

        chunk_probs = self._get_chunk_probs(spectrograms)

        # soft vote 
        avg_probs  = chunk_probs.mean(dim=0)
        soft_pred_idx = avg_probs.argmax().item()
        soft_confidence = avg_probs[soft_pred_idx].item()
        predicted_genre = self.index_to_genre[soft_pred_idx]

        # hard vote
        chunk_preds = chunk_probs.argmax(dim=1).tolist()
        hard_pred_idx = Counter(chunk_preds).most_common(1)[0][0]
        hard_votes  = chunk_preds.count(hard_pred_idx)
        hard_genre = self.index_to_genre[hard_pred_idx]

        all_probs = {
            self.index_to_genre[i]: avg_probs[i].item()
            for i in range(len(avg_probs))
        }
        all_probs_sorted = dict(
            sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        )

        if verbose:
            agreement = "✅ agree" if soft_pred_idx == hard_pred_idx else f"⚠️ disagree (hard vote → {hard_genre} {hard_votes}/{len(chunks)} chunks)"
            print(f"\n[PredictionPipeline]  {audio_path.name}")
            print(f"  Chunks processed : {len(chunks)}")
            print(f"  Soft vote        : {predicted_genre}  ({soft_confidence:.1%})")
            print(f"  Hard vote        : {hard_genre}  ({hard_votes}/{len(chunks)} chunks)  {agreement}")

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


    def predict_dataset(self, test_df: pd.DataFrame, MESSY_DATA_DIR: Path) -> pd.DataFrame:
        """
        Process a DataFrame of audio files and return the results as a DataFrame.
        """
        results = {}
        print(f"Starting inference on {test_df.shape[0]} files using {len(self.models)} models...")
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying Songs"):
            song_id = row["id"]
            
            try:
                filename = row["filename"]
                audio_path = MESSY_DATA_DIR / filename

                genre, _, _ = self.predict(audio_path=audio_path, verbose=False)
                results[song_id] = genre

            except Exception as e:
                print(f"\nError processing {row['filename']} (ID: {song_id}): {e}")
                results[song_id] = "blues"
                
        results_df = pd.DataFrame(list(results.items()), columns=['id', 'genre'])
        results_df.set_index("id", inplace=True)
        return results_df
    
