import torch
from pathlib import Path
import torchaudio
import numpy as np
from transformers import ASTFeatureExtractor
from scipy.signal import butter, filtfilt

feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")


def highpass_filter(audio: torch.Tensor, cutoff: int = 40, sample_rate: int = 16000):
    b, a = butter(5, cutoff / (sample_rate / 2), btype='high')

    audio_np = audio.squeeze(0).cpu().numpy()
    filtered = filtfilt(b, a, audio_np)

    filtered = np.ascontiguousarray(filtered)
    filtered = torch.from_numpy(filtered).unsqueeze(0)
    return filtered.type_as(audio)


def _load_and_preprocess(audio_path, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio, convert to mono, resample, and apply highpass filter."""
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    waveform = highpass_filter(waveform, sample_rate=sample_rate)
    return waveform


def _chunk_waveform(waveform: torch.Tensor, chunk_samples: int) -> list[torch.Tensor]:
    """
    Split waveform into non-overlapping chunks of exactly chunk_samples.
    The last chunk is zero-padded if shorter.
    Always returns at least 1 chunk.
    """
    total_samples = waveform.shape[1]
    n_chunks = max(1, total_samples // chunk_samples)
    chunks = []

    for c in range(n_chunks):
        start = c * chunk_samples
        end = start + chunk_samples

        if end > total_samples:
            chunk = waveform[:, start:]
            padding = chunk_samples - chunk.shape[1]
            if padding > 0:
                chunk = torch.nn.functional.pad(chunk, (0, padding))
        else:
            chunk = waveform[:, start:end]

        chunks.append(chunk)

    return chunks


def _normalize_chunk(chunk: torch.Tensor) -> torch.Tensor:
    """RMS normalization — same as training pipeline."""
    rms = torch.sqrt(torch.mean(chunk ** 2) + 1e-9)
    chunk = chunk / (rms * 3.0)
    return chunk.clamp(-1.0, 1.0)


def _chunks_to_spectrograms(chunks: list[torch.Tensor], sample_rate: int) -> torch.Tensor:
    """
    Convert a list of 1-D waveform chunks to a batched spectrogram tensor.
    Returns shape: [N, 1024, 128]
    Processing the whole song in one feature_extractor call is faster and
    ensures identical padding behaviour across all chunks.
    """
    numpy_chunks = [c.squeeze(0).cpu().numpy() for c in chunks]

    inputs = feature_extractor(
        numpy_chunks,
        sampling_rate=sample_rate,
        padding="max_length",
        return_tensors="pt"
    )
    return inputs.input_values   # [N, 1024, 128]


@torch.no_grad()
def predict_ast(model, audio_path, device, genre_map,
                batch_size: int = 8) -> tuple[str, float, dict]:
    """
    Predict the music genre of an audio file.
      - All chunks are processed in batches (faster, avoids repeated overhead)
      - Combines BOTH soft-vote (avg probabilities) and hard-vote (majority)
        and uses soft-vote as the primary result (more accurate)
      - Returns both voting results so callers can inspect agreement
      - Graceful handling of very short files (< 1 chunk)

    Args:
        model      : trained ASTGenreClassifier
        audio_path : path to the audio file
        device     : "cuda" or "cpu"
        genre_map  : dict mapping genre_name -> class_index
        batch_size : number of chunks to forward through model at once

    Returns:
        predicted_genre : str
        confidence      : float  (soft-vote probability of winning class)
        all_probs       : dict   genre_name -> probability (soft-vote)
    """
    model.eval()

    SAMPLE_RATE   = 16000
    CHUNK_SAMPLES = int(SAMPLE_RATE * 10.24)   # 163_840 samples = 1024 spectrogram frames

    index_to_genre = {v: k for k, v in genre_map.items()}

    waveform = _load_and_preprocess(audio_path, SAMPLE_RATE)

    chunks = _chunk_waveform(waveform, CHUNK_SAMPLES)
    chunks = [_normalize_chunk(c) for c in chunks]

    spectrograms = _chunks_to_spectrograms(chunks, SAMPLE_RATE)  # [N, 1024, 128]

    all_chunk_probs = []

    for i in range(0, len(spectrograms), batch_size):
        batch = spectrograms[i:i + batch_size].to(device)
        logits = model(batch)
        probs  = torch.softmax(logits, dim=1)          # [B, num_classes]
        all_chunk_probs.append(probs.cpu())

    all_chunk_probs = torch.cat(all_chunk_probs, dim=0)  # [N, num_classes]

    # Soft vote — average probabilities, then argmax
    avg_probs       = all_chunk_probs.mean(dim=0)          # [num_classes]
    soft_pred_idx   = avg_probs.argmax().item()
    soft_confidence = avg_probs[soft_pred_idx].item()

    # Hard vote — majority voting across chunk argmaxes
    chunk_preds    = all_chunk_probs.argmax(dim=1).tolist()
    hard_pred_idx  = max(set(chunk_preds), key=chunk_preds.count)
    hard_votes     = chunk_preds.count(hard_pred_idx)

    predicted_genre = index_to_genre[soft_pred_idx]
    hard_genre      = index_to_genre[hard_pred_idx]

    all_probs = {
        index_to_genre[i]: prob.item()
        for i, prob in enumerate(avg_probs)
    }

    # Sort by probability descending for readability
    all_probs_sorted = dict(
        sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    )

    # Log agreement between voting methods (useful for debugging)
    agreement = "✅ agree" if soft_pred_idx == hard_pred_idx else \
                f"⚠️  disagree (hard vote → {hard_genre} with {hard_votes}/{len(chunks)} chunks)"

    print(f"[predict] {Path(audio_path).name}")
    print(f"  Chunks processed : {len(chunks)}")
    print(f"  Soft vote        : {predicted_genre}  ({soft_confidence:.1%})")
    print(f"  Hard vote        : {hard_genre}  ({hard_votes}/{len(chunks)} chunks)  {agreement}")
    print(f"  Top-3            : " + "  |  ".join(
        f"{g}: {p:.1%}" for g, p in list(all_probs_sorted.items())[:3]
    ))

    return predicted_genre, soft_confidence, all_probs_sorted
