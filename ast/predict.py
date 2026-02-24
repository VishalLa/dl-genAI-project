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


def predict_ast(model, audio_path, device, genre_map):
    model.eval()

    SAMPLE_RATE = 16000  
    CHUNK_DURATION = 10.24   
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)  

    waveform, sr = torchaudio.load(audio_path)
    
    waveform = waveform.mean(dim=0, keepdim=True)
    
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    
    waveform = highpass_filter(waveform, sample_rate=SAMPLE_RATE)
    
    total_samples = waveform.shape[1]
    
    n_chunks = max(1, total_samples // CHUNK_SAMPLES)
    
    chunk_predictions = []
    
    with torch.no_grad():
        for c in range(n_chunks):
            start = c * CHUNK_SAMPLES
            end = start + CHUNK_SAMPLES
            
            # Extract chunk
            if end > total_samples:
                # Last chunk - pad if needed
                chunk = waveform[:, start:]
                padding = CHUNK_SAMPLES - chunk.shape[1]
                if padding > 0:
                    chunk = torch.nn.functional.pad(chunk, (0, padding))
            else:
                chunk = waveform[:, start:end]
            
            # Normalize (RMS normalization)
            rms = torch.sqrt(torch.mean(chunk**2) + 1e-9)
            chunk = chunk / (rms * 3.0)
            chunk = chunk.clamp(-1.0, 1.0)
            
            # Convert to numpy for feature extractor
            chunk = chunk.squeeze(0).cpu().numpy()
            
            # 3. Process with ASTFeatureExtractor (calculates spectrogram)
            inputs = feature_extractor(
                chunk,
                sampling_rate=SAMPLE_RATE,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_values = inputs.input_values.to(device)
            
            # 4. Forward pass (No attention mask needed for AST)
            logits = model(input_values)
            probs = torch.softmax(logits, dim=1)
            chunk_predictions.append(probs.cpu())

    # Average the probabilities across all chunks for the final prediction
    avg_probs = torch.stack(chunk_predictions).mean(dim=0).squeeze(0) 
    
    predicted_class = avg_probs.argmax().item()
    confidence = avg_probs[predicted_class].item()
    
    # Convert to genre name
    index_to_genre = {v: k for k, v in genre_map.items()}
    predicted_genre = index_to_genre[predicted_class]
    
    # Get all probabilities
    all_probs = {
        index_to_genre[i]: prob.item() 
        for i, prob in enumerate(avg_probs)
    }
    
    return predicted_genre, confidence, all_probs

