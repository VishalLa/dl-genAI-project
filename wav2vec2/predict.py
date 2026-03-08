import torch
from pathlib import Path
import torchaudio
from wav2vec2.wave2vec2_dataset import highpass_filter
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def predict(model, audio_path, device, genre_map):
    model.eval()


    SAMPLE_RATE = 16000  
    CHUNK_DURATION = 8   
    CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION  

    waveform, sr = torchaudio.load(audio_path)
    
    waveform = waveform.mean(dim=0, keepdim=True)
    
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
        # print(f"  Resampled: {sr}Hz -> {SAMPLE_RATE}Hz")
    
    waveform = highpass_filter(waveform, sample_rate=SAMPLE_RATE)
    
    total_samples = waveform.shape[1]
    duration_sec = total_samples / SAMPLE_RATE
    # print(f"  Duration: {duration_sec:.1f}s ({total_samples:,} samples)")
    

    n_chunks = max(1, total_samples // CHUNK_SAMPLES)
    # print(f"  Chunks: {n_chunks}")
    
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
            
            # Convert to numpy for processor
            chunk = chunk.squeeze(0).cpu().numpy()
            
            # Process with Wav2Vec2Processor
            inputs = processor(
                [chunk],
                sampling_rate=SAMPLE_RATE,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            # Get prediction for this chunk
            logits = model(input_values, attention_mask)
            probs = torch.softmax(logits, dim=1)
            chunk_predictions.append(probs.cpu())


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
