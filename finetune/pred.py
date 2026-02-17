import torch
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
from helper.golbal_variables import * 


def predict(model, audio_path, device, genre_map):
    """
    Predict genre for a single audio file
    
    Args:
        audio_path: Path to .wav file (e.g., "mashups/song0001.wav")
    """
    model.eval()
    
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    waveform = torch.mean(waveform, dim=0, keepdim=True) 
    
    if sr != 32000:
        resampler = torchaudio.transforms.Resample(sr, 32000)
        waveform = resampler(waveform)
    
    chunk_size = 32000 * 5  
    total_samples = waveform.shape[1]
    num_chunks = total_samples // chunk_size
    
    if num_chunks == 0:
        # Pad if shorter than 5 seconds
        if total_samples < chunk_size:
            padding = chunk_size - total_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        chunks = waveform.unsqueeze(0)  # [1, 1, 160000]
    else:
        # Extract chunks
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = waveform[:, start:end]
            chunks.append(chunk)
        chunks = torch.stack(chunks, dim=0)  # [num_chunks, 1, 160000]
    
    # Predict
    chunks = chunks.to(device)
    
    with torch.inference_mode():
        output_dict = model(chunks)
        logits = output_dict["clipwise_output"]
        
        # Average across chunks
        avg_logits = logits.mean(dim=0)
        probabilities = torch.softmax(avg_logits, dim=0)
        
        predicted_class = torch.argmax(probabilities).item()
    
    # Get genre name
    index_to_genre = {v: k for k, v in genre_map.items()}
    genre_name = index_to_genre[predicted_class]
    
    return genre_name

