import torch
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
from helper.golbal_variables import * 


class Mashup_Chunk(Dataset):
    def __init__(self, root_dir: Path, meta_data: Path):
        self.root_dir = root_dir

        self.SAMPLE_RATE = 32000
        self.DURATION = 5
        self.CHUNK_SAMPLES = self.SAMPLE_RATE * self.DURATION  

        self.meta_data = pd.read_csv(meta_data)
        print(f"Loaded {len(self.meta_data)} songs from metadata")

    
    def _fix_length(self, signal: torch.Tensor) -> torch.Tensor:
        length = signal.shape[1]

        if length > self.CHUNK_SAMPLES:
            signal = signal[:, :self.CHUNK_SAMPLES]
        elif length < self.CHUNK_SAMPLES:
            padding = self.CHUNK_SAMPLES - length
            signal = torch.nn.functional.pad(signal, (0, padding))

        return signal

    
    def __len__(self):
        return len(self.meta_data)
    

    def __getitem__(self, index):
        row = self.meta_data.iloc[index]
        song_id = row['id'] 
        filename = row['filename']

        file_path = self.root_dir / filename
        
        waveform, sr = torchaudio.load(file_path)
        waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            waveform = resampler(waveform)

        total_samples = waveform.shape[1]
        num_chunks = total_samples // self.CHUNK_SAMPLES 

        if num_chunks == 0:
            waveform = self._fix_length(waveform)
            chunks = waveform.unsqueeze(0)  # [1, 1, 160000]
        else: 
            chunks = []
            for i in range(num_chunks):
                start_idx = i * self.CHUNK_SAMPLES
                end_idx = start_idx + self.CHUNK_SAMPLES

                chunk = waveform[:, start_idx:end_idx]
                chunks.append(chunk)
            
            chunks = torch.stack(chunks, dim=0)  

        return chunks, song_id


def collate_fn_mashup(batch):
    all_chunks = []
    song_ids = []
    chunk_counts = []
    
    for chunks, song_id in batch:
        num_chunks = chunks.shape[0]
        all_chunks.append(chunks)
        song_ids.append(song_id)
        chunk_counts.append(num_chunks)
    
    all_chunks = torch.cat(all_chunks, dim=0)  # [total_chunks, 1, 160000]
    
    return all_chunks, song_ids, chunk_counts


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

