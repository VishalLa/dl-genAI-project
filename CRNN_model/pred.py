import torch
import torchaudio
from scipy.signal import butter, filtfilt


def highpass_filter(sample_rate, audio, cutoff=40):
    b, a = butter(5, cutoff / (sample_rate / 2), btype='high')
        
    audio_np = audio.numpy()
    filtered = filtfilt(b, a, audio_np)
        
    return torch.tensor(filtered.copy(), dtype=audio.dtype)


def predict(model, audio_path, device, genre_map):
    model.eval()
    
    # Load and Preprocess Waveform
    waveform, sr = torchaudio.load(audio_path)
    waveform = torch.mean(waveform, dim=0, keepdim=True) 
    
    if sr != 22050:
        resampler = torchaudio.transforms.Resample(sr, 22050)
        waveform = resampler(waveform)

    waveform = highpass_filter(sample_rate=22050, audio=waveform)
    
    # Chunk the Waveform
    chunk_size = 22050 * 5  
    total_samples = waveform.shape[1]
    num_chunks = total_samples // chunk_size
    
    if num_chunks == 0:
        padding = chunk_size - total_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        chunk_list = [waveform]
    else:
        chunk_list = [waveform[:, i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

    # Convert Chunks to Spectrograms
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050, n_mels=128, hop_length=512
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()

    processed_chunks = []
    for ch in chunk_list:
        mel = mel_transform(ch)
        mel_db = db_transform(mel)
        processed_chunks.append(mel_db)

    # Stack into [batch, 1, height, width]
    # Current processed_chunks: List of [1, 128, T]
    chunks = torch.stack(processed_chunks, dim=0)
    chunks = chunks.to(device)
    
    with torch.inference_mode():
        output = model(chunks)
        avg_logits = output.mean(dim=0)
        probabilities = torch.softmax(avg_logits, dim=0)
        predicted_class = torch.argmax(probabilities).item()
    
    index_to_genre = {v: k for k, v in genre_map.items()}
    return index_to_genre[predicted_class]
