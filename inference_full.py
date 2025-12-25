# Demo for inference on full audio. Only support the final model.
# Process like ans.py but accept parameters like inference.py.

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import librosa
import torch.nn.functional as F

from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from inference import load_config_and_state_dict, load_generator, process_audio


def demix_single(
    model: nn.Module,  
    mix: torch.Tensor, # (b, c, l)
) -> torch.Tensor:
    device = next(model.parameters()).device
    mix = mix.float().to(device)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
                with torch.no_grad():
                    output = model(mix)
    output = output.cpu()
    return F.pad(output, (0, mix.shape[-1]-output.shape[-1]), mode='constant', value=0)

def split_audio(
    audio: np.ndarray,
    chunk_size: int,
    overlap_size: int,
) -> list[np.ndarray]:
    """Split audio into overlapping chunks."""
    hop_size = chunk_size - overlap_size
    chunks = []
    start = 0
    while start + chunk_size <= audio.shape[-1]:
        chunks.append(audio[..., start:start + chunk_size])
        start += hop_size
    return chunks

def merge_chunks(
    chunks: list[np.ndarray],
    chunk_size: int,
    overlap_size: int,
) -> np.ndarray:
    """Merge chunks with overlap-add."""
    hop_size = chunk_size - overlap_size
    output = np.zeros((chunks[0].shape[0], hop_size * len(chunks) + overlap_size))
    for i, chunk in enumerate(chunks):
        start = i * hop_size
        end = start + chunk_size
        window = np.ones_like(chunk)
        if overlap_size > 0:
            fade_in = np.linspace(0, 1, overlap_size)
            fade_out = np.linspace(1, 0, overlap_size)
            window[..., :overlap_size] *= fade_in
            window[..., -overlap_size:] *= fade_out
        output[..., start:end] += chunk * window
    return output

def process_long_audio(
    model: nn.Module,
    mix: np.ndarray,  # (c, l)
    sr: int,
    chunk_duration: float,
    overlap: float,
    batch_size: int,
) -> np.ndarray:
    chunk_size = int(chunk_duration * sr)
    overlap_size = int(overlap * sr)
    hop_size = chunk_size - overlap_size
    
    l = mix.shape[-1]
    l_new = ((l - overlap_size + hop_size - 1) // hop_size) * hop_size + overlap_size
    padding_shape = (mix.shape[0], l_new - l)
    padding = np.zeros(padding_shape, dtype=mix.dtype)
    mix = np.concatenate([mix, padding], axis=-1)
    print(f"Processing long audio of length {mix.shape[-1]} samples")
    
    chunks = split_audio(mix, chunk_size, overlap_size)
    batched_chunks = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    processed_chunks = []
    for batch in batched_chunks:
        print(f"Processing chunks {len(processed_chunks) + 1}/{len(chunks)}")
        tensor = default_collate(batch) # Add batch dim
        processed = demix_single(model, tensor)
        processed_chunks.extend([processed[i].numpy() for i in range(processed.shape[0])])  # Remove batch dim
    merged = merge_chunks(processed_chunks, chunk_size, overlap_size)
    
    return merged[..., :l]

def inference(models, audio, sr, batch_size):
    # audio: (channels, samples)
    channels, samples = audio.shape
    for (config, model) in models:
        model_sr = config['data']['sample_rate']
        if sr != model_sr:
            audio = librosa.resample(audio, sr, model_sr)
        audio = process_long_audio(model, audio, sr, chunk_duration=10.0, overlap=1.0, batch_size=batch_size)
        if sr != model_sr:
            audio = librosa.resample(audio, model_sr, sr)
    if samples < audio.shape[1]:
        audio = audio[:, :samples]
    if samples > audio.shape[1]:
        audio = np.pad(audio, ((0, 0), (0, samples - audio.shape[1])), 'constant')
    return audio

def load_models(paths, device):
    models = []
    for path in paths:
        config, state_dict = load_config_and_state_dict(path, device)
        model = load_generator(config, state_dict, device=device)
        models.append((config, model))
    return models

def load_audio(input_path):
    audio, sr = sf.read(input_path, always_2d=True)
    audio = audio.T
    return audio, sr
    
def save_audio(audio, sr, output_path):
    audio = audio.T
    sf.write(output_path, audio, sr)

def calculate_rms(audio):
    rms = np.sqrt(np.mean(audio**2))
    rms_db = 20 * np.log10(rms + 1e-10)
    return rms_db

def inference_main(args):
    pre_models = load_models(args.checkpoint_pre, args.device)
    mss_models = load_models(args.checkpoint, args.device)
    post_models = load_models(args.checkpoint_post, args.device)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if input_dir.is_dir():
        # Get all audio files
        audio_files = sorted(input_dir.glob("*.flac")) + sorted(input_dir.glob("*.wav")) + sorted(input_dir.glob("*.mp3"))
        print(f"Found {len(audio_files)} audio files")
    else:
        audio_files = [input_dir]
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        audio, sr = load_audio(audio_file)
        print("Processing audio file:", audio_file)
        
        audio = inference(pre_models, audio, sr, batch_size=args.batch_size)
        
        audio = inference(mss_models, audio, sr, batch_size=args.batch_size)
        rms = calculate_rms(audio)
        print("RMS of MSS audio:", rms)
        
        audio_dereverb = inference(post_models, audio, sr, batch_size=args.batch_size)
        rms_dereverb = calculate_rms(audio_dereverb)
        print("RMS of dereverbed audio:", rms_dereverb)
        if rms - rms_dereverb > 10.0:
            print("Dereverb audio is too quiet, use original")
        else:
            audio = audio_dereverb
            
        output_path = output_dir / audio_file.name if input_dir.is_dir() else output_dir # corresponding to input_dir.is_dir()
        save_audio(audio, sr, output_path)
        print("Final result saved to:", output_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on audio files using trained generator")
    parser.add_argument("--checkpoint", '-c', nargs='*', default=[], type=str, help="model checkpoint (.ckpt or .pth)")
    parser.add_argument("--checkpoint_pre", '-p', nargs='*', default=[], type=str, help="pre-processing model checkpoint (.ckpt or .pth)")
    parser.add_argument("--checkpoint_post", '-P', nargs='*', default=[], type=str, help="post-processing model checkpoint (.ckpt or .pth)")
    parser.add_argument("--input_dir", '-i', type=str, help="Directory containing input files, or a single audio file")
    parser.add_argument("--output_dir", '-o', type=str, help="Directory to save processed audio, or a single audio file name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()
    inference_main(args)



