import argparse
from collections import OrderedDict
import copy
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
from tqdm import tqdm
import librosa
from models import MelRNN, MelRoFormer, UNet, UFormer
from models.bs_roformer import bs_roformer as BSRoformer
from models.bs_roformer import mel_band_roformer as MelBandRoformer

RAWSTEMS_TO_MSRBENCH = {
    'Voc': 'vox',
    'Gtr': 'gtr',
    'Kbs': 'key',
    'Synth': 'syn',
    'Bass': 'bass',
    'Rhy_DK': 'drums',
    'Rhy_PERC': 'perc',
    'Orch': 'orch',
}

def init_generator(model_cfg):
    if model_cfg['name'] == 'MelRNN':
        return MelRNN.MelRNN(**model_cfg['params'])
    elif model_cfg['name'] == 'MelRoFormer':
        return MelRoFormer.MelRoFormer(**model_cfg['params'])
    elif model_cfg['name'] == 'MelUNet':
        return UNet.MelUNet(**model_cfg['params'])
    elif model_cfg['name'] == 'UFormer':
        return UFormer.UFormer(UFormer.UFormerConfig(**model_cfg['params']))
    elif model_cfg['name'] == 'BSRoFormer':
        return BSRoformer.BSRoformer(**model_cfg['params'])
    elif model_cfg['name'] == 'MelBandRoformer':
        return MelBandRoformer.MelBandRoformer(**model_cfg['params'])
    else:
        raise ValueError(f"Unknown model name: {model_cfg['name']}")
    
class RoformerSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
    
    def forward(self, mixture, target=None):
        for module in self[:-1]:
            mixture = module(mixture) # only pass mixture
        return self[-1](mixture, target) # also pass target if present

def load_config_and_state_dict(path: str, map_location: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if path.endswith('.pth'):
        raise ValueError("Use .ckpt files instead of .pth files")
    print(f"Extracting state dict from {path}")
    full_checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    full_state_dict = full_checkpoint['state_dict']
    generator_state_dict = OrderedDict()
    prefix = 'generator.'
    prefix_len = len(prefix)
    for key, value in full_state_dict.items():
        if key.startswith(prefix):
            new_key = key[prefix_len:]
            generator_state_dict[new_key] = value
    return full_checkpoint['hyper_parameters'], generator_state_dict
            

def load_generator(config: Dict[str, Any], state_dict: Dict[str, Any], device: str = 'cuda') -> nn.Module:
    """Initialize and load the generator model from unwrapped checkpoint."""
    generator = init_generator(config['model'])
    
    if 'model1' in config:
        generator1 = copy.deepcopy(generator)
        generator = RoformerSequential(generator, generator1)
    
    # Load unwrapped generator weights
    generator.load_state_dict(state_dict)
    
    generator = generator.to(device)
    generator.eval()
    
    return generator


def process_audio(config, audio: np.ndarray, generator: nn.Module, device: str = 'cuda') -> np.ndarray:
    use_channel = config['model']['name'] in ['BSRoFormer', 'UFormer', 'MelBandRoformer']
    use_16_mix = config['model']['name'] in ['BSRoFormer', 'MelBandRoformer']
    """Process a single audio array through the generator."""
    # Convert to tensor: (channels, samples) -> (1, channels, samples)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]  # Add channel dimension for mono
    
    audio_tensor = torch.from_numpy(audio).float().to(device)
    if use_channel:
        audio_tensor = audio_tensor.unsqueeze(0) # Add batch dimension
    
    if use_16_mix:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
                with torch.no_grad():
                    output_tensor = generator(audio_tensor)
    else:
        with torch.no_grad():
            output_tensor = generator(audio_tensor)

    
    # Convert back to numpy: (1, channels, samples) -> (channels, samples)
    output_audio = output_tensor.cpu().numpy()
    if use_channel:
        output_audio = output_audio[0] # Remove batch dimension
    
    return output_audio


def main():
    parser = argparse.ArgumentParser(description="Run inference on audio files using trained generator")
    parser.add_argument("--checkpoint", '-c', type=str, required=True, help="Path to unwrapped generator weights (.ckpt)")
    parser.add_argument("--checkpoint_pre", '-p', type=str, help="pre-processing model checkpoint (.ckpt)")
    parser.add_argument("--checkpoint_post", '-P', type=str, help="post-processing model checkpoint (.ckpt)")
    parser.add_argument("--input_dir", '-i', type=str, help="Directory containing input .flac files")
    parser.add_argument("--output_dir", '-o', type=str, help="Directory to save processed audio")
    parser.add_argument("--instrument", type=str, help="Instrument to process (Vox/Gtr/Kbs/Synth/Bass/Rhy_DK/Rhy_PERC/Orch)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--no-eval", action="store_false", dest="eval", help="Skip evaluation after inference")
    parser.add_argument("--target_index", type=str, help="Index of target audio files, e.g. '11|12'")
    args = parser.parse_args()
    
    config, state_dict = load_config_and_state_dict(args.checkpoint, args.device)
    
    project_name = config['project_name']
    exp_name = config['exp_name']
    step = Path(args.checkpoint).stem
    instrument = RAWSTEMS_TO_MSRBENCH[config['data']['val_dataset']['target_stem']].capitalize() if args.instrument is None else args.instrument.capitalize()
    print(f"Project: {project_name}, Exp: {exp_name}, Step: {step}, Instrument: {instrument}")
    
    if not args.input_dir:
        args.input_dir = f"../../data/MSRBench/{instrument}/mixture/"
        print(f"No input directory specified, using default: {args.input_dir}")
    if not args.output_dir:
        args.output_dir = f"output/{project_name}/{exp_name}_{step}/"
        print(f"No output directory specified, using default: {args.output_dir}")
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    
    # Get all audio files
    audio_files = sorted(input_dir.glob("*.flac")) + sorted(input_dir.glob("*.wav"))
    if args.target_index is not None:
        regex = re.compile(rf"^\d+_DT({args.target_index})\.\w+$")
    else:
        regex = re.compile(rf"^.*\.\w+$")
    audio_files = [f for f in audio_files if regex.match(os.path.basename(f))]
    audio_files.sort()
    
    if len(audio_files) == 0:
        print(f"No .flac or .wav files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    generators = []
    
    if args.checkpoint_pre:
        print(f"Loading pre-processing model from {args.checkpoint_pre}...")
        config_pre, state_dict_pre = load_config_and_state_dict(args.checkpoint_pre, args.device)
        generator_pre = load_generator(config_pre, state_dict_pre, device=args.device)
        generators.append((generator_pre, "_pre"))
    
    # Load model
    print(f"Loading generator from {args.checkpoint}...")
    generator = load_generator(config, state_dict, device=args.device)
    generators.append((generator, "_sep" if args.checkpoint_post else ""))
    
    if args.checkpoint_post:
        print(f"Loading post-processing model from {args.checkpoint_post}...")
        config_post, state_dict_post = load_config_and_state_dict(args.checkpoint_post, args.device)
        generator_post = load_generator(config_post, state_dict_post, device=args.device)
        generators.append((generator_post, ""))
    
    # Process each file
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        input_path = audio_file
        for generator, postfix in generators:
            # Load audio
            audio, sr = sf.read(input_path)
            
            model_sr = config['data']['sample_rate']
            
            # Transpose if needed: soundfile loads as (samples, channels)
            if audio.ndim == 2:
                audio = audio.T  # Convert to (channels, samples)
                
            if sr != model_sr:
                audio = librosa.resample(audio, sr, model_sr)
            
            # Process through generator
            output_audio = process_audio(config, audio, generator, device=args.device)
            
            if sr != model_sr:
                output_audio = librosa.resample(output_audio, model_sr, sr)
            
            # Transpose back for saving: (channels, samples) -> (samples, channels)
            if output_audio.ndim == 2:
                output_audio = output_audio.T
            
            # Save with same filename
            output_path = output_dir / ((audio_file.stem + postfix) + audio_file.suffix)
            
            sf.write(output_path, output_audio, sr)
            input_path = output_path
    
    print(f"\nProcessing complete! Output saved to {output_dir}")
    
    if args.eval:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        program2_path = os.path.join(current_dir, "eval_plus.py")
        
        cmd = [sys.executable, program2_path]
        arg_eval = {
            '--target_dir': f"../../data/MSRBench/{instrument}/target/",
            '--output_dir': args.output_dir,
            '--target_index': args.target_index,
        }
        for key, value in arg_eval.items():
            if value is None:
                continue
            cmd.extend([key, str(value)])
        
        subprocess.run(cmd)


if __name__ == '__main__':
    main()