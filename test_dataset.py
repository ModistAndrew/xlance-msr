from pathlib import Path
import random
import logging
import numpy as np
import librosa
import soundfile as sf
import json
from typing import List, Optional, Dict, Union, Tuple, Any
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from data.augment import StemAugmentation, MixtureAugmentation
import argparse
import yaml
from data.dataset import InfiniteSampler, RawStems

if __name__ == "__main__":  
    
    parser = argparse.ArgumentParser(description="Train a Music Source Restoration Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--output_dir", type=str, default="test/test_dataset", help="Output dir")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = config['data']
        
    common_params = {
            "sr": config['sample_rate'],
            "clip_duration": config['clip_duration'],
        }
    dataset = RawStems(**config['train_dataset'], **common_params)
    val_dataset = RawStems(**config['val_dataset'], **common_params)
    
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {output_dir}")
        
    # Create a sampler
    sampler = InfiniteSampler(dataset)
    iterator = iter(sampler)
    
    # Sample for 5 iterations
    for i in tqdm(range(args.num_samples), desc="Sampling"):
        index = next(iterator)
        print(index)
        sample = dataset[index]
        print(sample["mixture"].shape)
        print(sample["target"].shape)
        sample["addition"] = sample["mixture"] - sample["target"]
        
        # Save the mixture and target
        mixture_path = Path(output_dir) / f"mixture_{i}.wav"
        target_path = Path(output_dir) / f"target_{i}.wav"
        addition_path = Path(output_dir) / f"addition_{i}.wav"
        
        sf.write(mixture_path, sample["mixture"].T, dataset.sr)
        sf.write(target_path, sample["target"].T, dataset.sr)
        sf.write(addition_path, sample["addition"].T, dataset.sr)