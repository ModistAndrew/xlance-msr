import argparse
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf
from inference import load_config_and_state_dict, load_generator, process_audio

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
    
def main():
    parser = argparse.ArgumentParser(description="Run inference on audio files using trained generator")
    parser.add_argument("--input_dir", '-i', type=str, required=True, help="Directory containing input .flac files")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda/cpu)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)

    audio_files = sorted(input_dir.glob("*.flac")) + sorted(input_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files")
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        audio, sr = load_audio(audio_file)
        rms = calculate_rms(audio)
        print("RMS of MSS audio:", audio_file, rms)


if __name__ == '__main__':
    main()