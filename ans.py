import argparse
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf
from inference import load_config_and_state_dict, load_generator, process_audio

MSS_MODEL_PATHS = {
    'vox': ['logs/sw/vox/checkpoints/00000010.ckpt'],
    'gtr': ['logs/sw/gtr/checkpoints/00000010.ckpt'],
    'key': ['logs/sw/key/checkpoints/00000010.ckpt'],
    'syn': ['logs/sw/syn/checkpoints/00000010.ckpt', 'logs/bsmoise/syn_mix_large/checkpoints/00240000.ckpt'],
    'bass': ['logs/sw/bass/checkpoints/00000010.ckpt'],
    'drums': ['logs/sw/drums/checkpoints/00000010.ckpt', 'logs/bsmoise/drums_mix_large_random/checkpoints/00630000.ckpt'],
    'perc': ['logs/sw/perc/checkpoints/00000010.ckpt', 'logs/bsmoise/perc_mix_large/checkpoints/00240000.ckpt'],
    'orch': ['logs/sw/orch/checkpoints/00000010.ckpt', 'logs/moisenew/orch_mix_large/checkpoints/00170000.ckpt'],
}
MSS_MODELS = {key: [] for key in MSS_MODEL_PATHS.keys()}

PRE_MODEL_PATHS = {'vox': ['logs/denoise/vox/checkpoints/00000010.ckpt']}
PRE_MODELS = {key: [] for key in PRE_MODEL_PATHS.keys()}
    
POST_MODEL_PATHS = {key: ['logs/dereverb/vox/checkpoints/00000010.ckpt'] if key in ['vox'] else [] for key in MSS_MODEL_PATHS.keys()}
POST_MODELS = {key: [] for key in POST_MODEL_PATHS.keys()}

OUTPUT_PATHS = {
    'vox': Path('xlancelab2/Vocals'),
    'gtr': Path('xlancelab2/Guitars'),
    'key': Path('xlancelab2/Keyboards'),
    'syn': Path('xlancelab2/Synthesizers'),
    'bass': Path('xlancelab2/Bass'),
    'drums': Path('xlancelab2/Drums'),
    'perc': Path('xlancelab2/Percussions'),
    'orch': Path('xlancelab2/Orchestral Elements'),
}

def load_models(dict, target_dict, device='cuda'):
    for key, value in dict.items():
        for path in value:
            config, state_dict = load_config_and_state_dict(path, device)
            model = load_generator(config, state_dict, device=device)
            target_dict[key].append((config, model))

def do_inference(dict, key, audio, sr, device='cuda'):
    # audio: (channels, samples)
    channels, samples = audio.shape
    for (config, model) in dict[key]:
        model_sr = config['data']['sample_rate']
        if sr != model_sr:
            audio = librosa.resample(audio, sr, model_sr)
        audio = process_audio(config, audio, model, device)
        if sr != model_sr:
            audio = librosa.resample(audio, model_sr, sr)
    if samples < audio.shape[1]:
        audio = audio[:, :samples]
    if samples > audio.shape[1]:
        audio = np.pad(audio, ((0, 0), (0, samples - audio.shape[1])), 'constant')
    return audio

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
    parser.add_argument("--output_dir", '-o', type=str, required=True, help="Directory to save processed audio")
    parser.add_argument("--instrument", type=str, required=True, help="Instrument to process (Vox/Gtr/Kbs/Synth/Bass/Rhy_DK/Rhy_PERC/Orch)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda/cpu)")
    args = parser.parse_args()
    
    load_models(PRE_MODEL_PATHS, PRE_MODELS, args.device)
    load_models(MSS_MODEL_PATHS, MSS_MODELS, args.device)
    load_models(POST_MODEL_PATHS, POST_MODELS, args.device)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    OUTPUT_PATHS[args.instrument].mkdir(parents=True, exist_ok=True)
    
    # Get all audio files
    audio_files = sorted(input_dir.glob("*.flac")) + sorted(input_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files")
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        audio, sr = load_audio(audio_file)
        print("Processing audio file:", audio_file)
        
        output_path= output_dir / ((audio_file.stem + "_denoised") + audio_file.suffix)
        audio = do_inference(PRE_MODELS, 'vox', audio, sr, device=args.device)
        save_audio(audio, sr, output_path)
        print("Denoised audio saved to:", output_path)
        
        output_path = output_dir / ((audio_file.stem + "_mssed") + audio_file.suffix)
        audio = do_inference(MSS_MODELS, args.instrument, audio, sr, device=args.device)
        rms = calculate_rms(audio)
        print("RMS of MSS audio:", rms)
        save_audio(audio, sr, output_path)
        print("MSS audio saved to:", output_path)
        
        output_path = output_dir / ((audio_file.stem + "_dereverbed") + audio_file.suffix)
        audio_dereverb = do_inference(POST_MODELS, args.instrument, audio, sr, device=args.device)
        rms_dereverb = calculate_rms(audio_dereverb)
        print("RMS of dereverbed audio:", rms_dereverb)
        if rms - rms_dereverb > 10.0:
            print("Dereverb audio is too quiet, use original")
        else:
            audio = audio_dereverb
        save_audio(audio, sr, output_path)
        print("Dereverb audio saved to:", output_path)
        save_audio(audio, sr, OUTPUT_PATHS[args.instrument] / audio_file.name)
        print("Final result saved to:", OUTPUT_PATHS[args.instrument] / audio_file.name)


if __name__ == '__main__':
    main()