import os
import soundfile as sf
import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
import argparse
import numpy as np
import warnings
from scipy.linalg import sqrtm
from tqdm import tqdm
import torchaudio.transforms as T
import statistics # <-- 新增导入，用于计算平均值和标准差

warnings.filterwarnings("ignore")

try:
    from transformers import ClapModel, ClapProcessor
except ImportError:
    print("Error: The 'transformers' library is not installed.")
    print("Please install it to run FAD-CLAP calculations:")
    print("pip install torch transformers")
    exit(1)


# --- (load_audio, get_clap_embeddings, calculate_frechet_distance 函数保持不变) ---
# 为保持简洁，此处省略了这三个函数，但它们应该放在完整代码中
# ----------------------------------------------------------------------------------

def load_audio(file_path, target_sr=48000):
    """加载音频文件，并将其重采样到目标采样率 (target_sr)。"""
    try:
        wav, samplerate = sf.read(file_path)
        
        if wav.ndim > 1:
            wav = wav.T
        else:
            wav = wav[np.newaxis, :]
            
        wav_tensor = torch.from_numpy(wav).float()

        if samplerate != target_sr:
            resampler = T.Resample(orig_freq=samplerate, new_freq=target_sr)
            wav_tensor = resampler(wav_tensor)
        
        return wav_tensor
    except Exception as e:
        return None

def get_clap_embeddings(file_paths, model, processor, device, batch_size=16):
    model.to(device)
    all_embeddings = []
    
    for i in tqdm(range(0, len(file_paths), batch_size), desc="  Calculating embeddings", ncols=100, leave=False):
        batch_paths = file_paths[i:i+batch_size]
        audio_batch = []
        for path in batch_paths:
            try:
                wav_tensor = load_audio(path, target_sr=48000)
                if wav_tensor is None:
                    continue
                
                for channel in wav_tensor:
                    audio_batch.append(channel.numpy())
            except Exception:
                continue

        if not audio_batch:
            continue

        try:
            inputs = processor(audios=audio_batch, sampling_rate=48000, return_tensors="pt", padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                audio_features = model.get_audio_features(**inputs)
            
            all_embeddings.append(audio_features.cpu().numpy())
        except Exception:
            continue
            
    if not all_embeddings:
        return np.array([])
        
    return np.concatenate(all_embeddings, axis=0)

def calculate_frechet_distance(embeddings1, embeddings2):
    if embeddings1.shape[0] < 2 or embeddings2.shape[0] < 2:
        return None

    mu1, mu2 = np.mean(embeddings1, axis=0), np.mean(embeddings2, axis=0)
    sigma1, sigma2 = np.cov(embeddings1, rowvar=False), np.cov(embeddings2, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    try:
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    except Exception:
        return None

    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fad_score = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fad_score


def main():
    parser = argparse.ArgumentParser(description="Calculate SI-SNR and FAD-CLAP for audio pairs. All audio is resampled to 48000Hz.")
    parser.add_argument("file_list", type=str, help="Path to a text file with the format: target_path|output_path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for FAD-CLAP embedding calculation.")
    parser.add_argument("--output_file", type=str, default="evaluation_results.txt", help="Filename to save all evaluation results.")
    args = parser.parse_args()

    if not os.path.exists(args.file_list):
        print(f"Error: Input file not found at {args.file_list}")
        return

    # 初始化文件写入
    RESULTS_FILENAME = args.output_file
    results_file = open(RESULTS_FILENAME, 'w', encoding='utf-8')
    results_file.write("--- Audio Evaluation Results ---\n")
    print(f"所有结果将被写入文件: {RESULTS_FILENAME}")

    sisnr_calculator = ScaleInvariantSignalNoiseRatio()
    all_target_paths = []
    all_output_paths = []
    all_sisnr_values = [] # <-- 新增列表，用于存储所有 SI-SNR 值

    
    print("\n--- Calculating SI-SNR (48kHz) for each pair ---")
    results_file.write("\n--- Pairwise SI-SNR (dB) ---\n")
    
    TARGET_SR = 48000 
    
    with open(args.file_list, 'r') as f:
        # 写入列名，使用文件名代替完整路径
        results_file.write(f"{'Target Filename':<30}|{'Output Filename':<30}|{'SI-SNR (dB)'}\n")
        
        for line in tqdm(f.readlines(), desc="  Processing audio pairs", ncols=100):
            line = line.strip()
            if not line or '|' not in line:
                continue

            try:
                target_path, output_path = [p.strip() for p in line.split('|')]

                if not os.path.exists(target_path) or not os.path.exists(output_path):
                    continue

                target_wav = load_audio(target_path, target_sr=TARGET_SR)
                output_wav = load_audio(output_path, target_sr=TARGET_SR)

                if target_wav is None or output_wav is None:
                    continue
                
                if target_wav.shape[0] != output_wav.shape[0]:
                    pass
                    
                min_len = min(target_wav.shape[-1], output_wav.shape[-1])
                target_wav = target_wav[..., :min_len]
                output_wav = output_wav[..., :min_len]

                if target_wav.shape[-1] == 0:
                    continue

                sisnr_val = sisnr_calculator(output_wav, target_wav)
                sisnr_item = sisnr_val.item()
                
                # 获取文件名 (只保留文件名，不含路径)
                target_filename = os.path.basename(target_path)
                output_filename = os.path.basename(output_path)
                
                # 构造输出行，使用格式化字符串保持对齐
                result_line = f"{target_filename:<30}|{output_filename:<30}|{sisnr_item:.4f}"
                
                # 打印到控制台 (只显示文件名和结果)
                # print(result_line) 
                # 写入文件
                results_file.write(result_line + "\n")
                
                all_sisnr_values.append(sisnr_item) # 存储 SI-SNR 值
                all_target_paths.append(target_path)
                all_output_paths.append(output_path)

            except Exception:
                continue

    # --- 总体统计参数计算 ---
    results_file.write("\n\n--- Overall Statistical Metrics ---\n")
    
    if all_sisnr_values:
        avg_sisnr = statistics.mean(all_sisnr_values)
        std_sisnr = statistics.stdev(all_sisnr_values) if len(all_sisnr_values) > 1 else 0.0
        
        # 写入平均值和标准差
        results_file.write(f"SI-SNR (dB) Average: {avg_sisnr:.4f}\n")
        results_file.write(f"SI-SNR (dB) Std Dev: {std_sisnr:.4f}\n")
    else:
        results_file.write("No valid SI-SNR values were calculated.\n")

    # --- FAD-CLAP 计算 ---
    
    print("\n--- Calculating FAD-CLAP (48kHz) ---")
    
    if not all_target_paths:
        results_file.write("\nFAD-CLAP: Skipped (No valid file pairs found).\n")
        results_file.close()
        return

    try:
        results_file.write(f"\nTotal pairs for FAD-CLAP: {len(all_target_paths)}\n")
        print("Loading CLAP model...")
        LOCAL_MODEL_PATH = "./clap-model"  # 您下载的模型路径
        clap_model = ClapModel.from_pretrained(LOCAL_MODEL_PATH)
        clap_processor = ClapProcessor.from_pretrained(LOCAL_MODEL_PATH)
        clap_model.eval()
        print("CLAP model loaded successfully.")
    except Exception as e:
        error_msg = f"Fatal Error: Could not load CLAP model. Error: {e}"
        print(error_msg)
        results_file.write(f"\nFAD-CLAP: {error_msg}\n")
        results_file.close()
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nCalculating embeddings for all target files...")
    target_embeddings = get_clap_embeddings(all_target_paths, clap_model, clap_processor, device, args.batch_size)

    print("Calculating embeddings for all output files...")
    output_embeddings = get_clap_embeddings(all_output_paths, clap_model, clap_processor, device, args.batch_size)

    if target_embeddings.size > 0 and output_embeddings.size > 0:
        print("Calculating Frechet Audio Distance (FAD)...")
        fad_score = calculate_frechet_distance(target_embeddings, output_embeddings)
        if fad_score is not None:
            final_fad_output = f"\nOverall FAD-CLAP Score: {fad_score:.4f}"
            print(final_fad_output)
            results_file.write(final_fad_output + "\n")
        else:
            msg = "\nCould not calculate FAD-CLAP score."
            print(msg)
            results_file.write(f"\nFAD-CLAP: {msg}\n")
    else:
        msg = "\nCould not calculate FAD-CLAP due to issues with embedding generation."
        print(msg)
        results_file.write(f"\nFAD-CLAP: {msg}\n")

    # 关闭文件句柄
    results_file.write("\n--- End of Report ---")
    results_file.close()
    print(f"\n评估完成，所有结果已保存到 {RESULTS_FILENAME}。")

if __name__ == "__main__":
    main()