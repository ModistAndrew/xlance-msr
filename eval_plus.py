import glob
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
from audiobox_aesthetics.infer import initialize_predictor
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
            print(f"Warning: Resampling audio from {samplerate} to {target_sr}")
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

def find_matching_pairs(target_dir, output_dir):
    """
    找到target和output文件夹中的匹配文件对
    假设target文件名为: 0.flac, 1.flac, ..., 249.flac
    output文件名为: {target_id}_DT{index}.flac
    """
    pairs = []
    
    target_files = glob.glob(os.path.join(target_dir, "*.*"))
    target_files.sort()
    
    print(f"Found {len(target_files)} target files in {target_dir}")
    
    for target_file in target_files:
        target_id = os.path.splitext(os.path.basename(target_file))[0]
        
        output_pattern = os.path.join(output_dir, f"{target_id}_DT*.*")
        matching_outputs = glob.glob(output_pattern)
        matching_outputs.sort()
        
        if matching_outputs:
            print(f"Target {target_id}: found {len(matching_outputs)} output files")
            for output_file in matching_outputs:
                pairs.append((target_file, output_file))
        else:
            print(f"Target {target_id}: no matching output files found")
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Calculate SI-SNR and FAD-CLAP for audio pairs. All audio is resampled to 48000Hz.")
    parser.add_argument("--file_list", type=str, help="Path to a text file with the format: target_path|output_path")
    parser.add_argument("--target_dir", type=str, help="Path to target audio directory")
    parser.add_argument("--output_dir", type=str, help="Path to output audio directory") 
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for FAD-CLAP embedding calculation.")
    parser.add_argument("--output_file", type=str, default="evaluation_results.txt", help="Filename to save all evaluation results.")
    args = parser.parse_args()
    
    if args.file_list and (args.target_dir or args.output_dir):
        print("Error: Please use either --file_list OR --target_dir/--output_dir, not both.")
        return
        
    if not args.file_list and (not args.target_dir or not args.output_dir):
        print("Error: Please provide either --file_list OR both --target_dir and --output_dir.")
        return
    
    # 初始化 AudioBox Aesthetics Predictor
    AXES_NAME = ["CE", "CU", "PC", "PQ"] 
    LOCAL_AESTHETICS_CKPT = "/inspire/hdd/global_user/chenxie-25019/HaoQiu/MSRKit/audiobox/audiobox_aes_checkpoint.pt"
    # try:
    #     print("\nLoading AudioBox Aesthetics predictor...")
    #     aesthetics_predictor = initialize_predictor(ckpt=None)
    #     print("AudioBox Aesthetics predictor loaded successfully.")
    # except Exception as e:
    #     print(f"Error loading AudioBox Aesthetics predictor: {e}. Aesthetics calculation will be skipped.")
    aesthetics_predictor = None
        
    # 初始化文件写入
    RESULTS_FILENAME = args.output_file
    results_file = open(RESULTS_FILENAME, 'w', encoding='utf-8')
    results_file.write("--- Audio Evaluation Results ---\n")
    print(f"所有结果将被写入文件: {RESULTS_FILENAME}")

    sisnr_calculator = ScaleInvariantSignalNoiseRatio()
    all_target_paths = []
    all_output_paths = []
    all_sisnr_values = []
    all_aesthetics_values = {axis: [] for axis in AXES_NAME}
    # ----------------------------------------------------
    # PHASE 1: 遍历文件列表，计算 SI-SNR，收集路径
    # ----------------------------------------------------
    
    print("\n--- Calculating SI-SNR (48kHz) for each pair ---")
    results_file.write("\n--- Pairwise SI-SNR (dB) ---\n")
    
    TARGET_SR = 48000 
    
    def calculate_sisnr(target_path, output_path):
        if not os.path.exists(target_path) or not os.path.exists(output_path):
            raise Exception(f"Skipping, file not found: {target_path} -> {output_path}")
        target_wav = load_audio(target_path, TARGET_SR)
        output_wav = load_audio(output_path, TARGET_SR)
        if target_wav is None or output_wav is None:
            raise Exception(f"Skipping, waveform not loaded: {target_path} -> {output_path}")
        if target_wav.shape[0] != output_wav.shape[0]:
            raise Exception(f"Skipping, shape mismatch: {target_path} -> {output_path}")
        min_len = min(target_wav.shape[-1], output_wav.shape[-1])
        target_wav = target_wav[..., :min_len]
        output_wav = output_wav[..., :min_len]
        if target_wav.shape[-1] == 0:
            raise Exception(f"Skipping, zero-length waveform: {target_path} -> {output_path}")
        sisnr_val = sisnr_calculator(output_wav, target_wav)
        all_sisnr_values.append(sisnr_val.item())
        print(f"{target_path}|{output_path}|{sisnr_val.item():.4f}")
        all_target_paths.append(target_path)
        all_output_paths.append(output_path)
    
    if args.file_list:
        with open(args.file_list, 'r') as f:
            for line in tqdm(f.readlines(), desc="  Processing audio pairs", ncols=100):
                line = line.strip()
                if not line or '|' not in line:
                    continue

                try:
                    target_path, output_path = [p.strip() for p in line.split('|')]
                    calculate_sisnr(target_path, output_path)
                except Exception:
                    print(f"Error processing a pair: {e}")
                    continue
    else:
        print("--- Finding matching file pairs ---")
        pairs = find_matching_pairs(args.target_dir, args.output_dir)
        print(f"Found {len(pairs)} file pairs")
        for target_path, output_path in pairs:
            try:
                calculate_sisnr(target_path, output_path)
            except Exception as e:
                print(f"Error processing {target_path} -> {output_path}: {e}")
                continue
            
    # ----------------------------------------------------
    # PHASE 2: 批量计算 AudioBox Aesthetics 分数
    # ----------------------------------------------------
    AESTHETICS_CHUNK_SIZE = 64 
    if aesthetics_predictor and all_output_paths:
        print("\n--- Calculating AudioBox Aesthetics Scores (Batch) ---")
        
        # 循环处理分块
    for i in tqdm(range(0, len(all_output_paths), AESTHETICS_CHUNK_SIZE), desc="  Aesthetics chunks"):
        
        # 提取当前批次的路径
        chunk_paths = all_output_paths[i:i + AESTHETICS_CHUNK_SIZE]
        aesthetics_input_list = [{"path": p} for p in chunk_paths]
        
        try:
            # 批量执行推理 (Chunked Inference)
            aesthetics_results = aesthetics_predictor.forward(aesthetics_input_list)
            
            # 结果匹配与收集 (与上一个回答的修正逻辑一致)
            num_outputs = len(chunk_paths)
            num_results = len(aesthetics_results)

            for j in range(num_outputs):
                if j < num_results and all(axis in aesthetics_results[j] for axis in AXES_NAME):
                    score_dict = aesthetics_results[j]
                    for axis in AXES_NAME:
                        all_aesthetics_values[axis].append(score_dict[axis])
                else:
                    for axis in AXES_NAME:
                        all_aesthetics_values[axis].append(float('nan'))
                            
        except Exception as e:
            # 仍然捕获 OOM 或其他异常
            print(f"\nError in chunk {i//AESTHETICS_CHUNK_SIZE}: {e}. Skipping chunk.")
            
            # 填充当前整个 chunk 为 NaN
            for axis in AXES_NAME:
                all_aesthetics_values[axis].extend([float('nan')] * len(chunk_paths))
            
            # 如果是 OOM 错误，可能需要提前停止，或者尝试更小的 AESTHETICS_CHUNK_SIZE
            if "CUDA out of memory" in str(e):
                print("FATAL OOM: Please reduce AESTHETICS_CHUNK_SIZE and restart.")
                # 这里可以考虑 break 或 sys.exit() 


    # ----------------------------------------------------
    # PHASE 3: 写入逐行结果 (SI-SNR 和 Aesthetics)
    # ----------------------------------------------------
    # 检查数据长度是否一致
    num_pairs = len(all_target_paths)
    if num_pairs != len(all_sisnr_values):
         raise RuntimeError("内部错误：指标计算结果数量不匹配。")
    
        # 写入新的列头
    results_file.write("\n--- Pairwise Metrics ---\n")
    
    # 动态构建列头字符串
    header_sisnr = f"{'Target Filename':<30}|{'Output Filename':<30}|{'SI-SNR (dB)':<15}"
    header_aesthetics = "".join([f"|{axis:<10}" for axis in AXES_NAME]) # CE, CU, PC, PQ
    results_file.write(header_sisnr + header_aesthetics + "\n")
    
    print("\n--- Writing results to file ---")
    
    # ... (循环 i in range(num_pairs) 不变)
    for i in tqdm(range(num_pairs), desc="  Writing results", ncols=100):
        # ... (路径和 SI-SNR 获取不变)
        target_path = all_target_paths[i]
        output_path = all_output_paths[i]
        sisnr_item = all_sisnr_values[i]
        
        target_filename = os.path.basename(target_path)
        output_filename = os.path.basename(output_path)
        # 构造 SI-SNR 部分
        result_line = f"{target_filename:<30}|{output_filename:<30}|{sisnr_item:<15.4f}"
        
        # 构造 Aesthetics 部分
        aesthetics_part = ""
        for axis in AXES_NAME:
            score = all_aesthetics_values[axis][i] # 从对应的列表中取出分数
            
            # 格式化 Aesthetics 分数
            aesthetics_str = f"{score:.4f}" if not np.isnan(score) else "N/A"
            aesthetics_part += f"|{aesthetics_str:<10}"

        # 写入文件
        results_file.write(result_line + aesthetics_part + "\n")

    # ----------------------------------------------------
    # PHASE 4: 总体统计参数计算 (SI-SNR, Aesthetics)
    # ----------------------------------------------------

    results_file.write("\n\n--- Overall Statistical Metrics ---\n")
     #  SI-SNR 统计
    if all_sisnr_values:
        avg_sisnr = statistics.mean(all_sisnr_values)
        std_sisnr = statistics.stdev(all_sisnr_values) if len(all_sisnr_values) > 1 else 0.0
        
        # 写入平均值和标准差
        results_file.write(f"SI-SNR (dB) Average: {avg_sisnr:.4f}\n")
        results_file.write(f"SI-SNR (dB) Std Dev: {std_sisnr:.4f}\n")
    else:
        results_file.write("No valid SI-SNR values were calculated.\n")

    # Aesthetics 统计
    # 2. Aesthetics 统计 (循环处理 4 个轴)
    results_file.write("\n--- Aesthetics MOS ---\n")
    for axis in AXES_NAME:
        scores = all_aesthetics_values[axis]
        valid_scores = [s for s in scores if not np.isnan(s)]
        
        if valid_scores:
            avg_aesthetics = statistics.mean(valid_scores)
            std_aesthetics = statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0
            
            # 写入结果
            results_file.write(f"  {axis} (Avg/Std): {avg_aesthetics:.4f} / {std_aesthetics:.4f} (from {len(valid_scores)} samples)\n")
        else:
            results_file.write(f"  {axis} (Avg/Std): N/A (No valid scores calculated)\n")
    
    # ----------------------------------------------------
    # --- FAD-CLAP 计算 ---
    # ----------------------------------------------------
    
    print("\n--- Calculating FAD-CLAP (48kHz) ---")
    
    if not all_target_paths:
        results_file.write("\nFAD-CLAP: Skipped (No valid file pairs found).\n")
        results_file.close()
        return

    try:
        results_file.write(f"\nTotal pairs for FAD-CLAP: {len(all_target_paths)}\n")
        print("Loading CLAP model...")
        LOCAL_MODEL_PATH = "/inspire/hdd/global_user/chenxie-25019/HaoQiu/MSRKit/clap-model"  # 您下载的模型路径
        clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused", local_files_only=True)
        clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused", local_files_only=True)
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
    print(f"\nDone!!!! Save the result into {RESULTS_FILENAME}。")

if __name__ == "__main__":
    main()