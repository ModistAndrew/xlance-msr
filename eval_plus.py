import glob
import os
import re
from pesq import pesq
import soundfile as sf
import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
import argparse
import numpy as np
import warnings
from scipy.linalg import sqrtm
from tqdm import tqdm
import torchaudio
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



def multi_mel_snr(reference, prediction, sr=48000):
    """Compute Multi-Mel-SNR between reference and prediction."""
    if not isinstance(reference, torch.Tensor):
        reference = torch.from_numpy(reference).float()
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.from_numpy(prediction).float()
    
    # Scale-invariant normalization
    alpha = torch.dot(reference, prediction) / (torch.dot(prediction, prediction) + 1e-8)
    prediction = alpha * prediction
    
    # Three mel configurations
    configs = [
        (512, 256, 80),    # (n_fft, hop_length, n_mels)
        (1024, 512, 128),
        (2048, 1024, 192)
    ]
    
    snrs = []
    for n_fft, hop, n_mels in configs:
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, 
            n_mels=n_mels, f_min=0, f_max=24000, power=2.0
        )
        M_ref = mel(reference)
        M_pred = mel(prediction)
        snr = 10 * torch.log10(M_ref.pow(2).sum() / ((M_ref - M_pred).pow(2).sum() + 1e-8))
        snrs.append(snr.item())
    
    return sum(snrs) / len(snrs)

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
    
    for i in tqdm(range(0, len(file_paths), batch_size), desc="  Calculating embeddings", ncols=100, leave=False):
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

def find_matching_pairs(target_dir, output_dir, target_index):
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
        if target_index is not None:
            regex = re.compile(rf"^{re.escape(target_id)}_DT({target_index})\.\w+$")
        else:
            regex = re.compile(rf"^{re.escape(target_id)}_DT\d+\.\w+$")
        matching_outputs = [f for f in matching_outputs if regex.match(os.path.basename(f))]
        matching_outputs.sort()
        
        if matching_outputs:
            print(f"Target {target_id}: found {len(matching_outputs)} output files")
            for output_file in matching_outputs:
                pairs.append((target_file, output_file))
        else:
            print(f"Target {target_id}: no matching output files found")
    
    return pairs

# --- 新增 PESQ 计算函数 ---
def calculate_pesq(target_wav, output_wav, target_sr=48000, pesq_sr=16000):
    """
    计算 PESQ 分数 (通常使用 16kHz 宽带模式)。
    target_wav 和 output_wav 必须是相同的单声道/双声道张量，且已对齐。
    """
    # 确保输入 Tensor 是单声道 (C=1)
    # WAV shape 通常是 [C, L]. 如果 C > 1, 我们将其转换为单声道。
    # 最简单的做法是取第一个声道 [0, :]
    if target_wav.ndim > 1 and target_wav.shape[0] > 1:
        # 提取第一个声道
        target_wav = target_wav[0:1, :]
    if output_wav.ndim > 1 and output_wav.shape[0] > 1:
        # 提取第一个声道
        output_wav = output_wav[0:1, :]
    # 将 Tensor 转换为 numpy 数组
    target_np = target_wav.squeeze(0).numpy()
    output_np = output_wav.squeeze(0).numpy()
    
    # 确保是单声道进行 PESQ 计算
    if target_np.ndim > 1:
        # 如果是多声道，取第一个声道或平均 (这里取第一个声道)
        target_np = target_np[0]
        output_np = output_np[0]
        
    # 重采样到 PESQ 要求的采样率 (16000 Hz)
    if target_sr != pesq_sr:
        resampler = T.Resample(orig_freq=target_sr, new_freq=pesq_sr)
        target_resampled = resampler(target_wav).squeeze(0).numpy()
        output_resampled = resampler(output_wav).squeeze(0).numpy()
    else:
        target_resampled = target_np
        output_resampled = output_np
    
    try:
        # 使用 wideband (wb) 模式，因为我们重采样到 16kHz
        score = pesq(pesq_sr, target_resampled, output_resampled, 'wb')
        return score
    except Exception as e:
        print(f"Warning: PESQ calculation failed for a pair. Error: {e}")
        return float('nan')

def main():
    parser = argparse.ArgumentParser(description="Calculate SI-SNR and FAD-CLAP for audio pairs. All audio is resampled to 48000Hz.")
    parser.add_argument("--target_dir", '-t', required=True, type=str, help="Path to target audio directory")
    parser.add_argument("--output_dir", '-o', required=True, type=str, help="Path to output audio directory")
    parser.add_argument("--target_index", '-i', type=str, help="Index of target audio files, e.g. '11|12'")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for FAD-CLAP embedding calculation.")
    parser.add_argument("--output_file", type=str, help="Filename to save all evaluation results.")
    # 测评指标开关
    parser.add_argument("--calc_sisnr", action="store_true", help="Calculate Scale-Invariant SNR (SI-SNR).")
    parser.add_argument("--calc_pesq", action="store_true", help="Calculate Perceptual Evaluation of Speech Quality (PESQ).")
    parser.add_argument("--calc_aesthetics", action="store_true", help="Calculate AudioBox Aesthetics MOS.")
    parser.add_argument("--calc_fad_clap", default=True, action="store_true", help="Calculate Frechet Audio Distance (FAD-CLAP).")
    parser.add_argument("--calc_mel_snr", default=True, action="store_true", help="Calculate Multi-Mel-SNR.") # <-- Multi-Mel-SNR 开关
    
    args = parser.parse_args()
    
    if not args.output_file:
        args.output_file = (args.output_dir[:-1] if args.output_dir.endswith('/') else args.output_dir)
        if args.target_index:
            args.output_file += f"_{args.target_index}"
        args.output_file += ".txt"
    
    # 初始化 AudioBox Aesthetics Predictor
    AXES_NAME = ["CE", "CU", "PC", "PQ"] 
    LOCAL_AESTHETICS_CKPT = "/inspire/hdd/global_user/chenxie-25019/HaoQiu/EVAL_MODEL/audiobox/audiobox_aes_checkpoint.pt"
    try:
        assert args.calc_aesthetics, "AudioBox Aesthetics is not enabled"
        print("\nLoading AudioBox Aesthetics predictor...")
        aesthetics_predictor = initialize_predictor(ckpt=LOCAL_AESTHETICS_CKPT)
        print("AudioBox Aesthetics predictor loaded successfully.")
    except Exception as e:
        print(f"Error loading AudioBox Aesthetics predictor: {e}. Aesthetics calculation will be skipped.")
        aesthetics_predictor = None
        
    # 初始化文件写入
    RESULTS_FILENAME = args.output_file
    if os.path.exists(RESULTS_FILENAME):
        raise Exception(f"Output file already exists: {RESULTS_FILENAME}")
    results_file = open(RESULTS_FILENAME, 'w', encoding='utf-8')
    results_file.write("--- Audio Evaluation Results ---\n")
    print(f"所有结果将被写入文件: {RESULTS_FILENAME}")

    sisnr_calculator = ScaleInvariantSignalNoiseRatio()
    all_target_paths = []
    all_output_paths = []
    all_sisnr_values = []
    all_pesq_values = []
    all_mel_snr_values = []
    all_aesthetics_values = {axis: [] for axis in AXES_NAME}
    # ----------------------------------------------------
    # PHASE 1: 遍历文件列表，计算 SI-SNR，收集路径
    # ----------------------------------------------------
    
    print("\n--- Calculating SI-SNR (48kHz) for each pair ---")
    results_file.write("\n--- Pairwise SI-SNR (dB) ---\n")
    
    TARGET_SR = 48000 
    
    def calculate_pairwise_metrics(target_path, output_path, args, results_list):
        if not os.path.exists(target_path) or not os.path.exists(output_path):
            raise Exception(f"Skipping, file not found: {target_path} -> {output_path}")
        target_wav = load_audio(target_path, TARGET_SR)
        output_wav = load_audio(output_path, TARGET_SR)
        if target_wav is None or output_wav is None:
            raise Exception(f"Skipping, waveform not loaded: {target_path} -> {output_path}")
        if target_wav.shape[0] != output_wav.shape[0]:
            print(f"Warning: shape mismatch: {target_path} -> {output_path}")
            if target_wav.shape[0] not in [1, 2]:
                raise Exception(f"Skipping, unsupported shape: {target_path} -> {output_path}")
            if output_wav.shape[0] not in [1, 2]:
                raise Exception(f"Skipping, unsupported shape: {target_path} -> {output_path}")
            if target_wav.shape[0] > output_wav.shape[0]: # 2 vs 1
                output_wav = output_wav.repeat(2, 1)
            else: # 1 vs 2
                output_wav = output_wav.mean(dim=0, keepdim=True)
        min_len = min(target_wav.shape[-1], output_wav.shape[-1])
        target_wav = target_wav[..., :min_len]
        output_wav = output_wav[..., :min_len]
        if target_wav.shape[-1] == 0:
            raise Exception(f"Skipping, zero-length waveform: {target_path} -> {output_path}")
        
        # --- SI-SNR part ---
        sisnr_val = float('nan')
        if args.calc_sisnr:
            sisnr_val = sisnr_calculator(output_wav, target_wav).item()
        results_list['sisnr'].append(sisnr_val)

        # --- PESQ part ---
        pesq_val = float('nan')
        if args.calc_pesq:
            pesq_val = calculate_pesq(target_wav, output_wav, TARGET_SR)
        results_list['pesq'].append(pesq_val)
        
        # --- Multi-Mel-SNR part ---
        mel_snr_val = float('nan')
        if args.calc_mel_snr:
            # Multi-Mel-SNR 假设单声道输入，故对每个声道计算并平均
            mel_snrs = []
            for ch in range(target_wav.shape[0]):
                 # 注意：multi_mel_snr 内部需要进行 SI-Norm，这里传入原始 wav
                mel_snr_val_ch = multi_mel_snr(target_wav[ch], output_wav[ch], sr=TARGET_SR)
                mel_snrs.append(mel_snr_val_ch)
            mel_snr_val = sum(mel_snrs) / len(mel_snrs) if mel_snrs else float('nan')
        results_list['mel_snr'].append(mel_snr_val)
        
        output_str = f"{target_path}|{output_path}"
        if args.calc_sisnr:
            output_str += f"|SI-SNR:{sisnr_val:.4f}"
        if args.calc_pesq:
            output_str += f"|PESQ:{pesq_val:.4f}"
        if args.calc_mel_snr:
            output_str += f"|Mel-SNR:{mel_snr_val:.4f}"
        print(output_str)
        
        all_target_paths.append(target_path)
        all_output_paths.append(output_path)
    
    
    all_pairwise_values = {
        'sisnr': [], 
        'pesq': [], 
        'mel_snr': [] 
    }
    
    print("--- Finding matching file pairs ---")
    pairs = find_matching_pairs(args.target_dir, args.output_dir, args.target_index)
    print(f"Found {len(pairs)} file pairs")
    for target_path, output_path in pairs:
        try:
            calculate_pairwise_metrics(target_path, output_path, args, all_pairwise_values)
        except Exception as e:
            print(f"Error processing {target_path} -> {output_path}: {e}")
            continue
            
    # ----------------------------------------------------
    # PHASE 2: 批量计算 AudioBox Aesthetics 分数
    # ----------------------------------------------------
    AESTHETICS_CHUNK_SIZE = 64 
    if args.calc_aesthetics and aesthetics_predictor and all_output_paths:
        print("\n--- Calculating AudioBox Aesthetics Scores (Batch) ---")
        
        # 循环处理分块
    for i in tqdm(range(0, len(all_output_paths), AESTHETICS_CHUNK_SIZE), desc="  Aesthetics chunks"):
        
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
    
    # 补全 Aesthetics 列表（如果未计算），确保长度与 num_pairs 匹配
    if not args.calc_aesthetics or not all_output_paths:
        if len(all_target_paths) > 0:
            for axis in AXES_NAME:
                # 只在列表长度不一致时进行填充（避免重复填充）
                if len(all_aesthetics_values[axis]) < len(all_target_paths):
                    all_aesthetics_values[axis].extend([float('nan')] * (len(all_target_paths) - len(all_aesthetics_values[axis])))

    # ----------------------------------------------------
    # PHASE 3: 写入逐行结果 (SI-SNR 和 Aesthetics)
    # ----------------------------------------------------
    # 检查数据长度是否一致
    num_pairs = len(all_target_paths)
    for metric_name, scores in all_pairwise_values.items():
        if len(scores) != num_pairs:
            # 如果某个列表的长度不匹配，说明计算或收集过程中出现了错误
            raise RuntimeError(f"内部错误：指标 '{metric_name}' 的结果数量 ({len(scores)}) 与文件对数量 ({num_pairs}) 不匹配。")

    # 检查 Aesthetics 指标的长度是否与文件对数量一致
    if args.calc_aesthetics:
        for axis in AXES_NAME:
            scores = all_aesthetics_values[axis]
            if len(scores) != num_pairs:
                raise RuntimeError(f"内部错误：Aesthetics 指标 '{axis}' 的结果数量 ({len(scores)}) 与文件对数量 ({num_pairs}) 不匹配。")
    
        # 写入新的列头
    results_file.write("\n--- Pairwise Metrics ---\n")
    
    # 动态构建列头字符串
    header_metrics = f"{'Target Filename':<30}|{'Output Filename':<30}"
    if args.calc_sisnr:
        header_metrics += f"|{'SI-SNR (dB)':<15}"
    if args.calc_pesq:
        header_metrics += f"|{'PESQ':<8}"
    if args.calc_mel_snr: # <-- 新增 Mel-SNR 列头
        header_metrics += f"|{'Mel-SNR (dB)':<15}"
    
    if args.calc_aesthetics:
        for axis in AXES_NAME:
            header_metrics += f"|{axis:<10}" # Aesthetics 的四个维度
            
    # 写入列头分隔线
    results_file.write(header_metrics + "\n") 
    results_file.write("-" * len(header_metrics) + "\n")
    
    print("\n--- Writing results to file ---")
    
    # ... (循环 i in range(num_pairs) 不变)
    for i in tqdm(range(num_pairs), desc="  Writing results", ncols=100):
        target_filename = os.path.basename(all_target_paths[i])
        output_filename = os.path.basename(all_output_paths[i])
        result_line = f"{target_filename:<30}|{output_filename:<30}"
        
        if args.calc_sisnr:
            sisnr_item = all_pairwise_values['sisnr'][i]
            result_line += f"|{sisnr_item:<15.4f}"
        if args.calc_pesq:
            pesq_item = all_pairwise_values['pesq'][i]
            pesq_str = f"{pesq_item:<8.4f}" if not np.isnan(pesq_item) else "N/A   "
            result_line += f"|{pesq_str}"
        if args.calc_mel_snr:
            mel_snr_item = all_pairwise_values['mel_snr'][i]
            mel_snr_str = f"{mel_snr_item:<15.4f}" if not np.isnan(mel_snr_item) else "N/A           "
            result_line += f"|{mel_snr_str}"
        
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
    if args.calc_sisnr and all_pairwise_values['sisnr']:
        scores = all_pairwise_values['sisnr']
        if scores:
            avg_sisnr = statistics.mean(scores)
            std_sisnr = statistics.stdev(scores) if len(scores) > 1 else 0.0
            
            # 写入平均值和标准差
            results_file.write(f"SI-SNR (dB) Average: {avg_sisnr:.4f}\n")
            results_file.write(f"SI-SNR (dB) Std Dev: {std_sisnr:.4f}\n")
        else:
            results_file.write("No valid SI-SNR values were calculated.\n")

    # PESQ 统计
    if args.calc_pesq and all_pairwise_values['pesq']:
        scores = all_pairwise_values['pesq']
        valid_pesq_scores = [s for s in scores if not np.isnan(s)]
        if valid_pesq_scores:
            avg_pesq = statistics.mean(valid_pesq_scores)
            std_pesq = statistics.stdev(valid_pesq_scores) if len(valid_pesq_scores) > 1 else 0.0
            results_file.write(f"\nPESQ Average: {avg_pesq:.4f}\n")
            results_file.write(f"PESQ Std Dev: {std_pesq:.4f} (from {len(valid_pesq_scores)} samples)\n")
        else:
            results_file.write("\nNo valid PESQ values were calculated.\n")
    
    #  Multi-Mel-SNR 统计 
    if args.calc_mel_snr and all_pairwise_values['mel_snr']:
        scores = all_pairwise_values['mel_snr']
        valid_scores = [s for s in scores if not np.isnan(s)]
        if valid_scores:
            avg_mel_snr = statistics.mean(valid_scores)
            std_mel_snr = statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0
            results_file.write(f"\nMulti-Mel-SNR Average: {avg_mel_snr:.4f}\n")
            results_file.write(f"Multi-Mel-SNR Std Dev: {std_mel_snr:.4f} (from {len(valid_scores)} samples)\n")
        else:
            results_file.write("\nNo valid Multi-Mel-SNR values were calculated.\n")        
    
    # Aesthetics 统计
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
    if args.calc_fad_clap:
        print("\n--- Calculating FAD-CLAP (48kHz) ---")
        
        if not all_target_paths:
            results_file.write("\nFAD-CLAP: Skipped (No valid file pairs found).\n")
        else:
            clap_model = None
            clap_processor = None

        try:
            results_file.write(f"\nTotal pairs for FAD-CLAP: {len(all_target_paths)}\n")
            print("Loading CLAP model...")
            LOCAL_MODEL_PATH = "/inspire/hdd/global_user/chenxie-25019/HaoQiu/EVAL_MODEL/clap-model"  # 您下载的模型路径
            clap_model = ClapModel.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
            clap_processor = ClapProcessor.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
            clap_model.eval()
            print("CLAP model loaded successfully.")
            
        except Exception as e:
            error_msg = f"Fatal Error: Could not load CLAP model. Error: {e}"
            print(error_msg)
            results_file.write(f"\nFAD-CLAP: {error_msg}\n")
    if clap_model and clap_processor:            
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