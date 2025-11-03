import torch
import torchaudio
import numpy as np
import random
import logging
from pathlib import Path
from tqdm import tqdm
import sys
# Path(__file__).resolve().parent 获取当前脚本所在的文件夹 (check/)
# .parent.parent 获取到 'ProjectRoot' 文件夹
# 将 'ProjectRoot' 路径添加到系统搜索路径中
sys.path.append(str(Path(__file__).resolve().parent.parent))
# 假设你的 MoisesDBAdapter, load_audio, contains_audio_signal, mix_to_target_snr 等工具函数
# 都在 data.dataset 中，你需要确保这里的导入路径正确。
# 如果它们在一个大文件中，你可能需要将它们复制到这个检查脚本中或确保能导入。
# 这里假设可以从 data.dataset 导入 MoisesDBAdapter
from data.dataset import MoisesDBAdapter # 替换为你实际的导入路径
# 假设你在 MoisesDBAdapter 中使用了 logger，这里也配置一下
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 辅助函数：将 Tensor 写入 WAV 文件 ---
def save_audio_to_file(
    audio_tensor: torch.Tensor, 
    file_path: Path, 
    sr: int
):
    """
    将 PyTorch Tensor 形状 (C, T) 或 (T) 的音频保存到 WAV 文件。
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    audio_tensor = audio_tensor.detach().cpu().float()
    
    # 转换为 (C, T) 格式
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0) # (1, T)
    elif audio_tensor.ndim == 2:
        # 如果是 (B, T)，我们只取第一个 B=1
        if audio_tensor.shape[0] > 1:
             audio_tensor = audio_tensor[0].unsqueeze(0)
        # 否则假设它是 (C, T) 或 (1, T)
        
    # 归一化到 [-1, 1]
    max_abs = audio_tensor.abs().max()
    if max_abs > 0:
        audio_tensor = audio_tensor / max_abs
            
    try:
        torchaudio.save(str(file_path), audio_tensor, sr)
        logger.info(f"Saved audio to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save audio to {file_path}: {e}")


def check_dataset_samples(
    root_directory: str, 
    target_stem: str,
    output_dir: str = "dataset_check_output",
    num_samples: int = 5
):
    """
    实例化 MoisesDBAdapter 并检查前几个随机样本。
    """
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- 1. 数据集配置 (与你的训练配置保持一致) ---
    SR = 48000
    CLIP_DURATION = 4.0
    
    # 注意: RMS_THRESHOLD 和 SNR_RANGE 必须匹配你的训练设置
    try:
        dataset = MoisesDBAdapter(
            target_stem=target_stem,
            root_directory=root_directory,
            sr=SR,
            clip_duration=CLIP_DURATION,
            snr_range=(0.0, 10.0),      # 保持与训练一致
            apply_augmentation=False,   # 检查时不进行增强，保持音频干净
            rms_threshold=-40.0         # 保持与训练一致
        )
        
    except FileNotFoundError as e:
        logger.error(f"Dataset initialization failed: {e}")
        return

    total_songs = len(dataset)
    logger.info(f"Dataset initialized with {total_songs} valid songs after filtering.")

    # 如果数据集为空，则退出
    if total_songs == 0:
        logger.warning("No songs left in the dataset. Check filtering/thresholds.")
        return

    # --- 2. 随机选择要检查的索引 ---
    random_indices = random.sample(range(total_songs), min(num_samples, total_songs))
    
    # --- 3. 循环加载和保存样本 ---
    for i, idx in enumerate(tqdm(random_indices, desc="Checking Dataset Samples")):
        try:
            # 调用 __getitem__ 获取一个样本 (mixture, target)
            sample = dataset[idx]
            
            mixture = sample['mixture']
            target = sample['target']
            
            # 将 numpy 数组转换为 torch Tensor (如果它们在 __getitem__ 中还是 numpy)
            if isinstance(mixture, np.ndarray):
                 # 假设你的音频是 (C, T) 或 (T)
                mixture = torch.from_numpy(mixture) 
                target = torch.from_numpy(target)
            
            # 保存到文件
            sample_dir = OUTPUT_DIR / f"sample_{i}_idx{idx}"
            save_audio_to_file(mixture, sample_dir / "01_mixture.wav", SR)
            save_audio_to_file(target, sample_dir / "02_target.wav", SR)
            
        except Exception as e:
            logger.error(f"Error loading sample index {idx}: {e}")

    logger.info(f"Dataset check complete. Check output directory: {OUTPUT_DIR.resolve()}")


if __name__ == '__main__':
    # --- ！！！请修改这里的参数 ！！！ ---
    # 替换为你的 MoisesDB 数据集根目录
    DATA_ROOT = "/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/data/moisesdb/canonical" 
    # 替换为你想要检查的目标 stem
    TARGET_STEM = "Bass" 
    # 检查样本数量
    NUM_SAMPLES_TO_CHECK = 5
    
    check_dataset_samples(
        root_directory=DATA_ROOT,
        target_stem=TARGET_STEM,
        num_samples=NUM_SAMPLES_TO_CHECK
    )