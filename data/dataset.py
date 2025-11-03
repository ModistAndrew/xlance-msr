import math
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
from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = ['.flac', '.mp3', '.wav']
DEFAULT_GAIN_RANGE = (0.5, 1.0)

def calculate_rms(audio: np.ndarray) -> float:
    return np.sqrt(np.mean(audio**2))

def contains_audio_signal(audio: np.ndarray, rms_threshold: float = 0.01) -> bool:
    if audio is None:
        return False
    return calculate_rms(audio) > rms_threshold

def fix_length(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    target_length, source_length = target.shape[-1], source.shape[-1]
    if target_length < source_length:
        return np.pad(target, ((0, 0), (0, source_length - target_length)), mode='constant')
    if target_length > source_length:
        return target[:, :source_length]
    return target

def fix_length_to_duration(target: np.ndarray, duration: float, sr: int) -> np.ndarray:
    target_length = target.shape[-1]
    required_length = int(duration * sr)
    if target_length < required_length:
        return np.pad(target, ((0, 0), (0, required_length - target_length)), mode='constant')
    if target_length > required_length:
        return target[:, :required_length]
    return target

def get_audio_duration(file_path: Path) -> float:
    try:
        return sf.info(file_path).duration
    except Exception as e:
        logger.error(f"Error getting duration for {file_path}: {e}")
        return 0.0

def load_audio(file_path: Path, offset: float, duration: float, sr: int) -> np.ndarray:
    try:
        audio, _ = librosa.load(file_path, sr=sr, offset=offset, duration=duration, mono=False)
        if len(audio.shape) == 1: audio = audio.reshape(1, -1)
        if audio.shape[1] == 0: return np.zeros((2, int(sr * duration)))
        if audio.shape[0] == 1: audio = np.vstack([audio, audio])
        return audio
    except Exception as e:
        logger.error(f"Error loading {file_path} at offset {offset}: {e}")
        return np.zeros((2, int(sr * duration)))
    
def _load_audio_mono(file_path: Path, sr: int) -> np.ndarray:
    """
    专门用于分析的加载函数：加载整个文件并转换为单声道。
    返回: 单声道 np.ndarray (samples,)
    """
    try:
        # 加载整个文件 (duration=None)，并指定输出采样率 sr
        # mono=True 确保直接输出单声道 (librosa 会进行 downmix)
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        return audio
    except Exception as e:
        logger.error(f"Error loading full audio for RMS analysis at {file_path}: {e}")
        return np.array([])
    
def calculate_rms_mask_from_file(file_path: Path, sr: int, clip_duration: float, rms_threshold: float) -> Tuple[np.ndarray, float]:
    """
    加载文件，计算 RMS (以秒为单位)，并生成布尔活动掩码。
    返回: (mask, file_duration_seconds)
    """
    # 1. 加载音频（单声道）
    audio = _load_audio_mono(file_path, sr)
    
    if audio.size == 0:
        return np.array([], dtype=bool), 0.0

    # 2. 计算每秒的 RMS/dB
    # 模仿 RawStems 的逻辑：计算每秒的 RMS/dB
    
    # 将音频分割成 1 秒的帧
    frame_len = sr
    num_frames = math.ceil(audio.size / frame_len)
    
    # 填充确保能整除
    padded_audio = np.pad(audio, (0, num_frames * frame_len - audio.size), mode='constant', constant_values=0)
    frames = padded_audio.reshape(num_frames, frame_len)
    
    # 计算每帧（即每秒）的 RMS (你的 calculate_rms 函数的原理)
    # 我们使用 20 * log10(RMS) 来得到 dB 值
    rms_values = np.sqrt(np.mean(frames**2, axis=1))
    
    # 转换为 dB 值 (使用一个小的参考值来防止 log(0))
    rms_db_per_second = 20 * np.log10(rms_values + 1e-10) 
    
    # 获取精确的文件持续时间（以秒为单位，用于返回）
    file_duration_sec = get_audio_duration(file_path) 
    
    # 3. 创建布尔活动掩码（与 RawStems 的 _compute_activity_masks 逻辑一致）
    
    window_size = int(np.ceil(clip_duration))
    is_loud = rms_db_per_second > rms_threshold
    
    if len(is_loud) < window_size:
        # 文件太短，无法进行完整的卷积，仅判断是否高于阈值
        mask = is_loud
    else:
        # 卷积平滑：判断窗口内是否至少 80% 活跃 (RawStems 的逻辑)
        sum_loud = np.convolve(is_loud, np.ones(window_size), 'valid')
        avg_loud_enough = sum_loud / window_size > 0.8 
        
        # 填充回原始的秒数长度
        mask = np.zeros(len(rms_db_per_second), dtype=bool)
        mask[:len(avg_loud_enough)] = avg_loud_enough
    
    return mask, file_duration_sec

def mix_to_target_snr(target: np.ndarray, noise: np.ndarray, target_snr_db: float) -> Tuple[np.ndarray, float, float]:
    target_power, noise_power = np.mean(target ** 2), np.mean(noise ** 2)
    if noise_power < 1e-8: return target.copy(), 1.0, 0.0
    if target_power < 1e-8: return noise * 0.001, 0.0, 0.001
    
    target_snr_linear = 10 ** (target_snr_db / 10)
    noise_scale = np.sqrt(target_power / (noise_power * target_snr_linear))
    scaled_noise = noise * noise_scale
    mixture = target + scaled_noise
    
    max_amplitude = np.max(np.abs(mixture))
    target_scale = 1.0
    if max_amplitude > 1.0:
        norm_factor = 0.95 / max_amplitude
        mixture *= norm_factor
        target_scale = norm_factor
        noise_scale *= norm_factor
    
    return mixture, target_scale, noise_scale

class RawStems(Dataset):
    def __init__(
        self,
        target_stem: str,
        root_directory: Union[str, Path],
        sr: int = 48000,
        clip_duration: float = 4.0,
        snr_range: Tuple[float, float] = (0.0, 10.0),
        apply_augmentation: bool = True,
        rms_threshold: float = -40.0,
    ) -> None:
        self.root_directory = Path(root_directory)
        self.sr = sr
        self.clip_duration = clip_duration
        self.snr_range = snr_range
        self.apply_augmentation = apply_augmentation
        self.rms_threshold = rms_threshold
        
        target_stem_parts = target_stem.split("_")
        self.target_stem_1 = target_stem_parts[0].strip()
        self.target_stem_2 = target_stem_parts[1].strip() if len(target_stem_parts) > 1 else None
        
        logger.info(f"Scanning '{self.root_directory}' for songs containing stem '{target_stem}'...")
        self.folders = []
        for song_dir in self.root_directory.iterdir():
            if song_dir.is_dir():
                target_path = song_dir / self.target_stem_1
                if self.target_stem_2:
                    target_path /= self.target_stem_2
                
                if target_path.exists() and target_path.is_dir():
                    self.folders.append(song_dir)
        
        if not self.folders:
            raise FileNotFoundError(f"No subdirectories in '{self.root_directory}' were found containing the stem path '{target_stem}'. "
                                    f"Please check your directory structure.")
        logger.info(f"Found {len(self.folders)} song folders.")

        self.audio_files = self._index_audio_files()
        if not self.audio_files: raise ValueError("No audio files found.")
            
        self.activity_masks = self._compute_activity_masks()
        
        self.stem_augmentation = StemAugmentation()
        self.mixture_augmentation = MixtureAugmentation()

    def _compute_activity_masks(self) -> Dict[str, np.ndarray]:
        rms_analysis_path = self.root_directory / "rms_analysis.jsonl"
        if not rms_analysis_path.exists():
            logger.warning("rms_analysis.jsonl not found. Non-silent selection will be disabled.")
            return {}
        
        logger.info(f"Loading and processing RMS data from {rms_analysis_path}")
        rms_data = {}
        with open(rms_analysis_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    rms_data[data['filepath']] = np.array(data['rms_db_per_second'])
                except (json.JSONDecodeError, KeyError):
                    continue

        logger.info("Computing activity masks for all indexed files...")
        activity_masks = {}
        window_size = int(np.ceil(self.clip_duration))

        all_indexed_files = set()
        for song in self.audio_files:
            all_indexed_files.update(p.relative_to(self.root_directory) for p in song["target_stems"])
            all_indexed_files.update(p.relative_to(self.root_directory) for p in song["others"])

        for relative_path in tqdm(all_indexed_files, desc="Computing Activity Masks"):
            path_str = str(relative_path)
            if path_str in rms_data:
                rms_values = rms_data[path_str]
                if len(rms_values) < window_size:
                    activity_masks[path_str] = np.array([False] * len(rms_values))
                    continue
                
                is_loud = rms_values > self.rms_threshold
                sum_loud = np.convolve(is_loud, np.ones(window_size), 'valid')
                avg_loud_enough = sum_loud / window_size > 0.8 
                
                mask = np.zeros(len(rms_values), dtype=bool)
                mask[:len(avg_loud_enough)] = avg_loud_enough
                activity_masks[path_str] = mask
            else:
                print(f"Warning: No RMS data found for {path_str}")
        return activity_masks

    def _find_common_valid_start_seconds(self, file_paths: List[Path]) -> List[int]:
        if not self.activity_masks: return []

        common_mask = None
        min_len = float('inf')

        masks_to_intersect = []
        for file_path in file_paths:
            path_str = str(file_path.relative_to(self.root_directory))
            mask = self.activity_masks.get(path_str)
            if mask is None: return []
            masks_to_intersect.append(mask)
            min_len = min(min_len, len(mask))
        
        if not masks_to_intersect: return []

        final_mask = np.ones(min_len, dtype=bool)
        for mask in masks_to_intersect:
            final_mask &= mask[:min_len]

        return np.where(final_mask)[0].tolist()

    def _index_audio_files(self) -> List[Dict[str, List[Path]]]:
        indexed_songs = []
        for folder in tqdm(self.folders, desc="Indexing audio files"):
            song_dict = {"target_stems": [], "others": []}
            target_folder = folder / self.target_stem_1
            if self.target_stem_2: target_folder /= self.target_stem_2
            
            if target_folder.exists():
                song_dict["target_stems"].extend(p for p in target_folder.rglob('*') if p.suffix.lower() in AUDIO_EXTENSIONS)
            
            for p in folder.rglob('*'):
                if p.suffix.lower() in AUDIO_EXTENSIONS:
                    try:
                        relative_path = p.relative_to(folder)
                        parts = relative_path.parts
                        is_target = len(parts) > 0 and parts[0] == self.target_stem_1 and \
                                    (self.target_stem_2 is None or (len(parts) > 1 and parts[1] == self.target_stem_2))
                        if not is_target:
                            song_dict["others"].append(p)
                    except ValueError:
                        continue
            
            if song_dict["target_stems"] and song_dict["others"]:
                indexed_songs.append(song_dict)
        return indexed_songs
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        song_dict = self.audio_files[index]
        
        for _ in range(100):
            num_targets = random.randint(1, min(len(song_dict["target_stems"]), 5))
            selected_targets = random.sample(song_dict["target_stems"], num_targets)
            
            num_others = random.randint(1, min(len(song_dict["others"]), 10))
            selected_others = random.sample(song_dict["others"], num_others)

            valid_starts = self._find_common_valid_start_seconds(selected_targets + selected_others)

            if valid_starts:
                start_second = random.choice(valid_starts)
                offset = start_second + random.uniform(0, 1.0 - (self.clip_duration % 1.0 or 1.0))
                
                target_mix = sum(load_audio(p, offset, self.clip_duration, self.sr) for p in selected_targets) / num_targets
                other_mix = sum(load_audio(p, offset, self.clip_duration, self.sr) for p in selected_others) / num_others

                if not contains_audio_signal(target_mix) or not contains_audio_signal(other_mix):
                    
                    continue

                target_clean = target_mix.copy()
                target_augmented = self.stem_augmentation.apply(target_mix, self.sr) if self.apply_augmentation else target_mix
                
                mixture, target_scale, _ = mix_to_target_snr(
                    target_augmented, other_mix, random.uniform(*self.snr_range)
                )
                target_clean *= target_scale
                
                mixture_augmented = self.mixture_augmentation.apply(mixture, self.sr) if self.apply_augmentation else mixture

                max_val = np.max(np.abs(mixture_augmented)) + 1e-8
                mixture_final = mixture_augmented / max_val
                target_final = target_clean / max_val
                
                rescale = np.random.uniform(*DEFAULT_GAIN_RANGE)

                mixture = np.nan_to_num(mixture_final * rescale)
                target = np.nan_to_num(target_final * rescale)
                
                target_length = int(self.clip_duration * self.sr)
                if target.shape[1] != target_length:
                    target = np.pad(target, (0, target_length - target.shape[1]), mode='constant')
                else:
                    target = target[:, :target_length]
                if mixture.shape[1] != target_length:
                    mixture = np.pad(mixture, (0, target_length - mixture.shape[1]), mode='constant')
                else:
                    mixture = mixture[:, :target_length]
                
                return {
                    "mixture": np.nan_to_num(mixture),
                    "target": np.nan_to_num(target)
                }

        return self.__getitem__(random.randint(0, len(self.audio_files) - 1))

    def __len__(self) -> int:
        return len(self.audio_files)

class MoisesDBAdapter(Dataset):
    """
    适配 MoisesDB 数据集，使其格式与 RawStems 一致。
    MoisesDB 结构：root_directory / song_id / stem_name.wav
    """
    # MoisesDB 常见的 Stems 列表
    MOISES_STEMS = ['vocals', 'drums', 'bass', 'other', 'piano', 'guitar']
    
    # 目标 stem 到 MoisesDB 文件名的映射
    STEM_ALIAS_MAP = {
        'voc': 'vocals',
        'drums': 'drums',
        'bass': 'bass',
        'gtr': 'guitar',
        'key': 'piano',
        'other': 'other',
    }

    def __init__(
        self,
        target_stem: str,
        root_directory: Union[str, Path],
        sr: int = 48000,
        clip_duration: float = 4.0,
        snr_range: Tuple[float, float] = (0.0, 10.0),
        apply_augmentation: bool = True,
        rms_threshold: float = -40.0, 
    ) -> None:
        self.root_directory = Path(root_directory)
        self.sr = sr
        self.clip_duration = clip_duration
        self.snr_range = snr_range
        self.apply_augmentation = apply_augmentation
        self.rms_threshold = rms_threshold
        target_stem_key = target_stem.split('_')[0].strip().lower() # 只取第一个部分
        self.target_filename = self.STEM_ALIAS_MAP.get(target_stem_key, target_stem_key)
        
        if self.target_filename not in self.MOISES_STEMS:
            raise ValueError(
                f"Target stem '{target_stem}' maps to '{self.target_filename}', "
                f"which is not a standard MoisesDB stem ({self.MOISES_STEMS})."
            )

        self.noise_filenames = [
            s for s in self.MOISES_STEMS if s != self.target_filename
        ]
        
        logger.info(f"Indexing MoisesDB for target stem: '{self.target_filename}'...")
        self.song_paths = self._index_songs()
        
        # 1. 索引歌曲结构
        if not self.song_paths:
            raise FileNotFoundError(f"No complete MoisesDB songs found in '{self.root_directory}'.")
        
        logger.info(f"Found {len(self.song_paths)} complete songs for target stem '{self.target_filename}'.")
        # 2. 计算并存储所有 Track 文件的活动掩码
        
        # --- 引入缓存机制 ---
        cache_filename = f"activity_masks_cache_{self.target_filename}_sr{self.sr}_clip{clip_duration:.1f}_rms{rms_threshold:.1f}.json"
        self.cache_path = self.root_directory / cache_filename
        # 尝试加载缓存
        self.activity_masks = self._load_activity_masks_cache()
        if self.activity_masks:
            logger.info(f"Loaded {len(self.activity_masks)} masks from cache: {self.cache_path.name}")
        else:
            # 2. 缓存不存在或无效，执行计算并保存
            logger.info("Cache not found or invalid. Analyzing RMS for all tracks to compute activity masks...")
            self.activity_masks = self._compute_all_track_masks(self.sr, clip_duration, rms_threshold)
            self._save_activity_masks_cache()
        

        # 3. 预先计算每首歌的有效起始秒（作为索引优化）
        self.song_valid_starts = self._precompute_valid_starts_per_song()
        logger.info(f"Precomputed valid start times for {len(self.song_valid_starts)} songs.")
        # 过滤掉没有任何有效片段的歌曲 TODO
        self._filter_invalid_songs()
        
        self.stem_augmentation = StemAugmentation()
        self.mixture_augmentation = MixtureAugmentation()
        
    def _filter_invalid_songs(self):
        """
        过滤掉那些在预计算后没有任何有效起始秒（valid_starts = []）的歌曲。
        更新 self.song_paths 和 self.song_valid_starts 列表。
        """
        initial_count = len(self.song_paths)
        
        # 使用列表推导式来构建新的、有效的列表
        # (valid_starts, song_data) 元组
        valid_data_pairs = [
            (valid_starts, song_data) 
            for valid_starts, song_data in zip(self.song_valid_starts, self.song_paths) 
            if valid_starts # 检查列表是否非空
        ]
        
        # 分别更新属性
        self.song_valid_starts = [pair[0] for pair in valid_data_pairs]
        self.song_paths = [pair[1] for pair in valid_data_pairs]
        
        filtered_count = len(self.song_paths)
        skipped_count = initial_count - filtered_count
        
        logger.info(f"Filtered out {skipped_count} songs with no common active segments.")
        logger.info(f"Final usable dataset size: {filtered_count} songs.")
        
        if filtered_count == 0:
            raise FileNotFoundError(
                "After filtering, no songs with common active segments remain. "
                "Check your RMS threshold and clip duration settings."
            )
    def _load_activity_masks_cache(self) -> Dict[Path, np.ndarray]:
        """尝试从 JSON 文件加载活动掩码缓存。"""
        if not self.cache_path.exists():
            return {}
        
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            masks = {}
            for path_str, mask_list in cached_data.items():
                # 关键：将路径字符串转回 Path 对象
                file_path = Path(path_str)
                # 关键：将列表转回布尔 NumPy 数组
                masks[file_path] = np.array(mask_list, dtype=bool)
                
            return masks
            
        except (json.JSONDecodeError, KeyError, TypeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load activity mask cache from {self.cache_path.name}. Error: {e}")
            # 如果加载失败，返回空字典，触发重新计算
            return {}
        
    def _save_activity_masks_cache(self):
        """将计算出的活动掩码保存到 JSON 文件中。"""
        if not self.activity_masks:
            logger.warning("No masks computed, skipping cache save.")
            return

        serializable_data = {}
        for file_path, mask in self.activity_masks.items():
            # 关键：将 Path 对象转换为字符串
            path_str = str(file_path)
            # 关键：将 NumPy 数组转换为 Python 列表 (JSON 可序列化)
            serializable_data[path_str] = mask.tolist()
            
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=4)
            logger.info(f"Successfully saved activity mask cache to {self.cache_path.name}")
        except Exception as e:
            logger.error(f"Failed to save activity mask cache: {e}")
        
    
    def _compute_all_track_masks(self, sr: int, clip_duration: float, rms_threshold: float) -> Dict[Path, np.ndarray]:
        """为所有找到的 Track 文件计算活动掩码。"""
        masks = {}
        all_files = set()
        
        # 收集所有需要分析的唯一文件路径
        for song_data in self.song_paths:
            all_files.update(song_data["target_tracks"])
            all_files.update(song_data["noise_tracks"])

        logging.info(f"Analyzing {len(all_files)} unique audio files for activity.")
        
        for file_path in tqdm(all_files, desc="Computing Activity Masks"):
            try:
                # 调用核心分析函数
                mask, duration = calculate_rms_mask_from_file(
                    file_path, sr, clip_duration, rms_threshold
                )
                if mask.size > 0:
                    masks[file_path] = mask
                else:
                    logging.warning(f"Mask computation resulted in an empty mask for: {file_path.name}")
            except Exception as e:
                logging.error(f"Failed to compute mask for {file_path.name}: {e}")
                
        return masks
    def _get_combined_mask(self, file_paths: List[Path], stem_type: str, operation: str = "AND") -> Tuple[Union[np.ndarray, None], int]:
        """
        根据操作符计算一组文件路径的掩码的交集 (AND) 或并集 (OR)。
        返回: (合并后的掩码, 最短长度)
        """
        if not file_paths:
            return None, 0
        
        masks = []
        min_len = float('inf')
        
        for file_path in file_paths:
            mask = self.activity_masks.get(file_path)
            # 检查掩码是否存在
            if mask is None or len(mask) == 0:
                logger.debug(f"Missing mask for {stem_type} track: {file_path.name}")
                return None, 0
            masks.append(mask)
            min_len = min(min_len, len(mask))
            
        if not masks:
            return None, 0

        if operation == "AND":
            # 交集：所有都必须活跃 (默认用于 Target Tracks)
            combined_mask = np.ones(min_len, dtype=bool)
            for mask in masks:
                combined_mask &= mask[:min_len]
        
        elif operation == "OR":
            # 并集：至少一个活跃 (用于 Noise Tracks)
            combined_mask = np.zeros(min_len, dtype=bool)
            for mask in masks:
                combined_mask |= mask[:min_len]
                
        else:
            raise ValueError("Invalid operation for mask combination.")

        return combined_mask, min_len
    def _precompute_valid_starts_per_song(self) -> List[List[int]]:
        """
        为每首歌找到所有音轨共同活跃的起始秒。
        使用 (Target Tracks 的交集) AND (Noise Tracks 的并集) 逻辑。
        """
        valid_starts_list = []
        
        # 初始化统计计数器
        total_songs = len(self.song_paths)
        skipped_missing_mask_count = 0
        skipped_no_common_activity_count = 0
        
        for song_index, song_data in enumerate(tqdm(self.song_paths, desc="Precomputing Valid Start Times")):
            target_paths = song_data["target_tracks"]
            noise_paths = song_data["noise_tracks"]
            
            # 1. 计算 Target Tracks 的交集 (Target AND)
            # Target 必须同时活跃，确保目标信号是干净的。
            target_mask, _ = self._get_combined_mask(target_paths, "target", operation="AND")
            
            # 2. 计算 Noise Tracks 的并集 (Noise OR)
            # 只要有一个 Noise Track 活跃，就认为 Noise Mix 是活跃的。
            noise_mask, _ = self._get_combined_mask(noise_paths, "noise", operation="OR")
            
            
            # --- 检查和处理缺失/损坏的掩码 ---
            # 如果任一掩码（Target 或 Noise）在内部返回 None，说明有 Track 文件丢失或损坏。
            if target_mask is None or noise_mask is None:
                valid_starts_list.append([])
                skipped_missing_mask_count += 1
                # 内部的 _get_combined_mask 已经打印了 debug 信息
                continue

            # 3. 最终交集：(Target AND) AND (Noise OR)
            
            # 找出两个结果中最短的长度，以确保交集操作的安全
            final_min_len = min(len(target_mask), len(noise_mask))
            
            # 最终的共同活跃掩码：两个组合掩码在最短长度上的交集
            common_active_mask = target_mask[:final_min_len] & noise_mask[:final_min_len]

            # 找到有效起始秒数
            valid_starts = np.where(common_active_mask)[0].tolist()
            
            if not valid_starts:
                # 记录交集为空导致的跳过
                skipped_no_common_activity_count += 1
                logger.debug(f"Song {song_index} (No Activity): Target AND Noise OR resulted in no common segments.")
                
            valid_starts_list.append(valid_starts)
            
        # 统计报告
        successful_songs = total_songs - skipped_missing_mask_count - skipped_no_common_activity_count
        
        logger.info("--- Valid Start Times Precomputation Summary ---")
        logger.info(f"Total Songs Indexed: {total_songs}")
        logger.info(f"Songs with Valid Starts: {successful_songs} ({successful_songs / total_songs:.1%} of total)")
        logger.info(f"Skipped (Missing Mask/Track Error): {skipped_missing_mask_count}")
        logger.info(f"Skipped (No Common Activity Found): {skipped_no_common_activity_count}")
        logger.info("--------------------------------------------------")
        
        return valid_starts_list

    def _index_songs(self) -> List[Dict[str, Any]]:
        """
        索引 MoisesDB 歌曲，记录每个 Stem 文件夹及其包含的 Track 文件路径。
        """
        indexed_songs = []
        # 确定目标 Stem 的名称（例如 'bass'）
        target_stem_name = self.target_filename 
        # 确定所有潜在 Stem 的名称（例如 'vocals', 'drums', 'bass', ...）
        all_stem_names = self.MOISES_STEMS
        for song_dir in tqdm(self.root_directory.iterdir(), desc="Indexing MoisesDB Tracks"):
            if not song_dir.is_dir():
                continue
            song_data = {"target_tracks": [], "noise_tracks": []}
            # 记录每个 Stem 文件夹下找到的 Track 文件路径
            found_stems = {}
            
            # 遍历所有期望的 Stem 文件夹
            for stem_name in all_stem_names:
                stem_folder = song_dir / stem_name
                if stem_folder.is_dir():
                    # 收集该 Stem 文件夹内所有 .wav 文件 (即 Tracks)
                    tracks = [p for p in stem_folder.rglob('*.wav') if p.is_file()]
                    if tracks:
                        found_stems[stem_name] = tracks

            # 检查是否同时找到了目标 Stem 和至少一个噪声 Stem
            if target_stem_name in found_stems and len(found_stems) >= 2:
                
                # 组织 target 和 noise tracks
                for stem_name, tracks in found_stems.items():
                    if stem_name == target_stem_name:
                        song_data["target_tracks"].extend(tracks)
                    else:
                        song_data["noise_tracks"].extend(tracks)

                # 确保目标和噪声 Track 都不为空
                if song_data["target_tracks"] and song_data["noise_tracks"]:
                    indexed_songs.append(song_data)
                else:
                    logger.warning(f"Skipping song {song_dir.name}: Target or Noise track list is empty.")
                    
            # 否则跳过
            else:
                logger.warning(f"Skipping song {song_dir.name}: Missing target stem '{target_stem_name}' or insufficient noise stems.")
                
        return indexed_songs

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        
        # 获取索引到的歌曲数据，包含所有 Track 文件路径
        song_data = self.song_paths[index] 
        valid_starts = self.song_valid_starts[index]
        # 检查是否有有效的起始秒
        if not valid_starts:
            logger.warning(f"Song {index} has no common active segments. Retrying with a new index.")
            index = random.randint(0, len(self.song_paths) - 1)
            return self.__getitem__(index)
        
        # --- 引入 RawStems 式的动态轨道选择与 10 次尝试 ---
        # 从 song_data 中获取所有可用的 Target 和 Noise 文件路径列表
        all_target_paths = song_data["target_tracks"]
        all_noise_paths = song_data["noise_tracks"]
        # 尝试加载 10 次不同的随机轨道组合
        for attempt in range(10):
            try:
                # 1. 动态随机选择音轨组合 (模仿 RawStems 逻辑)
                # 确保我们选择的数量不超过实际可用的数量
                num_targets = random.randint(1, min(len(all_target_paths), 5))
                selected_targets = random.sample(all_target_paths, num_targets)
                
                num_others = random.randint(1, min(len(all_noise_paths), 10))
                selected_others = random.sample(all_noise_paths, num_others)
                # 2. 选择时间点（使用预计算的有效时间点）
                #logger.info(f"Successfully get valid_start with {index}")
                start_second = random.choice(valid_starts)
                offset = start_second + random.uniform(0, 1.0 - (self.clip_duration % 1.0 or 1.0))

                # 3. 加载并混合 (使用随机选择的音轨)
                target_mix = sum(load_audio(p, offset, self.clip_duration, self.sr) for p in selected_targets) / num_targets
                other_mix = sum(load_audio(p, offset, self.clip_duration, self.sr) for p in selected_others) / num_others

                # 4. 确保音频信号有效 (RMS Check)
                REALTIME_RMS_THRESHOLD = 0.002
                if not contains_audio_signal(target_mix, rms_threshold=REALTIME_RMS_THRESHOLD) or \
                   not contains_audio_signal(other_mix, rms_threshold=REALTIME_RMS_THRESHOLD):
                #    logger.info(f"Attempt {attempt+1} (Song {index}): RMS check failed for dynamic mix. Retrying track selection.")
                    continue # ➔ 失败，继续下一次 for 循环尝试新的轨道组合

                # 5. 增强、SNR 混合、归一化和长度修正 (保持原逻辑)
                # ... (这部分逻辑与您原始代码中的 MoisesDBAdapter.__getitem__ 保持一致)
                target_clean = target_mix.copy()
                target_augmented = self.stem_augmentation.apply(target_mix, self.sr) if self.apply_augmentation else target_mix
                
                mixture, target_scale, _ = mix_to_target_snr(
                    target_augmented, other_mix, random.uniform(*self.snr_range)
                )
                target_clean *= target_scale
                
                mixture_augmented = self.mixture_augmentation.apply(mixture, self.sr) if self.apply_augmentation else mixture

                max_val = np.max(np.abs(mixture_augmented)) + 1e-8
                mixture_final = mixture_augmented / max_val
                target_final = target_clean / max_val
                rescale = np.random.uniform(*DEFAULT_GAIN_RANGE)

                mixture = np.nan_to_num(mixture_final * rescale)
                target = np.nan_to_num(target_final * rescale)
                
                target = fix_length_to_duration(target, self.clip_duration, self.sr)
                mixture = fix_length_to_duration(mixture, self.clip_duration, self.sr)

                return {
                    "mixture": np.nan_to_num(mixture),
                    "target": np.nan_to_num(target)
                }

            except Exception as e:
                # 捕获加载错误（如 load_audio 失败）或其他意外错误
                logger.error(f"MoisesDB Load Error on dynamic attempt {attempt+1} for song {index}: {type(e).__name__}: {e}. Retrying track selection.")
                # 如果是加载错误，我们仍然尝试下一次随机轨道组合（因为时间点是确定的）
                continue

        # 如果 10 次动态尝试都失败（可能是轨道组合都不够响亮，或加载错误一直出现）
        logger.error(f"Failed to load data after 10 dynamic retries for song index {index}. Falling back to random index.")
        # ➔ 最终失败：回归到原有的“重试另一首歌”的策略
        index = random.randint(0, len(self.song_paths) - 1)
        return self.__getitem__(index) # 递归调用，希望下一首歌成功

    def __len__(self) -> int:
        return len(self.song_paths)
    
    
class InfiniteSampler(Sampler):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset_size = len(dataset)
        self.indexes = list(range(self.dataset_size))
        self.reset()
    
    def reset(self) -> None:
        random.shuffle(self.indexes)
        self.pointer = 0
        
    def __iter__(self):
        while True:
            if self.pointer >= self.dataset_size: self.reset()
            yield self.indexes[self.pointer]
            self.pointer += 1
            
def create_weighted_dataloader(
    rawstems_dataset: Dataset, 
    moisesdb_dataset: Dataset, 
    target_ratio: float = 0.5, # 期望 RawStems 占总抽样机会的比例
    batch_size: int = 8,
    num_workers = 32
) -> DataLoader:
    
    # 1. 拼接数据集
    combined_dataset = ConcatDataset([rawstems_dataset, moisesdb_dataset])
    
    # 2. 计算数据集的长度和起始/结束索引
    len_a = len(rawstems_dataset)
    len_b = len(moisesdb_dataset)
    total_len = len_a + len_b

    # 计算 RawStems 的权重
    weight_a = target_ratio / len_a 
    weights_a = [weight_a] * len_a

    # 计算 MoisesDB 的权重
    weight_b = (1.0 - target_ratio) / len_b
    weights_b = [weight_b] * len_b
    
    # 组合所有权重
    all_weights = weights_a + weights_b
    
    logger.info(f"Dataset A (RawStems) size: {len_a}, per-sample weight: {weight_a:.6f}")
    logger.info(f"Dataset B (MoisesDB) size: {len_b}, per-sample weight: {weight_b:.6f}")
    logger.info(f"Target ratio (A:B): {target_ratio * 100:.1f}% : {(1-target_ratio) * 100:.1f}%")
    
    # 4. 创建采样器
    # num_samples 应该非常大，以模拟无限循环的训练过程
    num_samples = int(1e6) 
    
    # Replacement=True 是 WeightedRandomSampler 的典型用法，允许重复抽样
    sampler = WeightedRandomSampler(
        weights=all_weights, 
        num_samples=num_samples, 
        replacement=True
    )
    
    # 5. 创建 DataLoader
    # 由于我们使用了 WeightedRandomSampler，不需要使用 InfiniteSampler 和 shuffle=True
    dataloader = DataLoader(
        dataset=combined_dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers
    )
    
    return dataloader