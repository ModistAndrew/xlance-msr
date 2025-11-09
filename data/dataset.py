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
from data.moise_taxonomy import get_banned_other_pairs, get_target_stem_pairs

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
        clip_duration: float = 3.0,
        snr_range: Tuple[float, float] = (0.0, 10.0),
        apply_augmentation: bool = True,
        rms_threshold: float = -40.0,
        no_mixture: bool = False,
        moisesdb: bool = False,
        random_mixture: bool = False,
    ) -> None:
        self.root_directory = Path(root_directory)
        self.sr = sr
        self.clip_duration = clip_duration
        self.snr_range = snr_range
        self.apply_augmentation = apply_augmentation
        self.rms_threshold = rms_threshold
        self.no_mixture = no_mixture
        self.random_mixture = random_mixture
        
        if moisesdb:
            self.target_stems = get_target_stem_pairs(target_stem)
            self.allowed_others = ["vocals", "bass", "drums", "guitar", "other_plucked", "percussion", "piano", "other_keys", "bowed_strings", "wind", "other"]
            self.banned_others = get_banned_other_pairs(target_stem)
        else:
            target_stem_parts = target_stem.split("_")
            target_stem_1 = target_stem_parts[0].strip()
            target_stem_2 = target_stem_parts[1].strip() if len(target_stem_parts) > 1 else None
            self.target_stems = [(target_stem_1, target_stem_2)]
            self.allowed_others = ["Kbs", "Gtr", "Bass", "Voc", "Synth", "Rhy", "Orch"]
            assert target_stem_1 in self.allowed_others
            self.banned_others = []
        
        logger.info(f"Scanning '{self.root_directory}' for songs containing stem '{target_stem}'...")
        self.folders = []
        for song_dir in self.root_directory.iterdir():
            if song_dir.is_dir():
                for (target_stem_1, target_stem_2) in self.target_stems:
                    target_path = song_dir / target_stem_1
                    if target_stem_2:
                        target_path /= target_stem_2

                    if target_path.exists() and target_path.is_dir():
                        self.folders.append(song_dir)
                        break
        
        if not self.folders:
            raise FileNotFoundError(f"No subdirectories in '{self.root_directory}' were found containing the stem path '{target_stem}'. "
                                    f"Please check your directory structure.")
        logger.info(f"Found {len(self.folders)} song folders.")

        self.audio_files = self._index_audio_files()
        if not self.audio_files: raise ValueError("No audio files found.")
        logger.info(f"Indexed {len(self.audio_files)} audio files.")
            
        self.activity_masks = self._compute_activity_masks()
        self._filter_activity_masks()
        logger.info(f"{len(self.audio_files)} audio files after filtering.")
        
        self.stem_augmentation = StemAugmentation()
        self.mixture_augmentation = MixtureAugmentation()
        
    def load_audio(self, file_path: Path, offset: float, duration: float, sr: int, aug: bool) -> np.ndarray:
        audio, _ = librosa.load(file_path, sr=sr, offset=offset, duration=duration, mono=False)
        if len(audio.shape) == 1: audio = audio.reshape(1, -1)
        if audio.shape[1] == 0: return np.zeros((2, int(sr * duration)))
        if audio.shape[0] == 1: audio = np.vstack([audio, audio])
        if aug and self.apply_augmentation:
            audio = self.stem_augmentation.apply(audio, self.sr)
        return audio

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
                logger.warning(f"No RMS data found for {path_str}")
        return activity_masks
    
    def _filter_activity_masks(self) -> None:
        def filter_stem(stem: Path) -> bool:
            if not self._find_common_valid_start_seconds([stem]):
                # logger.warning(f"Skipping {stem} due to silence.")
                return False
            return True
        def filter_song(song: Dict[str, List[Path]]) -> bool:
            if song["target_stems"] and song["others"]:
                return True
            # logger.warning(f"Skipping {song} due to empty or invalid audio.")
            return False
        for song in self.audio_files:
            song["target_stems"] = list(filter(filter_stem, song["target_stems"]))
            song["others"] = list(filter(filter_stem, song["others"]))
        self.audio_files = list(filter(filter_song, self.audio_files))

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
            for (target_stem_1, target_stem_2) in self.target_stems:
                target_folder = folder / target_stem_1
                if target_stem_2: target_folder /= target_stem_2
                
                if target_folder.exists():
                    song_dict["target_stems"].extend(p for p in target_folder.rglob('*') if p.suffix.lower() in AUDIO_EXTENSIONS)
                
            for p in folder.rglob('*'):
                if p.suffix.lower() in AUDIO_EXTENSIONS:
                    try:
                        relative_path = p.relative_to(folder)
                        parts = relative_path.parts
                        if not (len(parts) > 0 and parts[0] in self.allowed_others):
                            # logger.warning(f"Skipping {p} due to unknown stem.")
                            raise ValueError
                        for (target_stem_1, target_stem_2) in self.target_stems + self.banned_others:
                            if len(parts) > 0 and parts[0] == target_stem_1 and (target_stem_2 is None or (len(parts) > 1 and parts[1] == target_stem_2)):
                                raise ValueError
                        song_dict["others"].append(p)
                    except ValueError:
                        continue
            
            if song_dict["target_stems"] and song_dict["others"]:
                indexed_songs.append(song_dict)
            # else:
                # logger.warning(f"Skipping {folder} due to empty or invalid audio.")
        return indexed_songs
    
    def load_other_audio_randomly(self, index: int, offset: float, duration: float, sr: int, aug: bool) -> np.ndarray:
        song_dict = self.audio_files[index]
        selected = random.choice(song_dict["others"])
        valid_starts = self._find_common_valid_start_seconds([selected])
        start_second = random.choice(valid_starts)
        offset = start_second + random.uniform(0, 1.0 - (self.clip_duration % 1.0 or 1.0))
        return self.load_audio(selected, offset, duration, sr, aug)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        song_dict = self.audio_files[index]
        
        for _ in range(100):
            num_targets = random.randint(1, min(len(song_dict["target_stems"]), 5))
            selected_targets = random.sample(song_dict["target_stems"], num_targets)
            
            if not self.no_mixture and not self.random_mixture:
                num_others = random.randint(1, min(len(song_dict["others"]), 10))
                selected_others = random.sample(song_dict["others"], num_others)
                valid_starts = self._find_common_valid_start_seconds(selected_targets + selected_others)
            else:
                valid_starts = self._find_common_valid_start_seconds(selected_targets)

            if valid_starts:
                start_second = random.choice(valid_starts)
                offset = start_second + random.uniform(0, 1.0 - (self.clip_duration % 1.0 or 1.0))
                
                target_mix = sum(self.load_audio(p, offset, self.clip_duration, self.sr, False) for p in selected_targets) / num_targets # aug later
                if not self.no_mixture and not self.random_mixture:
                    other_mix = sum(self.load_audio(p, offset, self.clip_duration, self.sr, True) for p in selected_others) / num_others
                elif self.random_mixture:
                    num_others = random.randint(1, 10)
                    selected_indices = random.sample(range(len(self.audio_files)), num_others)
                    other_mix = sum(self.load_other_audio_randomly(index, offset, self.clip_duration, self.sr, True) for index in selected_indices) / num_others
                else:
                    other_mix = np.zeros_like(target_mix)
                
                if not contains_audio_signal(target_mix) or not (contains_audio_signal(other_mix) or self.no_mixture):
                    # logger.warning(f"Skipping {song_dict} due to empty or invalid audio.")
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

        # logger.warning(f"No valid audio found for {song_dict}. Skipping.")
        return self.__getitem__(random.randint(0, len(self.audio_files) - 1))

    def __len__(self) -> int:
        return len(self.audio_files)


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