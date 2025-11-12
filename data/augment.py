import numpy as np
from data.eq_utils import apply_random_eq
from pedalboard import Pedalboard, Resample, Compressor, Distortion, Reverb, Limiter, MP3Compressor
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio
from scipy.signal import butter, lfilter

def fix_length_to_duration(target: np.ndarray, duration: float) -> np.ndarray:
    target_duration = target.shape[-1]

    if target_duration < duration:
        target = np.pad(target, ((0, 0), (0, int(duration - target_duration))), mode='constant')
    elif target_duration > duration:
        target = target[:, :int(duration)]

    return target

def calculate_rms(audio: np.ndarray) -> float:
    return np.sqrt(np.mean(audio**2))

def apply_fm_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    应用 FM 电台模拟效果：低通滤波 (带宽限制) + 噪声叠加。
    """
    
    # 1. 随机带宽限制参数 (Cutoff Freq)
    # 模拟接收不良的信号，截止频率在 8kHz 到 14kHz 之间
    cutoff_freq = np.random.uniform(8000, 14000) 
    order = 5 # 滤波器阶数，越高衰减越陡峭
    
    # 2. 噪声参数
    # 噪声幅度，模拟信号弱时的嘶嘶声
    noise_level = np.random.uniform(0.0005, 0.005) # 噪声电平，需根据您的数据进行调整
    
    # --- 低通滤波 (带宽限制) ---
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    b, a = butter_lowpass(cutoff_freq, sample_rate, order=order)
    
    # 注意：lfilter 默认只处理一维数组。如果 audio 是多通道 (C, L)，需要逐通道处理。
    if audio.ndim == 2:
        # (C, L) 格式
        filtered_audio = np.array([lfilter(b, a, channel) for channel in audio])
    else:
        # (L,) 格式
        filtered_audio = lfilter(b, a, audio)

    # --- 噪声叠加 ---
    
    # 生成白噪音，并乘以噪声电平
    noise = np.random.normal(0, 1, filtered_audio.shape) * noise_level

    # 叠加
    fm_audio = filtered_audio + noise
    
    # 确保幅度不会溢出，但由于噪声幅度小，通常不会成为问题
    np.clip(fm_audio, -1.0, 1.0, out=fm_audio)
    
    return fm_audio

class StemAugmentation:
    def __init__(self):
        pass
    
    def apply(self, audio: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        if np.max(np.abs(audio)) == 0:
            return audio
        
        original_length = audio.shape[-1]
        original_rms = calculate_rms(audio)
        if original_rms == 0:
            return audio
        
        normalize_scale = np.max(np.abs(audio)) + 1e-6
        audio = audio / normalize_scale
        
        do_eq, do_resample, do_compressor, do_distortion, do_reverb = np.random.randint(0, 2, 5)  # 5 random choices
        
        if do_eq:
            audio = apply_random_eq(audio, sample_rate)  # Assuming this preserves length
        
        board = Pedalboard()
        
        if do_resample:
            board.append(Resample(target_sample_rate=np.random.randint(8000, 32000)))
        
        if do_compressor:
            board.append(Compressor(
                threshold_db=np.random.uniform(-20, 0),
                ratio=np.random.uniform(1.5, 10.0),
                attack_ms=np.random.uniform(1, 10),
                release_ms=np.random.uniform(50, 200)
            ))
        
        if do_distortion:
            board.append(Distortion(drive_db=np.random.uniform(0, 5)))
            
        if do_reverb:
            board.append(Reverb(
                room_size=np.random.uniform(0.1, 1.0),
                damping=np.random.uniform(0.1, 1.0),
                wet_level=np.random.uniform(0.1, 0.5),
                width=np.random.uniform(0.1, 1.0)
            ))
        
        if len(board) > 0:
            audio = board(audio, sample_rate=sample_rate)
        
        audio = fix_length_to_duration(audio, original_length)
        
        new_rms = calculate_rms(audio)
        
        return audio * (original_rms / new_rms)


class MixtureAugmentation:
    
    def __init__(self):
        self.encodec_model = EncodecModel.encodec_model_48khz()
        self.encodec_model.eval()
        self.encodec_available = True
            
        self.encodec_bandwidths = [6.0, 12.0, 24.0] 
        self.p_encodec = 0
        self.p_mp3 = 0
        self.p_fm = 1
        self.is_cuda_initialized = False
    
    def apply(self, audio: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        if np.max(np.abs(audio)) == 0:
            return audio
        
        original_length = audio.shape[-1]
        original_rms = calculate_rms(audio)
        if original_rms == 0:
            return audio
        
        normalize_scale = np.max(np.abs(audio)) + 1e-6
        audio = audio / normalize_scale
        
        do_limiter, do_resample, do_codec = np.random.randint(0, 2, 3)  # 2 random choices
        
        board = Pedalboard()
        if do_limiter:
            board.append(Limiter(
                threshold_db=np.random.uniform(-10, 0),
                release_ms=np.random.uniform(50, 200)
            ))
            
        if do_resample:
            board.append(Resample(target_sample_rate=np.random.randint(8000, 32000)))

        if do_codec:
            # Encodec Part
            if self.encodec_available and np.random.rand() < self.p_encodec:
                device = 'cpu'
                # device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if device == 'cuda' and not self.is_cuda_initialized:
                    self.encodec_model = self.encodec_model.to(device)
                    self.is_cuda_initialized = True
                model = self.encodec_model
                # print(" DEBUG:Using Encodec augmentation")
                target_bw = np.random.choice(self.encodec_bandwidths)
                model.set_target_bandwidth(target_bw)
                
                wav_tensor = torch.from_numpy(audio).float().to(device)
                wav_processed = convert_audio(wav_tensor, sample_rate, model.sample_rate, model.channels)
                wav_input = wav_processed.unsqueeze(0)
                with torch.no_grad():
                    # 编码 -> 解码 (引入神经失真)
                    reconstructed_tensor = model(wav_input).squeeze(0)
                    # 将结果转回 numpy
                    audio = reconstructed_tensor.cpu().numpy()
                    # 重要：更新 sample_rate 以便后续的 Pedalboard 步骤使用 Encodec 的采样率
                    sample_rate = model.sample_rate
            # MP3 Part
            elif np.random.rand() < self.p_mp3:
                board.append(MP3Compressor(vbr_quality=np.random.uniform(1.0, 9.0)))
            # FM part
            elif np.random.rand() < self.p_fm:
                # print(" DEBUG: Using FM augmentation")
                audio = apply_fm_effect(audio, sample_rate)
            
        if len(board) > 0:
            audio = board(audio, sample_rate=sample_rate)
            
        audio = fix_length_to_duration(audio, original_length)
        new_rms = calculate_rms(audio)
        
        return audio * (original_rms / new_rms)