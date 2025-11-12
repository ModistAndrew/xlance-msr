import numpy as np
from data.eq_utils import apply_random_eq
from pedalboard import Pedalboard, Resample, Compressor, Distortion, Reverb, Limiter, MP3Compressor, HighpassFilter, LowpassFilter
import torch
from scipy.signal import butter, lfilter
try:
    import pyroomacoustics as pra
except Exception as e:
    print(f"[WARN] Failed to import pyroomacoustics. Reverb effects will be disabled. Reason: {e}")
else:
    from encodec import EncodecModel
    from encodec.utils import convert_audio

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

def apply_random_room_reverb(audio, sr):
    # audio 为 (C, L)，若是 (L,) 则 reshape
    if audio.ndim == 1:
        audio = audio[None, :]  # -> (1, L)

    C, L = audio.shape

    # 随机房间大小 (更大 → 更多混响尾巴)
    room_dim = np.random.uniform(3, 9, size=3)

    # 随机选择麦克风&声源位置
    room = pra.ShoeBox(room_dim, fs=sr, max_order=np.random.randint(4, 7), absorption=np.random.uniform(0.2, 0.7))

    mic_loc = np.array([
    np.random.uniform(0.5, room_dim[0]-0.5),
    np.random.uniform(0.5, room_dim[1]-0.5),
    np.random.uniform(1.0, 2.0),  # 麦克风高度 ~ 人耳高度
    ])

    source_loc = np.array([
    np.random.uniform(0.5, room_dim[0]-0.5),
    np.random.uniform(0.5, room_dim[1]-0.5),
    np.random.uniform(1.0, 2.0),  # 声源高度不必和人同高，但保持现实
    ])
    room.add_microphone(mic_loc)
    room.add_source(source_loc, signal=audio.mean(axis=0))  # 用 mean 保持左右一致的空间信息

    room.compute_rir()
    
    WET_LEVEL = np.random.uniform(0.1, 0.6)
    DRY_LEVEL = np.random.uniform(0.5, 1.0)
    wet_audio = np.vstack([
        np.convolve(audio[ch], room.rir[0][0], mode="full")[:L]
        for ch in range(C)
    ])
    wet_norm = np.max(np.abs(wet_audio)) + 1e-8

    # 最终输出 = 干声 * Dry 比例 + 归一化湿声 * Wet 比例
    out = (audio * DRY_LEVEL) + (wet_audio * (WET_LEVEL / wet_norm))
    max_out = np.max(np.abs(out)) + 1e-8
    out_normalized = out / max_out
    
    return out_normalized

class MasteringEnhancer:
    def __init__(self):
        pass

    def __call__(self, audio: np.ndarray, sr: int):
        board = Pedalboard()

        # 1) 高频空气感（温和提升）
        if np.random.rand() < 0.5:
            board.append(LowpassFilter(np.random.uniform(14000, 19000)))

        # 2) 低频收紧（避免boom）
        if np.random.rand() < 0.5:
            board.append(HighpassFilter(np.random.uniform(20, 60)))

        # 3) 轻柔总线压缩（Glue）
        if np.random.rand() < 0.7:
            board.append(Compressor(
                threshold_db=np.random.uniform(-12, -6),
                ratio=np.random.uniform(1.2, 2.0),
                attack_ms=np.random.uniform(10, 30),
                release_ms=np.random.uniform(100, 300)
            ))

        # 4) Tape 饱和感（质感 & 谐波）
        if np.random.rand() < 0.6:
            # 使用一个很小的 drive_db (例如 0.5 到 2.0 dB) 来模拟轻微的饱和
            board.append(Distortion(drive_db=np.random.uniform(0.5, 2.0)))

        # 5) 最后一层安全限制（保护不削顶）
        board.append(Limiter(threshold_db=np.random.uniform(-3, -0.1)))

        return board(audio, sample_rate=sr)
    
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
        self.p_encodec = 0.2
        self.p_mp3 = 0.3
        self.p_fm = 0.2
        self.p_room = 0.3
        self.p_limiter = 0.4
        self.p_resample = 0.3
        self.is_cuda_initialized = False
        self.mastering = MasteringEnhancer()
        self.p_mastering = 0.3
    
    def apply(self, audio: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        if np.max(np.abs(audio)) == 0:
            return audio
        
        original_length = audio.shape[-1]
        original_rms = calculate_rms(audio)
        if original_rms == 0:
            return audio
        
        normalize_scale = np.max(np.abs(audio)) + 1e-6
        audio = audio / normalize_scale
        
        board = Pedalboard()
               
        if np.random.rand() < self.p_limiter:
            board.append(Limiter(
                threshold_db=np.random.uniform(-10, 0),
                release_ms=np.random.uniform(50, 200)
            ))
            
        if np.random.rand() < self.p_resample:
            board.append(Resample(target_sample_rate=np.random.randint(16000, 44100)))
            
        if np.random.rand() < self.p_mastering:
            audio = self.mastering(audio, sample_rate)
               
        # Encodec Part
        if np.random.rand() < self.p_encodec:
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
            audio = apply_fm_effect(audio, sample_rate)
        # Room part
        elif np.random.rand() < self.p_room: 
            audio = apply_random_room_reverb(audio, sample_rate)
            
        if len(board) > 0:
            audio = board(audio, sample_rate=sample_rate)
            
        audio = fix_length_to_duration(audio, original_length)
        new_rms = calculate_rms(audio)
        
        return audio * (original_rms / new_rms)