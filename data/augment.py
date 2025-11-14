import numpy as np
from data.eq_utils import apply_random_eq
from pedalboard import Pedalboard, Resample, Compressor, Distortion, Reverb, Limiter, MP3Compressor, HighpassFilter, LowpassFilter
import torch
from scipy.signal import butter, lfilter, sosfilt
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
    cutoff_freq = np.random.uniform(8000, 14000) 
    order = 5
    noise_level = np.random.uniform(0.0005, 0.005)
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    b, a = butter_lowpass(cutoff_freq, sample_rate, order=order)
    filtered_audio = np.array([lfilter(b, a, channel) for channel in audio])
    noise = np.random.normal(0, 1, filtered_audio.shape) * noise_level
    fm_audio = filtered_audio + noise
    np.clip(fm_audio, -1.0, 1.0, out=fm_audio) 
    return fm_audio

def apply_random_room_reverb(audio, sr):
    C, L = audio.shape
    room_dim = np.random.uniform(3, 9, size=3)
    room = pra.ShoeBox(room_dim, fs=sr, max_order=np.random.randint(4, 7), absorption=np.random.uniform(0.2, 0.7))
    mic_loc = np.array([
    np.random.uniform(0.5, room_dim[0]-0.5),
    np.random.uniform(0.5, room_dim[1]-0.5),
    np.random.uniform(1.0, 2.0),
    ])
    source_loc = np.array([
    np.random.uniform(0.5, room_dim[0]-0.5),
    np.random.uniform(0.5, room_dim[1]-0.5),
    np.random.uniform(1.0, 2.0),
    ])
    room.add_microphone(mic_loc)
    room.add_source(source_loc, signal=audio.mean(axis=0))
    room.compute_rir()
    WET_LEVEL = np.random.uniform(0.1, 0.6)
    DRY_LEVEL = np.random.uniform(0.5, 1.0)
    wet_audio = np.vstack([
        np.convolve(audio[ch], room.rir[0][0], mode="full")[:L]
        for ch in range(C)
    ])
    wet_norm = np.max(np.abs(wet_audio)) + 1e-8
    out = (audio * DRY_LEVEL) + (wet_audio * (WET_LEVEL / wet_norm))
    max_out = np.max(np.abs(out)) + 1e-8
    out_normalized = out / max_out
    return out_normalized

def apply_live_dt4_simple(audio: np.ndarray, sample_rate: int, snr_db: float = 20.0) -> np.ndarray:
    audio = apply_random_room_reverb(audio, sample_rate)
    audio = _apply_phone_filter(audio, sample_rate)
    audio = _add_environmental_noise(audio, sample_rate, snr_db)
    return audio

def _apply_phone_filter(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    lowcut = 300.0
    highcut = 3400.0

    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(4, [low, high], btype='band', output='sos')
    
    filtered = np.array([sosfilt(sos, channel) for channel in audio])
    return filtered

def _add_environmental_noise(audio: np.ndarray, sample_rate: int, snr_db: float) -> np.ndarray:
    C, L = audio.shape
    
    noise = _generate_noise(L, sample_rate)
    
    if C > 1:
        noise = np.tile(noise, (C, 1))

    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power > 0:
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        scale = np.sqrt(target_noise_power / noise_power)
        noise = noise * scale

    mixed = audio + noise
    
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
    
    return mixed

def _generate_noise(length: int, sample_rate: int) -> np.ndarray:
    t = np.arange(length) / sample_rate
    
    noise = np.random.normal(0, 1, length)
    
    low_freq = np.random.uniform(50, 120)
    noise += 0.3 * np.sin(2 * np.pi * low_freq * t)
    
    mid_freq = np.random.uniform(200, 800)
    noise += 0.2 * np.sin(2 * np.pi * mid_freq * t + np.random.uniform(0, 2*np.pi))
    
    b = [0.1, 0.2, 0.4, 0.2, 0.1]
    noise = lfilter(b, 1, noise)
    
    return noise

class MasteringEnhancer:
    def __init__(self):
        pass

    def __call__(self, audio: np.ndarray, sr: int):
        board = Pedalboard()

        if np.random.rand() < 0.5:
            board.append(LowpassFilter(np.random.uniform(14000, 19000)))

        if np.random.rand() < 0.5:
            board.append(HighpassFilter(np.random.uniform(20, 60)))

        if np.random.rand() < 0.7:
            board.append(Compressor(
                threshold_db=np.random.uniform(-12, -6),
                ratio=np.random.uniform(1.2, 2.0),
                attack_ms=np.random.uniform(10, 30),
                release_ms=np.random.uniform(100, 300)
            ))

        if np.random.rand() < 0.6:
            board.append(Distortion(drive_db=np.random.uniform(0.5, 2.0)))

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
        self.encodec_bandwidths = [3.0, 6.0, 12.0, 24.0]
        self.p_resample = 0.5
        self.p_mastering = 0.5
        self.p_mp3 = 0.5
        self.p_fm = 0.5
        self.p_live = 0.5
        self.p_encodec = 0.5
        self.is_cuda_initialized = False
        self.mastering = MasteringEnhancer()

    
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
            
        if np.random.rand() < self.p_resample:
            board.append(Resample(target_sample_rate=np.random.randint(16000, 44100)))
            
        if np.random.rand() < self.p_mastering:
            audio = self.mastering(audio, sample_rate)
            
        if np.random.rand() < self.p_mp3:
            board.append(MP3Compressor(vbr_quality=np.random.uniform(1.0, 9.0)))
            
        if np.random.rand() < self.p_fm:
            audio = apply_fm_effect(audio, sample_rate)

        if np.random.rand() < self.p_live: 
            audio = apply_live_dt4_simple(audio, sample_rate)
               
        if np.random.rand() < self.p_encodec:
            device = 'cpu'
            model = self.encodec_model
            target_bw = np.random.choice(self.encodec_bandwidths)
            model.set_target_bandwidth(target_bw)
            wav_tensor = torch.from_numpy(audio).float().to(device)
            wav_processed = convert_audio(wav_tensor, sample_rate, model.sample_rate, model.channels)
            wav_input = wav_processed.unsqueeze(0)
            with torch.no_grad():
                reconstructed_tensor = model(wav_input).squeeze(0)
                audio = reconstructed_tensor.cpu().numpy()
                sample_rate = model.sample_rate
            
        if len(board) > 0:
            audio = board(audio, sample_rate=sample_rate)
            
        audio = fix_length_to_duration(audio, original_length)
        new_rms = calculate_rms(audio)
        
        return audio * (original_rms / new_rms)