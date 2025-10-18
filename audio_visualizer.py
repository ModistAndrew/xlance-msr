import streamlit as st
import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict

# 设置页面配置
st.set_page_config(page_title="多音频对比与频谱可视化工具 (v5.2 - 最终稳定版)", layout="wide")

# --- 辅助函数 ---

def get_safe_prefix(prefix_input):
    """从逗号分隔的字符串中安全地获取第一个有效前缀，否则返回空字符串。"""
    prefix_list = [p.strip() for p in prefix_input.split(',') if p.strip()]
    return prefix_list[0] if prefix_list else ""

def get_prefix_list(prefix_input):
    """从逗号分隔的字符串中获取所有有效前缀的列表。"""
    return [p.strip() for p in prefix_input.split(',') if p.strip()]

def get_universal_ids_regex(folder_path, prefix_list, regex_pattern, extensions=['.wav', '.mp3', '.flac']):
    # ... (此函数体保持不变) ...
    if not os.path.isdir(folder_path):
        return set() 
    
    ids = set()
    try:
        compiled_regex = re.compile(regex_pattern)
        
        for filename in os.listdir(folder_path):
            base, ext = os.path.splitext(filename)
            if ext.lower() in extensions:
                
                current_base = base
                for prefix in prefix_list:
                    if current_base.startswith(prefix):
                        current_base = current_base[len(prefix):]
                        break
                
                match = compiled_regex.match(current_base)
                
                if match and 'x' in match.groupdict():
                    file_id_x = match.group('x')
                    if file_id_x:
                        ids.add(file_id_x)
                        
    except re.error as e:
        return set()
    except Exception:
        pass
        
    return ids

@st.cache_data(show_spinner="正在扫描文件并匹配通用 ID (x)...")
def find_matched_ids(output_path, mix_path, tar_path, out_pfx_str, mix_pfx_str, tar_pfx_str, out_pat, mix_pat, tar_pat):
    """获取三个文件夹中所有通用 ID (x) 的交集，并缓存结果。"""
    
    out_prefixes_list = get_prefix_list(out_pfx_str)
    mix_prefixes_list = get_prefix_list(mix_pfx_str)
    tar_prefixes_list = get_prefix_list(tar_pfx_str)

    # 获取三个集合的通用 ID (x)
    out_ids = get_universal_ids_regex(output_path, out_prefixes_list, out_pat)
    mix_ids = get_universal_ids_regex(mix_path, mix_prefixes_list, mix_pat)
    tar_ids = get_universal_ids_regex(tar_path, tar_prefixes_list, tar_pat)
    
    # 找到三个集合的交集
    matched_x_ids = sorted(list(out_ids & mix_ids & tar_ids))
    return matched_x_ids


@st.cache_data(show_spinner="正在加载音频并生成频谱图...")
def generate_spectrogram(audio_path, title):
    # --- 修复点 2: 移除中文标题 (中文乱码问题) ---
    try:
        y, sr = librosa.load(audio_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
        ax.set(title=f'Mel Spectrogram: {title}', xlabel='Time', ylabel='Mel Freq') # <-- 标题改为英文
        ax.tick_params(labelsize=8)
        
        return fig
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"处理音频文件失败: {str(e)}")
        return None

# --- Streamlit 状态初始化 ---
if 'output_path' not in st.session_state: st.session_state.output_path = ""
if 'mixture_path' not in st.session_state: st.session_state.mixture_path = ""
if 'target_path' not in st.session_state: st.session_state.target_path = ""
if 'output_pattern' not in st.session_state: st.session_state.output_pattern = r"(?P<x>\d+)_DT(?P<y>\d+)" 
if 'mix_pattern' not in st.session_state: st.session_state.mix_pattern = r"(?P<x>\d+)_DT(?P<y>\d+)" 
if 'tar_pattern' not in st.session_state: st.session_state.tar_pattern = r"(?P<x>\d+)" 
if 'output_prefixes' not in st.session_state: st.session_state.output_prefixes = ""
if 'mix_prefixes' not in st.session_state: st.session_state.mix_prefixes = ""
if 'tar_prefixes' not in st.session_state: st.session_state.tar_prefixes = ""
if 'separator' not in st.session_state: st.session_state.separator = "_DT" # <--- 新增状态
if 'selected_y' not in st.session_state: st.session_state.selected_y = "0" 
if 'available_x_ids' not in st.session_state: st.session_state.available_x_ids = [] 
if 'selected_x_id' not in st.session_state: st.session_state.selected_x_id = None


# --- 主体 UI ---

st.title("🎼 多音频对比与频谱可视化工具 (v5.2 - 最终稳定版)")

# 1. 文件夹输入
st.header("1. 输入文件夹路径")
col_out, col_mix, col_tar = st.columns(3)
with col_out: st.session_state.output_path = st.text_input("Output 文件夹路径", st.session_state.output_path)
with col_mix: st.session_state.mixture_path = st.text_input("Mixture 文件夹路径", st.session_state.mixture_path)
with col_tar: st.session_state.target_path = st.text_input("Target 文件夹路径", st.session_state.target_path)


# 2. 模式和前缀配置
st.header("2. 配置文件名匹配模式")
st.markdown("请为每种文件类型配置**独立**的模式和前缀。模式必须包含 `(?P<x>...)`。")

# --- Output 配置 ---
col_out_p, col_out_r = st.columns(2)
with col_out_p: st.session_state.output_prefixes = st.text_input("Output 前缀", st.session_state.output_prefixes)
with col_out_r: st.session_state.output_pattern = st.text_input("Output 模式 (提取 x)", st.session_state.output_pattern, key='out_pat', help="例如: `(?P<x>\d+)_DT\d+`")

# --- Mixture 配置 ---
col_mix_p, col_mix_r = st.columns(2)
with col_mix_p: st.session_state.mix_prefixes = st.text_input("Mixture 前缀", st.session_state.mix_prefixes)
with col_mix_r: st.session_state.mix_pattern = st.text_input("Mixture 模式 (提取 x)", st.session_state.mix_pattern, key='mix_pat', help="例如: `(?P<x>\d+)_DT\d+`")

# --- Target 配置 (修复点 1: 确保 Target 模式在这里配置) ---
col_tar_p, col_tar_r = st.columns(2)
with col_tar_p: st.session_state.tar_prefixes = st.text_input("Target 前缀", st.session_state.tar_prefixes)
with col_tar_r: st.session_state.tar_pattern = st.text_input("Target 模式 (提取 x)", st.session_state.tar_pattern, key='tar_pat', help="例如: `(?P<x>\d+)`")


# 3. 核心版本号配置 (UI 结构调整，避免重复标题)
st.header("3. 核心版本号与 ID 加载")
col_sep, col_y, col_btn_y = st.columns([1, 1, 2])
with col_sep:
    st.session_state.separator = st.text_input("X和Y之间的分隔符", st.session_state.separator, help="例如：`_DT`，`_V_` 等。**注意：修改此项后必须相应修改第2部分的模式！**") 
with col_y:
    st.session_state.selected_y = st.text_input("目标小版本号 (y)", st.session_state.selected_y)

with col_btn_y:
    st.write(" ")
    if st.button("加载/刷新通用ID列表 (x)", help="清除缓存，根据新的模式和前缀重新匹配通用大序号 (x)"):
        st.cache_data.clear()
        st.session_state.selected_x_id = None 
        st.rerun()


# 4. 文件列表加载逻辑 (通用 ID x)
if st.session_state.output_path and st.session_state.mixture_path and st.session_state.target_path:
    
    # 核心：调用缓存函数，如果参数不变，它会立即返回结果
    matched_x_ids = find_matched_ids(
        st.session_state.output_path, st.session_state.mixture_path, st.session_state.target_path,
        st.session_state.output_prefixes, st.session_state.mix_prefixes, st.session_state.tar_prefixes,
        st.session_state.output_pattern, st.session_state.mix_pattern, st.session_state.tar_pattern
    )
    
    st.session_state.available_x_ids = matched_x_ids

    if not st.session_state.available_x_ids:
        st.warning("在三个文件夹中未找到匹配的通用音频 ID (x)。请检查路径、文件格式或正则表达式模式。")
    else:
        st.success(f"成功找到 {len(st.session_state.available_x_ids)} 个匹配的通用 ID (x)。")
        if st.session_state.selected_x_id not in st.session_state.available_x_ids:
            st.session_state.selected_x_id = st.session_state.available_x_ids[0] if st.session_state.available_x_ids else None


# 5. 选择音频对 (通用 ID x)
if st.session_state.available_x_ids:
    st.header("4. 选择音频 ID (x)")
    col_select, col_random = st.columns([3, 1])

    with col_select:
        new_selected_x_id = st.selectbox(
            "手动选择一个通用音频 ID (x)",
            st.session_state.available_x_ids,
            index=st.session_state.available_x_ids.index(st.session_state.selected_x_id) if st.session_state.selected_x_id in st.session_state.available_x_ids else 0
        )
        if new_selected_x_id and new_selected_x_id != st.session_state.selected_x_id:
            st.session_state.selected_x_id = new_selected_x_id
            st.rerun()

    with col_random:
        st.write(" ")
        if st.button("随机抽取 ID (x)"):
            st.session_state.selected_x_id = random.choice(st.session_state.available_x_ids)
            st.rerun()


# 6. 展示结果
if st.session_state.selected_x_id:
    selected_x_id = st.session_state.selected_x_id
    selected_y = st.session_state.selected_y
    separator = st.session_state.separator 
    
    st.header(f"5. 展示结果：ID(x) - {selected_x_id}, 版本(y) - {selected_y}")

    # --- 文件路径构造逻辑 ---
    
    def find_full_path(folder, base_name):
        """尝试查找常用扩展名，返回完整路径"""
        for ext in ['.wav', '.flac', '.mp3']:
            full_path = os.path.join(folder, base_name + ext)
            if os.path.exists(full_path):
                return full_path
        return None

    # ✅ 安全获取前缀
    out_prefix = get_safe_prefix(st.session_state.output_prefixes)
    mix_prefix = get_safe_prefix(st.session_state.mix_prefixes)
    tar_prefix = get_safe_prefix(st.session_state.tar_prefixes)

    
    # 1. Output 文件：[Output Prefix]x[SEPARATOR]y
    output_base_name = f"{out_prefix}{selected_x_id}{separator}{selected_y}"
    output_file_path = find_full_path(st.session_state.output_path, output_base_name)
    
    # 2. Mixture 文件：[Mixture Prefix]x[SEPARATOR]y
    mixture_base_name = f"{mix_prefix}{selected_x_id}{separator}{selected_y}"
    mixture_file_path = find_full_path(st.session_state.mixture_path, mixture_base_name)
    
    # 3. Target 文件：[Target Prefix]x (简化的命名)
    target_base_name = f"{tar_prefix}{selected_x_id}"
    target_file_path = find_full_path(st.session_state.target_path, target_base_name)
    
    
    # --- 三列展示 ---
    col_out, col_mix, col_tar = st.columns(3)
    
    def display_audio_col(col, title, file_path, base_name):
        with col:
            st.subheader(title)
            if file_path:
                st.markdown(f"**File:** `{os.path.basename(file_path)}`")
                
                try:
                    # 尝试用 flac 格式播放，如果文件是 flac
                    ext = os.path.splitext(file_path)[1].lower()
                    st.audio(file_path, format=f'audio/{ext.strip(".")}' if ext else 'audio/wav')
                except Exception as e:
                    st.error(f"Playback failed for {title}: {str(e)}")
                
                fig = generate_spectrogram(file_path, title)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.warning(f"File not found. Attempted base name: `{base_name}`")

    # 1. Output 列
    display_audio_col(col_out, "Output (Result)", output_file_path, output_base_name)
    
    # 2. Mixture 列
    display_audio_col(col_mix, "Mixture (Original)", mixture_file_path, mixture_base_name)

    # 3. Target 列
    display_audio_col(col_tar, "Target (Ground Truth)", target_file_path, target_base_name)