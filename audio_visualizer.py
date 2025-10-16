import streamlit as st
import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 设置页面配置
st.set_page_config(page_title="音频对比与频谱可视化工具 (v2.0)", layout="wide")

# --- 辅助函数 ---

# 定义一个可以清理文件名的函数，以便提取通用ID
def clean_filename_for_id(filename_base, prefixes):
    """从文件名基础部分剥离指定前缀，获取通用ID"""
    cleaned_name = filename_base
    for prefix in prefixes:
        if cleaned_name.startswith(prefix):
            cleaned_name = cleaned_name[len(prefix):]
    return cleaned_name

def get_audio_files_v2(folder_path, prefixes, extensions=['.wav', '.mp3', '.flac']):
    """获取文件夹内所有指定扩展名的音频文件的通用ID (通过剥离前缀)"""
    if not os.path.isdir(folder_path):
        return {} # 返回 ID 到 完整文件名基础 (base name) 的映射
    
    file_ids = {}
    for filename in os.listdir(folder_path):
        base, ext = os.path.splitext(filename)
        if ext.lower() in extensions:
            # 获取通用 ID
            file_id = clean_filename_for_id(base, prefixes)
            # 存储通用 ID 到 文件的基础名称 (例如: '2' -> 'mixture_2')
            file_ids[file_id] = base
    return file_ids

@st.cache_data(show_spinner="正在加载音频并生成频谱图...")
def generate_spectrogram(audio_path, title):
    """加载音频并生成梅尔频谱图，返回Matplotlib Figure对象"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        # 使用'viridis'作为默认色图，如果'magma'在某些环境下不适用
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
        ax.set(title=f'Mel-spectrogram: {title}')
        
        return fig
    except FileNotFoundError:
        return None
    except Exception:
        return None

# --- Streamlit 状态初始化 ---

if 'mixture_path' not in st.session_state:
    st.session_state.mixture_path = "/inspire/hdd/global_user/chenxie-25019/HaoQiu/music-source-restoration/msr_test_set/Bass" # 预设您的路径
if 'target_path' not in st.session_state:
    st.session_state.target_path = "/inspire/hdd/global_user/chenxie-25019/HaoQiu/music-source-restoration/Result/Bass_gan_35k" # 预设您的路径
if 'mix_prefixes' not in st.session_state:
    st.session_state.mix_prefixes = "mixture_,source_" # 混合文件常见前缀
if 'tar_prefixes' not in st.session_state:
    st.session_state.tar_prefixes = "restored_,pred_,target_" # 目标文件常见前缀
if 'matched_ids' not in st.session_state:
    st.session_state.matched_ids = {} # 存储匹配的通用ID -> (mix_base, tar_base)
if 'available_keys' not in st.session_state:
    st.session_state.available_keys = [] # 存储通用ID列表
if 'selected_key' not in st.session_state:
    st.session_state.selected_key = None

# --- 主体 UI ---

st.title("🎼 音频对比与频谱可视化工具 (v2.0)")
st.markdown("此版本包含**智能文件名匹配**功能，用于音源分离/恢复数据的可视化。")

# 1. 文件夹输入
st.header("1. 输入文件夹路径")
col_mix, col_tar = st.columns(2)
with col_mix:
    mixture_path_input = st.text_input("输入 **Mixture/原始音频** 文件夹路径", st.session_state.mixture_path)
with col_tar:
    target_path_input = st.text_input("输入 **Target/结果音频** 文件夹路径", st.session_state.target_path)
st.session_state.mixture_path = mixture_path_input
st.session_state.target_path = target_path_input


# 2. 前缀配置
st.header("2. 配置文件名匹配前缀")
st.markdown("请配置文件名中需要被**剥离**的前缀，以获得通用ID进行匹配。")
col_mix_p, col_tar_p, col_btn = st.columns([1, 1, 0.5])

with col_mix_p:
    mix_prefixes_input = st.text_input("Mixture 文件前缀 (逗号分隔)", st.session_state.mix_prefixes, key="mix_p_input")

with col_tar_p:
    tar_prefixes_input = st.text_input("Target 文件前缀 (逗号分隔)", st.session_state.tar_prefixes, key="tar_p_input")

with col_btn:
    st.write(" ") # 占位
    if st.button("加载/刷新文件列表", help="清除缓存，根据前缀重新匹配音频对"):
        st.session_state.mix_prefixes = mix_prefixes_input
        st.session_state.tar_prefixes = tar_prefixes_input
        st.cache_data.clear()
        # 重置选择，触发后续匹配逻辑
        st.session_state.selected_key = None 
        st.rerun()

# 3. 文件列表加载逻辑
if st.session_state.mixture_path and st.session_state.target_path:
    
    # 将逗号分隔的前缀字符串转换为列表
    mix_prefixes_list = [p.strip() for p in st.session_state.mix_prefixes.split(',') if p.strip()]
    tar_prefixes_list = [p.strip() for p in st.session_state.tar_prefixes.split(',') if p.strip()]

    # 获取两个文件夹的 ID -> 文件基础名称 映射
    mix_id_to_base = get_audio_files_v2(st.session_state.mixture_path, mix_prefixes_list)
    tar_id_to_base = get_audio_files_v2(st.session_state.target_path, tar_prefixes_list)
    
    # 找到共同存在的通用 ID
    matched_ids = mix_id_to_base.keys() & tar_id_to_base.keys()
    
    st.session_state.matched_ids = {
        file_id: (mix_id_to_base[file_id], tar_id_to_base[file_id])
        for file_id in matched_ids
    }
    st.session_state.available_keys = sorted(list(matched_ids))

    if not st.session_state.available_keys:
        st.warning("在两个文件夹中未找到匹配的音频对。请检查路径、文件格式或**前缀配置**。")
    else:
        st.success(f"成功找到 {len(st.session_state.available_keys)} 对匹配的音频文件ID。")
        # 确保选中的key仍然可用
        if st.session_state.selected_key not in st.session_state.available_keys:
            st.session_state.selected_key = st.session_state.available_keys[0] if st.session_state.available_keys else None


# 4. 选择音频对
st.header("3. 选择音频对")
if st.session_state.available_keys:
    
    col_select, col_random = st.columns([3, 1])

    with col_select:
        # 手动选择
        new_selected_key = st.selectbox(
            "手动选择一个通用音频ID",
            st.session_state.available_keys,
            index=st.session_state.available_keys.index(st.session_state.selected_key) if st.session_state.selected_key in st.session_state.available_keys else 0
        )
        # 如果新选择的 key 存在，则更新
        if new_selected_key and new_selected_key != st.session_state.selected_key:
            st.session_state.selected_key = new_selected_key

    with col_random:
        # 随机抽取
        st.write(" ") # 占位
        if st.button("随机抽取"):
            st.session_state.selected_key = random.choice(st.session_state.available_keys)
            st.rerun()

# 5. 展示结果
if st.session_state.selected_key and st.session_state.matched_ids:
    selected_id = st.session_state.selected_key
    
    if selected_id in st.session_state.matched_ids:
        mix_base, tar_base = st.session_state.matched_ids[selected_id]
        
        st.header(f"4. 展示结果：通用 ID - {selected_id}")
        st.markdown(f"**Mixture 文件基础名:** `{mix_base}` | **Target 文件基础名:** `{tar_base}`")

        # 路径查找函数
        def get_full_path(folder, base_name):
            """尝试查找常用扩展名，返回完整路径"""
            for ext in ['.wav', '.mp3', '.flac']:
                full_path = os.path.join(folder, base_name + ext)
                if os.path.exists(full_path):
                    return full_path
            return None

        mix_file_path = get_full_path(st.session_state.mixture_path, mix_base)
        tar_file_path = get_full_path(st.session_state.target_path, tar_base)
        
        if mix_file_path and tar_file_path:
            
            col_mix, col_tar = st.columns(2)
            
            # --- Mixture 音频展示 ---
            with col_mix:
                st.subheader("Mixture (原始/输入)")
                st.markdown(f"**路径:** `{mix_file_path}`")
                
                try:
                    st.audio(mix_file_path, format='audio/wav') # 尝试指定格式
                except Exception as e:
                    st.error(f"播放混合音频失败: {str(e)}")
                
                fig_mix = generate_spectrogram(mix_file_path, f"Mixture ({mix_base})")
                if fig_mix:
                    st.pyplot(fig_mix)
                    plt.close(fig_mix) # 避免内存泄漏
                    
            # --- Target 音频展示 ---
            with col_tar:
                st.subheader("Target (恢复/结果)")
                st.markdown(f"**路径:** `{tar_file_path}`")
                
                try:
                    st.audio(tar_file_path, format='audio/wav') # 尝试指定格式
                except Exception as e:
                    st.error(f"播放目标音频失败: {str(e)}")
                
                fig_tar = generate_spectrogram(tar_file_path, f"Target ({tar_base})")
                if fig_tar:
                    st.pyplot(fig_tar)
                    plt.close(fig_tar) # 避免内存泄漏
        else:
            st.error("无法找到选定音频对的完整文件路径，请检查文件扩展名或路径权限。")
    else:
        st.error(f"内部错误：通用 ID '{selected_id}' 未在匹配列表中找到。请刷新页面。")