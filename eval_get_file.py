import os
import glob
import re

def create_file_list(
    output_filename="file_list.txt",
    output_dir_path="./outputs", # 您的 output/mixture 文件所在的文件夹路径
    target_dir_path="./targets", # 您的 target 文件所在的文件夹路径
    output_name_pattern="mixture_(\d+)\.wav", # output/mixture 文件的命名模式，(\d+)用于捕获数字
    target_name_pattern="target_(\d+)\.wav", # target 文件的命名模式，(\d+)用于捕获数字
):
    """
    根据文件命名模式，在两个文件夹中查找匹配的文件对，并生成 target_path|output_path 列表。
    
    Args:
        output_filename (str): 生成的列表文件的名称。
        output_dir_path (str): 包含 output/mixture 文件的目录路径。
        target_dir_path (str): 包含 target 文件的目录路径。
        output_name_pattern (str): output/mixture 文件的正则表达式模式。
        target_name_pattern (str): target 文件的正则表达式模式。
    """
    
    # 检查文件夹是否存在
    if not os.path.isdir(output_dir_path):
        print(f"错误：输出文件目录不存在：{output_dir_path}")
        return
    if not os.path.isdir(target_dir_path):
        print(f"错误：目标文件目录不存在：{target_dir_path}")
        return

    # 1. 获取所有 output/mixture 文件的路径和匹配的数字
    output_files_map = {}
    output_regex = re.compile(output_name_pattern)
    
    # 使用 glob 查找所有 .wav 文件
    for filename in glob.glob(os.path.join(output_dir_path, "*.wav")):
        match = output_regex.search(os.path.basename(filename))
        if match:
            # 捕获文件名的数字部分作为键
            key = match.group(1)
            output_files_map[key] = filename

    # 2. 查找匹配的 target 文件并生成列表
    file_pairs = []
    target_regex = re.compile(target_name_pattern)

    # 使用 glob 查找所有 .wav 文件
    for filename in glob.glob(os.path.join(target_dir_path, "*.wav")):
        match = target_regex.search(os.path.basename(filename))
        if match:
            # 捕获文件名的数字部分
            key = match.group(1)
            
            # 检查 output/mixture 文件夹中是否有匹配的文件
            if key in output_files_map:
                target_path = filename
                output_path = output_files_map[key]
                
                # 注意：格式是 target_path|output_path
                file_pairs.append(f"{target_path}|{output_path}\n")

    # 3. 将结果写入文件
    if file_pairs:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.writelines(file_pairs)
        print(f"成功生成文件列表：{output_filename}")
        print(f"共找到 {len(file_pairs)} 对匹配的音频文件。")
    else:
        print("未找到任何匹配的音频文件对，请检查文件夹路径和命名模式。")

# ====================================================================
# 【请在这里修改您的配置】
# ====================================================================

# 您的输出（例如：去噪后的文件，在原metric代码中作为 output_path/preds）
# 文件夹路径
MIXTURE_DIR = "/inspire/hdd/global_user/chenxie-25019/HaoQiu/music-source-restoration/Result/Plus_Guitars_gan_1"  
# 文件命名模式。(\d+) 代表任意长度的数字。
MIXTURE_PATTERN = r"restored_mixture_(\d+)\.wav" 

# 您的目标（例如：干净的原始文件，在原metric代码中作为 target_path/target）
# 文件夹路径
TARGET_DIR = "/inspire/hdd/global_user/chenxie-25019/HaoQiu/music-source-restoration/msr_test_set/Gtr" 
# 文件命名模式
TARGET_PATTERN = r"target_(\d+)\.wav"

# 生成的列表文件的名称
OUTPUT_FILE = "audio_file_list.txt"

# ====================================================================

if __name__ == "__main__":
    # 假设您的文件都在当前工作目录下的 'outputs' 和 'targets' 文件夹中，
    # 并且文件名形如 'mixture_001.wav' 和 'target_001.wav'
    # 
    # **重要提示：运行前请修改上面的配置变量！**
    
    create_file_list(
        output_filename=OUTPUT_FILE,
        output_dir_path=MIXTURE_DIR,
        target_dir_path=TARGET_DIR,
        output_name_pattern=MIXTURE_PATTERN,
        target_name_pattern=TARGET_PATTERN,
    )