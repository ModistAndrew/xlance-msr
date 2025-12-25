import os
import torch
import gradio as gr
import tempfile
import spaces
from inference_full import inference_main
from huggingface_hub import hf_hub_download

# ===== Basic config =====
USE_CUDA = torch.cuda.is_available()
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "12"))
REPO_ID = os.getenv("MODEL_REPO_ID", "chenxie95/xlance-msr-ckpt")

# Instrument to checkpoint mapping
INSTRUMENT_MAP = {
    'vox': {
        'pre': ['denoise.pth'],
        'mss': ['vox_mss.pth'],
        'post': ['dereverb.pth']
    },
    'gtr': {
        'pre': ['denoise.pth'],
        'mss': ['gtr_mss.pth'],
        'post': []
    },
    'key': {
        'pre': ['denoise.pth'],
        'mss': ['key_mss.pth'],
        'post': []
    },
    'syn': {
        'pre': ['denoise.pth'],
        'mss': ['syn_mss.pth', 'syn_mss1.pth'],
        'post': []
    },
    'bass': {
        'pre': ['denoise.pth'],
        'mss': ['bass_mss.pth'],
        'post': []
    },
    'drums': {
        'pre': ['denoise.pth'],
        'mss': ['drums_mss.pth', 'drums_mss1.pth'],
        'post': []
    },
    'perc': {
        'pre': ['denoise.pth'],
        'mss': ['perc_mss.pth', 'perc_mss1.pth'],
        'post': []
    },
    'orch': {
        'pre': ['denoise.pth'],
        'mss': ['orch_mss.pth', 'orch_mss1.pth'],
        'post': []
    }
}

# Cache for downloaded models
MODEL_CACHE = {}

def download_model(filename):
    """Download and cache a model file"""
    if filename not in MODEL_CACHE:
        print(f"Downloading {filename}...")
        MODEL_CACHE[filename] = hf_hub_download(
            repo_id=REPO_ID, 
            filename=filename,
            local_dir_use_symlinks=False
        )
    return MODEL_CACHE[filename]

def prepare_models(instrument):
    """Download all required models for the instrument"""
    config = INSTRUMENT_MAP[instrument]
    pre_models = [download_model(m) for m in config['pre']]
    mss_models = [download_model(m) for m in config['mss']]
    post_models = [download_model(m) for m in config['post']]
    return pre_models, mss_models, post_models

@spaces.GPU
def separate_audio(audio_path, instrument):
    temp_dir = tempfile.gettempdir()
    output_filename = os.path.basename(audio_path).replace('.wav', '_inference.wav')
    output_path = os.path.join(temp_dir, output_filename)
    
    pre_models, mss_models, post_models = prepare_models(instrument)

    class Args:
        checkpoint_pre = pre_models
        checkpoint = mss_models
        checkpoint_post = post_models
        input_dir = audio_path
        output_dir = output_path
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = BATCH_SIZE
    args = Args()
    
    inference_main(args)
    return output_path

# ===== Gradio UI =====
with gr.Blocks() as demo:
    gr.Markdown("# 🎵 Audio Separation Tool")
    gr.Markdown("Extract instrument tracks from audio using XLance-MSR model")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Input Audio"
            )
            
            instrument_select = gr.Dropdown(
                choices=list(INSTRUMENT_MAP.keys()),
                value="vox",
                label="Select Instrument",
                info="Supports 8 instrument types"
            )
            
            process_btn = gr.Button("Start Separation", variant="primary")
            
        with gr.Column():
            audio_output = gr.Audio(
                type="filepath",
                label="Separated Result",
                show_download_button=True
            )
            
            status = gr.Textbox(
                label="Status",
                value="Ready",
                interactive=False
            )
    
    # Button click
    process_btn.click(
        fn=separate_audio,
        inputs=[audio_input, instrument_select],
        outputs=[audio_output]
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["examples/forget.mp3", "vox"],
            ["examples/forget.mp3", "drums"],
            ["examples/sonata.mp3", "key"],
            ["examples/sonata.mp3", "orch"],
        ],
        inputs=[audio_input, instrument_select],
        label="Example Audios",
        examples_per_page=4,
    )

# Queue: keep a small queue to avoid OOM
demo.queue(max_size=8)
demo.launch()