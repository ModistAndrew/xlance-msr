# train
python train.py \
--config configs/melunet/bass.yaml

# inference
python inference.py \
--config configs/melrnn/vox.yaml \
--checkpoint logs/melrnn/vox/checkpoints/00090000.ckpt \
--input_dir /inspire/hdd/global_user/chenxie-25019/HaoQiu/MSRBench/Vocals/mixture \
--output_dir output/melrnn/90k

# eval
python eval_plus.py \
--target_dir /inspire/hdd/global_user/chenxie-25019/HaoQiu/DATASET/MSRBench/Vocals/target \
--output_dir output/melrnn/vox/90k \
--output_file output/melrnn/vox/90k.txt \
--calc_sisnr \
--calc_pesq \
--calc_aesthetics \
--calc_fad_clap \
--calc_mel_snr 

# Visualization
streamlit run audio_visualizer.py

# demo
# target::/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/data/MSRBench/Bass/target
# output::/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/jinxuanzhu/MSRKit/output/uformer/bass_00090000
# mixture::/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/data/MSRBench/Bass/mixture
# 根据MSRBench的设置，不用填写其他label
# 根据需要选择 通用音频号 及其 最小版本号
