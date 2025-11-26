# train
python train.py \
--config configs/melunet/bass.yaml

# inference
python inference.py \
--config configs/melunet/bass.yaml \
--checkpoint logs/melunet/bass/checkpoints/00260000.ckpt \
--input_dir ../../data/MSRBench/Bass/mixture/ \
--output_dir output/melunet/bass_00260000/

python inference.py \
--config configs/bsmoise/bass.yaml \
--checkpoint logs/bsmoise/bass/checkpoints/00230000.ckpt \
--input_dir ../../data/MSRBench/Bass/mixture/ \
--output_dir output/bsmoise/bass_00230000/

# eval
python eval_plus.py \
--target_dir ../../data/MSRBench/Bass/target/ \
--output_dir output/bsmoise/bass_00230000 \
--output_file output/bsmoise/bass_00230000.txt

# 1
python inference.py \
--config configs/bsrestore/orch.yaml \
--checkpoint logs/bsroformer/orch_v2/checkpoints/00300000.ckpt \
--input_dir ../../data/MSRBench/Orch/mixture/ \
--output_dir output/bsroformer/orch_v2_00300000

# 2
python inference.py \
--config configs/bsmoise/vox_mix.yaml \
--checkpoint logs/bsmoise/vox_mix/checkpoints/00480000.ckpt \
--input_dir output/bsrestore/vox_mix_large_00150000 \
--output_dir output/bsrestore_seq/vox_mix_large_00150000_vox_mix_00480000

# 3
python inference.py \
--config configs/bsrestore/vox_all.yaml \
--checkpoint logs/bsrestore/vox_stem_all/checkpoints/00670000.ckpt \
--input_dir output/bsrestore_seq/vox_mix_large_00150000_vox_mix_00480000 \
--output_dir output/bsrestore_seq/vox_mix_large_00150000_vox_mix_00480000

# direct
python inference.py \
--config configs/bsrestore/vox_hard_gan.yaml \
--checkpoint logs/bsrestore/vox_hard_large_gan/checkpoints/00240000.ckpt \
--input_dir ../../data/MSRBench/Vox/mixture/ \
--output_dir output/bsrestore/vox_hard_large_gan_00240000

python eval_plus.py \
--target_dir ../../data/MSRBench/Vox/target/ \
--output_dir output/sw/vox_00000010

python inference.py \
-c logs/denoise/vox/checkpoints/00000010.ckpt \
-i output/sw/vox_00000010 \
-o output/denoise/vox_00000010 \
--instrument vox

python inference.py \
-c logs/bsmoise/orch_mix_large_random/checkpoints/00250000.ckpt \
-i output/sw/orch_00000010 \
-o output/bsmoise_seq/orch_mix_large_random_00250000 \
--target_index '0'

python inference.py \
-c logs/bsrestore_mix/vox_codec_l1/checkpoints/00120000.ckpt \
-P logs/sw/vox/checkpoints/00000010.ckpt \
-i ../../data/MSRBench/Vox/mixture/ \
-o output/bsrestore_mix/vox_codec_l1_00120000 \
--target_index '9|10|11|12'

python inference.py \
-c logs/bsrestore_mix/vox_codec_l1/checkpoints/00120000.ckpt \
-i output/sw/vox_00000010 \
-o output/bsrestore_mix_seq/vox_codec_l1_00120000 \
--target_index '9|10|11|12'

python eval_plus.py \
-t ../../data/MSRBench/Key/target/ \
-o output/bsmoise/key_mix_00320000 \
-i '0'

python inference.py \
-p logs/denoise/vox/checkpoints/00000010.ckpt \
-c logs/sw/vox/checkpoints/00000010.ckpt \
-P logs/dereverb/vox/checkpoints/00000010.ckpt \
-i OrganizersMixture/Vocals \
-o Answer/Vocals/default \
--no-eval

python inference.py -c logs/bsrestore_mix/vox_codec_large/checkpoints/00050000.ckpt -P logs/sw/vox/checkpoints/00000010.ckpt