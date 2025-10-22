# train
python train.py \
--config configs/melunet/bass.yaml

# inference
python inference.py \
--config configs/melunet/bass.yaml \
--checkpoint logs/melunet/bass/checkpoints/00260000.ckpt \
--input_dir ../../data/MSRBench/Bass/mixture/ \
--output_dir output/melunet/bass_00260000/

# eval
python eval_plus.py \
--target_dir ../../data/MSRBench/Bass/target/ \
--output_dir output/melunet/bass_00260000/ \
--output_file output/melunet/bass_00260000.txt