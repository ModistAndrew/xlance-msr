# inference
python inference.py \
--config configs/melunet/bass.yaml \
--checkpoint logs/melunet/bass/checkpoints/00260000.ckpt \
--input_dir ../../data/MSRBench/Bass/mixture/ \
--output_dir output/melunet_00260000 