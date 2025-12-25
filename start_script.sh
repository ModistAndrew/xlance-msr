# inference_full (for inference on full audio)
python inference_full.py \
-p checkpoints/denoise.pth \
-c checkpoints/vox_mss.pth \
-P checkpoints/dereverb.pth \ # only vox need it
-i test/input \
-o test/output \
--device cuda \
--batch_size 4

# another example (eight instruments are supported in total)
python inference_full.py \
-p checkpoints/denoise.pth \
-c checkpoints/drums_mss.pth checkpoints/drums_mss1.pth \ # use multiple checkpoints for some instruments
-i test/input \
-o test/output \
--device cuda \
--batch_size 4

# ans (for reproducing our results)
python ans.py -i OrganizersMixture/Vocals/ -o Answer/Vocals/ans --instrument vox

# train (see the original MSRKit repo for more details)
python train.py \
--config configs/bsmoise/bass_mix_large.yaml

# the following commands are internal functions, just for reference

# eval_plus
python eval_plus.py \
--target_dir ../../data/MSRBench/Bass/target/ \
--output_dir output/bsmoise/bass_00230000 \
--output_file output/bsmoise/bass_00230000.txt

# inference
python inference.py \
-p logs/denoise/vox/checkpoints/00000010.ckpt \
-c logs/sw/vox/checkpoints/00000010.ckpt \
-P logs/dereverb/vox/checkpoints/00000010.ckpt \
-i OrganizersMixture/Vocals \
-o Answer/Vocals/default \
--no-eval

# unwrap
python unwrap.py -i checkpoints/bass_mss.ckpt -o checkpoints/bass_mss.pth