# System Card: OrganizersMixture

## Team Information
- **Team Name:** xlancelab
- **Members:** 
  - **Jinxuan Zhu**
    - Role: Technical Lead
    - Affiliation: X-LANCE Lab, School of Computer Science, Shanghai Jiao Tong University
    - Email: zhujinxuan@sjtu.edu.cn
  - **Hao Qiu**
    - Role: Technical Support
    - Affiliation: X-LANCE Lab, School of Computer Science, Shanghai Jiao Tong University
    - Email: quarter_peach@sjtu.edu.cn
  - **Haina Zhu**
    - Role: Lead Organizer
    - Affiliation: X-LANCE Lab, School of Computer Science, Shanghai Jiao Tong University
    - Email: hainazhu@sjtu.edu.cn
  - **Jianwei Yu**
    - Role: Advisor
    - Affiliation: Microsoft
    - Email: jianweiyu@microsoft.com
  - **Xie Chen**
    - Role: Advisor
    - Affiliation: X-LANCE Lab, School of Computer Science, Shanghai Jiao Tong University, Shanghai Innovation Institute and Jiangsu Key Lab of Language Computing
    - Email: chenxie95@sjtu.edu.cn
  - **Kai Yu**
    - Role: Advisor
    - Affiliation: X-LANCE Lab, School of Computer Science, Shanghai Jiao Tong University
    - Email: kai.yu@sjtu.edu.cn


## Model Information
- **Model Architecture:** Sequential BS-Roformers.
- **Training Methodology:** Use l1_loss together with multi_stft_resolution_loss. Hyper-parameters are in configs. Random mixture of music pieces is partially used for training.
- **Training Data:** [MoisesDB](https://github.com/moises-ai/moises-db/blob/main/moisesdb/dataset.py) and [RawStems](https://huggingface.co/datasets/yongyizang/RawStems/viewer). RawStems is cleaned and filtered by us manually.
- **Computational Resources:** We use a single H200 GPU for large scale training and a single 4090 GPU for small scale training. Inference can be done on a single 4090 GPU.
- **Other Information:** We use many pretrained models from the repo [MSST](https://github.com/ZFTurbo/Music-Source-Separation-Training), including [Roformer-SW](https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/tree/main), [dereverb](https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt) and [denoise](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.7/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt). Inference details can be found in ans.py.

