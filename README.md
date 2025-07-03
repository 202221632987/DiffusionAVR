# DiffusionAVR: Automated Software Vulnerability Repair Based on Diffusion Models

<p aligh="center"> This repository contains the code and data for <b>DiffusionAVR: Automated Software Vulnerability Repair Based on Diffusion Models</b> </p>

## Introduction

Recent approaches based on pre-trained models have achieved strong results in automated vulnerability repair. However, these methods heavily rely on the autoregressive generation paradigm and beam search. We observe that their performance drops significantly when the beam size is reduced, indicating vulnerability to error accumulation during sequence generation.

To address this issue, we propose <b>DiffusionAVR</b>, a novel method for automated software vulnerability repair based on diffusion models. Our approach eliminates the need for autoregressive decoding by employing an encoder-only architecture and a reverse denoising process. Through iterative refinement, DiffusionAVR can generate robust and diverse patch candidates without suffering from error accumulation.

Extensive experiments on real-world C/C++ vulnerability repair datasets show that DiffusionAVR outperforms existing methods in both effectiveness and efficiency. The number of correctly repaired vulnerabilities increases from 79 (with beam size = 1) to 117, and the average patch generation time is reduced from 0.4s to 0.09s.

----------

## Contents  
1. [Dataset](#Dataset)   
2. [Requirement](#Requirement)  
3. [Code](#Code)  

## Dataset

The Dataset we used in the paper:

1.CVEFixes (Bhandari et al.) 

2.Big-Vul (Fan et al.)

## Requirement

```bash
git clone https://github.com/202221632987/DiffusionAVR.git
cd DiffusionAVR
pip install -r requirements.txt
```


## Code

1.Launch Training 

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=12231 train.py \
  --checkpoint_path diffusion_models/diffuAVR \
  --dataset CVEfixed \
  --data_dir MickyMike/cvefixes_bigvul \
  --data_split_num 0 \
  --vocab codebert-base \
  --use_plm_init no \
  --lr 0.0001 \
  --use_fp16 True \
  --batch_size 128 \
  --microbatch 20 \
  --diffusion_steps 2000 \
  --noise_schedule sqrt \
  --schedule_sampler lossaware \
  --resume_checkpoint none \
  --seq_len 768 \
  --hidden_t_dim 768 \
  --seed 110 \
  --hidden_dim 768 \
  --learning_steps 50000 \
  --save_interval 5000 \
  --config_name codebert-base \
  --notes none \
  --learned_mean_embed True \
  --denoise True \
  --denoise_rate 0.5 \
  --reg_rate 0.0
```

2.Run Inference

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 sample.py \
  --model_path diffusion_models/ddiffuAVR/ema_0.9999_0500000.pt \
  --step 10 \
  --batch_size 5 \
  --start_n 0 \
  --seed2 110 \
  --split test \
  --out_dir generation_outputs \
  --top_p -1 \
  --rejection_rate 0.0 \
  --clamp_step 0 \
  --note none
```
