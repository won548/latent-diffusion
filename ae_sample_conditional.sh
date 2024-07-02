#!/bin/bash

# Define the range of values for ddim_eta and scale
ddim_eta_values=(0.0 0.2 0.5 1.0)  # You can add more values if needed
scale_values=(0.0 1.0 3.0 5.0)    # You can add more values if needed

# Define the list of checkpoint files
ckpt_files=(
  "2024-06-19T14-18-06_cspine-ldm-vq-f8-cond"
#  "2024-06-18T15-09-49_cspine-ldm-vq-f8-cond"
)

# Loop through each combination of ckpt, ddim_eta, and scale
for ckpt in "${ckpt_files[@]}"
do
  for ddim_eta in "${ddim_eta_values[@]}"
  do
    for scale in "${scale_values[@]}"
    do
      echo "Running with ckpt=$ckpt, ddim_eta=$ddim_eta and scale=$scale"
      CUDA_VISIBLE_DEVICES=0 python sample_conditional.py \
        --ckpt_path /media/NAS/nas_32/dongkyu/Workspace/latent-diffusion/logs \
        --ckpt $ckpt \
        --ddim_steps 200 \
        --ddim_eta $ddim_eta \
        --scale $scale \
        --n_samples 40
    done
  done
done