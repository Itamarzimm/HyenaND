<div align="center">
<h1> Hyena N-D: Multi-Dimensional Hyena for Spatial Inductive Bias </h1>
<h3>Itamar Zimerman and Lior Wolf </h3>
<h4>Tel Aviv University </h4>
</div>

## Official PyTorch Implementation of "Multi-Dimensional Hyena for Spatial Inductive Bias" (AISTATS 2024)

## Set Up Environment:
- conda create --p yourenv python=3.7
- conda activate yourenv
- pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

- git clone git@github.com:Itamarzimm/HyenaND.git 
- cd pytorch-image-models (based on "git clone https://github.com/rwightman/pytorch-image-models.git --branch v0.4.12 --depth 1")
- pip install -e .
- pip install einops
- pip install pandas

## Training:
Hyena-DeiT:

python main.py --model deit_tiny_patch16_224 --batch-size 256 --seed 0 --method hyena2d --directional_mode seq --hyena2d_filter --data-set CELEB --bce-loss --epochs 20 --warmup-epochs 1

Hyena-DeiT Hybrid:

python main.py --model deit_tiny_patch16_224 --batch-size 256 --seed 0 --method hyena2dattn --directional_mode seq --hyena2d_filter --data-set CELEB --bce-loss --epochs 20 --warmup-epochs 1

## Acknowledgement:
This repository is heavily based on [DEIT](https://github.com/facebookresearch/deit) and [Hyena](https://github.com/HazyResearch/safari). Thanks for their wonderful works.