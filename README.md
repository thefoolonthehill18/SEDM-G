# Mitigating Exposure Bias in Discriminator Guided Diffusion Models

Codebase for SEDM-G++.

<img src="https://github.com/thefoolonthehill18/SEDM-G/blob/main/fig/overview.png" alt="overview" width="500"/>

## Abstract

Diffusion Models have demonstrated remarkable performance in image generation. However, their demanding computational requirements for training have prompted ongoing efforts to enhance the quality of generated images through modifications in the sampling process. A recent approach, known as Discriminator Guidance, seeks to bridge the gap between the model score and the data score by incorporating an auxiliary term, derived from a discriminator network. We show that despite significantly improving sample quality, this technique has not resolved the persistent issue of Exposure Bias and we propose SEDM-G++, which incorporates a modified sampling approach, combining Discriminator Guidance and Epsilon Scaling. Our proposed approach outperforms the current state-of-the-art, by achieving an FID score of 1.73 on the unconditional CIFAR-10 dataset.


## Getting Started

### 1. Download pre-trained EDM score model
List of available pre-trained models [here](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/). 

```
wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
```

### 2. Download pre-trained discriminator and classifier network (from [DG](https://github.com/alsdudrla10/DG) repo)

```
gdown https://drive.google.com/uc?id=1wNIVxC2EEn4nRujTn_lhfdfoGtJVCtkB
gdown https://drive.google.com/uc?id=1nGx9wfamOMJIjaQTGtk47fPZl_pPFa79
```

### 3. Download CIFAR-10 stat files for FID evaluation

```
gdown https://drive.google.com/uc?id=1PMGteKHkXxmh5bWtOCo3AcUyHqfyzZ4g
```

### 4. Generate samples using SEDM-G++

```
torchrun --standalone --nproc_per_node=2 generate.py --network /path/to/edm-cifar10-32x32-uncond-vp.pkl \\
--outdir=generated_samples --seeds=0-49999 --discriminator_ckpt=/path/to/discriminator_60.pt \\
 --pretrained_classifier_ckpt=/path/to/32x32_classifier.pt --eps_scaler=1.004 --dg_weight_1st_order=1.67
```

### 5. Evaluate FID
```
python3 fid_npzs.py --ref=/path/to/cifar10-32x32.npz --num_samples=50000 --images=/samples
```

## Uncoordinated samples from SEDM-G++
<img src="https://github.com/thefoolonthehill18/SEDM-G/blob/main/fig/uncoordinated_samples.jpg" alt="uncoordinated_samples" width="700"/>

## References

This codebase builds heavily upon the work of:
- Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35, 26565-26577.
- Kim, D., Kim, Y., Kwon, S.J., Kang, W. and Moon, I.C. (2023). Refining generative process with discriminator guidance in score-based diffusion models. In Proceedings of the 40th International Conference on Machine Learning, 2023.
- Ning, M., Li, M., Su, J., Salah, A.A. and Ertugrul, I.O., (2023). Elucidating the Exposure Bias in Diffusion Models. arXiv preprint arXiv:2308.15321.
