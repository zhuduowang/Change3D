<div align="center">

<h2>Change3D: Revisiting Change Detection and Captioning from A Video Modeling Perspective</h2>

**_A simple and efficient framework for change detection and captioning tasks._**

[Duowang Zhu](https://scholar.google.com/citations?user=9qk9xhoAAAAJ&hl=en)<sup>1</sup>, [Xiaohu Huang](https://scholar.google.com/citations?user=sBjFwuQAAAAJ&hl=en)<sup>2</sup>, [Haiyan Huang](https://www.researchgate.net/profile/Haiyan-Huang-11)<sup>1</sup>, [Hao Zhou](https://scholar.google.com/citations?user=xZ-0R3cAAAAJ&hl=zh-CN)<sup>3</sup>, and [Zhenfeng Shao](http://www.lmars.whu.edu.cn/prof_web/shaozhenfeng/index.html)<sup>1*</sup>  

<sup>1</sup> Wuhan University&nbsp;&nbsp; <sup>2</sup> The University of Hong Kong&nbsp;&nbsp; <sup>3</sup> Bytedance

<div align="center">

[<img src="https://img.shields.io/badge/Paper-PDF-red" height="25">](https://arxiv.org/pdf/2503.18803)
[<img src="https://img.shields.io/badge/arXiv-2503.18803-green" height="25">](https://arxiv.org/pdf/2503.18803)
[<img src="https://img.shields.io/badge/Webpage-Demo-blue?logo=globe" height="25">](https://zhuduowang.github.io/Change3D)

</div>

</div>
<div align="center">
<img src="assets/parameter_distribution_with_Change3D.png" alt="Visualization" style="width: 60%; max-width: 800px;">
</div>

## ✨ Highlights

- **Unified Framework:** Supports multiple change detection and captioning tasks.
- **Highly Efficient: Uses ~6–13% of the parameters and ~8–34% of the FLOPs** compared to SOTA.
- **SOTA Performance:** Achieves SOTA performance **without complex structures**, offering an alternative to 2D models.

## 📰 News

- **[2025.03.25]** We have released all the training codes of Change3D!

- **[2025.02.27]** **Change3D has been accepted by CVPR 2025!** 🎉🎉

## 📄 Abstract

We present **Change3D**, a unified video-based framework for change detection and captioning. Unlike traditional methods that use separate image encoders and multiple change extractors, Change3D treats bi-temporal images as a short video with learnable perception frames. A video encoder enables direct interaction and difference detection, simplifying the architecture. Our approach supports various tasks, including binary change detection (BCD), semantic change detection (SCD), building damage assessment (BDA), and change captioning (CC). Evaluated on eight benchmarks, Change3D outperforms SOTA methods while using only **~6%–13%** of the parameters and **~8%–34%** of the FLOPs.

## 🎮 Framework
![Framework](assets/framework.png)

Figure 1. Overall architectures of Change3D for Binary Change Detection, Semantic Change Detection, Building Damage Assessment, and Change Captioning.

## 📝 Performance
We conduct extensive experiments on eight public datasets: LEVIR-CD, WHU-CD, CLCD, HRSCD, SECOND, xBD, LEVIR-CC, and DUBAI-CC.

![result_of_BCD](assets/result_of_BCD.png)

![result_of_SCD](assets/result_of_SCD.png)

![result_of_BDA](assets/result_of_BDA.png)

![result_of_CC](assets/result_of_CC.png)

## 🎯 How to Use

### Installation

```
conda create -n Change3D python=3.11.0
conda activate Change3D
pip install -r requirements.txt
```

#### Pretrained Weight

Download the [X3D-L](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_L.pyth) weight and put it into the root directory.

### Data Preparation

- For BCD: 
Download [LEVIR-CD](https://chenhao.in/LEVIR/), [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html) and [CLCD](https://github.com/liumency/CropLand-CD) datasets. Prepare the dataset into the following structure and crop each image into 256x256 patches.
```
    ├─Train
        ├─t1          jpg/png
        ├─t2          jpg/png
        └─label       jpg/png
    ├─Val
        ├─t1 
        ├─t2
        └─label
    ├─Test
        ├─t1
        ├─t2
        └─label
```

- For SCD:
Download [HRSCD](https://rcdaudt.github.io/hrscd/) and [SECOND](https://captain-whu.github.io/SCD/) datasets. Prepare the dataset into the following structure and crop each image into 256x256 patches.
```
    ├─Train
        ├─t1          jpg/png
        ├─t2          jpg/png
        ├─label1      jpg/png
        ├─label2      jpg/png
        └─change      jpg/png
    ├─Val
        ├─t1
        ├─t2
        ├─label1
        ├─label2
        └─change
    ├─Test
        ├─t1
        ├─t2
        ├─label1
        ├─label2
        └─change
```

- For BDA:
Download [xBD](https://xview2.org/dataset) dataset. Prepare the dataset into the following structure and crop each image into 256x256 patches.
```
    ├─Train
        ├─t1          jpg/png
        ├─t2          jpg/png
        ├─label1      jpg/png
        └─label2      jpg/png
    ├─Val
        ├─t1
        ├─t2
        ├─label1
        └─label2
    ├─Test
        ├─t1
        ├─t2
        ├─label1
        └─label2
```

- For CC:
Download [LEVIR-CC](https://github.com/Chen-Yang-Liu/RSICC) and [DUBAI-CC](https://disi.unitn.it/~melgani/datasets.html) datasets.

Prepare the dataset into the following structure and set its path in the get_dataset_path function.


### Inference with Pre-trained Models

- Download weights and data infos:

    - Download pre-trained models
        | Tokenizer | Generation Model | FID | FID cfg |
        |:---------:|:----------------|:----:|:---:|
        | [VA-VAE](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/blob/main/vavae-imagenet256-f16d32-dinov2.pt) | [LightningDiT-XL-800ep](https://huggingface.co/hustvl/lightningdit-xl-imagenet256-800ep/blob/main/lightningdit-xl-imagenet256-800ep.pt) | 2.17 | 1.35 |
        |           | [LightningDiT-XL-64ep](https://huggingface.co/hustvl/lightningdit-xl-imagenet256-64ep/blob/main/lightningdit-xl-imagenet256-64ep.pt) | 5.14 | 2.11 |

    - Download [latent statistics](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/blob/main/latents_stats.pt). This file contains the channel-wise mean and standard deviation statistics.

    - Modify config file in ``configs/reproductions`` as required. 

- Fast sample demo images:

    Run:
    ```
    bash bash run_fast_inference.sh ${config_path}
    ```
    Images will be saved into ``demo_images/demo_samples.png``, e.g. the following one:
    <div align="center">
    <img src="images/demo_samples.png" alt="Demo Samples" width="600">
    </div>

- Sample for FID-50k evaluation:
    
    Run:
    ```
    bash run_inference.sh ${config_path}
    ```
    NOTE: The FID result reported by the script serves as a reference value. The final FID-50k reported in paper is evaluated with ADM:

    ```
    git clone https://github.com/openai/guided-diffusion.git
    
    # save your npz file with tools/save_npz.py
    bash run_fid_eval.sh /path/to/your.npz
    ```

## 🎮 Train Your Own Models

 
- **We provide a 👆[detailed tutorial](docs/tutorial.md) for training your own models of 2.1 FID score within only 64 epochs. It takes only about 10 hours with 8 x H800 GPUs.** 


## ❤️ Acknowledgements

This repo is mainly built on [DiT](https://github.com/facebookresearch/DiT), [FastDiT](https://github.com/chuanyangjin/fast-DiT) and [SiT](https://github.com/willisma/SiT). Our VAVAE codes are mainly built with [LDM](https://github.com/CompVis/latent-diffusion) and [MAR](https://github.com/LTH14/mar). Thanks for all these great works.

## 📝 Citation

If you find our work useful, please consider to cite our related paper:

```
# CVPR 2025
@article{vavae,
  title={Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models},
  author={Yao, Jingfeng and Wang, Xinggang},
  journal={arXiv preprint arXiv:2501.01423},
  year={2025}
}

# NeurIPS 24
@article{fasterdit,
  title={FasterDiT: Towards Faster Diffusion Transformers Training without Architecture Modification},
  author={Yao, Jingfeng and Wang, Cheng and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2410.10356},
  year={2024}
}
```
