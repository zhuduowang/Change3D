# Change3D
The official code of **[Change3D: Revisiting Change Detection and Captioning from A Video Modeling Perspective](https://arxiv.org/pdf/2406.12847)**.  
Duowang Zhu<sup>1</sup>, Xiaohu Huang<sup>2</sup>, Haiyan Huang<sup>1</sup>, Hao Zhou<sup>3</sup>, and Zhenfeng Shao<sup>1</sup>

<sup>1</sup> LIESMARS, Wuhan University&nbsp;&nbsp; <sup>2</sup> Visual AI Lab, The University of Hong Kong&nbsp;&nbsp; <sup>3</sup> Department of Computer Vision Technology (VIS), Baidu Inc.

[[paper]](https://arxiv.org/pdf/2406.12847)

# Abstract
We present **Change3D**, a unified video-based framework for change detection and captioning. Unlike traditional methods that use separate image encoders and multiple change extractors, Change3D treats bi-temporal images as a short video with learnable perception frames. A video encoder enables direct interaction and difference detection, simplifying the architecture. Our approach supports various tasks, including binary change detection (BCD), semantic change detection (SCD), building damage assessment (BDA), and change captioning (CC). Evaluated on eight benchmarks, Change3D outperforms SOTA methods while using only **~6%–13%** of the parameters and **~8%–34%** of the FLOPs.

# Framework
![Framework](assets/framework.png)

Figure 1. Overall architectures of Change3D for Binary Change Detection, Semantic Change Detection, Building Damage Assessment, and Change Captioning.

# Performance
We conduct extensive experiments on eight public datasets: LEVIR-CD, WHU-CD, CLCD, HRSCD, SECOND, xBD, LEVIR-CC, and DUBAI-CC.

![result_of_BCD](assets/result_of_BCD.png)

![result_of_SCD](assets/result_of_SCD.png)

![result_of_BDA](assets/result_of_BDA.png)

![result_of_CC](assets/result_of_CC.png)

# TODO
- [x] Code release of Change3D for BCD.
- [ ] Code release of Change3D for SCD.
- [ ] Code release of Change3D for BDA.
- [ ] Code release of Change3D for CC.


# Usage

### Data Preparation
- Download the [LEVIR-CD](https://chenhao.in/LEVIR/), [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html), [CLCD](https://github.com/liumency/CropLand-CD), and [OSCD](https://rcdaudt.github.io/oscd/) datasets. (You can also download the processed WHU-CD dataset from [here](https://www.dropbox.com/scl/fi/8gczkg78fh95yofq5bs7p/WHU.zip?rlkey=05bpczx0gdp99hl6o2xr1zvyj&dl=0))

- Crop each image in the dataset into 256x256 patches.

- Prepare the dataset into the following structure and set its path in the [config](https://github.com/zhuduowang/ChangeViT/blob/5e08b4b2bdc94de282588562b85bb4bb6e0cd610/main.py#L146) file.
    ```
    ├─Train
        ├─A          jpg/png
        ├─B          jpg/png
        └─label      jpg/png
    ├─Val
        ├─A 
        ├─B
        └─label
    ├─Test
        ├─A
        ├─B
        └─label
    ```

## Dependency
```
pip install -r requirements.txt
```

## Training
```
python main.py --file_root LEVIR --max_steps 80000 --model_type small --batch_size 16 --lr 2e-4 --gpu_id 0
```

## Inference
```
python eval.py --file_root LEVIR --max_steps 80000 --model_type small --batch_size 16 --lr 2e-4 --gpu_id 0
```

## License
ChangeViT is released under the [CC BY-NC-SA 4.0 license](LICENSE).


## Acknowledgement
This repository is built upon [pytorchvideo](https://github.com/facebookresearch/pytorchvideo) and [A2Net](https://github.com/guanyuezhen/A2Net). Thanks for those well-organized codebases.


## Citation
```bibtex
@article{zhu2024changevit,
  title={ChangeViT: Unleashing Plain Vision Transformers for Change Detection},
  author={Zhu, Duowang and Huang, Xiaohu and Huang, Haiyan and Shao, Zhenfeng and Cheng, Qimin},
  journal={arXiv preprint arXiv:2406.12847},
  year={2024}
}
```
