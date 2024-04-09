# DGA

This is the official repository for “DGA: Direction-guided Attack Against Optical Aerial Detection in Camera Shooting Direction Agnostic Scenarios”.

## Requirements

```
conda create -n dga python=3.8 -y
conda activate dga

conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch

pip install -U openmim
mim install mmengine==0.6.0
mim install "mmcv==2.0.0rc4" 

cd mmdetection
pip install -v -e .

cd ../mmyolo
pip install -v -e .

cd ../mmrotate
pip install -v -e .


mmengine == 0.6.0
mmcv == 2.0.0rc4
mmdet == 3.0.0rc6
mmyolo == 3.0.0rc6
mmrotate == 1.0.0rc1
```

## Install

- Install mmrotate
- Move the camors folder to the `mmrotate/projects/` directory.


## Digital Adversarial Attack

The objective is to make adversarial attack samples absolutely similar to the original images, with minimal pixel-level differences from the source image, while requiring perturbations that are characterized by wide-ranging yet subtle disturbances.

Plan of Models:

- :heavy_check_mark: FGSM (2014)
- :heavy_check_mark: BIM (2016)
- :heavy_check_mark: PGD (2017)
- :heavy_check_mark: MIFGSM (2017)
- :heavy_check_mark: VMIFGSM (CVPR'2021)
- :clock3: [TPA](https://github.com/plpl2019/TPA) (TGRS'2023)
- :clock3: [UEA](https://github.com/LiangSiyuan21/Adversarial-Attacks-for-Image-and-Video-Object-Detection/tree/master/img_attack_with_attention) (IJCAI'2019)
- :heavy_plus_sign: [DAG](https://arxiv.org/abs/1703.08603v2) (2017)
- :heavy_plus_sign: [BAA](https://ieeexplore.ieee.org/document/9522031) (2021)
- :heavy_plus_sign: [RPAttack](https://github.com/VDIGPKU/RPAttack) (ICME'2021)

Execute the PGD digital attack on a single image and generate an adversarial sample
```bash
python projects/camors/camors/digital_attack.py \
    data/DOTA/split_ss_dota/val/images/P0170__1024__824___57.png \
    projects/camors/configs/vfnet_r50_fpn_1x_dota_airplane.py \
    --weights work_dirs/vfnet_r50_fpn_1x_dota_airplane/epoch_12.pth \
    --out-dir /public/zhouyue/outputs \
    --attack-method PGD
```

Perform the PGD digital attack on a single image and visualize the detection results post-attack
```bash
python projects/camors/camors/digital_attack.py \
    data/DOTA/split_ss_dota/val/images/P0170__1024__824___57.png \
    projects/camors/configs/vfnet_r50_fpn_1x_dota_airplane.py \
    --weights work_dirs/vfnet_r50_fpn_1x_dota_airplane/epoch_12.pth \
    --out-dir /public/zhouyue/outputs \
    --attack-method PGD \
    --draw-pred
```

Perform the PGD digital attack on the dataset and generate adversarial samples
```bash
python projects/camors/camors/digital_attack.py \
    data/DOTA/split_ss_dota/val/images/ \
    projects/camors/configs/vfnet_r50_fpn_1x_dota_airplane.py \
    --weights work_dirs/vfnet_r50_fpn_1x_dota_airplane/epoch_12.pth \
    --out-dir /public/zhouyue/outputs \
    --attack-method PGD
```

## Physical Adversarial Attack

Attacks on neural networks are often conducted through the use of stickers or patches, where the color of the patches or stickers does not need to resemble the background. However, the area covered by the stickers should not be too large, with the perturbation characterized by narrow scope and significant disturbance.

Plan of Models:

- :heavy_check_mark: [DPatch](https://github.com/veralauee/DPatch) (AAAIW'2019)
- :heavy_check_mark: [OBJ](https://gitlab.com/EAVISE/adversarial-yolo) (2019)
- :heavy_check_mark: [APPA](https://ieeexplore.ieee.org/abstract/document/9965436) (TGRS'2022)
- :clock3: [DGA](https://ieeexplore.ieee.org/abstract/document/) (TGRS'2024)
- :clock3: [Patch-Noobj](https://www.mdpi.com/2072-4292/13/20/4078) (RS'2021)
- :clock3: [APA](https://www.mdpi.com/2072-4292/14/21/5298) (RS'2022)
- :heavy_plus_sign: [APC](https://repository.uantwerpen.be/docman/irua/16bd0a/p177670.pdf)  (2020)
- :heavy_plus_sign: [AerialAttack](https://openaccess.thecvf.com/content/WACV2022/html/Du_Physical_Adversarial_Attacks_on_an_Aerial_Imagery_Object_Detector_WACV_2022_paper.html?ref=https://githubhelp.com) (WACV'2022)
- :heavy_plus_sign: [AdvSticker](https://github.com/jinyugy21/Adv-Stickers_RHDE) (TPAMI'2022)
- :heavy_plus_sign: [SOPP](https://github.com/shighghyujie/newpatch-rl) (TPAMI'2022)
- :heavy_plus_sign: [Adversarial Defense in Aerial Detection](https://robustart.github.io/long_paper/08.pdf)

Training the OBJ model
```bash
python tools/train.py projects/camors/configs/rtmdet_tiny_1x_dota_obj_airplane.py
```

Test and generate adversarial examples
```bash
python tools/test.py projects/camors/configs/rtmdet_tiny_1x_dota_obj_airplane.py \
    /home/zytx121/mmrotate/work_dirs/rtmdet_tiny_3x_dota_airplane/epoch_36.pth
```


## SJTU-4K Dataset

<img width="489" alt="image" src="https://github.com/zytx121/DGA/assets/10410257/5711a95a-f33a-4724-bc2b-2a120da3f695">

<img width="489" alt="image" src="https://github.com/zytx121/DGA/assets/10410257/6a07b3a2-80c1-4434-a5c8-581e1965343b">



To collect the data we require, we design a reasonable scheme for data capture. First, we chose 20 scenes on our campus as our experimental site, including streets and car parks, where many kinds of cars are often seen. Afterward, we used the DJI Mini 3 drone as the capturing tool. In our scheme, the flight height ranges from 20 m to 120 m, and there were 9 flight heights in total. The resolution to the raw images is 4000 $\times$ 2260 pixels. We also provide aerial images captured at different pitch angles and directions, which are closer to real scenes. Figure 1 shows several images captured by our drone with different pitch angles from the heights of 20 m, 40 m, 80 m, and 110 m, respectively. The size of the objects is quite diverse, and the number of objects in a single image is large, especially for images captured at a great height. The detailed distribution of data in the SJTU-4K dataset is shown in Figure 2. However, this size is too large, being unsuitable for our experiment. Therefore, we need to conduct some processing of the raw images. They are tailored into a smaller size of 1024 $\times$ 1024 when being attacked. Ultimately, there are 10260 training images with 57172 car objects and 6345 testing images with 19969 car objects. To cooperate with the new patch-based attack evaluation method proposed, we used rotated boxes to label vehicles. Unlike the common rotated box representation, we use a 360-degree angle representation to retain the direction information of vehicles. 

The dataset is available now. 

[GoogleDrive](https://drive.google.com/drive/folders/1LWXC-a7OM2kGbXeCp-frrMcBOFD127Hb?usp=sharing)

[jbox](https://jbox.sjtu.edu.cn/l/j1vS3y)(Code：jlfr)

## Reference

1. [APPA](https://github.com/JiaweiLian/AP-PA)
2. [mmrotate](https://github.com/open-mmlab/mmrotate)
3. [mmyolo](https://github.com/open-mmlab/mmyolo)
4. [mmdetection](https://github.com/open-mmlab/mmdetection)


## Citation

If you use this project for attacks in your research, please consider citing

```
@article{zhou2024dga,
  title={DGA: Direction-guided Attack Against Optical Aerial Detection in Camera Shooting Direction Agnostic Scenarios},
  author={Zhou, Yue and Sun, Shuqi and Jiang, Xue and Xu, Guozheng and Hu, Fengyuan and Zhang, Ze and Liu, Xingzhao},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={},
  pages={1--22},
  year={2024},
  publisher={IEEE}
}

@article{zhou2023camonet,
  title={CamoNet: A Target Camouflage Network for Remote Sensing Images Based on Adversarial Attack},
  author={Zhou, Yue and Jiang, Wanghan and Jiang, Xue and Chen, Lin and Liu, Xingzhao},
  journal={Remote Sensing},
  volume={15},
  year={2023},
  number={21}
}

```

