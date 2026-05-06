# SemiSAM-O1: Semi-Supervised 3D Medical Image Segmentation with Only One Labeled Template


The official repo of "[SemiSAM-O1: How far can we push the boundary of annotation-efficient medical image segmentation?](https://arxiv.org/pdf/2604.24109)".

![image](https://github.com/YichiZhang98/SemiSAM-O1/blob/main/figs/comparison.png)

## 👀 Overview

SemiSAM-O1 is an annotation-efficient semi-supervised medical image segmentation framework that learns from only one labeled volume. Instead of relying on repeated interaction with foundation models during training, SemiSAM-O1 uses the foundation model once offline as a feature extractor, propagates annotation via prototype similarity, and teratively improves pseudo-labels with uncertainty-guided refinement.

*  🚀 Strong performance under extreme low-label setting
*  ⚡️ Significantly reduced training cost
*  🔄 A self-improving training loop

## 🧩 Method

![image](https://github.com/YichiZhang98/SemiSAM-O1/blob/main/figs/framework.png)

## 🔧 Usage

### Requirements

- Python 3.10+
- PyTorch 2.0+
- SAM-Med3D-turbo checkpoint (`pretrained_ckpt/sam_med3d_turbo.pth`)
- Dependencies: `h5py`, `tensorboardX`, `SimpleITK`, `medpy`, `tqdm`, `scipy`, `scikit-image`

### Data Preparation

Organize data in HDF5 format under `data/<dataset>/`:

```
data/<dataset>/
├── data/
│   ├── sample1.h5    # keys: 'image' (3D float), 'label' (3D uint8)
│   ├── sample2.h5
│   └── ...
├── train.txt         # one sample name per line (without .h5 extension)
├── val.txt
└── test.txt
```

### Training

All commands run from `code/` directory.

```bash
cd code

# Mean Teacher (MT) backbone, LA dataset, 3 rounds
python train_SemiSAM_O1.py \
    --root_path ../data/LA \
    --exp SemiSAM_O1/LA \
    --backbone mt \
    --max_iterations 15000 \
    --num_rounds 3 \
    --seed 2025

# Uncertainty-Aware Mean Teacher (UAMT)
python train_SemiSAM_O1.py \
    --root_path ../data/LA \
    --exp SemiSAM_O1/LA \
    --backbone uamt \
    --max_iterations 15000 \
    --num_rounds 3 \
    --seed 2025

# Deep Adversarial Network (DAN)
python train_SemiSAM_O1.py \
    --root_path ../data/LA \
    --exp SemiSAM_O1/LA \
    --backbone dan \
    --max_iterations 15000 \
    --num_rounds 3 \
    --seed 2025

# Dual-Task Consistency (DTC) — automatically uses unet_3D_sdf
python train_SemiSAM_O1.py \
    --root_path ../data/LA \
    --exp SemiSAM_O1/LA \
    --backbone dtc \
    --max_iterations 15000 \
    --num_rounds 3 \
    --seed 2025
```


Models are saved to `../model/{exp}_{BACKBONE}_1label_1/{model}/`.

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--backbone` | `uamt` | SSL backbone: `mt`, `uamt`, `dan`, `dtc` |
| `--max_iterations` | `15000` | Training iterations per round |
| `--num_rounds` | `3` | Number of iterative refinement rounds |
| `--labeled_num` | `1` | Number of labeled samples |
| `--knn_k` | `5` | K for KNN pseudo-label refinement |
| `--uncertainty_quantile` | `0.9` | Uncertainty threshold for KNN refinement |
| `--num_classes` | `2` | Number of segmentation classes |
| `--sam_ckpt` | `pretrained_ckpt/sam_med3d_turbo.pth` | SAM-Med3D checkpoint |


### Testing

```bash
python test_baseline.py \
    --root_path ../data/LA \
    --exp SemiSAM_O1/LA \
    --num_classes 2 \
    --ckpt [roundN_best.pth] \
    --test_list test.txt \
    --save_pred output_dir/
```

For DTC backbone, add `--model unet_3D_sdf`.

Metrics reported: Dice, Jaccard, 95% Hausdorff Distance (95HD), Average Surface Distance (ASD).


## 🖋️ Citation

```
@article{zhang2026semisam-o1,
  title={SemiSAM-O1: How far can we push the boundary of annotation-efficient medical image segmentation?},
  author={Zhang, Yichi and Xue, Le and Xu, Bichun and Luo, Judong and Wu, Zhigang and Fu, Yu and Hu, Zixin and Cheng, Yuan and Qi, Yuan},
  journal={arXiv preprint arXiv:2604.24109},
  year={2026}
}
```
