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

Coming soon.

## 🖋️ Citation

```
@article{zhang2026semisam-o1,
  title={SemiSAM-O1: How far can we push the boundary of annotation-efficient medical image segmentation?},
  author={Zhang, Yichi and Xue, Le and Xu, Bichun and Luo, Judong and Wu, Zhigang and Fu, Yu and Hu, Zixin and Cheng, Yuan and Qi, Yuan},
  journal={arXiv preprint arXiv:2604.24109},
  year={2026}
}
```
