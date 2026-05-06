"""
SemiSAM-O1: Iterative Pseudo-Label Refinement for One-Label Semi-Supervised 3D Segmentation

Supports multiple SSL backbones: MT, UAMT, DAN, DTC
Shared pipeline: SAM-Med3D feature extraction -> prototype pseudo-labels -> iterative training + KNN refinement
"""

import argparse
import logging
import os
import random
import sys
import time
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dataloaders.brats2019 import (RandomCrop, RandomRotFlip,
                                   ToTensor, TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, ramps

from segment_anything.build_sam3D import sam_model_registry3D

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LA')
parser.add_argument('--exp', type=str, default='SemiSAM_O1/LA')
parser.add_argument('--backbone', type=str, default='uamt',
                    choices=['mt', 'uamt', 'dan', 'dtc'],
                    help='SSL backbone: mt, uamt, dan, dtc')
parser.add_argument('--model', type=str, default='unet_3D')
parser.add_argument('--max_iterations', type=int, default=15000)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--DAN_lr', type=float, default=0.0001)
parser.add_argument('--patch_size', nargs=3, type=int, default=[128, 128, 128])
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--labeled_bs', type=int, default=1)
parser.add_argument('--labeled_num', type=int, default=1)
parser.add_argument('--ema_decay', type=float, default=0.99)
parser.add_argument('--consistency', type=float, default=0.1)
parser.add_argument('--consistency_rampup', type=float, default=200.0)
parser.add_argument('--num_rounds', type=int, default=3)
parser.add_argument('--knn_k', type=int, default=5)
parser.add_argument('--uncertainty_quantile', type=float, default=0.9)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--sam_ckpt', type=str, default='pretrained_ckpt/sam_med3d_turbo.pth')
parser.add_argument('--prompt', type=str, default='unc')
parser.add_argument('--image_key', type=str, default='image')
parser.add_argument('--binary_label', action='store_true')

args = parser.parse_args()

# Auto-select model for DTC
if args.backbone == 'dtc':
    args.model = 'unet_3D_sdf'


def _binarize(label):
    """Merge all FG classes into class 1 when --binary_label is set."""
    if args.binary_label:
        return (label > 0).astype(np.uint8)
    return label.astype(np.uint8)


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


# ============================================================
# SAM-Med3D Feature Extraction (shared across all backbones)
# ============================================================

def load_sam_encoder(ckpt_path, device='cuda'):
    sam_model = sam_model_registry3D['vit_b_ori'](checkpoint=None).to(device)
    model_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = model_dict['model_state_dict']
    sam_model.load_state_dict(state_dict)
    sam_model.eval()
    return sam_model


def extract_sam_features_single(sam_model, image_3d, patch_size=[128, 128, 128], device='cuda', crop_center=None):
    w, h, d = image_3d.shape
    pw = max(patch_size[0] - w, 0)
    ph = max(patch_size[1] - h, 0)
    pd = max(patch_size[2] - d, 0)
    if pw > 0 or ph > 0 or pd > 0:
        image_3d = np.pad(image_3d,
                          [(pw // 2, pw - pw // 2),
                           (ph // 2, ph - ph // 2),
                           (pd // 2, pd - pd // 2)],
                          mode='constant', constant_values=0)
    ww, hh, dd = image_3d.shape
    if crop_center is not None:
        cw, ch, cd = crop_center[0] + pw//2, crop_center[1] + ph//2, crop_center[2] + pd//2
        w1 = max(min(cw - patch_size[0]//2, ww - patch_size[0]), 0)
        h1 = max(min(ch - patch_size[1]//2, hh - patch_size[1]), 0)
        d1 = max(min(cd - patch_size[2]//2, dd - patch_size[2]), 0)
    else:
        w1 = (ww - patch_size[0]) // 2
        h1 = (hh - patch_size[1]) // 2
        d1 = (dd - patch_size[2]) // 2
    patch = image_3d[w1:w1 + patch_size[0], h1:h1 + patch_size[1], d1:d1 + patch_size[2]]
    patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = sam_model.image_encoder(patch_tensor)
    spatial_feat = feat[0].cpu()
    global_feat = spatial_feat.mean(dim=[1, 2, 3])
    global_feat = F.normalize(global_feat, dim=0)
    return spatial_feat, global_feat


def extract_all_features(sam_model, data_dir, image_list, patch_size, device='cuda', image_key='image'):
    all_spatial = []
    all_global = []
    for name in tqdm(image_list, desc="Extracting SAM features"):
        h5f = h5py.File(os.path.join(data_dir, "data", name + ".h5"), 'r')
        image = h5f[image_key][:]
        h5f.close()
        spatial, glob = extract_sam_features_single(sam_model, image, patch_size, device)
        all_spatial.append(spatial)
        all_global.append(glob)
    all_global = torch.stack(all_global)
    return all_spatial, all_global


# ============================================================
# Round 0: Prototype-based Pseudo-Label Generation (shared)
# ============================================================

def _crop_patch_to_original(pseudo_patch, image_shape, patch_size):
    """Map a patch-sized pseudo-label back to original image coordinates."""
    w, h, d = image_shape
    full_pseudo = np.zeros(image_shape, dtype=np.uint8)
    pw_img = max(patch_size[0] - w, 0)
    ph_img = max(patch_size[1] - h, 0)
    pd_img = max(patch_size[2] - d, 0)
    pad_w, pad_h, pad_d = pw_img // 2, ph_img // 2, pd_img // 2
    ww_p = max(w + pw_img, patch_size[0])
    hh_p = max(h + ph_img, patch_size[1])
    dd_p = max(d + pd_img, patch_size[2])
    w1 = (ww_p - patch_size[0]) // 2
    h1 = (hh_p - patch_size[1]) // 2
    d1 = (dd_p - patch_size[2]) // 2
    sw = max(pad_w - w1, 0); sh = max(pad_h - h1, 0); sd = max(pad_d - d1, 0)
    dw = max(w1 - pad_w, 0); dh = max(h1 - pad_h, 0); dd2 = max(d1 - pad_d, 0)
    cw = min(patch_size[0] - sw, w - dw)
    ch = min(patch_size[1] - sh, h - dh)
    cd = min(patch_size[2] - sd, d - dd2)
    if cw > 0 and ch > 0 and cd > 0:
        full_pseudo[dw:dw+cw, dh:dh+ch, dd2:dd2+cd] = \
            pseudo_patch[sw:sw+cw, sh:sh+ch, sd:sd+cd]
    return full_pseudo


def generate_initial_pseudo_labels(sam_model, data_dir, image_list, labeled_num,
                                   patch_size, num_classes=2, device='cuda', image_key='image'):
    logging.info("=== Round 0: Generating initial pseudo-labels (num_classes=%d) ===", num_classes)
    all_spatial, all_global = extract_all_features(
        sam_model, data_dir, image_list, patch_size, device, image_key=image_key)

    # Compute per-class prototypes from labeled sample(s) using center crop
    class_protos_list = {c: [] for c in range(num_classes)}
    for idx in range(labeled_num):
        h5f = h5py.File(os.path.join(data_dir, "data", image_list[idx] + ".h5"), 'r')
        label = _binarize(h5f['label'][:])
        h5f.close()

        spatial_feat = all_spatial[idx]
        feat_res = spatial_feat.shape[1:]
        w, h, d = label.shape
        pw = max(patch_size[0] - w, 0)
        ph = max(patch_size[1] - h, 0)
        pd = max(patch_size[2] - d, 0)
        if pw > 0 or ph > 0 or pd > 0:
            label_padded = np.pad(label,
                                  [(pw // 2, pw - pw // 2),
                                   (ph // 2, ph - ph // 2),
                                   (pd // 2, pd - pd // 2)],
                                  mode='constant', constant_values=0)
        else:
            label_padded = label
        ww, hh, dd = label_padded.shape
        w1 = (ww - patch_size[0]) // 2
        h1 = (hh - patch_size[1]) // 2
        d1 = (dd - patch_size[2]) // 2
        label_crop = label_padded[w1:w1+patch_size[0], h1:h1+patch_size[1], d1:d1+patch_size[2]]
        for c in range(1, num_classes):
            binary_label = (np.abs(label_crop - c) < 0.5).astype(np.float32)
            binary_low = torch.from_numpy(binary_label).float().unsqueeze(0).unsqueeze(0)
            binary_low = F.interpolate(binary_low, size=list(feat_res), mode='nearest')[0, 0]
            fg_mask = (binary_low > 0.5).float()
            if fg_mask.sum() > 0:
                proto = (spatial_feat * fg_mask.unsqueeze(0)).sum(dim=[1, 2, 3]) / (fg_mask.sum() + 1e-8)
                class_protos_list[c].append(F.normalize(proto, dim=0))
        bg_label = (label_crop < 0.5).astype(np.float32)
        bg_low = torch.from_numpy(bg_label).float().unsqueeze(0).unsqueeze(0)
        bg_low = F.interpolate(bg_low, size=list(feat_res), mode='nearest')[0, 0]
        bg_mask = (bg_low > 0.5).float()
        if bg_mask.sum() > 0:
            proto = (spatial_feat * bg_mask.unsqueeze(0)).sum(dim=[1, 2, 3]) / (bg_mask.sum() + 1e-8)
            class_protos_list[0].append(F.normalize(proto, dim=0))

    prototypes = {}
    for c in range(num_classes):
        if len(class_protos_list[c]) > 0:
            prototypes[c] = F.normalize(torch.stack(class_protos_list[c]).mean(dim=0), dim=0)
    feat_dim = all_spatial[0].shape[0]
    valid_classes = sorted([c for c in range(1, num_classes) if c in prototypes])
    logging.info("Round 0: valid prototypes for %d/%d FG classes: %s",
                 len(valid_classes), num_classes - 1, valid_classes)

    # Per-organ binary prototype matching, then merge via per-voxel argmax
    pseudo_labels = {}
    for idx in tqdm(range(len(image_list)), desc="R0 pseudo-labels"):
        name = image_list[idx]
        h5f = h5py.File(os.path.join(data_dir, "data", name + ".h5"), 'r')
        image = h5f[image_key][:]
        if idx < labeled_num:
            pseudo_labels[name] = _binarize(h5f['label'][:])
            h5f.close()
            continue
        h5f.close()

        spatial_feat = all_spatial[idx]
        feat_norm = F.normalize(spatial_feat, dim=0)
        img_shape = image.shape

        # Collect per-organ raw cosine similarity at feature resolution
        fg_scores = {}
        for c in valid_classes:
            fg_proto = prototypes[c]
            sim_fg = torch.sum(feat_norm * fg_proto.view(feat_dim, 1, 1, 1), dim=0)
            fg_scores[c] = sim_fg

        # Build multi-class score map using raw similarity + BG similarity
        feat_res = spatial_feat.shape[1:]
        score_feat = torch.zeros(num_classes, *feat_res)
        if 0 in prototypes:
            bg_proto = prototypes[0]
            score_feat[0] = torch.sum(feat_norm * bg_proto.view(feat_dim, 1, 1, 1), dim=0)
        for c, sc in fg_scores.items():
            score_feat[c] = sc
        # Upsample to patch size and argmax
        score_high = F.interpolate(score_feat.unsqueeze(0), size=patch_size,
                                   mode='nearest')
        pseudo_patch = torch.argmax(score_high[0], dim=0).numpy().astype(np.uint8)

        # Map patch back to original volume
        pseudo_labels[name] = _crop_patch_to_original(pseudo_patch, img_shape, patch_size)

    fg_ratios = [(pseudo_labels[n] > 0).sum() / pseudo_labels[n].size
                 for n in image_list[labeled_num:]]
    logging.info("Round 0 pseudo-labels: avg FG ratio = %.4f", np.mean(fg_ratios))
    return pseudo_labels, all_spatial, all_global


# ============================================================
# KNN Refinement (shared)
# ============================================================

def refine_pseudo_labels_knn(pseudo_labels_raw, uncertainties, all_global,
                             image_list, labeled_num, k=5, q_unc=0.9, num_classes=2):
    N = len(image_list)
    unc_values = np.array([uncertainties[image_list[i]] for i in range(labeled_num, N)])
    if len(unc_values) == 0:
        return pseudo_labels_raw
    threshold = np.quantile(unc_values, q_unc)
    certain_indices = list(range(labeled_num))
    uncertain_indices = []
    for i in range(labeled_num, N):
        if uncertainties[image_list[i]] < threshold:
            certain_indices.append(i)
        else:
            uncertain_indices.append(i)
    logging.info("KNN refine: certain=%d, uncertain=%d, threshold=%.6f",
                 len(certain_indices), len(uncertain_indices), threshold)
    if len(uncertain_indices) == 0 or len(certain_indices) < k:
        return pseudo_labels_raw
    certain_feats = F.normalize(all_global[certain_indices], dim=1)
    refined = dict(pseudo_labels_raw)
    for idx in uncertain_indices:
        name = image_list[idx]
        query_feat = F.normalize(all_global[idx:idx + 1], dim=1)
        sims = torch.mm(query_feat, certain_feats.t())[0]
        topk_sims, topk_idx = torch.topk(sims, min(k, len(certain_indices)))
        weights = torch.clamp(topk_sims, min=0)
        if weights.sum() < 1e-8:
            continue
        target_shape = pseudo_labels_raw[name].shape
        vote_map = np.zeros((num_classes,) + target_shape, dtype=np.float32)
        for j in range(len(topk_idx)):
            neighbor_name = image_list[certain_indices[topk_idx[j]]]
            neighbor_label = pseudo_labels_raw[neighbor_name]
            w = weights[j].item()
            if neighbor_label.shape != target_shape:
                nl_tensor = torch.from_numpy(neighbor_label.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                nl_resized = F.interpolate(nl_tensor, size=target_shape, mode='nearest')[0, 0].numpy().astype(int)
            else:
                nl_resized = neighbor_label.astype(int)
            for c in range(num_classes):
                vote_map[c] += w * (nl_resized == c).astype(np.float32)
        refined[name] = np.argmax(vote_map, axis=0).astype(np.uint8)
    return refined


# ============================================================
# Dataset with Pseudo-Labels (shared)
# ============================================================

class LADatasetWithPseudo(Dataset):
    def __init__(self, base_dir, image_list, pseudo_labels, labeled_num,
                 transform=None, image_key='image'):
        self._base_dir = base_dir
        self.image_list = image_list
        self.pseudo_labels = pseudo_labels
        self.labeled_num = labeled_num
        self.transform = transform
        self.image_key = image_key

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx]
        h5f = h5py.File(os.path.join(self._base_dir, "data", name + ".h5"), 'r')
        image = h5f[self.image_key][:]
        if idx < self.labeled_num:
            label = _binarize(h5f['label'][:])
        else:
            label = self.pseudo_labels[name]
        h5f.close()
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


# ============================================================
# Pseudo-label generation from trained model (backbone-aware)
# ============================================================

def generate_pseudo_labels_from_model(net, data_dir, image_list, labeled_num,
                                      num_classes, patch_size, backbone='uamt',
                                      stride_xy=64, stride_z=64, image_key='image'):
    logging.info("=== Generating pseudo-labels from trained model ===")
    net.eval()
    pseudo_labels = {}
    uncertainties = {}
    is_dtc = (backbone == 'dtc')

    for idx, name in enumerate(tqdm(image_list, desc="Model inference")):
        h5f = h5py.File(os.path.join(data_dir, "data", name + ".h5"), 'r')
        image = h5f[image_key][:]
        h5f.close()
        if idx < labeled_num:
            h5f = h5py.File(os.path.join(data_dir, "data", name + ".h5"), 'r')
            pseudo_labels[name] = _binarize(h5f['label'][:])
            h5f.close()
            uncertainties[name] = 0.0
            continue

        w, h, d = image.shape
        add_pad = False
        w_pad = max(patch_size[0] - w, 0)
        h_pad = max(patch_size[1] - h, 0)
        d_pad = max(patch_size[2] - d, 0)
        if w_pad > 0 or h_pad > 0 or d_pad > 0:
            add_pad = True
            image_padded = np.pad(image,
                                  [(w_pad // 2, w_pad - w_pad // 2),
                                   (h_pad // 2, h_pad - h_pad // 2),
                                   (d_pad // 2, d_pad - d_pad // 2)],
                                  mode='constant', constant_values=0)
        else:
            image_padded = image

        ww, hh, dd = image_padded.shape
        score_map = np.zeros((num_classes,) + image_padded.shape, dtype=np.float32)
        cnt = np.zeros(image_padded.shape, dtype=np.float32)
        sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

        for x in range(sx):
            xs = min(stride_xy * x, ww - patch_size[0])
            for y in range(sy):
                ys = min(stride_xy * y, hh - patch_size[1])
                for z in range(sz):
                    zs = min(stride_z * z, dd - patch_size[2])
                    test_patch = image_padded[xs:xs + patch_size[0],
                                              ys:ys + patch_size[1],
                                              zs:zs + patch_size[2]]
                    test_patch = torch.from_numpy(
                        test_patch[np.newaxis, np.newaxis].astype(np.float32)).cuda()
                    with torch.no_grad():
                        if is_dtc:
                            _, y1 = net(test_patch)
                            prob = torch.softmax(y1, dim=1).cpu().numpy()[0]
                        else:
                            y1 = net(test_patch)
                            prob = torch.softmax(y1, dim=1).cpu().numpy()[0]
                    score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1],
                              zs:zs + patch_size[2]] += prob
                    cnt[xs:xs + patch_size[0], ys:ys + patch_size[1],
                        zs:zs + patch_size[2]] += 1

        score_map = score_map / np.expand_dims(cnt + 1e-8, axis=0)
        label_map = np.argmax(score_map, axis=0).astype(np.uint8)
        prob_map = score_map / (score_map.sum(axis=0, keepdims=True) + 1e-8)
        entropy = -np.sum(prob_map * np.log(prob_map + 1e-8), axis=0)
        avg_entropy = entropy.mean()

        if add_pad:
            label_map = label_map[w_pad // 2:w_pad // 2 + w,
                                  h_pad // 2:h_pad // 2 + h,
                                  d_pad // 2:d_pad // 2 + d]
        pseudo_labels[name] = label_map
        uncertainties[name] = float(avg_entropy)

    net.train()
    return pseudo_labels, uncertainties


# ============================================================
# Pseudo-label quality evaluation on val set
# ============================================================

def evaluate_pseudo_label_quality(pseudo_labels, data_dir, val_list, image_list,
                                  labeled_num, num_classes=2, image_key='image'):
    """Evaluate pseudo-label per-class Dice against GT on unlabeled training samples."""
    per_class_dice = {c: [] for c in range(1, num_classes)}
    count = 0
    for name in image_list[labeled_num:]:
        if name not in pseudo_labels:
            continue
        h5f = h5py.File(os.path.join(data_dir, "data", name + ".h5"), 'r')
        gt = _binarize(h5f['label'][:])
        h5f.close()
        pred = pseudo_labels[name].astype(np.uint8)
        if pred.shape != gt.shape:
            from scipy.ndimage import zoom
            zoom_factors = [g / p for g, p in zip(gt.shape, pred.shape)]
            pred = zoom(pred.astype(np.float32), zoom_factors, order=0).astype(np.uint8)
        for c in range(1, num_classes):
            inter = np.sum((pred == c) & (gt == c))
            denom = np.sum(pred == c) + np.sum(gt == c)
            dice = 2.0 * inter / (denom + 1e-8) if denom > 0 else 0.0
            per_class_dice[c].append(dice)
        count += 1
    class_avg = []
    for c in range(1, num_classes):
        avg = np.mean(per_class_dice[c]) if per_class_dice[c] else 0.0
        class_avg.append(avg)
        logging.info("  Pseudo-label class %2d: Dice = %.4f", c, avg)
    mean_dice = np.mean(class_avg) if class_avg else 0.0
    logging.info("Pseudo-label quality: mean Dice = %.4f (over %d samples, %d classes)",
                 mean_dice, count, num_classes - 1)
    return mean_dice


# ============================================================
# DTC helper: compute SDF
# ============================================================

def compute_sdf(img_gt, out_shape):
    from scipy.ndimage import distance_transform_edt as distance
    from skimage import segmentation as skimage_seg
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)
    for b in range(out_shape[0]):
        posmask = img_gt[b].astype(bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis) + 1e-8) - \
                  (posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis) + 1e-8)
            sdf[boundary == 1] = 0
            normalized_sdf[b] = sdf
    return normalized_sdf


# ============================================================
# Backbone-specific training rounds
# ============================================================

def _make_dataloader(args, pseudo_labels, image_list, round_num):
    db_train = LADatasetWithPseudo(
        base_dir=args.root_path,
        image_list=image_list,
        pseudo_labels=pseudo_labels,
        labeled_num=args.labeled_num,
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(args.patch_size),
            ToTensor(),
        ]),
        image_key=args.image_key)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id + round_num * 100)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, len(image_list)))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    return trainloader


def _validate(model, args, backbone):
    if backbone == 'dtc':
        from val_3D_sdf import test_all_case
    else:
        from val_3D import test_all_case
    return test_all_case(model, args.root_path, test_list="val.txt",
                         num_classes=args.num_classes, patch_size=args.patch_size,
                         stride_xy=96, stride_z=96, image_key=args.image_key,
                         binary_label=args.binary_label)


def _save_best(model, snapshot_path, round_num, iter_num, best_performance):
    save_path = os.path.join(
        snapshot_path, f'round{round_num}_iter{iter_num}_dice{best_performance:.4f}.pth')
    torch.save(model.state_dict(), save_path)
    torch.save(model.state_dict(),
               os.path.join(snapshot_path, f'round{round_num}_best.pth'))


def _log_validation(avg_metric, round_num, iter_num, num_classes):
    """Log per-class Dice and mean Dice from validation."""
    mean_dice = avg_metric[:, 0].mean()
    mean_hd95 = avg_metric[:, 1].mean()
    parts = []
    for c in range(num_classes - 1):
        parts.append(f'c{c+1}={avg_metric[c, 0]:.4f}')
    logging.info('Round %d iter %d : val_mean_dice=%.4f, val_mean_hd95=%.4f | %s',
                 round_num, iter_num, mean_dice, mean_hd95, ', '.join(parts))
    return mean_dice


# ---- MT ----
def train_one_round_mt(args, round_num, snapshot_path, pseudo_labels, image_list, writer):
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    logging.info("=== Round %d: Training MT model ===", round_num)

    def create_model(ema=False):
        net = net_factory_3d(net_type='unet_3D', in_chns=1, class_num=num_classes).cuda()
        if ema:
            for param in net.parameters():
                param.detach_()
        return net

    model = create_model()
    ema_model = create_model(ema=True)
    trainloader = _make_dataloader(args, pseudo_labels, image_list, round_num)
    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_model_state = None

    iterator = tqdm(range(max_epoch), ncols=70, desc=f"Round {round_num}")
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            # Supervised loss
            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:args.labeled_bs])
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs],
                                  label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)

            # Consistency loss
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_loss = torch.mean(
                (outputs_soft[args.labeled_bs:] - ema_output_soft) ** 2)

            # Pseudo-label loss
            pseudo_ce = ce_loss(outputs[args.labeled_bs:], label_batch[args.labeled_bs:])
            pseudo_dice = dice_loss(outputs_soft[args.labeled_bs:],
                                    label_batch[args.labeled_bs:].unsqueeze(1))
            pseudo_loss = 0.5 * (pseudo_ce + pseudo_dice)
            pseudo_weight = min(1.0, iter_num / (max_iterations * 0.3))

            loss = supervised_loss + consistency_weight * consistency_loss + \
                pseudo_weight * pseudo_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num += 1

            if iter_num % 100 == 0:
                logging.info('Round %d iter %d : loss=%.4f, sup=%.4f, cons=%.4f, pseudo=%.4f',
                             round_num, iter_num, loss.item(), supervised_loss.item(),
                             consistency_loss.item(), pseudo_loss.item())

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = _validate(model, args, 'mt')
                cur_dice = _log_validation(avg_metric, round_num, iter_num, num_classes)
                if cur_dice > best_performance:
                    best_performance = cur_dice
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    _save_best(model, snapshot_path, round_num, iter_num, best_performance)
                model.train()

            if iter_num % 3000 == 0:
                torch.save(model.state_dict(),
                           os.path.join(snapshot_path, f'round{round_num}_iter{iter_num}.pth'))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    logging.info("Round %d: Best validation Dice = %.4f", round_num, best_performance)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_performance


# ---- UAMT ----
def train_one_round_uamt(args, round_num, snapshot_path, pseudo_labels, image_list, writer):
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    logging.info("=== Round %d: Training UAMT model ===", round_num)

    def create_model(ema=False):
        net = net_factory_3d(net_type='unet_3D', in_chns=1, class_num=num_classes).cuda()
        if ema:
            for param in net.parameters():
                param.detach_()
        return net

    model = create_model()
    ema_model = create_model(ema=True)
    trainloader = _make_dataloader(args, pseudo_labels, image_list, round_num)
    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_model_state = None

    iterator = tqdm(range(max_epoch), ncols=70, desc=f"Round {round_num}")
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)

            # UAMT T=8 MC uncertainty
            T = 8
            _, _, dv, wv, hv = unlabeled_volume_batch.shape
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, num_classes, dv, wv, hv]).cuda()
            for i in range(T // 2):
                ema_inputs_t = volume_batch_r + \
                    torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs_t)
            preds = torch.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, num_classes, dv, wv, hv)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)

            # Supervised loss
            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:args.labeled_bs])
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs],
                                  label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)

            # Consistency with uncertainty masking
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist = losses.softmax_mse_loss(outputs[args.labeled_bs:], ema_output)
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_loss = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)

            # Pseudo-label loss
            pseudo_ce = ce_loss(outputs[args.labeled_bs:], label_batch[args.labeled_bs:])
            pseudo_dice = dice_loss(outputs_soft[args.labeled_bs:],
                                    label_batch[args.labeled_bs:].unsqueeze(1))
            pseudo_loss = 0.5 * (pseudo_ce + pseudo_dice)
            pseudo_weight = min(1.0, iter_num / (max_iterations * 0.3))

            loss = supervised_loss + consistency_weight * consistency_loss + \
                pseudo_weight * pseudo_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num += 1

            if iter_num % 100 == 0:
                logging.info('Round %d iter %d : loss=%.4f, sup=%.4f, cons=%.4f, pseudo=%.4f',
                             round_num, iter_num, loss.item(), supervised_loss.item(),
                             consistency_loss.item(), pseudo_loss.item())

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = _validate(model, args, 'uamt')
                cur_dice = _log_validation(avg_metric, round_num, iter_num, num_classes)
                if cur_dice > best_performance:
                    best_performance = cur_dice
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    _save_best(model, snapshot_path, round_num, iter_num, best_performance)
                model.train()

            if iter_num % 3000 == 0:
                torch.save(model.state_dict(),
                           os.path.join(snapshot_path, f'round{round_num}_iter{iter_num}.pth'))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    logging.info("Round %d: Best validation Dice = %.4f", round_num, best_performance)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_performance


# ---- DAN ----
def train_one_round_dan(args, round_num, snapshot_path, pseudo_labels, image_list, writer):
    from networks.discriminator import FC3DDiscriminator
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    logging.info("=== Round %d: Training DAN model ===", round_num)

    model = net_factory_3d(net_type='unet_3D', in_chns=1, class_num=num_classes).cuda()
    DAN = FC3DDiscriminator(num_classes=num_classes).cuda()

    trainloader = _make_dataloader(args, pseudo_labels, image_list, round_num)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    DAN_optimizer = optim.Adam(DAN.parameters(), lr=args.DAN_lr, betas=(0.9, 0.99))
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_model_state = None

    iterator = tqdm(range(max_epoch), ncols=70, desc=f"Round {round_num}")
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            DAN_target = torch.tensor([1, 0]).cuda()

            # Phase 1: Train segmentation network
            model.train()
            DAN.eval()
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:args.labeled_bs])
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs],
                                  label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            DAN_outputs = DAN(outputs_soft[args.labeled_bs:], volume_batch[args.labeled_bs:])
            consistency_loss = F.cross_entropy(DAN_outputs, DAN_target[:args.labeled_bs].long())

            # Pseudo-label loss
            pseudo_ce = ce_loss(outputs[args.labeled_bs:], label_batch[args.labeled_bs:])
            pseudo_dice = dice_loss(outputs_soft[args.labeled_bs:],
                                    label_batch[args.labeled_bs:].unsqueeze(1))
            pseudo_loss = 0.5 * (pseudo_ce + pseudo_dice)
            pseudo_weight = min(1.0, iter_num / (max_iterations * 0.3))

            loss = supervised_loss + consistency_weight * consistency_loss + \
                pseudo_weight * pseudo_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Phase 2: Train discriminator
            model.eval()
            DAN.train()
            with torch.no_grad():
                outputs = model(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
            DAN_outputs = DAN(outputs_soft, volume_batch)
            DAN_loss = F.cross_entropy(DAN_outputs, DAN_target.long())
            DAN_optimizer.zero_grad()
            DAN_loss.backward()
            DAN_optimizer.step()

            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num += 1

            if iter_num % 100 == 0:
                logging.info('Round %d iter %d : loss=%.4f, sup=%.4f, cons=%.4f, pseudo=%.4f',
                             round_num, iter_num, loss.item(), supervised_loss.item(),
                             consistency_loss.item(), pseudo_loss.item())

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = _validate(model, args, 'dan')
                cur_dice = _log_validation(avg_metric, round_num, iter_num, num_classes)
                if cur_dice > best_performance:
                    best_performance = cur_dice
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    _save_best(model, snapshot_path, round_num, iter_num, best_performance)
                model.train()

            if iter_num % 3000 == 0:
                torch.save(model.state_dict(),
                           os.path.join(snapshot_path, f'round{round_num}_iter{iter_num}.pth'))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    logging.info("Round %d: Best validation Dice = %.4f", round_num, best_performance)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_performance


# ---- DTC ----
def train_one_round_dtc(args, round_num, snapshot_path, pseudo_labels, image_list, writer):
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    logging.info("=== Round %d: Training DTC model ===", round_num)

    model = net_factory_3d(net_type='unet_3D_sdf', in_chns=1, class_num=num_classes).cuda()
    trainloader = _make_dataloader(args, pseudo_labels, image_list, round_num)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_model_state = None

    iterator = tqdm(range(max_epoch), ncols=70, desc=f"Round {round_num}")
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

            outputs_dis, outputs_seg = model(volume_batch)
            softmask_seg = torch.sigmoid(outputs_seg)

            # SDF loss
            with torch.no_grad():
                gt_dis = compute_sdf(label_batch.cpu().numpy(), outputs_dis.shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()
            loss_dis = torch.norm(outputs_dis - gt_dis, 1) / torch.numel(outputs_dis)

            # Segmentation loss (labeled)
            loss_ce = 0.01 * ce_loss(outputs_seg[:args.labeled_bs],
                                     label_batch[:args.labeled_bs].long())
            loss_dice = losses.dice_loss(
                softmask_seg[:args.labeled_bs, :, :, :], label_batch[:args.labeled_bs] == 1)
            loss_seg = 0.5 * (loss_ce + loss_dice)

            # SDF-to-mask dice (labeled)
            dis_to_mask = torch.sigmoid(-1500 * outputs_dis)
            loss_dis_dice = losses.dice_loss(
                dis_to_mask[:args.labeled_bs, :, :, :], label_batch[:args.labeled_bs] == 1)

            # Consistency (SDF-mask vs seg-mask)
            consistency_loss = torch.mean((dis_to_mask - softmask_seg) ** 2)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            # Pseudo-label loss (unlabeled)
            pseudo_ce = 0.01 * ce_loss(outputs_seg[args.labeled_bs:],
                                       label_batch[args.labeled_bs:].long())
            pseudo_dice = losses.dice_loss(
                softmask_seg[args.labeled_bs:, :, :, :], label_batch[args.labeled_bs:] == 1)
            pseudo_loss = 0.5 * (pseudo_ce + pseudo_dice)
            pseudo_weight = min(1.0, iter_num / (max_iterations * 0.3))

            loss = loss_seg + loss_dis_dice + consistency_weight * consistency_loss + \
                pseudo_weight * pseudo_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num += 1

            if iter_num % 100 == 0:
                logging.info('Round %d iter %d : loss=%.4f, seg=%.4f, cons=%.4f, pseudo=%.4f',
                             round_num, iter_num, loss.item(), loss_seg.item(),
                             consistency_loss.item(), pseudo_loss.item())

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = _validate(model, args, 'dtc')
                cur_dice = _log_validation(avg_metric, round_num, iter_num, num_classes)
                if cur_dice > best_performance:
                    best_performance = cur_dice
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    _save_best(model, snapshot_path, round_num, iter_num, best_performance)
                model.train()

            if iter_num % 3000 == 0:
                torch.save(model.state_dict(),
                           os.path.join(snapshot_path, f'round{round_num}_iter{iter_num}.pth'))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    logging.info("Round %d: Best validation Dice = %.4f", round_num, best_performance)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_performance


# ============================================================
# Main Pipeline
# ============================================================

BACKBONE_TRAINERS = {
    'mt': train_one_round_mt,
    'uamt': train_one_round_uamt,
    'dan': train_one_round_dan,
    'dtc': train_one_round_dtc,
}


def main():
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_1label_1/{}".format(
        args.exp, args.backbone.upper(), args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Args: %s", str(args))
    logging.info("Backbone: %s, Model: %s", args.backbone, args.model)

    start_time = time.time()

    # Load data list
    with open(os.path.join(args.root_path, 'train.txt'), 'r') as f:
        image_list = [line.strip().split(",")[0] for line in f.readlines()]
    logging.info("Total training samples: %d (labeled: %d)", len(image_list), args.labeled_num)

    writer = SummaryWriter(snapshot_path + '/log')

    # Load SAM-Med3D encoder for feature extraction
    logging.info("Loading SAM-Med3D encoder from %s", args.sam_ckpt)
    sam_model = load_sam_encoder(args.sam_ckpt)

    # Round 0: Initial pseudo-labels
    pseudo_labels, all_spatial, all_global = generate_initial_pseudo_labels(
        sam_model, args.root_path, image_list, args.labeled_num, args.patch_size,
        num_classes=args.num_classes, image_key=args.image_key)

    # Free SAM model GPU memory
    del sam_model
    torch.cuda.empty_cache()

    train_fn = BACKBONE_TRAINERS[args.backbone]
    round_results = []

    for round_num in range(1, args.num_rounds + 1):
        round_start = time.time()
        logging.info("=" * 60)
        logging.info("Starting Round %d / %d", round_num, args.num_rounds)

        model, best_dice = train_fn(
            args, round_num, snapshot_path, pseudo_labels, image_list, writer)
        round_results.append({'round': round_num, 'best_dice': best_dice})

        round_elapsed = time.time() - round_start
        logging.info("Round %d finished in %.1f min (best val dice=%.4f)",
                     round_num, round_elapsed / 60, best_dice)

        if round_num < args.num_rounds:
            new_pseudo_labels, uncertainties = generate_pseudo_labels_from_model(
                model, args.root_path, image_list, args.labeled_num,
                num_classes=args.num_classes, patch_size=args.patch_size,
                backbone=args.backbone, image_key=args.image_key)
            pseudo_labels = refine_pseudo_labels_knn(
                new_pseudo_labels, uncertainties, all_global,
                image_list, args.labeled_num,
                k=args.knn_k, q_unc=args.uncertainty_quantile,
                num_classes=args.num_classes)
            fg_ratios = [pseudo_labels[n].sum() / pseudo_labels[n].size
                         for n in image_list[args.labeled_num:]]
            logging.info("Round %d refined pseudo-labels: avg FG ratio = %.4f",
                         round_num, np.mean(fg_ratios))

        del model
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    logging.info("=" * 60)
    logging.info("Training complete (%.1f hours). Results by round:", total_time / 3600)
    for r in round_results:
        logging.info("  Round %d: Best Dice = %.4f", r['round'], r['best_dice'])
    logging.info("=" * 60)
    writer.close()

    with open(os.path.join(snapshot_path, 'results_summary.txt'), 'w') as f:
        f.write(f"SemiSAM-O1 Results Summary ({args.backbone.upper()})\n")
        f.write("=" * 40 + "\n")
        f.write(f"Backbone: {args.backbone}\nModel: {args.model}\n")
        f.write(f"Labeled: {args.labeled_num}, Rounds: {args.num_rounds}\n")
        f.write(f"Total time: {total_time/3600:.1f} hours\n\n")
        for r in round_results:
            f.write(f"Round {r['round']}: Best Val Dice = {r['best_dice']:.4f}\n")

    import json
    tracking = {
        'test_dice': {r['round']: float(r['best_dice']) for r in round_results},
    }
    json_path = os.path.join(snapshot_path, 'round_tracking.json')
    with open(json_path, 'w') as f:
        json.dump(tracking, f, indent=2)
    logging.info("Saved round tracking to %s", json_path)


if __name__ == "__main__":
    main()
