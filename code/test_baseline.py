import argparse
import os
import math

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from networks.net_factory_3d import net_factory_3d
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, required=True)
parser.add_argument('--exp', type=str, required=True)
parser.add_argument('--model', type=str, default='unet_3D')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--patch_size', nargs=3, type=int, default=[128, 128, 128])
parser.add_argument('--stride_xy', type=int, default=64)
parser.add_argument('--stride_z', type=int, default=64)
parser.add_argument('--ckpt', type=str, default=None,
                    help='Checkpoint name, default: {model}_best_model.pth')
parser.add_argument('--test_list', type=str, default='test.txt')
parser.add_argument('--labeled_num', type=int, default=None,
                    help='If set, snapshot_path uses {exp}_{labeled_num}/{model} format')
parser.add_argument('--save_pred', type=str, default=None,
                    help='Save predictions as nii.gz to this directory')
args = parser.parse_args()


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes):
    w, h, d = image.shape
    w_pad = max(patch_size[0] - w, 0)
    h_pad = max(patch_size[1] - h, 0)
    d_pad = max(patch_size[2] - d, 0)
    add_pad = (w_pad > 0 or h_pad > 0 or d_pad > 0)
    if add_pad:
        image = np.pad(image,
                       [(w_pad//2, w_pad - w_pad//2),
                        (h_pad//2, h_pad - h_pad//2),
                        (d_pad//2, d_pad - d_pad//2)],
                       mode='constant', constant_values=0)
    ww, hh, dd = image.shape
    score_map = np.zeros((num_classes,) + image.shape, dtype=np.float32)
    cnt = np.zeros(image.shape, dtype=np.float32)

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])
                patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                patch_t = torch.from_numpy(patch[np.newaxis, np.newaxis].astype(np.float32)).cuda()
                with torch.no_grad():
                    y1 = net(patch_t)
                    if isinstance(y1, tuple):
                        _, y1 = y1
                    out = torch.softmax(y1, dim=1).cpu().numpy()[0]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += out
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

    score_map /= np.expand_dims(cnt, 0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[w_pad//2:w_pad//2+w, h_pad//2:h_pad//2+h, d_pad//2:d_pad//2+d]
    return label_map


def cal_metric(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return np.array([dice, jc, hd95, asd])
    elif pred.sum() == 0 and gt.sum() == 0:
        return np.array([1.0, 1.0, 0.0, 0.0])
    else:
        return np.array([0.0, 0.0, 200.0, 100.0])


def main():
    if args.labeled_num is not None:
        snapshot_path = f"../model/{args.exp}_{args.labeled_num}/{args.model}"
    else:
        snapshot_path = f"../model/{args.exp}/{args.model}"

    ckpt_name = args.ckpt or f'{args.model}_best_model.pth'
    ckpt_path = os.path.join(snapshot_path, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return

    net = net_factory_3d(net_type=args.model, in_chns=1, class_num=args.num_classes).cuda()
    net.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
    net.eval()
    print(f"Loaded: {ckpt_path}")

    with open(os.path.join(args.root_path, args.test_list)) as f:
        test_list = [l.strip().split(",")[0] for l in f.readlines()]

    all_metrics = []
    print(f"Testing {len(test_list)} cases...")
    for name in tqdm(test_list):
        h5f = h5py.File(os.path.join(args.root_path, "data", name + ".h5"), 'r')
        image, label = h5f['image'][:], h5f['label'][:]
        h5f.close()
        pred = test_single_case(net, image, args.stride_xy, args.stride_z,
                                args.patch_size, args.num_classes)
        for c in range(1, args.num_classes):
            m = cal_metric(pred == c, label == c)
            all_metrics.append(m)

        if args.save_pred:
            os.makedirs(args.save_pred, exist_ok=True)
            pred_itk = sitk.GetImageFromArray(pred.astype(np.uint8))
            pred_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(pred_itk, os.path.join(args.save_pred, f"{name}_pred.nii.gz"))
            img_itk = sitk.GetImageFromArray(image)
            img_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(img_itk, os.path.join(args.save_pred, f"{name}_img.nii.gz"))
            lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
            lab_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(lab_itk, os.path.join(args.save_pred, f"{name}_lab.nii.gz"))

    all_metrics = np.array(all_metrics)
    mean = all_metrics.mean(axis=0)
    std = all_metrics.std(axis=0)
    print(f"\n{'='*50}")
    print(f"Results: {args.exp} on {args.test_list}")
    print(f"  Dice:    {mean[0]:.4f} ± {std[0]:.4f}")
    print(f"  Jaccard: {mean[1]:.4f} ± {std[1]:.4f}")
    print(f"  HD95:    {mean[2]:.2f} ± {std[2]:.2f}")
    print(f"  ASD:     {mean[3]:.2f} ± {std[3]:.2f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
