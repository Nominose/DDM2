"""
DDM2 Inference Script - Generate full volume prediction
生成完整的 50 slice nii.gz 文件，并转换回原始 HU 空间

Usage:
    python inference_ddm2.py -c config/ct_denoise.json --patient_idx 8
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

sys.path.insert(0, '.')

import data as Data
import model as Model
import core.logger as Logger


def apply_histogram_equalization(img, bins, bins_mapped):
    """正向 HE：原始 HU → HE 空间"""
    if bins is None or bins_mapped is None:
        return img
    flat_img = img.flatten()
    bin_indices = np.digitize(flat_img, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins_mapped) - 1)
    equalized = bins_mapped[bin_indices]
    return equalized.reshape(img.shape).astype(np.float32)


def inverse_histogram_equalization(img, bins, bins_mapped):
    """
    逆向 HE：HE 空间 → 原始 HU
    
    原理：
    - 正向：原始值 → 查 bins 位置 → 用 bins_mapped 取值
    - 逆向：HE 值 → 查 bins_mapped 位置 → 用 bins 取值
    """
    if bins is None or bins_mapped is None:
        return img
    
    flat_img = img.flatten()
    
    # 在 bins_mapped 中找位置（逆向查找）
    bin_indices = np.digitize(flat_img, bins_mapped) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 1)
    
    # 用 bins 取回原始值
    original = bins[bin_indices]
    
    return original.reshape(img.shape).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description='DDM2 Inference')
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--patient_idx', type=int, default=8, help='Patient index')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--save_first', action='store_true', default=True, help='Save first-step')
    parser.add_argument('--save_final', action='store_true', default=True, help='Save final')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--no_inverse_he', action='store_true', help='Skip inverse HE')
    return parser.parse_args()


def find_latest_checkpoint(experiments_dir='experiments'):
    """自动查找最新的 checkpoint"""
    latest_dir = None
    latest_time = 0
    
    for d in os.listdir(experiments_dir):
        if d.startswith('ct_denoise_'):
            ckpt_path = os.path.join(experiments_dir, d, 'checkpoint', 'latest_gen.pth')
            if os.path.exists(ckpt_path):
                mtime = os.path.getmtime(ckpt_path)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_dir = os.path.join(experiments_dir, d, 'checkpoint', 'latest')
    
    return latest_dir


def main():
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载 config
    with open(args.config, 'r') as f:
        opt = json.load(f)
    
    HU_MIN = opt['datasets']['val'].get('HU_MIN', -1000.0)
    HU_MAX = opt['datasets']['val'].get('HU_MAX', 2000.0)
    
    # 加载 HE bins
    bins_file = opt['datasets']['val'].get('bins_file')
    bins_mapped_file = opt['datasets']['val'].get('bins_mapped_file')
    bins = None
    bins_mapped = None
    use_inverse_he = False
    
    if bins_file and bins_mapped_file:
        if os.path.exists(bins_file) and os.path.exists(bins_mapped_file):
            bins = np.load(bins_file).astype(np.float32)
            bins_mapped = np.load(bins_mapped_file).astype(np.float32)
            use_inverse_he = not args.no_inverse_he
            print(f"Histogram Equalization bins loaded")
            print(f"  bins range: [{bins.min():.1f}, {bins.max():.1f}]")
            print(f"  bins_mapped range: [{bins_mapped.min():.1f}, {bins_mapped.max():.1f}]")
    
    print("=" * 60)
    print("DDM2 Inference")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"HU range: [{HU_MIN}, {HU_MAX}]")
    print(f"Inverse HE: {use_inverse_he}")
    print(f"Patient index: {args.patient_idx}")
    
    # 查找 checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
    
    if checkpoint is None:
        print("[ERROR] No checkpoint found!")
        return
    
    print(f"Checkpoint: {checkpoint}")
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.dirname(checkpoint).replace('/checkpoint', '/inference')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output dir: {args.output_dir}")
    
    # 创建数据集
    print("\n[1/3] Loading dataset...")
    val_opt = opt['datasets']['val'].copy()
    val_opt['val_volume_idx'] = args.patient_idx
    val_opt['val_slice_idx'] = 'all'
    
    val_set = Data.create_dataset(val_opt, 'val', stage2_file=opt.get('stage2_file'))
    
    num_slices = len(val_set)
    print(f"Total slices: {num_slices}")
    
    # 获取患者信息
    if hasattr(val_set, 'n2n_pairs') and args.patient_idx < len(val_set.n2n_pairs):
        pair = val_set.n2n_pairs[args.patient_idx]
        patient_id = pair['patient_id']
        patient_subid = pair['patient_subid']
        print(f"Patient ID: {patient_id}, SubID: {patient_subid}")
    else:
        patient_id = f"patient_{args.patient_idx}"
        patient_subid = "0"
    
    # 加载模型
    print("\n[2/3] Loading model...")
    opt_model = Logger.dict_to_nonedict(opt)
    opt_model['path']['resume_state'] = checkpoint
    
    diffusion = Model.create_model(opt_model)
    diffusion.set_new_noise_schedule(opt_model['model']['beta_schedule']['val'], schedule_phase='val')
    print("Model loaded!")
    
    # 推理
    print("\n[3/3] Running inference...")
    
    first_results = []
    final_results = []
    noisy_inputs = []
    
    for idx in tqdm(range(num_slices), desc="Inference"):
        sample = val_set[idx]
        
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                 for k, v in sample.items()}
        
        diffusion.feed_data(batch)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()
        
        all_imgs = visuals['denoised'].numpy()
        
        # 从 [-1, 1] 转换到 [0, 1]
        noisy = (all_imgs[0].squeeze() + 1) / 2
        first = (all_imgs[1].squeeze() + 1) / 2
        final = (all_imgs[-1].squeeze() + 1) / 2
        
        # 转换到 HE 空间的 HU 值
        noisy_hu = noisy * (HU_MAX - HU_MIN) + HU_MIN
        first_hu = first * (HU_MAX - HU_MIN) + HU_MIN
        final_hu = final * (HU_MAX - HU_MIN) + HU_MIN
        
        # 逆向 HE：转换回原始 HU 空间
        if use_inverse_he:
            noisy_hu = inverse_histogram_equalization(noisy_hu, bins, bins_mapped)
            first_hu = inverse_histogram_equalization(first_hu, bins, bins_mapped)
            final_hu = inverse_histogram_equalization(final_hu, bins, bins_mapped)
        
        noisy_inputs.append(noisy_hu)
        first_results.append(first_hu)
        final_results.append(final_hu)
    
    # 堆叠成 3D volume
    noisy_volume = np.stack(noisy_inputs, axis=-1).astype(np.float32)
    first_volume = np.stack(first_results, axis=-1).astype(np.float32)
    final_volume = np.stack(final_results, axis=-1).astype(np.float32)
    
    print(f"\nVolume shape: {first_volume.shape}")
    
    # 获取 affine
    affine = np.eye(4)
    if hasattr(val_set, 'n2n_pairs') and args.patient_idx < len(val_set.n2n_pairs):
        pair = val_set.n2n_pairs[args.patient_idx]
        noise_path = pair['noise_0']
        if hasattr(val_set, '_fix_path'):
            noise_path = val_set._fix_path(noise_path)
        if os.path.exists(noise_path):
            try:
                orig_nii = nib.load(noise_path)
                affine = orig_nii.affine
            except:
                pass
    
    # 保存文件
    pid_str = f"{int(patient_id):08d}" if isinstance(patient_id, (int, float)) else str(patient_id)
    psid_str = f"{int(patient_subid):010d}" if isinstance(patient_subid, (int, float)) else str(patient_subid)
    
    output_subdir = os.path.join(args.output_dir, pid_str, psid_str)
    os.makedirs(output_subdir, exist_ok=True)
    
    # 保存 noisy input
    noisy_nii = nib.Nifti1Image(noisy_volume, affine)
    noisy_path = os.path.join(output_subdir, 'noisy_input.nii.gz')
    nib.save(noisy_nii, noisy_path)
    print(f"Saved: {noisy_path}")
    
    # 保存 first-step
    if args.save_first:
        first_nii = nib.Nifti1Image(first_volume, affine)
        first_path = os.path.join(output_subdir, 'ddm2_first_step.nii.gz')
        nib.save(first_nii, first_path)
        print(f"Saved: {first_path}")
    
    # 保存 final
    if args.save_final:
        final_nii = nib.Nifti1Image(final_volume, affine)
        final_path = os.path.join(output_subdir, 'ddm2_final.nii.gz')
        nib.save(final_nii, final_path)
        print(f"Saved: {final_path}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("Statistics (HU values, after inverse HE)" if use_inverse_he else "Statistics (HU values, HE space)")
    print("-" * 60)
    print(f"{'Image':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 60)
    print(f"{'Noisy Input':<20} {noisy_volume.min():>10.1f} {noisy_volume.max():>10.1f} {noisy_volume.mean():>10.1f} {noisy_volume.std():>10.1f}")
    print(f"{'DDM2 First-step':<20} {first_volume.min():>10.1f} {first_volume.max():>10.1f} {first_volume.mean():>10.1f} {first_volume.std():>10.1f}")
    print(f"{'DDM2 Final':<20} {final_volume.min():>10.1f} {final_volume.max():>10.1f} {final_volume.mean():>10.1f} {final_volume.std():>10.1f}")
    print("=" * 60)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
