"""
DDM2 Batch Inference Script - Generate predictions for all patients
批量推理所有患者，生成 nii.gz 文件，并转换回原始 HU 空间

Usage:
    python inference_ddm2_batch.py -c config/ct_denoise.json --save_mode final
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


def inverse_histogram_equalization(img, bins, bins_mapped):
    """
    逆向 HE：HE 空间 → 原始 HU
    """
    if bins is None or bins_mapped is None:
        return img
    
    flat_img = img.flatten()
    bin_indices = np.digitize(flat_img, bins_mapped) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 1)
    original = bins[bin_indices]
    
    return original.reshape(img.shape).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description='DDM2 Batch Inference')
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--output_dir', type=str, default='/host/d/file/pre/ddm2/pred_images', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--save_mode', type=str, default='final', choices=['first', 'final', 'both'], help='Which result to save')
    parser.add_argument('--no_inverse_he', action='store_true', help='Skip inverse HE')
    return parser.parse_args()


def find_latest_checkpoint(experiments_dir='experiments'):
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


def inference_single_patient(diffusion, val_set, patient_idx, HU_MIN, HU_MAX, 
                             output_dir, save_mode, bins, bins_mapped, use_inverse_he):
    """推理单个患者"""
    
    if not hasattr(val_set, 'n2n_pairs') or patient_idx >= len(val_set.n2n_pairs):
        return None
    
    pair = val_set.n2n_pairs[patient_idx]
    patient_id = pair['patient_id']
    patient_subid = pair['patient_subid']
    
    pid_str = f"{int(patient_id):08d}"
    psid_str = f"{int(patient_subid):010d}"
    
    output_subdir = os.path.join(output_dir, pid_str, psid_str)
    first_path = os.path.join(output_subdir, 'ddm2_first_step.nii.gz')
    final_path = os.path.join(output_subdir, 'ddm2_final.nii.gz')
    
    # 检查是否已存在
    if save_mode == 'first' and os.path.exists(first_path):
        return f"Skip {pid_str}/{psid_str} (exists)"
    if save_mode == 'final' and os.path.exists(final_path):
        return f"Skip {pid_str}/{psid_str} (exists)"
    if save_mode == 'both' and os.path.exists(first_path) and os.path.exists(final_path):
        return f"Skip {pid_str}/{psid_str} (exists)"
    
    # 获取该患者的所有 slices
    patient_samples = [(i, s) for i, s in enumerate(val_set.samples) if s[0] == patient_idx]
    
    if len(patient_samples) == 0:
        return f"Skip {pid_str}/{psid_str} (no samples)"
    
    first_results = []
    final_results = []
    
    for sample_idx, (vol_idx, slice_idx) in patient_samples:
        sample = val_set[sample_idx]
        
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                 for k, v in sample.items()}
        
        diffusion.feed_data(batch)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()
        
        all_imgs = visuals['denoised'].numpy()
        first = (all_imgs[1].squeeze() + 1) / 2
        final = (all_imgs[-1].squeeze() + 1) / 2
        
        first_hu = first * (HU_MAX - HU_MIN) + HU_MIN
        final_hu = final * (HU_MAX - HU_MIN) + HU_MIN
        
        # 逆向 HE
        if use_inverse_he:
            first_hu = inverse_histogram_equalization(first_hu, bins, bins_mapped)
            final_hu = inverse_histogram_equalization(final_hu, bins, bins_mapped)
        
        first_results.append(first_hu)
        final_results.append(final_hu)
    
    # 堆叠成 3D volume
    first_volume = np.stack(first_results, axis=-1).astype(np.float32)
    final_volume = np.stack(final_results, axis=-1).astype(np.float32)
    
    # 获取 affine
    affine = np.eye(4)
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
    os.makedirs(output_subdir, exist_ok=True)
    
    if save_mode in ['first', 'both']:
        first_nii = nib.Nifti1Image(first_volume, affine)
        nib.save(first_nii, first_path)
    
    if save_mode in ['final', 'both']:
        final_nii = nib.Nifti1Image(final_volume, affine)
        nib.save(final_nii, final_path)
    
    return f"Saved {pid_str}/{psid_str} ({first_volume.shape})"


def main():
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
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
    
    print("=" * 60)
    print("DDM2 Batch Inference")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"HU range: [{HU_MIN}, {HU_MAX}]")
    print(f"Inverse HE: {use_inverse_he}")
    print(f"Output: {args.output_dir}")
    print(f"Save mode: {args.save_mode}")
    
    # 查找 checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
    
    if checkpoint is None:
        print("[ERROR] No checkpoint found!")
        return
    
    print(f"Checkpoint: {checkpoint}")
    
    # 创建数据集
    print("\n[1/3] Loading dataset...")
    val_opt = opt['datasets']['val'].copy()
    val_opt['val_volume_idx'] = 'all'
    val_opt['val_slice_idx'] = 'all'
    
    val_set = Data.create_dataset(val_opt, 'val', stage2_file=opt.get('stage2_file'))
    
    if hasattr(val_set, 'n2n_pairs'):
        patient_indices = list(range(len(val_set.n2n_pairs)))
    else:
        patient_indices = list(set(s[0] for s in val_set.samples))
    
    print(f"Total patients: {len(patient_indices)}")
    print(f"Total samples: {len(val_set)}")
    
    # 加载模型
    print("\n[2/3] Loading model...")
    opt_model = Logger.dict_to_nonedict(opt)
    opt_model['path']['resume_state'] = checkpoint
    
    diffusion = Model.create_model(opt_model)
    diffusion.set_new_noise_schedule(opt_model['model']['beta_schedule']['val'], schedule_phase='val')
    print("Model loaded!")
    
    # 批量推理
    print("\n[3/3] Running batch inference...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    for patient_idx in tqdm(patient_indices, desc="Patients"):
        result = inference_single_patient(
            diffusion, val_set, patient_idx, 
            HU_MIN, HU_MAX, args.output_dir, args.save_mode,
            bins, bins_mapped, use_inverse_he
        )
        if result:
            tqdm.write(result)
    
    print("\n" + "=" * 60)
    print("Done!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
