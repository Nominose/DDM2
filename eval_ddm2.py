"""
DDM2 Quantitative Evaluation Script
量化评估脚本：对比 GT、Noisy、N2N、DDM2

Usage:
    # 评估单个患者
    python eval_ddm2.py -c config/ct_denoise.json --patient_idx 8
    
    # 评估所有患者并输出 Excel
    python eval_ddm2.py -c config/ct_denoise.json --all --output results.xlsx
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, '.')

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("[WARNING] lpips not installed, LPIPS metric will be skipped")


def calc_psnr(img, ref, data_range):
    mse = np.mean((img - ref) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range ** 2 / mse)


def calc_ssim(img, ref, data_range):
    return ssim(img, ref, data_range=data_range)


def calc_mae(img, ref):
    return np.mean(np.abs(img - ref))


def calc_lpips(img, ref, loss_fn, device, vmin, vmax):
    if loss_fn is None:
        return None
    
    img_clip = np.clip(img, vmin, vmax).astype(np.float32)
    ref_clip = np.clip(ref, vmin, vmax).astype(np.float32)
    
    img_norm = (img_clip - vmin) / (vmax - vmin) * 2 - 1
    ref_norm = (ref_clip - vmin) / (vmax - vmin) * 2 - 1
    
    img_3ch = np.stack([img_norm, img_norm, img_norm], axis=0)[np.newaxis, ...]
    ref_3ch = np.stack([ref_norm, ref_norm, ref_norm], axis=0)[np.newaxis, ...]
    
    img_tensor = torch.from_numpy(img_3ch).to(device)
    ref_tensor = torch.from_numpy(ref_3ch).to(device)
    
    with torch.no_grad():
        lpips_val = loss_fn(img_tensor, ref_tensor)
    
    return lpips_val.item()


def calc_metrics(img, ref, vmin, vmax, loss_fn=None, device=None):
    img_clip = np.clip(img, vmin, vmax)
    ref_clip = np.clip(ref, vmin, vmax)
    data_range = vmax - vmin
    
    metrics = {
        'MAE': calc_mae(img_clip, ref_clip),
        'PSNR': calc_psnr(img_clip, ref_clip, data_range),
        'SSIM': calc_ssim(img_clip, ref_clip, data_range),
    }
    
    if loss_fn is not None:
        metrics['LPIPS'] = calc_lpips(img, ref, loss_fn, device, vmin, vmax)
    
    return metrics


def load_nii(path):
    if not os.path.exists(path):
        return None
    return nib.load(path).get_fdata().astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description='DDM2 Evaluation')
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file path')
    parser.add_argument('--patient_idx', type=int, default=8, help='Patient index')
    parser.add_argument('--all', action='store_true', help='Evaluate all patients')
    parser.add_argument('--output', type=str, default=None, help='Output Excel file')
    parser.add_argument('--eval_window', type=float, nargs=2, default=[0, 100], help='HU window for evaluation')
    parser.add_argument('--gt_dir', type=str, default='/host/d/file/gt', help='GT directory')
    parser.add_argument('--ddm2_dir', type=str, default=None, help='DDM2 results directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--use_final', action='store_true', default=True, help='Use DDM2 final result')
    parser.add_argument('--slice_offset', type=int, default=30, help='Slice offset for noise data')
    return parser.parse_args()


def evaluate_patient(patient_id, patient_subid, gt_dir, noise_dir, n2n_dir, ddm2_dir, 
                     eval_vmin, eval_vmax, loss_fn, device, use_final=True, 
                     slice_range=(30, 80), slice_offset=30):
    """评估单个患者"""
    
    pid_str = f"{int(patient_id):08d}"
    psid_str = f"{int(patient_subid):010d}"
    
    # 文件路径
    gt_path = os.path.join(gt_dir, pid_str, psid_str, 'gt_img.nii.gz')
    if not os.path.exists(gt_path):
        gt_path = os.path.join(gt_dir, 'gt_img.nii.gz')
    
    noise_path = os.path.join(noise_dir, pid_str, psid_str, 'gaussian_random_0', 'recon.nii.gz')
    n2n_path = os.path.join(n2n_dir, pid_str, psid_str, 'random_0', 'epoch78', 'pred_img.nii.gz')
    
    ddm2_first_path = os.path.join(ddm2_dir, pid_str, psid_str, 'ddm2_first_step.nii.gz')
    ddm2_final_path = os.path.join(ddm2_dir, pid_str, psid_str, 'ddm2_final.nii.gz')
    
    # 也尝试不带前导零的路径
    if not os.path.exists(ddm2_first_path):
        alt_pid = str(int(patient_id))
        alt_psid = str(int(patient_subid))
        ddm2_first_path = os.path.join(ddm2_dir, alt_pid, alt_psid, 'ddm2_first_step.nii.gz')
        ddm2_final_path = os.path.join(ddm2_dir, alt_pid, alt_psid, 'ddm2_final.nii.gz')
    
    # 加载数据
    gt_data = load_nii(gt_path)
    noise_data = load_nii(noise_path)
    n2n_data = load_nii(n2n_path)
    ddm2_first_data = load_nii(ddm2_first_path)
    ddm2_final_data = load_nii(ddm2_final_path)
    
    # 检查数据
    missing = []
    if gt_data is None:
        missing.append(f'GT: {gt_path}')
    if noise_data is None:
        missing.append(f'Noise: {noise_path}')
    if n2n_data is None:
        missing.append(f'N2N: {n2n_path}')
    if ddm2_first_data is None and ddm2_final_data is None:
        missing.append(f'DDM2: {ddm2_first_path}')
    
    if gt_data is None or noise_data is None or n2n_data is None:
        return None, missing
    
    # 确定 slice 数量
    num_slices = min(
        gt_data.shape[2],
        n2n_data.shape[2],
        ddm2_first_data.shape[2] if ddm2_first_data is not None else 999,
        ddm2_final_data.shape[2] if ddm2_final_data is not None else 999
    )
    
    # 计算每个 slice 的指标
    results = {
        'Noisy': {'MAE': [], 'PSNR': [], 'SSIM': [], 'LPIPS': []},
        'N2N': {'MAE': [], 'PSNR': [], 'SSIM': [], 'LPIPS': []},
        'DDM2_First': {'MAE': [], 'PSNR': [], 'SSIM': [], 'LPIPS': []},
        'DDM2_Final': {'MAE': [], 'PSNR': [], 'SSIM': [], 'LPIPS': []},
    }
    
    for s in range(num_slices):
        gt_slice = gt_data[:, :, s]
        noise_slice = noise_data[:, :, s + slice_offset]
        n2n_slice = n2n_data[:, :, s]
        
        # Noisy vs GT
        m = calc_metrics(noise_slice, gt_slice, eval_vmin, eval_vmax, loss_fn, device)
        for k, v in m.items():
            if v is not None:
                results['Noisy'][k].append(v)
        
        # N2N vs GT
        m = calc_metrics(n2n_slice, gt_slice, eval_vmin, eval_vmax, loss_fn, device)
        for k, v in m.items():
            if v is not None:
                results['N2N'][k].append(v)
        
        # DDM2 First vs GT
        if ddm2_first_data is not None:
            ddm2_first_slice = ddm2_first_data[:, :, s]
            m = calc_metrics(ddm2_first_slice, gt_slice, eval_vmin, eval_vmax, loss_fn, device)
            for k, v in m.items():
                if v is not None:
                    results['DDM2_First'][k].append(v)
        
        # DDM2 Final vs GT
        if ddm2_final_data is not None:
            ddm2_final_slice = ddm2_final_data[:, :, s]
            m = calc_metrics(ddm2_final_slice, gt_slice, eval_vmin, eval_vmax, loss_fn, device)
            for k, v in m.items():
                if v is not None:
                    results['DDM2_Final'][k].append(v)
    
    # 计算平均值
    avg_results = {}
    for method, metrics in results.items():
        avg_results[method] = {}
        for metric, values in metrics.items():
            if values:
                avg_results[method][metric] = np.mean(values)
                avg_results[method][f'{metric}_std'] = np.std(values)
    
    return avg_results, missing


def main():
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.config, 'r') as f:
        opt = json.load(f)
    
    eval_vmin, eval_vmax = args.eval_window
    
    noise_dir = opt['datasets']['val'].get('data_root', '/host/d/file/simulation/')
    n2n_dir = opt['datasets']['val'].get('teacher_n2n_root', '/host/d/file/pre/noise2noise/pred_images/')
    
    if args.ddm2_dir is None:
        for d in sorted(os.listdir('experiments'), reverse=True):
            if d.startswith('ct_denoise_'):
                infer_dir = os.path.join('experiments', d, 'inference')
                if os.path.exists(infer_dir):
                    args.ddm2_dir = infer_dir
                    break
    
    if args.ddm2_dir is None:
        args.ddm2_dir = '/host/d/file/pre/ddm2/pred_images'
    
    print("=" * 70)
    print("DDM2 Quantitative Evaluation")
    print("=" * 70)
    print(f"GT dir: {args.gt_dir}")
    print(f"Noise dir: {noise_dir}")
    print(f"N2N dir: {n2n_dir}")
    print(f"DDM2 dir: {args.ddm2_dir}")
    print(f"Eval window: [{eval_vmin}, {eval_vmax}] HU")
    print(f"Slice offset: {args.slice_offset}")
    
    loss_fn = None
    if HAS_LPIPS:
        loss_fn = lpips.LPIPS(net='alex').to(device)
        print("LPIPS initialized")
    
    # 获取患者列表
    import data as Data
    val_opt = opt['datasets']['val'].copy()
    val_opt['val_volume_idx'] = 'all'
    val_opt['val_slice_idx'] = 'all'
    val_set = Data.create_dataset(val_opt, 'val', stage2_file=opt.get('stage2_file'))
    
    slice_range = opt['datasets']['val'].get('slice_range', [30, 80])
    
    if args.all:
        patient_indices = list(range(len(val_set.n2n_pairs)))
    else:
        patient_indices = [args.patient_idx]
    
    print(f"\nEvaluating {len(patient_indices)} patient(s)...")
    
    all_results = []
    
    for patient_idx in tqdm(patient_indices, desc="Evaluating"):
        if patient_idx >= len(val_set.n2n_pairs):
            continue
        
        pair = val_set.n2n_pairs[patient_idx]
        patient_id = pair['patient_id']
        patient_subid = pair['patient_subid']
        
        results, missing = evaluate_patient(
            patient_id, patient_subid,
            args.gt_dir, noise_dir, n2n_dir, args.ddm2_dir,
            eval_vmin, eval_vmax, loss_fn, device,
            args.use_final, slice_range, args.slice_offset
        )
        
        if results is None:
            tqdm.write(f"[Skip] Patient {patient_id}: missing files")
            for m in missing:
                tqdm.write(f"  - {m}")
            continue
        
        row = {
            'Patient_ID': patient_id,
            'Patient_SubID': patient_subid,
        }
        
        for method in ['Noisy', 'N2N', 'DDM2_First', 'DDM2_Final']:
            for metric in ['MAE', 'PSNR', 'SSIM', 'LPIPS']:
                if method in results and metric in results[method]:
                    row[f'{method}_{metric}'] = results[method][metric]
        
        all_results.append(row)
        
        if not args.all:
            print("\n" + "=" * 70)
            print(f"Patient: {patient_id} / {patient_subid}")
            print("-" * 70)
            print(f"{'Method':<15} {'MAE ↓':>10} {'PSNR ↑':>10} {'SSIM ↑':>10} {'LPIPS ↓':>10}")
            print("-" * 70)
            for method in ['Noisy', 'N2N', 'DDM2_First', 'DDM2_Final']:
                if method in results:
                    mae = results[method].get('MAE', float('nan'))
                    psnr = results[method].get('PSNR', float('nan'))
                    ssim_val = results[method].get('SSIM', float('nan'))
                    lpips_val = results[method].get('LPIPS', float('nan'))
                    print(f"{method:<15} {mae:>10.4f} {psnr:>10.2f} {ssim_val:>10.4f} {lpips_val:>10.4f}")
            print("=" * 70)
    
    if args.all and all_results:
        df = pd.DataFrame(all_results)
        
        print("\n" + "=" * 70)
        print("Average Results (across all patients)")
        print("-" * 70)
        print(f"{'Method':<15} {'MAE ↓':>10} {'PSNR ↑':>10} {'SSIM ↑':>10} {'LPIPS ↓':>10}")
        print("-" * 70)
        
        for method in ['Noisy', 'N2N', 'DDM2_First', 'DDM2_Final']:
            mae = df[f'{method}_MAE'].mean() if f'{method}_MAE' in df.columns else float('nan')
            psnr = df[f'{method}_PSNR'].mean() if f'{method}_PSNR' in df.columns else float('nan')
            ssim_val = df[f'{method}_SSIM'].mean() if f'{method}_SSIM' in df.columns else float('nan')
            lpips_val = df[f'{method}_LPIPS'].mean() if f'{method}_LPIPS' in df.columns else float('nan')
            print(f"{method:<15} {mae:>10.4f} {psnr:>10.2f} {ssim_val:>10.4f} {lpips_val:>10.4f}")
        
        print("=" * 70)
        
        if args.output:
            avg_row = {'Patient_ID': 'Average', 'Patient_SubID': ''}
            for col in df.columns:
                if col not in ['Patient_ID', 'Patient_SubID']:
                    avg_row[col] = df[col].mean()
            
            df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
            df.to_excel(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
