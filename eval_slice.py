"""
DDM2 Single Slice Evaluation - 快速评估单个 slice
对比 GT、Noisy、N2N、DDM2 并生成可视化

Usage:
    python eval_slice.py --slice_idx 26
"""

import os
import sys
import argparse
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, '.')

try:
    import lpips
    HAS_LPIPS = True
except:
    HAS_LPIPS = False


def calc_metrics(img, ref, vmin, vmax, loss_fn=None, device=None):
    img_clip = np.clip(img, vmin, vmax)
    ref_clip = np.clip(ref, vmin, vmax)
    data_range = vmax - vmin
    
    mae = np.mean(np.abs(img_clip - ref_clip))
    mse = np.mean((img_clip - ref_clip) ** 2)
    psnr = 10 * np.log10(data_range ** 2 / mse) if mse > 0 else float('inf')
    ssim_val = ssim(img_clip, ref_clip, data_range=data_range)
    
    lpips_val = None
    if loss_fn is not None:
        img_norm = (img_clip - vmin) / (vmax - vmin) * 2 - 1
        ref_norm = (ref_clip - vmin) / (vmax - vmin) * 2 - 1
        img_3ch = np.stack([img_norm, img_norm, img_norm], axis=0)[np.newaxis, ...].astype(np.float32)
        ref_3ch = np.stack([ref_norm, ref_norm, ref_norm], axis=0)[np.newaxis, ...].astype(np.float32)
        with torch.no_grad():
            lpips_val = loss_fn(torch.from_numpy(img_3ch).to(device), 
                               torch.from_numpy(ref_3ch).to(device)).item()
    
    return {'MAE': mae, 'PSNR': psnr, 'SSIM': ssim_val, 'LPIPS': lpips_val}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slice_idx', type=int, default=26)
    parser.add_argument('--gt_file', type=str, default='/host/d/file/gt/gt_img.nii.gz')
    parser.add_argument('--noise_file', type=str, default='/host/d/file/simulation/00214841/0000455418/gaussian_random_0/recon.nii.gz')
    parser.add_argument('--n2n_file', type=str, default='/host/d/file/pre/noise2noise/pred_images/00214841/0000455418/random_0/epoch78/pred_img.nii.gz')
    parser.add_argument('--ddm2_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='/host/c/Users/ROG/Desktop/ddm2_eval')
    parser.add_argument('--eval_window', type=float, nargs=2, default=[0, 100])
    parser.add_argument('--display_window', type=float, nargs=2, default=[0, 80])
    parser.add_argument('--slice_offset', type=int, default=30)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    eval_vmin, eval_vmax = args.eval_window
    disp_vmin, disp_vmax = args.display_window
    
    print("=" * 70)
    print(f"Evaluating Slice {args.slice_idx}")
    print("=" * 70)
    
    print("\nLoading data...")
    gt_data = nib.load(args.gt_file).get_fdata()
    noise_data = nib.load(args.noise_file).get_fdata()
    n2n_data = nib.load(args.n2n_file).get_fdata()
    
    gt_slice = gt_data[:, :, args.slice_idx]
    noise_slice = noise_data[:, :, args.slice_idx + args.slice_offset]
    n2n_slice = n2n_data[:, :, args.slice_idx]
    
    ddm2_first = None
    ddm2_final = None
    
    if args.ddm2_dir is None:
        for d in sorted(os.listdir('experiments'), reverse=True):
            if d.startswith('ct_denoise_'):
                infer_dir = os.path.join('experiments', d, 'inference')
                if os.path.exists(infer_dir):
                    args.ddm2_dir = infer_dir
                    break
    
    if args.ddm2_dir:
        for pid_dir in os.listdir(args.ddm2_dir):
            pid_path = os.path.join(args.ddm2_dir, pid_dir)
            if os.path.isdir(pid_path):
                for psid_dir in os.listdir(pid_path):
                    psid_path = os.path.join(pid_path, psid_dir)
                    
                    first_path = os.path.join(psid_path, 'ddm2_first_step.nii.gz')
                    final_path = os.path.join(psid_path, 'ddm2_final.nii.gz')
                    
                    if os.path.exists(first_path):
                        ddm2_first = nib.load(first_path).get_fdata()[:, :, args.slice_idx]
                        print(f"Loaded DDM2 first: {first_path}")
                    
                    if os.path.exists(final_path):
                        ddm2_final = nib.load(final_path).get_fdata()[:, :, args.slice_idx]
                        print(f"Loaded DDM2 final: {final_path}")
                    
                    break
                break
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = lpips.LPIPS(net='alex').to(device) if HAS_LPIPS else None
    
    print("\nCalculating metrics...")
    
    results = {}
    results['Noisy'] = calc_metrics(noise_slice, gt_slice, eval_vmin, eval_vmax, loss_fn, device)
    results['N2N'] = calc_metrics(n2n_slice, gt_slice, eval_vmin, eval_vmax, loss_fn, device)
    
    if ddm2_first is not None:
        results['DDM2_First'] = calc_metrics(ddm2_first, gt_slice, eval_vmin, eval_vmax, loss_fn, device)
    
    if ddm2_final is not None:
        results['DDM2_Final'] = calc_metrics(ddm2_final, gt_slice, eval_vmin, eval_vmax, loss_fn, device)
    
    print("\n" + "=" * 70)
    print(f"{'Method':<15} {'MAE ↓':>10} {'PSNR ↑':>10} {'SSIM ↑':>10} {'LPIPS ↓':>10}")
    print("-" * 70)
    
    for method, m in results.items():
        lpips_str = f"{m['LPIPS']:.4f}" if m['LPIPS'] is not None else "N/A"
        print(f"{method:<15} {m['MAE']:>10.4f} {m['PSNR']:>10.2f} {m['SSIM']:>10.4f} {lpips_str:>10}")
    
    print("=" * 70)
    
    # 可视化
    images = [('GT', gt_slice, None), ('Noisy', noise_slice, 'Noisy'), ('N2N', n2n_slice, 'N2N')]
    if ddm2_first is not None:
        images.append(('DDM2_First', ddm2_first, 'DDM2_First'))
    if ddm2_final is not None:
        images.append(('DDM2_Final', ddm2_final, 'DDM2_Final'))
    
    n_cols = len(images)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    
    for i, (display_name, img, result_key) in enumerate(images):
        axes[0, i].imshow(img, cmap='gray', vmin=disp_vmin, vmax=disp_vmax)
        if result_key and result_key in results:
            axes[0, i].set_title(f'{display_name}\nSSIM={results[result_key]["SSIM"]:.4f}')
        else:
            axes[0, i].set_title(display_name)
        axes[0, i].axis('off')
        
        if result_key and result_key in results:
            diff = img - gt_slice
            axes[1, i].imshow(diff, cmap='RdBu', vmin=-30, vmax=30)
            axes[1, i].set_title(f'{display_name} - GT\nMAE={results[result_key]["MAE"]:.2f}')
        else:
            axes[1, i].imshow(np.zeros_like(gt_slice), cmap='gray')
            axes[1, i].set_title('Reference')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, f'slice{args.slice_idx}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")
    
    import json
    json_path = os.path.join(args.output_dir, f'slice{args.slice_idx}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved: {json_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
