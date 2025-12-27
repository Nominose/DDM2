# %%
import nibabel as nb
import os
import lpips
import torch
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

# %%
# ========== 修改这里的路径 ==========
# 患者信息（目前只有一个患者）
patient_list = [
    {'patient_id': '00214841', 'patient_subid': '0000455418', 'random_n': 0}
]

# 文件路径配置
GT_DIR = '/host/d/file/gt'
NOISE_DIR = '/host/d/file/simulation'
N2N_DIR = '/host/d/file/pre/noise2noise/pred_images'
DDM2_DIR = 'experiments/ct_denoise_251226_164507/inference'  # 改成你的实验目录

# 评估窗口
VMIN = 0
VMAX = 100

# 输出文件
OUTPUT_EXCEL = '/host/c/Users/ROG/Desktop/ddm2_quantitative_results.xlsx'

# Noise slice offset
SLICE_OFFSET = 30

# %%
def calc_mae_with_ref_window(img, ref, vmin, vmax):
    """计算 MAE（只在参考图像的 [vmin, vmax] 范围内计算）"""
    maes = []
    for slice_num in range(0, img.shape[-1]):
        slice_img = img[:,:,slice_num]
        slice_ref = ref[:,:,slice_num]
        mask = np.where((slice_ref >= vmin) & (slice_ref <= vmax), 1, 0)
        if np.sum(mask) > 0:
            mae = np.sum(np.abs(slice_img - slice_ref) * mask) / np.sum(mask)
            maes.append(mae)
    return np.mean(maes), np.std(maes)

# %%
def calc_ssim_with_ref_window(img, ref, vmin, vmax):
    """计算 SSIM（只在参考图像的 [vmin, vmax] 范围内计算）"""
    ssims = []
    for slice_num in range(0, img.shape[-1]):
        slice_img = img[:,:,slice_num]
        slice_ref = ref[:,:,slice_num]
        mask = np.where((slice_ref >= vmin) & (slice_ref <= vmax), 1, 0)
        if np.sum(mask) > 0:
            _, ssim_map = structural_similarity(slice_img, slice_ref, data_range=vmax - vmin, full=True)
            ssim = np.sum(ssim_map * mask) / np.sum(mask)
            ssims.append(ssim)
    return np.mean(ssims), np.std(ssims)

# %%
def calc_lpips(imgs1, imgs2, vmin, vmax, loss_fn, device):
    """计算 LPIPS"""
    lpipss = []
    for slice_num in range(0, imgs1.shape[-1]):
        slice1 = imgs1[:,:,slice_num]
        slice2 = imgs2[:,:,slice_num]

        slice1 = np.clip(slice1, vmin, vmax).astype(np.float32)
        slice2 = np.clip(slice2, vmin, vmax).astype(np.float32)

        slice1 = (slice1 - vmin) / (vmax - vmin) * 2 - 1
        slice2 = (slice2 - vmin) / (vmax - vmin) * 2 - 1

        slice1 = np.stack([slice1, slice1, slice1], axis=-1)
        slice2 = np.stack([slice2, slice2, slice2], axis=-1)

        slice1 = np.transpose(slice1, (2, 0, 1))[np.newaxis, ...]
        slice2 = np.transpose(slice2, (2, 0, 1))[np.newaxis, ...]

        slice1 = torch.from_numpy(slice1).to(device)
        slice2 = torch.from_numpy(slice2).to(device)

        lpips_val = loss_fn(slice1, slice2)
        lpipss.append(lpips_val.item())

    return np.mean(lpipss), np.std(lpipss)

# %%
# 初始化 LPIPS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = lpips.LPIPS().to(device)
print(f"LPIPS initialized on {device}")

# %%
## 主循环：计算指标
results = []

for patient in patient_list:
    patient_id = patient['patient_id']
    patient_subid = patient['patient_subid']
    random_n = patient['random_n']
    
    print(f"\n{'='*60}")
    print(f"Processing: {patient_id} / {patient_subid} / random_{random_n}")
    print('='*60)

    # ========== 加载数据 ==========
    
    # GT (ground truth)
    gt_file = os.path.join(GT_DIR, 'gt_img.nii.gz')
    # 如果 GT 在患者子目录下，用这个：
    # gt_file = os.path.join(GT_DIR, patient_id, patient_subid, 'gt_img.nii.gz')
    gt_img = nb.load(gt_file).get_fdata()
    print(f"GT: {gt_file}, shape: {gt_img.shape}")

    # Noisy image
    noise_file = os.path.join(NOISE_DIR, patient_id, patient_subid, f'gaussian_random_{random_n}', 'recon.nii.gz')
    noise_img = nb.load(noise_file).get_fdata()
    # 对齐 slice（noise 有 100 slices，GT/N2N/DDM2 有 50 slices）
    noise_img = noise_img[:, :, SLICE_OFFSET:SLICE_OFFSET+gt_img.shape[-1]]
    print(f"Noisy: {noise_file}, shape after offset: {noise_img.shape}")

    # Noise2Noise
    n2n_file = os.path.join(N2N_DIR, patient_id, patient_subid, f'random_{random_n}', 'epoch78', 'pred_img.nii.gz')
    n2n_img = nb.load(n2n_file).get_fdata()
    print(f"N2N: {n2n_file}, shape: {n2n_img.shape}")

    # DDM2 First-step
    # 注意：DDM2 输出的 patient_id 格式可能不同（无前导零）
    ddm2_pid = str(int(patient_id))  # 去掉前导零
    ddm2_psid = str(int(patient_subid))
    
    ddm2_first_file = os.path.join(DDM2_DIR, ddm2_pid, ddm2_psid, 'ddm2_first_step.nii.gz')
    if not os.path.exists(ddm2_first_file):
        # 尝试带前导零的路径
        ddm2_first_file = os.path.join(DDM2_DIR, patient_id, patient_subid, 'ddm2_first_step.nii.gz')
    ddm2_first_img = nb.load(ddm2_first_file).get_fdata()
    print(f"DDM2 First: {ddm2_first_file}, shape: {ddm2_first_img.shape}")

    # DDM2 Final
    ddm2_final_file = os.path.join(DDM2_DIR, ddm2_pid, ddm2_psid, 'ddm2_final.nii.gz')
    if not os.path.exists(ddm2_final_file):
        ddm2_final_file = os.path.join(DDM2_DIR, patient_id, patient_subid, 'ddm2_final.nii.gz')
    ddm2_final_img = nb.load(ddm2_final_file).get_fdata()
    print(f"DDM2 Final: {ddm2_final_file}, shape: {ddm2_final_img.shape}")

    # 确保所有数据形状一致
    min_slices = min(gt_img.shape[-1], noise_img.shape[-1], n2n_img.shape[-1], 
                     ddm2_first_img.shape[-1], ddm2_final_img.shape[-1])
    gt_img = gt_img[:,:,:min_slices]
    noise_img = noise_img[:,:,:min_slices]
    n2n_img = n2n_img[:,:,:min_slices]
    ddm2_first_img = ddm2_first_img[:,:,:min_slices]
    ddm2_final_img = ddm2_final_img[:,:,:min_slices]
    
    print(f"\nAligned to {min_slices} slices")

    # ========== 计算指标 ==========
    print(f"\nCalculating metrics (window: [{VMIN}, {VMAX}])...")
    
    # MAE
    mae_noise, mae_noise_std = calc_mae_with_ref_window(noise_img, gt_img, VMIN, VMAX)
    mae_n2n, mae_n2n_std = calc_mae_with_ref_window(n2n_img, gt_img, VMIN, VMAX)
    mae_ddm2_first, mae_ddm2_first_std = calc_mae_with_ref_window(ddm2_first_img, gt_img, VMIN, VMAX)
    mae_ddm2_final, mae_ddm2_final_std = calc_mae_with_ref_window(ddm2_final_img, gt_img, VMIN, VMAX)
    
    # SSIM
    ssim_noise, ssim_noise_std = calc_ssim_with_ref_window(noise_img, gt_img, VMIN, VMAX)
    ssim_n2n, ssim_n2n_std = calc_ssim_with_ref_window(n2n_img, gt_img, VMIN, VMAX)
    ssim_ddm2_first, ssim_ddm2_first_std = calc_ssim_with_ref_window(ddm2_first_img, gt_img, VMIN, VMAX)
    ssim_ddm2_final, ssim_ddm2_final_std = calc_ssim_with_ref_window(ddm2_final_img, gt_img, VMIN, VMAX)
    
    # LPIPS
    lpips_noise, _ = calc_lpips(noise_img, gt_img, VMIN, VMAX, loss_fn, device)
    lpips_n2n, _ = calc_lpips(n2n_img, gt_img, VMIN, VMAX, loss_fn, device)
    lpips_ddm2_first, _ = calc_lpips(ddm2_first_img, gt_img, VMIN, VMAX, loss_fn, device)
    lpips_ddm2_final, _ = calc_lpips(ddm2_final_img, gt_img, VMIN, VMAX, loss_fn, device)

    # ========== 打印结果 ==========
    print(f"\n{'Method':<15} {'MAE ↓':>12} {'SSIM ↑':>12} {'LPIPS ↓':>12}")
    print('-' * 55)
    print(f"{'Noisy':<15} {mae_noise:>12.4f} {ssim_noise:>12.4f} {lpips_noise:>12.4f}")
    print(f"{'N2N':<15} {mae_n2n:>12.4f} {ssim_n2n:>12.4f} {lpips_n2n:>12.4f}")
    print(f"{'DDM2_First':<15} {mae_ddm2_first:>12.4f} {ssim_ddm2_first:>12.4f} {lpips_ddm2_first:>12.4f}")
    print(f"{'DDM2_Final':<15} {mae_ddm2_final:>12.4f} {ssim_ddm2_final:>12.4f} {lpips_ddm2_final:>12.4f}")

    # ========== 保存结果 ==========
    results.append([
        patient_id, patient_subid, random_n,
        mae_noise, mae_n2n, mae_ddm2_first, mae_ddm2_final,
        ssim_noise, ssim_n2n, ssim_ddm2_first, ssim_ddm2_final,
        lpips_noise, lpips_n2n, lpips_ddm2_first, lpips_ddm2_final,
        # 标准差
        mae_noise_std, mae_n2n_std, mae_ddm2_first_std, mae_ddm2_final_std,
        ssim_noise_std, ssim_n2n_std, ssim_ddm2_first_std, ssim_ddm2_final_std,
    ])

# %%
# 保存到 Excel
df = pd.DataFrame(results, columns=[
    'patient_id', 'patient_subid', 'random_n',
    'mae_noisy', 'mae_n2n', 'mae_ddm2_first', 'mae_ddm2_final',
    'ssim_noisy', 'ssim_n2n', 'ssim_ddm2_first', 'ssim_ddm2_final',
    'lpips_noisy', 'lpips_n2n', 'lpips_ddm2_first', 'lpips_ddm2_final',
    'mae_noisy_std', 'mae_n2n_std', 'mae_ddm2_first_std', 'mae_ddm2_final_std',
    'ssim_noisy_std', 'ssim_n2n_std', 'ssim_ddm2_first_std', 'ssim_ddm2_final_std',
])

# 添加平均行（如果有多个患者）
if len(results) > 1:
    avg_row = ['Average', '', '']
    for col in df.columns[3:]:
        avg_row.append(df[col].mean())
    df.loc[len(df)] = avg_row

df.to_excel(OUTPUT_EXCEL, index=False)
print(f"\n{'='*60}")
print(f"Results saved to: {OUTPUT_EXCEL}")
print('='*60)

# %%
# 打印最终汇总
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\n{'Method':<15} {'MAE ↓':>12} {'SSIM ↑':>12} {'LPIPS ↓':>12}")
print('-' * 55)
print(f"{'Noisy':<15} {df['mae_noisy'].mean():>12.4f} {df['ssim_noisy'].mean():>12.4f} {df['lpips_noisy'].mean():>12.4f}")
print(f"{'N2N':<15} {df['mae_n2n'].mean():>12.4f} {df['ssim_n2n'].mean():>12.4f} {df['lpips_n2n'].mean():>12.4f}")
print(f"{'DDM2_First':<15} {df['mae_ddm2_first'].mean():>12.4f} {df['ssim_ddm2_first'].mean():>12.4f} {df['lpips_ddm2_first'].mean():>12.4f}")
print(f"{'DDM2_Final':<15} {df['mae_ddm2_final'].mean():>12.4f} {df['ssim_ddm2_final'].mean():>12.4f} {df['lpips_ddm2_final'].mean():>12.4f}")
print("=" * 60)
