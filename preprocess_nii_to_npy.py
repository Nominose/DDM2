"""
预处理脚本：将 .nii.gz 转换为 .npy 格式
.npy 读取速度比 .nii.gz 快 10-50 倍
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

EXCEL_PATH = "/host/d/file/fixedCT_static_simulation_train_test_gaussian_local.xlsx"
DATA_ROOT = "/host/d/file/simulation/"
OUTPUT_DIR = "/host/d/file/simulation_npy/"

df = pd.read_excel(EXCEL_PATH)
noise_files = df['noise_file'].unique()
print(f"Found {len(noise_files)} files")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for nii_path in tqdm(noise_files, desc="Converting"):
    full_path = nii_path.replace('/host/d/file/simulation/', DATA_ROOT)
    rel_path = nii_path.replace('/host/d/file/simulation/', '')
    npy_path = os.path.join(OUTPUT_DIR, rel_path.replace('.nii.gz', '.npy'))
    
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    
    if os.path.exists(full_path):
        try:
            data = nib.load(full_path).get_fdata().astype(np.float32)
            np.save(npy_path, data)
        except Exception as e:
            print(f"Error: {full_path}: {e}")

print(f"Done! Output: {OUTPUT_DIR}")