import numpy as np
import os

npy_dir = '/host/d/file/simulation_npy/'
for root, dirs, files in os.walk(npy_dir):
    for f in files[:5]:
        if f.endswith('.npy'):
            path = os.path.join(root, f)
            data = np.load(path)
            print(f'{f}: min={data.min():.4f}, max={data.max():.4f}, nan={np.isnan(data).sum()}, inf={np.isinf(data).sum()}')
    break
