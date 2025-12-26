"""
CT Dataset for DDM2 - with histogram equalization and auto slice offset detection
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import nibabel as nib


def apply_histogram_equalization(img, bins, bins_mapped):
    """
    Apply histogram equalization using pre-computed bins mapping.
    """
    if bins is None or bins_mapped is None:
        return img
    
    flat_img = img.flatten()
    bin_indices = np.digitize(flat_img, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins_mapped) - 1)
    equalized = bins_mapped[bin_indices]
    
    return equalized.reshape(img.shape).astype(np.float32)  # 确保 float32


class CTDataset(Dataset):

    def __init__(
        self,
        dataroot,
        valid_mask,
        phase='train',
        image_size=512,
        in_channel=1,
        val_volume_idx=0,
        val_slice_idx=25,
        padding=3,
        lr_flip=0.5,
        stage2_file=None,
        data_root=None,
        train_batches=(0, 1, 2, 3, 4),
        val_batches=(5,),
        slice_range=None,
        HU_MIN=-1000.0,
        HU_MAX=2000.0,
        teacher_n2n_root=None,
        teacher_n2n_epoch=78,
        histogram_equalization=True,
        bins_file=None,
        bins_mapped_file=None,
        **kwargs
    ):
        self.phase = phase
        self.image_size = image_size
        self.in_channel = in_channel
        self.padding = padding // 2
        self.lr_flip = lr_flip
        self.data_root = data_root
        self.HU_MIN = HU_MIN
        self.HU_MAX = HU_MAX
        self.teacher_n2n_root = teacher_n2n_root
        self.teacher_n2n_epoch = teacher_n2n_epoch
        
        # Histogram equalization settings
        self.histogram_equalization = histogram_equalization
        self.bins = None
        self.bins_mapped = None
        
        if histogram_equalization:
            if bins_file is not None and bins_mapped_file is not None:
                if os.path.exists(bins_file) and os.path.exists(bins_mapped_file):
                    self.bins = np.load(bins_file).astype(np.float32)
                    self.bins_mapped = np.load(bins_mapped_file).astype(np.float32)
                    print(f'[{phase}] Histogram equalization enabled:')
                    print(f'    bins: {bins_file} (shape: {self.bins.shape})')
                    print(f'    bins_mapped: {bins_mapped_file} (shape: {self.bins_mapped.shape})')
                else:
                    print(f'[WARNING] Histogram equalization files not found:')
                    print(f'    bins_file: {bins_file} (exists: {os.path.exists(bins_file) if bins_file else False})')
                    print(f'    bins_mapped_file: {bins_mapped_file} (exists: {os.path.exists(bins_mapped_file) if bins_mapped_file else False})')
                    self.histogram_equalization = False
            else:
                print(f'[WARNING] Histogram equalization enabled but no bins files specified, disabled.')
                self.histogram_equalization = False

        assert isinstance(valid_mask, (list, tuple)) and len(valid_mask) == 2

        self.df = pd.read_excel(dataroot)
        target_batches = train_batches if phase == 'train' else val_batches
        self.df = self.df[self.df['batch'].isin(target_batches)].reset_index(drop=True)

        self.n2n_pairs = self._build_n2n_pairs(self.df)
        self.data_shape = self._infer_shape()
        W, H, S = self.data_shape

        v_start = max(int(valid_mask[0]), 0)
        v_end = min(int(valid_mask[1]), len(self.n2n_pairs))
        self.n2n_pairs = self.n2n_pairs[v_start:v_end]

        if slice_range is None:
            self.slice_start, self.slice_end = 0, S
        else:
            self.slice_start = max(int(slice_range[0]), 0)
            self.slice_end = min(int(slice_range[1]), S)
        self.num_slices = self.slice_end - self.slice_start

        # Auto detect slice offset
        if self.teacher_n2n_root is not None and len(self.n2n_pairs) > 0:
            detected_offset = self._detect_slice_offset()
            if detected_offset is not None:
                if detected_offset != self.slice_start:
                    print(f'[WARNING] Detected offset ({detected_offset}) != config slice_start ({self.slice_start})')
                    print(f'[WARNING] Consider setting slice_range to [{detected_offset}, {detected_offset + self.num_slices}]')
                else:
                    print(f'[OK] Slice offset verified: {self.slice_start}')

        V = len(self.n2n_pairs)
        if val_volume_idx == 'all':
            self.val_volume_idx = list(range(V))
        elif isinstance(val_volume_idx, int):
            self.val_volume_idx = [val_volume_idx]
        else:
            self.val_volume_idx = list(val_volume_idx)
        self.val_volume_idx = [x for x in self.val_volume_idx if x < V]

        if val_slice_idx == 'all':
            self.val_slice_idx = list(range(self.num_slices))
        elif isinstance(val_slice_idx, int):
            self.val_slice_idx = [val_slice_idx]
        else:
            self.val_slice_idx = list(val_slice_idx)
        self.val_slice_idx = [x for x in self.val_slice_idx if x < self.num_slices]

        self.samples = self._build_sample_indices()
        self.matched_state = self._parse_stage2_file(stage2_file) if stage2_file else None

        self.data_size_before_padding = (W, H, self.num_slices, V)
        
        class FakeRawData:
            def __init__(self, shape):
                self.shape = shape
        self.raw_data = FakeRawData(self.data_size_before_padding)

        if phase == 'train':
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomVerticalFlip(lr_flip),
                transforms.RandomHorizontalFlip(lr_flip),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ])

        print(f'[{phase}] CTDataset: pairs={V}, slices={self.num_slices}, samples={len(self.samples)}')
        print(f'[{phase}] Noise slice_range: [{self.slice_start}, {self.slice_end})')
        print(f'[{phase}] HU range: [{self.HU_MIN}, {self.HU_MAX}]')
        print(f'[{phase}] Histogram equalization: {self.histogram_equalization}')
        if self.teacher_n2n_root:
            print(f'[{phase}] Using teacher N2N from: {self.teacher_n2n_root}')

    def _fix_path(self, path):
        if self.data_root is not None:
            return path.replace('/host/d/file/simulation/', self.data_root)
        return path

    def _get_npy_path(self, nii_path):
        return nii_path.replace('.nii.gz', '.npy').replace('/simulation/', '/simulation_npy/')

    def _teacher_n2n_exists(self, pred_path):
        if pred_path is None:
            return False
        npy_path = pred_path.replace('.nii.gz', '.npy')
        return os.path.exists(pred_path) or os.path.exists(npy_path)

    def _detect_slice_offset(self):
        """Auto detect slice offset between noise and teacher N2N"""
        for pair in self.n2n_pairs:
            pred_path = self._get_teacher_n2n_path(pair['patient_id'], pair['patient_subid'])
            if not self._teacher_n2n_exists(pred_path):
                continue
            
            noise_path = self._fix_path(pair['noise_0'])
            npy_noise = self._get_npy_path(noise_path)
            
            try:
                if os.path.exists(npy_noise):
                    noise_data = np.load(npy_noise)
                elif os.path.exists(noise_path):
                    noise_data = nib.load(noise_path).get_fdata()
                else:
                    continue
                
                npy_teacher = pred_path.replace('.nii.gz', '.npy')
                if os.path.exists(npy_teacher):
                    teacher_data = np.load(npy_teacher)
                elif os.path.exists(pred_path):
                    teacher_data = nib.load(pred_path).get_fdata()
                else:
                    continue
            except Exception as e:
                print(f"[WARNING] Failed to load data: {e}")
                continue
            
            noise_norm = np.clip((noise_data - self.HU_MIN) / (self.HU_MAX - self.HU_MIN), 0, 1)
            teacher_norm = np.clip((teacher_data - self.HU_MIN) / (self.HU_MAX - self.HU_MIN), 0, 1)
            
            noise_slices = noise_data.shape[2]
            teacher_slices = teacher_data.shape[2]
            
            print(f"[Slice Detection] Noise: {noise_slices} slices, Teacher: {teacher_slices} slices")
            
            best_offset = 0
            best_corr = -1
            
            for offset in range(0, noise_slices - teacher_slices + 1):
                corrs = []
                test_slices = [0, teacher_slices//4, teacher_slices//2, 3*teacher_slices//4, min(teacher_slices-1, 49)]
                for ts in test_slices:
                    ns = ts + offset
                    if ns < noise_slices and ts < teacher_slices:
                        corr = np.corrcoef(noise_norm[:,:,ns].flatten(), teacher_norm[:,:,ts].flatten())[0,1]
                        if not np.isnan(corr):
                            corrs.append(corr)
                
                if corrs:
                    mean_corr = np.mean(corrs)
                    if mean_corr > best_corr:
                        best_corr = mean_corr
                        best_offset = offset
            
            print(f"[Slice Detection] Best offset: {best_offset}, correlation: {best_corr:.4f}")
            
            if best_corr < 0.8:
                print(f"[WARNING] Low correlation! Slice alignment may be incorrect.")
            
            return best_offset
        
        return None

    def _build_n2n_pairs(self, df):
        bad_patients = {8527, 10431, 10461, 10536, 11638, 11640, 19591, 30597,
                        76624, 102364, 104563, 109021, 139437, 148611, 154227, 172147}
        
        pairs = []
        grouped = df.groupby(['Patient_ID', 'Patient_subID'])
        
        for (pid, psid), g in grouped:
            try:
                pid_int = int(pid)
            except:
                pid_int = -1
            
            if pid_int in bad_patients:
                continue

            g0 = g[g['random_num'] == 0]
            g1 = g[g['random_num'] == 1]
            
            if len(g0) > 0 and len(g1) > 0:
                pairs.append({
                    'noise_0': g0.iloc[0]['noise_file'],
                    'noise_1': g1.iloc[0]['noise_file'],
                    'patient_id': pid,
                    'patient_subid': psid
                })
        
        print(f'Found {len(pairs)} N2N pairs')
        return pairs

    def _infer_shape(self):
        default = (512, 512, 100)
        if len(self.n2n_pairs) == 0:
            return default

        p = self._fix_path(self.n2n_pairs[0]['noise_0'])
        npy_path = self._get_npy_path(p)
        
        if os.path.exists(npy_path):
            data = np.load(npy_path, mmap_mode='r')
            if len(data.shape) >= 3:
                return (int(data.shape[0]), int(data.shape[1]), int(data.shape[2]))
        
        if os.path.exists(p):
            nii = nib.load(p)
            if len(nii.shape) >= 3:
                return (int(nii.shape[0]), int(nii.shape[1]), int(nii.shape[2]))
        
        return default

    def _build_sample_indices(self):
        samples = []
        
        if self.phase in ('train', 'test'):
            for vol_idx in range(len(self.n2n_pairs)):
                if self.teacher_n2n_root is not None:
                    pair = self.n2n_pairs[vol_idx]
                    pred_path = self._get_teacher_n2n_path(pair['patient_id'], pair['patient_subid'])
                    if not self._teacher_n2n_exists(pred_path):
                        continue
                
                for slice_idx in range(self.num_slices):
                    samples.append((vol_idx, slice_idx))
        else:
            for vol_idx in self.val_volume_idx:
                if self.teacher_n2n_root is not None:
                    if vol_idx < len(self.n2n_pairs):
                        pair = self.n2n_pairs[vol_idx]
                        pred_path = self._get_teacher_n2n_path(pair['patient_id'], pair['patient_subid'])
                        if not self._teacher_n2n_exists(pred_path):
                            continue
                
                for slice_idx in self.val_slice_idx:
                    samples.append((vol_idx, slice_idx))
        
        return samples

    def _parse_stage2_file(self, file_path):
        if file_path is None or not os.path.exists(file_path):
            return None
        
        results = {}
        with open(file_path, 'r') as f:
            for line in f:
                info = line.strip().split('_')
                if len(info) >= 3:
                    v, s, t = int(info[0]), int(info[1]), int(info[2])
                    results.setdefault(v, {})[s] = t
        return results

    def _preprocess_image(self, img):
        """
        Preprocess image: histogram equalization -> HU cutoff -> normalize
        
        Args:
            img: Raw image data (原始 HU 值)
        
        Returns:
            Preprocessed image in [0, 1] range, dtype=float32
        """
        # 确保输入是 float32
        img = img.astype(np.float32)
        
        # Step 1: Histogram equalization (在 HU cutoff 之前)
        if self.histogram_equalization and self.bins is not None:
            img = apply_histogram_equalization(img, self.bins, self.bins_mapped)
        
        # Step 2: HU cutoff and normalization
        img = np.clip(img, self.HU_MIN, self.HU_MAX)
        img = (img - self.HU_MIN) / (self.HU_MAX - self.HU_MIN)
        
        # 确保输出是 float32
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def _load_slice(self, nii_path, slice_idx):
        """Load noise slice - adds slice_start offset"""
        nii_path = self._fix_path(nii_path)
        npy_path = self._get_npy_path(nii_path)
        
        # Add offset for noise data
        actual_idx = slice_idx + self.slice_start
        
        if os.path.exists(npy_path):
            vol_mmap = np.load(npy_path, mmap_mode='r')
            if vol_mmap.ndim >= 3:
                actual_idx = max(0, min(actual_idx, vol_mmap.shape[2] - 1))
                img = np.array(vol_mmap[:, :, actual_idx], dtype=np.float32)
            else:
                img = np.array(vol_mmap, dtype=np.float32)
                
        elif os.path.exists(nii_path):
            nii = nib.load(nii_path)
            if nii.ndim >= 3:
                actual_idx = max(0, min(actual_idx, nii.shape[2] - 1))
                img = np.asarray(nii.dataobj[:, :, actual_idx], dtype=np.float32)
            else:
                img = nii.get_fdata().astype(np.float32)
        else:
            return np.zeros((self.data_shape[0], self.data_shape[1]), dtype=np.float32)

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply preprocessing (histogram equalization + normalization)
        return self._preprocess_image(img)

    def _get_teacher_n2n_path(self, patient_id, patient_subid):
        if self.teacher_n2n_root is None:
            return None
        
        pid_str = f"{int(patient_id):08d}"
        psid_str = f"{int(patient_subid):010d}"
        
        return os.path.join(
            self.teacher_n2n_root,
            pid_str,
            psid_str,
            "random_0",
            f"epoch{self.teacher_n2n_epoch}",
            "pred_img.nii.gz"
        )

    def _load_teacher_denoised(self, patient_id, patient_subid, slice_idx):
        """Load teacher N2N result - no offset"""
        pred_path = self._get_teacher_n2n_path(patient_id, patient_subid)
        
        if pred_path is None:
            return None
        
        npy_path = pred_path.replace('.nii.gz', '.npy')
        
        if os.path.exists(npy_path):
            vol_mmap = np.load(npy_path, mmap_mode='r')
            if vol_mmap.ndim >= 3:
                slice_idx = max(0, min(slice_idx, vol_mmap.shape[2] - 1))
                img = np.array(vol_mmap[:, :, slice_idx], dtype=np.float32)
            else:
                img = np.array(vol_mmap, dtype=np.float32)
        elif os.path.exists(pred_path):
            nii = nib.load(pred_path)
            data = nii.get_fdata().astype(np.float32)
            if data.ndim >= 3:
                slice_idx = max(0, min(slice_idx, data.shape[2] - 1))
                img = data[:, :, slice_idx]
            else:
                img = data
        else:
            return None

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply preprocessing (histogram equalization + normalization)
        return self._preprocess_image(img)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        volume_idx, slice_idx = self.samples[index]
        pair = self.n2n_pairs[volume_idx]

        n0 = self._load_slice(pair['noise_0'], slice_idx)
        n1 = self._load_slice(pair['noise_1'], slice_idx)

        teacher_denoised = None
        if self.teacher_n2n_root is not None:
            teacher_denoised = self._load_teacher_denoised(
                pair['patient_id'], 
                pair['patient_subid'], 
                slice_idx
            )

        if self.phase == 'train' and random.random() > 0.5:
            input_img, target_img = n1, n0
        else:
            input_img, target_img = n0, n1

        if self.padding > 0:
            cond_ch = 2 * self.padding
            channels = [input_img] * cond_ch + [target_img]
        else:
            channels = [input_img, target_img]

        has_teacher = teacher_denoised is not None
        if has_teacher:
            channels.append(teacher_denoised)

        raw_input = np.stack(channels, axis=-1).astype(np.float32)  # 确保 float32
        raw_input = self.transforms(raw_input)

        if has_teacher:
            denoised_tensor = raw_input[[-1], :, :]
            raw_input = raw_input[:-1, :, :]

        ret = {
            'X': raw_input[[-1], :, :].float(),  # 确保 float32
            'condition': raw_input[:-1, :, :].float()  # 确保 float32
        }

        if self.matched_state is not None:
            if volume_idx in self.matched_state and slice_idx in self.matched_state[volume_idx]:
                ret['matched_state'] = torch.tensor([float(self.matched_state[volume_idx][slice_idx])])
            else:
                ret['matched_state'] = torch.tensor([500.0])
        else:
            ret['matched_state'] = torch.tensor([500.0])

        if self.teacher_n2n_root is not None:
            if has_teacher:
                ret['denoised'] = denoised_tensor.float()  # 确保 float32
            else:
                ret['denoised'] = ret['X'].clone()
                ret['matched_state'] = torch.tensor([1.0])

        if torch.isnan(ret['X']).any() or torch.isinf(ret['X']).any():
            ret['X'] = torch.zeros_like(ret['X'])
        if torch.isnan(ret['condition']).any() or torch.isinf(ret['condition']).any():
            ret['condition'] = torch.zeros_like(ret['condition'])

        return ret
