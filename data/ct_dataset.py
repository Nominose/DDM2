"""
CT Dataset for DDM² - v2
支持 use_random_num 参数选择噪声实现

数据结构:
    Excel (200条) → 100 个 N2N pairs
    ├── 每个 pair 有 random_num=0 和 random_num=1 两个噪声实现
    ├── Batch 0-4: 84 pairs (训练)
    └── Batch 5: 16 pairs (验证)

use_random_num 选项:
    - 'both': 两个都用，N2N 训练时随机选择输入/目标 (默认)
    - 0: 只用 random_num=0
    - 1: 只用 random_num=1
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
    
    return equalized.reshape(img.shape)


class CTDataset(Dataset):
    """
    CT 图像去噪数据集
    
    主要特性:
    1. 支持 Noise2Noise 训练（使用两个噪声实现互相监督）
    2. 支持选择使用哪个 random_num
    3. 支持 histogram equalization
    4. 支持 teacher N2N denoised 结果
    5. 自动过滤坏样本
    """

    # 已知的坏样本 Patient_ID（数据中有 NaN）
    BAD_PATIENTS = {
        8527, 10431, 10461, 10536,
        11638, 11640, 19591, 30597,
        76624, 102364, 104563, 109021,
        139437, 148611, 154227, 172147
    }

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
        use_random_num='both',  # 新增: 'both', 0, 或 1
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
        self.use_random_num = use_random_num
        
        # Histogram equalization settings
        self.histogram_equalization = histogram_equalization
        self.bins = None
        self.bins_mapped = None
        
        if histogram_equalization:
            if bins_file is not None and bins_mapped_file is not None:
                if os.path.exists(bins_file) and os.path.exists(bins_mapped_file):
                    self.bins = np.load(bins_file)
                    self.bins_mapped = np.load(bins_mapped_file)
                    print(f'[{phase}] Histogram equalization enabled')
                else:
                    print(f'[WARNING] Histogram equalization files not found, disabled.')
                    self.histogram_equalization = False
            else:
                print(f'[WARNING] Histogram equalization enabled but no bins files specified, disabled.')
                self.histogram_equalization = False

        assert isinstance(valid_mask, (list, tuple)) and len(valid_mask) == 2

        # 读取 Excel
        self.df = pd.read_excel(dataroot)
        target_batches = train_batches if phase == 'train' else val_batches
        print(f'[DEBUG] Phase: {phase}, target_batches: {target_batches}')
        print(f'[DEBUG] Original df rows: {len(self.df)}')
        self.df = self.df[self.df['batch'].isin(target_batches)].reset_index(drop=True)
        print(f'[DEBUG] After batch filter: {len(self.df)} rows')

        # 构建 N2N pairs（根据 use_random_num 筛选）
        self.n2n_pairs = self._build_n2n_pairs()
        self.data_shape = self._infer_shape()
        W, H, S = self.data_shape

        # valid_mask 用于筛选 pairs（通常不需要，保留兼容性）
        v_start = max(int(valid_mask[0]), 0)
        v_end = min(int(valid_mask[1]), len(self.n2n_pairs)) if valid_mask[1] <= 100 else len(self.n2n_pairs)
        # 如果 valid_mask 像 [0, 100] 这样是 slice 范围，就不截取 pairs
        if valid_mask[1] > len(self.n2n_pairs):
            # valid_mask 是 slice 范围
            self.slice_start = max(int(valid_mask[0]), 0)
            self.slice_end = min(int(valid_mask[1]), S)
        else:
            # valid_mask 是 pair 范围（旧行为）
            self.n2n_pairs = self.n2n_pairs[v_start:v_end]
            self.slice_start, self.slice_end = 0, S

        # slice_range 覆盖 valid_mask
        if slice_range is not None:
            self.slice_start = max(int(slice_range[0]), 0)
            self.slice_end = min(int(slice_range[1]), S)
        
        self.num_slices = self.slice_end - self.slice_start

        # Auto detect slice offset
        if self.teacher_n2n_root is not None and len(self.n2n_pairs) > 0:
            detected_offset = self._detect_slice_offset()
            if detected_offset is not None and detected_offset != self.slice_start:
                print(f'[WARNING] Detected offset ({detected_offset}) != config slice_start ({self.slice_start})')

        V = len(self.n2n_pairs)
        
        # 处理 val_volume_idx
        if val_volume_idx == 'all' or val_volume_idx == ['all'] or val_volume_idx == ('all',):
            self.val_volume_idx = list(range(V))
        elif isinstance(val_volume_idx, int):
            self.val_volume_idx = [val_volume_idx]
        elif isinstance(val_volume_idx, (list, tuple)):
            # 过滤掉非整数
            self.val_volume_idx = [x for x in val_volume_idx if isinstance(x, int)]
        else:
            self.val_volume_idx = list(range(V))
        self.val_volume_idx = [x for x in self.val_volume_idx if isinstance(x, int) and x < V]

        # 处理 val_slice_idx
        if val_slice_idx == 'all' or val_slice_idx == ['all'] or val_slice_idx == ('all',):
            self.val_slice_idx = list(range(self.num_slices))
        elif isinstance(val_slice_idx, int):
            self.val_slice_idx = [val_slice_idx]
        elif isinstance(val_slice_idx, (list, tuple)):
            # 过滤掉非整数
            self.val_slice_idx = [x for x in val_slice_idx if isinstance(x, int)]
        else:
            self.val_slice_idx = list(range(self.num_slices))
        self.val_slice_idx = [x for x in self.val_slice_idx if isinstance(x, int) and x < self.num_slices]
        
        # 调试输出
        print(f'[{phase}] val_volume_idx: {self.val_volume_idx[:5]}... (total {len(self.val_volume_idx)})')
        print(f'[{phase}] val_slice_idx: {self.val_slice_idx[:5]}... (total {len(self.val_slice_idx)})')
        print(f'[{phase}] n2n_pairs: {V}, num_slices: {self.num_slices}')

        # 构建样本索引
        self.samples = self._build_sample_indices()
        self.matched_state = self._parse_stage2_file(stage2_file) if stage2_file else None

        # 兼容原代码的 raw_data.shape 访问
        self.data_size_before_padding = (W, H, self.num_slices, V)
        
        class FakeRawData:
            def __init__(self, shape):
                self.shape = shape
        self.raw_data = FakeRawData(self.data_size_before_padding)

        # 数据增强
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

        # 打印信息
        random_num_desc = {
            'both': '两个都用 (N2N)',
            0: '只用 random_num=0',
            1: '只用 random_num=1'
        }.get(use_random_num, str(use_random_num))
        
        print(f'[{phase}] CTDataset v2:')
        print(f'    pairs={V}, slices={self.num_slices}, samples={len(self.samples)}')
        print(f'    slice_range: [{self.slice_start}, {self.slice_end})')
        print(f'    use_random_num: {random_num_desc}')
        print(f'    HU range: [{self.HU_MIN}, {self.HU_MAX}]')
        print(f'    histogram_equalization: {self.histogram_equalization}')

    def _build_n2n_pairs(self):
        """
        构建 N2N pairs
        
        根据 use_random_num 参数决定如何构建:
        - 'both': 每个 pair 包含 noise_0 和 noise_1
        - 0: 每个 pair 只有 noise_0（用于单独训练或评估）
        - 1: 每个 pair 只有 noise_1
        """
        pairs = []
        grouped = self.df.groupby(['Patient_ID', 'Patient_subID'])
        skipped = 0
        
        print(f'[DEBUG] _build_n2n_pairs: df has {len(self.df)} rows, {len(grouped)} groups')
        print(f'[DEBUG] use_random_num = {self.use_random_num}')
        
        for (pid, psid), group in grouped:
            # 跳过坏样本
            try:
                pid_int = int(pid)
            except:
                pid_int = -1
            
            if pid_int in self.BAD_PATIENTS:
                skipped += 1
                continue
            
            noise_0_rows = group[group['random_num'] == 0]
            noise_1_rows = group[group['random_num'] == 1]
            
            if self.use_random_num == 'both':
                # 需要两个噪声实现都存在
                if len(noise_0_rows) > 0 and len(noise_1_rows) > 0:
                    pairs.append({
                        'noise_0': noise_0_rows.iloc[0]['noise_file'],
                        'noise_1': noise_1_rows.iloc[0]['noise_file'],
                        'gt': noise_0_rows.iloc[0]['ground_truth_file'],
                        'patient_id': pid,
                        'patient_subid': psid
                    })
            elif self.use_random_num == 0:
                # 只用 random_num=0
                if len(noise_0_rows) > 0:
                    pairs.append({
                        'noise_0': noise_0_rows.iloc[0]['noise_file'],
                        'noise_1': noise_0_rows.iloc[0]['noise_file'],  # 复制自己
                        'gt': noise_0_rows.iloc[0]['ground_truth_file'],
                        'patient_id': pid,
                        'patient_subid': psid
                    })
            elif self.use_random_num == 1:
                # 只用 random_num=1
                if len(noise_1_rows) > 0:
                    pairs.append({
                        'noise_0': noise_1_rows.iloc[0]['noise_file'],  # 用 noise_1 填充
                        'noise_1': noise_1_rows.iloc[0]['noise_file'],
                        'gt': noise_1_rows.iloc[0]['ground_truth_file'],
                        'patient_id': pid,
                        'patient_subid': psid
                    })
        
        print(f'Found {len(pairs)} N2N pairs (skipped {skipped} bad samples)')
        return pairs

    def _infer_shape(self):
        """推断数据维度"""
        default = (512, 512, 100)
        
        if len(self.n2n_pairs) == 0:
            return default
        
        path = self._fix_path(self.n2n_pairs[0]['noise_0'])
        npy_path = self._get_npy_path(path)
        
        try:
            if os.path.exists(npy_path):
                data = np.load(npy_path, mmap_mode='r')
                if data.ndim >= 3:
                    return (data.shape[0], data.shape[1], data.shape[2])
            elif os.path.exists(path):
                nii = nib.load(path)
                return nii.shape[:3]
        except:
            pass
        
        return default

    def _fix_path(self, path):
        """修复路径前缀"""
        if self.data_root is not None:
            return path.replace('/host/d/file/simulation/', self.data_root)
        return path

    def _get_npy_path(self, nii_path):
        """获取 npy 缓存路径"""
        return nii_path.replace('.nii.gz', '.npy').replace('/simulation/', '/simulation_npy/')

    def _teacher_n2n_exists(self, pred_path):
        """检查 teacher N2N 结果是否存在"""
        if pred_path is None:
            return False
        npy_path = pred_path.replace('.nii.gz', '.npy')
        return os.path.exists(pred_path) or os.path.exists(npy_path)

    def _detect_slice_offset(self):
        """自动检测 slice offset"""
        for pair in self.n2n_pairs:
            pred_path = self._get_teacher_n2n_path(pair['patient_id'], pair['patient_subid'])
            if not self._teacher_n2n_exists(pred_path):
                continue
            
            noise_path = self._fix_path(pair['noise_0'])
            npy_noise = self._get_npy_path(noise_path)
            
            try:
                if os.path.exists(npy_noise):
                    noise_data = np.load(npy_noise)
                else:
                    nii = nib.load(noise_path)
                    noise_data = nii.get_fdata()
                
                npy_pred = pred_path.replace('.nii.gz', '.npy')
                if os.path.exists(npy_pred):
                    pred_data = np.load(npy_pred)
                else:
                    pred_nii = nib.load(pred_path)
                    pred_data = pred_nii.get_fdata()
                
                noise_slices = noise_data.shape[2] if noise_data.ndim >= 3 else 1
                pred_slices = pred_data.shape[2] if pred_data.ndim >= 3 else 1
                
                if noise_slices > pred_slices:
                    return noise_slices - pred_slices
                return 0
            except:
                continue
        
        return None

    def _build_sample_indices(self):
        """构建样本索引列表"""
        samples = []
        
        print(f'[DEBUG] _build_sample_indices: phase={self.phase}')
        print(f'[DEBUG] n2n_pairs={len(self.n2n_pairs)}, num_slices={self.num_slices}')
        print(f'[DEBUG] val_volume_idx={self.val_volume_idx[:5] if len(self.val_volume_idx) > 5 else self.val_volume_idx}...')
        print(f'[DEBUG] val_slice_idx={self.val_slice_idx[:5] if len(self.val_slice_idx) > 5 else self.val_slice_idx}...')
        
        if self.phase in ('train', 'test'):
            for vol_idx in range(len(self.n2n_pairs)):
                if self.teacher_n2n_root is not None:
                    pair = self.n2n_pairs[vol_idx]
                    pred_path = self._get_teacher_n2n_path(pair['patient_id'], pair['patient_subid'])
                    if not self._teacher_n2n_exists(pred_path):
                        continue
                
                for slice_idx in range(self.num_slices):
                    samples.append({
                        'pair_idx': vol_idx,
                        'slice_idx': slice_idx
                    })
        else:
            # 验证阶段：不检查 teacher N2N 是否存在
            # 因为验证只需要用模型去噪，不需要 teacher 结果
            for vol_idx in self.val_volume_idx:
                for slice_idx in self.val_slice_idx:
                    samples.append({
                        'pair_idx': vol_idx,
                        'slice_idx': slice_idx
                    })
        
        print(f'[DEBUG] Built {len(samples)} samples')
        return samples

    def _parse_stage2_file(self, file_path):
        """解析 stage2_matched.txt 文件"""
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
        """预处理图像: histogram equalization -> HU cutoff -> normalize"""
        # Step 1: Histogram equalization
        if self.histogram_equalization and self.bins is not None:
            img = apply_histogram_equalization(img, self.bins, self.bins_mapped)
        
        # Step 2: HU cutoff and normalization
        img = np.clip(img, self.HU_MIN, self.HU_MAX)
        img = (img - self.HU_MIN) / (self.HU_MAX - self.HU_MIN)
        
        return np.clip(img, 0.0, 1.0)

    def _load_slice(self, nii_path, slice_idx):
        """加载单个 slice"""
        nii_path = self._fix_path(nii_path)
        npy_path = self._get_npy_path(nii_path)
        
        # 加上 slice offset
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
        
        return self._preprocess_image(img)

    def _get_teacher_n2n_path(self, patient_id, patient_subid):
        """获取 teacher N2N 结果路径"""
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
        """加载 teacher N2N 去噪结果"""
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
        
        return self._preprocess_image(img)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_info = self.samples[index]
        volume_idx = sample_info['pair_idx']
        slice_idx = sample_info['slice_idx']
        pair = self.n2n_pairs[volume_idx]

        # 加载噪声图像
        n0 = self._load_slice(pair['noise_0'], slice_idx)
        n1 = self._load_slice(pair['noise_1'], slice_idx)

        # 加载 teacher denoised（如果有）
        teacher_denoised = None
        if self.teacher_n2n_root is not None:
            teacher_denoised = self._load_teacher_denoised(
                pair['patient_id'], 
                pair['patient_subid'], 
                slice_idx
            )

        # N2N 训练时随机选择输入/目标
        if self.use_random_num == 'both' and self.phase == 'train' and random.random() > 0.5:
            input_img, target_img = n1, n0
        else:
            input_img, target_img = n0, n1

        # 构建 condition channels
        if self.padding > 0:
            cond_ch = 2 * self.padding
            channels = [input_img] * cond_ch + [target_img]
        else:
            channels = [input_img, target_img]

        has_teacher = teacher_denoised is not None
        if has_teacher:
            channels.append(teacher_denoised)

        raw_input = np.stack(channels, axis=-1)
        raw_input = self.transforms(raw_input)

        if has_teacher:
            denoised_tensor = raw_input[[-1], :, :]
            raw_input = raw_input[:-1, :, :]

        ret = {
            'X': raw_input[[-1], :, :].float(),           # 确保 float32
            'condition': raw_input[:-1, :, :].float()     # 确保 float32
        }

        # matched_state
        if self.matched_state is not None:
            if volume_idx in self.matched_state and slice_idx in self.matched_state[volume_idx]:
                ret['matched_state'] = torch.tensor([float(self.matched_state[volume_idx][slice_idx])], dtype=torch.float32)
            else:
                ret['matched_state'] = torch.tensor([500.0], dtype=torch.float32)
        else:
            ret['matched_state'] = torch.tensor([500.0], dtype=torch.float32)

        # teacher denoised
        if self.teacher_n2n_root is not None:
            if has_teacher:
                ret['denoised'] = denoised_tensor.float()  # 确保 float32
            else:
                ret['denoised'] = ret['X'].clone()
                ret['matched_state'] = torch.tensor([1.0], dtype=torch.float32)

        # NaN/Inf 检查
        if torch.isnan(ret['X']).any() or torch.isinf(ret['X']).any():
            ret['X'] = torch.zeros_like(ret['X'])
        if torch.isnan(ret['condition']).any() or torch.isinf(ret['condition']).any():
            ret['condition'] = torch.zeros_like(ret['condition'])

        return ret


def create_ct_dataloader(opt, phase='train', stage2_file=None):
    """
    创建 CT DataLoader 的工厂函数
    
    Args:
        opt: 配置字典，包含 datasets.train 或 datasets.val
        phase: 'train' 或 'val'
        stage2_file: stage2_matched.txt 路径
    
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader
    
    dataset_opt = opt['datasets'][phase]
    
    dataset = CTDataset(
        dataroot=dataset_opt['dataroot'],
        valid_mask=dataset_opt.get('valid_mask', [0, 100]),
        phase=phase,
        image_size=dataset_opt.get('image_size', 512),
        in_channel=dataset_opt.get('in_channel', 1),
        val_volume_idx=dataset_opt.get('val_volume_idx', 0),
        val_slice_idx=dataset_opt.get('val_slice_idx', 25),
        padding=dataset_opt.get('padding', 3),
        lr_flip=dataset_opt.get('lr_flip', 0.5),
        stage2_file=stage2_file or opt.get('stage2_file'),
        data_root=dataset_opt.get('data_root'),
        train_batches=dataset_opt.get('train_batches', [0, 1, 2, 3, 4]),
        val_batches=dataset_opt.get('val_batches', [5]),
        slice_range=dataset_opt.get('slice_range'),
        HU_MIN=dataset_opt.get('HU_MIN', -1000.0),
        HU_MAX=dataset_opt.get('HU_MAX', 2000.0),
        teacher_n2n_root=dataset_opt.get('teacher_n2n_root'),
        teacher_n2n_epoch=dataset_opt.get('teacher_n2n_epoch', 78),
        histogram_equalization=dataset_opt.get('histogram_equalization', True),
        bins_file=dataset_opt.get('bins_file'),
        bins_mapped_file=dataset_opt.get('bins_mapped_file'),
        use_random_num=dataset_opt.get('use_random_num', 'both'),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=dataset_opt.get('batch_size', 1),
        shuffle=dataset_opt.get('use_shuffle', phase == 'train'),
        num_workers=dataset_opt.get('num_workers', 0),
        pin_memory=True,
        drop_last=phase == 'train'
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("CTDataset v2 测试")
    print("=" * 60)
    
    # 模拟配置
    test_config = {
        'datasets': {
            'train': {
                'dataroot': '/path/to/excel.xlsx',
                'valid_mask': [0, 100],
                'train_batches': [0, 1, 2, 3, 4],
                'val_batches': [5],
                'use_random_num': 'both',  # 或 0 或 1
                'batch_size': 1,
            },
            'val': {
                'dataroot': '/path/to/excel.xlsx',
                'valid_mask': [0, 100],
                'train_batches': [0, 1, 2, 3, 4],
                'val_batches': [5],
                'use_random_num': 'both',
                'val_volume_idx': 'all',
                'val_slice_idx': 'all',
                'batch_size': 1,
            }
        }
    }
    
    print("\n配置示例:")
    print(f"  use_random_num='both': 两个噪声实现都用 (N2N 训练)")
    print(f"  use_random_num=0: 只用 random_num=0")
    print(f"  use_random_num=1: 只用 random_num=1")
    
    print("\n使用方法:")
    print("  from ct_dataset_v2 import CTDataset, create_ct_dataloader")
    print("  ")
    print("  # 方法1: 直接创建 Dataset")
    print("  dataset = CTDataset(")
    print("      dataroot='path/to/excel.xlsx',")
    print("      valid_mask=[0, 100],")
    print("      use_random_num='both',  # 或 0 或 1")
    print("      ...)")
    print("  ")
    print("  # 方法2: 使用工厂函数")
    print("  train_loader = create_ct_dataloader(config, phase='train')")
