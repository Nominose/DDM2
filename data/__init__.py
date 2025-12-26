import logging
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt.get('use_shuffle', True),
            num_workers=dataset_opt.get('num_workers', 0),
            pin_memory=True,
            drop_last=True
        )
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=dataset_opt.get('num_workers', 1),
            pin_memory=True
        )
    else:
        raise NotImplementedError(f'Dataloader [{phase}] is not found.')


def create_dataset(dataset_opt, phase, stage2_file=None):
    """
    根据 dataset_opt['name'] 自动选择 MRI 或 CT 数据集
    """
    logger = logging.getLogger('base')
    dataset_name = dataset_opt.get('name', 'mri').lower()

    if 'ct' in dataset_name:
        from data.ct_dataset import CTDataset

        volume_mask = dataset_opt.get('volume_mask', None)
        if volume_mask is None:
            volume_mask = [0, 10**9]

        slice_range = dataset_opt.get('slice_range', None)
        if slice_range is None:
            slice_range = dataset_opt.get('valid_mask', None)

        dataset = CTDataset(
            dataroot=dataset_opt['dataroot'],
            valid_mask=volume_mask,
            phase=phase,
            image_size=dataset_opt.get('image_size', 512),
            in_channel=dataset_opt.get('in_channel', 1),
            padding=dataset_opt.get('padding', 3),
            lr_flip=dataset_opt.get('lr_flip', 0.5),
            stage2_file=stage2_file,
            data_root=dataset_opt.get('data_root', None),
            train_batches=dataset_opt.get('train_batches', [0, 1, 2, 3, 4]),
            val_batches=dataset_opt.get('val_batches', [5]),
            slice_range=slice_range,
            val_volume_idx=dataset_opt.get('val_volume_idx', 0),
            val_slice_idx=dataset_opt.get('val_slice_idx', 25),
            HU_MIN=dataset_opt.get('HU_MIN', -1000.0),
            HU_MAX=dataset_opt.get('HU_MAX', 2000.0),
            # ====== 导师N2N参数 ======
            teacher_n2n_root=dataset_opt.get('teacher_n2n_root', None),
            teacher_n2n_epoch=dataset_opt.get('teacher_n2n_epoch', 78),
            # ====== Histogram Equalization 参数 ======
            histogram_equalization=dataset_opt.get('histogram_equalization', False),
            bins_file=dataset_opt.get('bins_file', None),
            bins_mapped_file=dataset_opt.get('bins_mapped_file', None),
            # =========================================
        )

        logger.info(f'CT dataset [{dataset_opt.get("name","ct")}] is created. Size: {len(dataset)}')
        return dataset

    # MRI
    from data.mri_dataset import MRIDataset

    dataset = MRIDataset(
        dataroot=dataset_opt['dataroot'],
        valid_mask=dataset_opt['valid_mask'],
        phase=phase,
        val_volume_idx=dataset_opt.get('val_volume_idx', 0),
        val_slice_idx=dataset_opt.get('val_slice_idx', 25),
        padding=dataset_opt.get('padding', 3),
        in_channel=dataset_opt.get('in_channel', 1),
        image_size=dataset_opt.get('image_size', 128),
        lr_flip=dataset_opt.get('lr_flip', 0.5),
        stage2_file=stage2_file
    )

    logger.info(f'MRI dataset [{dataset_opt.get("name","mri")}] is created. Size: {len(dataset)}')
    return dataset
