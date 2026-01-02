# DDM² CT Denoising Notebooks

直接调用 DDM² 原始脚本的 Jupyter Notebook 工作流。

## 术语说明

### CT 数据结构

```
一位患者的一次 CT 扫描 = 1 个 Volume（3D 数据）
                         ├── Slice 0   (2D 图像, 512×512)
                         ├── Slice 1
                         ├── ...
                         └── Slice 99  (共 100 张切片)
```

| 术语 | 含义 | 示例 |
|------|------|------|
| **Volume** | 一个完整的 3D CT 扫描 | 一位患者的一次扫描 |
| **Slice** | Volume 中的单个 2D 横截面图像 | 512×512 的单张图像 |
| **N2N Pair** | 同一扫描的两个不同噪声实现 | random_num=0 和 random_num=1 |
| **volume_idx** | Volume 的索引编号 | 0, 1, 2, ... |
| **slice_idx** | 切片的索引编号 | 0, 1, ..., 99 |

### 你的数据

```
Excel 文件 (200 条记录)
├── 100 个 N2N pairs (每个 pair 有 2 条记录)
│   ├── Batch 0-4: 84 pairs → 训练集
│   └── Batch 5:   16 pairs → 验证集
│
└── 每个 pair 的 NIfTI 文件: (512, 512, 100)
    └── 100 张 slices
```

所以：
- 训练样本数 = 84 volumes × 100 slices = **8,400**
- 验证样本数 = 16 volumes × 100 slices = **1,600**

---

## Volume/Slice 选择参数

### 在 config 文件中的位置

```json
{
    "datasets": {
        "train": {
            "train_batches": [0, 1, 2, 3, 4],  // 哪些 batch 用于训练
            "val_batches": [5],                 // 哪些 batch 用于验证
            "valid_mask": [0, 100],             // slice 范围 [start, end)
            "val_volume_idx": "all",            // Stage 2 处理哪些 volume
            "val_slice_idx": "all"              // Stage 2 处理哪些 slice
        }
    }
}
```

### 参数详解

| 参数 | 作用 | 可选值 |
|------|------|--------|
| `train_batches` | 选择训练用的 batch | 列表，如 `[0,1,2,3,4]` |
| `val_batches` | 选择验证用的 batch | 列表，如 `[5]` |
| `valid_mask` | 限制 slice 范围 | `[start, end)`，如 `[0, 100]` |
| `val_volume_idx` | Stage 2 处理的 volume | `"all"`, 数字, 或列表 `[0,1,5]` |
| `val_slice_idx` | Stage 2 处理的 slice | `"all"`, 数字, 或列表 `[25,50,75]` |

### 使用场景

**完整训练**:
```python
TRAIN_BATCHES = [0, 1, 2, 3, 4]
SLICE_START, SLICE_END = 0, 100
VAL_VOLUME_IDX = 'all'
VAL_SLICE_IDX = 'all'
# → 8,400 训练样本
```

**快速测试** (单图验证流程):
```python
TRAIN_BATCHES = [0]
SLICE_START, SLICE_END = 50, 51  # 只用 1 张 slice
VAL_VOLUME_IDX = 0
VAL_SLICE_IDX = 50
# → ~17 训练样本
```

**中等规模** (调参用):
```python
TRAIN_BATCHES = [0, 1]
SLICE_START, SLICE_END = 25, 75  # 中间 50 张
VAL_VOLUME_IDX = 'all'
VAL_SLICE_IDX = 'all'
# → ~1,700 训练样本
```

---

## Notebooks 说明

### 1. Stage 2: `1_stage_match/stage_match.ipynb`

**功能**: 为每个训练样本计算对应的扩散时间步 `matched_t`

**你可以在 notebook 里直接设置**:
```python
# 数据选择
TRAIN_BATCHES = [0, 1, 2, 3, 4]
SLICE_START = 0
SLICE_END = 100
VAL_VOLUME_IDX = 'all'   # 处理所有 volume
VAL_SLICE_IDX = 'all'    # 处理所有 slice
```

**调用的命令**: `python match_state.py -p train -c config.json`

**输出**: `stage2_matched.txt`，每行格式 `volume_idx_slice_idx_matched_t`

### 2. Stage 3: `2_train/train_ddm2.ipynb`

**功能**: 训练扩散模型

**你可以在 notebook 里直接设置**:
```python
# 数据选择
TRAIN_BATCHES = [0, 1, 2, 3, 4]
VAL_BATCHES = [5]
SLICE_START, SLICE_END = 0, 100
VAL_VOLUME_IDX = 0       # 验证用的 volume
VAL_SLICE_IDX = 50       # 验证用的 slice

# 训练参数
N_ITER = 100000
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

# 关键路径
NOISE_MODEL_CHECKPOINT = 'experiments/xxx/checkpoint/latest'
STAGE2_FILE = 'experiments/xxx/stage2_matched.txt'
```

**调用的命令**: `python train_diff_model.py -p train -c config.json`

### 3. Inference: `3_inference/inference_ddm2.ipynb`

**功能**: 使用训练好的模型进行去噪

---

## 工作流程

```
Stage 1 (命令行)                    Stage 2 (Notebook)              Stage 3 (Notebook)
python train_noise_model.py    →   1_stage_match/stage_match.ipynb → 2_train/train_ddm2.ipynb
       ↓                                    ↓                              ↓
  Noise Model                        stage2_matched.txt              Diffusion Model
  checkpoint                                                          checkpoint
```

---

## 快速开始

1. **确保 Stage 1 已完成**
   ```bash
   python train_noise_model.py -p train -c config/ct_denoise.json
   ```

2. **运行 Stage 2** (打开 notebook)
   - 设置 `PROJECT_ROOT` 和 `CONFIG_FILE`
   - 设置 Volume/Slice 选择参数
   - 运行所有 cell

3. **运行 Stage 3** (打开 notebook)
   - 设置关键路径（Noise Model checkpoint, Stage 2 文件）
   - 设置训练参数
   - 运行训练

4. **推理**
   - 使用 `3_inference/inference_ddm2.ipynb`

---

## 注意事项

- Notebooks 通过 `subprocess` 直接调用原始 Python 脚本
- 所有配置修改会保存到临时文件，不会修改原始 config
- Beta schedule 等参数完全从 config 读取，不需要手动设置
