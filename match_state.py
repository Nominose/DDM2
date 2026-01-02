import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from functools import partial
from scipy.stats import norm
import math
from tqdm import tqdm
import nibabel as nib

class Object:
    def __init__(self, config):
        self.config = config
        self.phase = 'train'
        self.gpu_ids = None
        self.debug = False

def _rev_warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_start * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[n_timestep - warmup_time:] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                    help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                    help='Run either train(training) or val(generation)', default='train')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('--debug', action='store_true')

# parse configs
args = parser.parse_args()

opt = Logger.parse(args)

# Convert to NoneDict, which return None for missing key.
opt = Logger.dict_to_nonedict(opt)
# logging
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

Logger.setup_logger(None, opt['path']['log'],
                    'train', level=logging.INFO, screen=True)
Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
logger = logging.getLogger('base')
logger.info('[Stage 2] Markov chain state matching (using teacher N2N)!')

# dataset - 只创建训练集用于 Stage 2 匹配
for phase, dataset_opt in opt['datasets'].items():
    dataset_opt['initial_stage_file'] = None
    if phase == 'train':
        # Stage 2 需要遍历所有训练样本，所以设置 shuffle=False
        dataset_opt['use_shuffle'] = False
        train_set = Data.create_dataset(dataset_opt, phase)
        train_loader = Data.create_dataloader(
            train_set, dataset_opt, phase)
        break  # 只需要训练集

logger.info('Initial Dataset Finished')
logger.info(f'Total training samples to match: {len(train_set)}')
logger.info('Using teacher N2N results from ct_dataset')

#######
to_torch = partial(torch.tensor, dtype=torch.float32, device='cuda:0')

betas = _rev_warmup_beta(opt['noise_model']['beta_schedule']['linear_start'], opt['noise_model']['beta_schedule']['linear_end'],
                             1000, 0.7)

alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
sqrt_alphas_cumprod_prev_np = np.sqrt(
            np.append(1., alphas_cumprod))
sqrt_alphas_cumprod_prev = to_torch(np.sqrt(
            np.append(1., alphas_cumprod)))

idx = 0
stage_file = open(opt['stage2_file'],'w+')

# ====== 修复：使用 train_loader 而不是 val_loader ======
for _,  data in tqdm(enumerate(train_loader), total=len(train_loader), desc="Matching"):
    idx += 1
    
    # ====== 修复：从 train_set.samples 获取正确的 pair_idx 和 slice_idx ======
    sample_info = train_set.samples[idx - 1]
    volume_idx = sample_info['pair_idx']
    slice_idx = sample_info['slice_idx']
    
    # ====== 修复：直接使用 ct_dataset 加载的 denoised（已处理 slice 偏移）======
    if 'denoised' not in data:
        logger.warning(f'No denoised for volume {volume_idx}, slice {slice_idx}, using default t=500')
        stage_file.write('%d_%d_%d\n' % (volume_idx, slice_idx, 500))
        continue
    
    denoised = data['denoised'].cuda()
    # ========================================================================

    max_lh = -1
    max_t = -1
    min_lh = 999
    min_t = -1
    prev_diff = 999.
    
    for t in range(sqrt_alphas_cumprod_prev.shape[0]): # linear search with early stopping
        noise = data['X'].cuda() - sqrt_alphas_cumprod_prev[t] * denoised
        noise_mean = torch.mean(noise)
        noise = noise - noise_mean

        mu, std = norm.fit(noise.cpu().numpy())

        diff = np.abs((1 - sqrt_alphas_cumprod_prev[t]**2).sqrt().cpu().numpy() - std)

        if diff < min_lh:
            min_lh = diff
            min_t = t

        if diff > prev_diff:
            break # find a match!
        else:
            prev_diff = diff

    if idx == 30 and args.debug:
        noise = torch.randn_like(denoised)
        result = sqrt_alphas_cumprod_prev[min_t] * denoised.detach() + (1. - sqrt_alphas_cumprod_prev[min_t]**2).sqrt() * noise
        denoised_np = denoised.detach().cpu().numpy()[0,0]
        input_np = data['X'].detach().cpu().numpy()[0,0]
        result_np = result.detach().cpu().numpy()[0,0]

        result_np = (result_np + 1.) / 2.
        input_np = (input_np + 1.) / 2.
        plt.imshow(np.hstack((input_np, result_np, denoised_np)), cmap='gray')
        plt.show()
        
        print(min_t, np.max(result_np), np.min(result_np))
        break
    
    stage_file.write('%d_%d_%d\n' % (int(volume_idx), int(slice_idx), int(min_t)))

stage_file.close()
logger.info(f'Stage 2 matching complete! Total samples: {idx}')
logger.info(f'Output saved to: {opt["stage2_file"]}')
print('done!')
