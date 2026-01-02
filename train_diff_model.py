import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

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
    logger.info(Logger.dict2str(opt))

    stage2_file = opt['stage2_file']
    logger.info(f'Stage 2 file: {stage2_file}')

    # dataset
    train_loader = None
    val_loader = None
    train_set = None
    val_set = None
    
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase, stage2_file=stage2_file)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
            logger.info(f'[DEBUG] train_set: {len(train_set)} samples')
            logger.info(f'[DEBUG] train_loader: {len(train_loader)} batches')
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase, stage2_file=stage2_file)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
            logger.info(f'[DEBUG] val_set: {len(val_set)} samples')
            logger.info(f'[DEBUG] val_loader: {len(val_loader)} batches')
    
    logger.info('Initial Dataset Finished')
    
    # 检查 val_loader 是否可用
    if val_loader is None or len(val_loader) == 0:
        logger.warning('WARNING: val_loader is empty or None! Validation will be skipped.')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    val_freq = opt['train']['val_freq']
    print_freq = opt['train']['print_freq']
    save_freq = opt['train']['save_checkpoint_freq']
    
    logger.info(f'Training config: n_iter={n_iter}, val_freq={val_freq}, print_freq={print_freq}, save_freq={save_freq}')

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                
                # log
                if current_step % print_freq == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                    logger.info(message)

                # validation
                if current_step % val_freq == 0:
                    logger.info(f'')
                    logger.info(f'{"="*60}')
                    logger.info(f'VALIDATION START at step {current_step}')
                    logger.info(f'{"="*60}')
                    
                    if val_loader is None or len(val_loader) == 0:
                        logger.warning('val_loader is empty, skipping validation')
                        continue
                    
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    logger.info(f'Result path: {result_path}')
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    
                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        logger.info(f'Processing validation sample {idx}/{len(val_loader)}')
                        
                        try:
                            diffusion.feed_data(val_data)
                            diffusion.test(continous=True)
                            
                            visuals = diffusion.get_current_visuals()
                            logger.info(f'  Visuals keys: {list(visuals.keys())}')
                            
                            # 检查 denoised 的形状
                            denoised_tensor = visuals['denoised']
                            logger.info(f'  Denoised tensor shape: {denoised_tensor.shape}')
                            
                            denoised_img = Metrics.tensor2img(denoised_tensor, out_type=np.float32)
                            input_img = Metrics.tensor2img(visuals['X'], out_type=np.float32)
                            
                            logger.info(f'  Denoised img shape: {denoised_img.shape}, range: [{denoised_img.min():.3f}, {denoised_img.max():.3f}]')
                            logger.info(f'  Input img shape: {input_img.shape}, range: [{input_img.min():.3f}, {input_img.max():.3f}]')

                            # 保存图片
                            denoised_path = '{}/{}_{}_denoised.png'.format(result_path, current_step, idx)
                            input_path = '{}/{}_{}_input.png'.format(result_path, current_step, idx)
                            
                            Metrics.save_img(denoised_img[:,:], denoised_path)
                            Metrics.save_img(input_img[:,:], input_path)
                            
                            logger.info(f'  Saved: {denoised_path}')
                            
                        except Exception as e:
                            logger.error(f'  Error processing sample {idx}: {str(e)}')
                            import traceback
                            traceback.print_exc()
                        
                        # 只验证前 5 个样本（加速）
                        if idx >= 5:
                            logger.info(f'  Stopping validation early (only {idx} samples)')
                            break
                    
                    logger.info(f'{"="*60}')
                    logger.info(f'VALIDATION END - saved {idx} images to {result_path}')
                    logger.info(f'{"="*60}')
                    logger.info(f'')
                    
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')

                # save checkpoint
                if current_step % save_freq == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step, save_last_only=True)

        # save model
        logger.info('End of training.')
    
    else:
        # Evaluation mode
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        
        for _, val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals()

            denoised_img = Metrics.tensor2img(visuals['denoised'], out_type=np.float32)
            input_img = Metrics.tensor2img(visuals['X'], out_type=np.float32)

            Metrics.save_img(
                denoised_img, '{}/{}_{}_denoised.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                input_img, '{}/{}_{}_input.png'.format(result_path, current_step, idx))
            
            if idx % 10 == 0:
                logger.info(f'Processed {idx} samples')

        logger_val = logging.getLogger('val')
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> Total samples: {}'.format(
            current_epoch, current_step, idx))
