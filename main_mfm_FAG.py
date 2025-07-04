import os
import random
import time
import argparse
import datetime
import numpy as np
import subprocess

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from timm.utils import AverageMeter
from collections import defaultdict
import numpy as np
from data.moire_dataset import TestDataset
from torch.utils.data import DataLoader
from utils import ComputeMetric

from config import get_config
from models import build_model_mask
import models
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('MFM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument('--launcher', choices=['pytorch', 'slurm'], default='pytorch', help='job launcher')
    parser.add_argument('--port', type=int, default=29501, help='port only works when launcher=="slurm"')

    args = parser.parse_args()

    config = get_config(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args, config


def main(config, log_writer):
    data_loader_train = build_loader(config, logger, is_pretrain=True)
    dataset_eval = TestDataset([config.EVAL_DATASET], test=True)
    eval_loader = DataLoader(dataset_eval, 12, \
                            num_workers=4, pin_memory=True, drop_last=False)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model_mask(config, is_pretrain=True)
    
    model.FAG = getattr(models, config.MODEL_FAG)(config.MODEL_NAME)
    model.FAG.load_state_dict(torch.load(config.FAG_CHECKPOINT, map_location='cpu'), strict=True)
    for param in model.FAG.parameters():
        param.requires_grad=False
    model.FAG.eval()
    
    model.cuda()
    
    # logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler, log_writer)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, logger)
        if dist.get_rank() == 0:
            if log_writer is not None:
                log_writer.flush()
        
        eval_once(config, model, eval_loader, epoch)
        torch.cuda.empty_cache()
        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def eval_once(config, model, data_loader, epoch):
    model.eval()
    num_steps = len(data_loader)
    img_dict = defaultdict(dict)
    
    start = time.time()
    for idx, (img_t, img_path, img_name, label) in enumerate(data_loader):
        img_t = img_t.cuda(non_blocking=True)
        pred = model(img_t, None)
        probability = torch.nn.functional.softmax(pred,dim=1)[:,1].detach().tolist()
    
        batch_results = [(label_.item(), probability_,  name_) 
                     for label_, probability_, name_ in zip(label, probability, img_name)]
        
        for data in batch_results:
            label = int(data[0])
            score = float(data[1])
            img_name = data[2]
            if not img_name in img_dict.keys():
                img_dict[img_name] = {'label': label, 'num': 1, 'score':[score]}
            else:
                if not img_dict[img_name]['label'] == label:
                    print ('false 1')
                img_dict[img_name]['num'] += 1
                img_dict[img_name]['score'] += [score]
                
        # if idx == 2:
        #     break
        if idx % 200 == 0:
            logger.info(f"cur idx: {idx}")

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} Evaluation takes {datetime.timedelta(seconds=int(epoch_time))}")
    
    logger.info(f"img_dict len: {len(img_dict)}")
    y_ture = np.array([])
    y_score = np.array([])
    for k, v in img_dict.items():
        if not len(v['score']) == v['num']:
            print ('false 2')
        score_averagy = sum(v['score']) / v['num']
        y_ture = np.append(y_ture, v['label'])
        y_score = np.append(y_score, score_averagy)

    auc, eer, best_thresh = ComputeMetric(y_ture, y_score, isPlot=False, model_name='', fig_path='./results/')
    logger.info (f"metric all shape:  {y_ture.shape}")
    logger.info (f" auc:  {auc},  eer:  {eer},  best_thresh:  {best_thresh}")
        
    logger.info(f"legal score mean: {np.mean(y_score)}")
    logger.info("-------------------------------------")
    
    

def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, log_writer):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_FAG_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (img, img_target, mask, label) in enumerate(data_loader):
        # if idx == 3:
        #     break
        # print (model.module.FAG.model.heads[0].weight)
        # print ('mask.shape: ', mask.shape)
        img = img.cuda(non_blocking=True)
        if img_target is not None:
            img_target = img_target.cuda(non_blocking=True)
        if mask is not None:
            mask = mask.cuda(non_blocking=True)
        if label is not None:
            label = label.cuda(non_blocking=True)

        loss, loss_FAG = model(img, img_target, mask, label)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), img.size(0))
        loss_FAG_meter.update(loss_FAG.item(), img.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        lr = optimizer.param_groups[0]["lr"]
        loss_value_reduce = reduce_tensor(loss).item()

        if log_writer is not None and (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((idx / num_steps + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('grad_norm', grad_norm, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'loss_FAG {loss_FAG_meter.val:.4f} ({loss_FAG_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    args, config = parse_option()
    print ('config.AMP_OPT_LEVEL: ', config.AMP_OPT_LEVEL)
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if args.launcher == 'slurm':
        ## initialize slurm distributed training environment
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        # specify master port
        if args.port is not None:
            os.environ['MASTER_PORT'] = str(args.port)
        elif 'MASTER_PORT' in os.environ:
            pass  # use MASTER_PORT in the environment variable
        else:
            # 29500 is torch.distributed default port
            os.environ['MASTER_PORT'] = '29500'
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        dist.init_process_group(backend='nccl')
    elif args.launcher == 'pytorch':
        ## initialize pytorch distributed training environment
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        else:
            rank = -1
            world_size = -1
        torch.cuda.set_device(config.LOCAL_RANK)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    else:
        raise ValueError(f'Invalid launcher type: {args.launcher}')
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
        log_writer = SummaryWriter(log_dir=config.OUTPUT)
    else:
        log_writer = None

    # print config
    logger.info(config.dump())

    main(config, log_writer)
