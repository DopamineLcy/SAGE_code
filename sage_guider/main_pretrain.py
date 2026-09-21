# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

import pretrain_datasets as pretrain_dataset_mod
import models

from engine_pretrain import train_one_epoch, evaluate



def get_args_parser():
    parser = argparse.ArgumentParser('Pre-training', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='BaseModel', type=str, metavar='MODEL',
                        help='Name of model to train (default: BaseModel)')
    parser.add_argument('--use_vision_cls_token', action='store_true',
                        help='Use vision cls token')
    parser.set_defaults(use_vision_cls_token=False)
    parser.add_argument('--proj_dim', type=int, default=512,
                        help='projection dimension (default: 512)')
    parser.add_argument('--num_hidden_layers', type=int, default=2,
                        help='number of hidden layers (default: 2)')
    parser.add_argument('--rad_dino_output_layer', type=int, default=-1,
                        help='output layer of rad-dino (default: -1)')
    parser.add_argument('--use_extra_pos_embed', action='store_true',
                        help='Use extra pos embed')
    parser.set_defaults(use_extra_pos_embed=False)
    
    # Optimizer parameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of training epochs')
    parser.add_argument('--lr_decay_epochs', default=100, type=int,
                        help='cosine learning-rate decay horizon in epochs')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus, default: 256 from memory.txt)')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr, default: 1e-4 from memory.txt)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_iterations', type=int, default=5000, metavar='N',
                        help='iterations to warmup LR (default: 5000 from memory.txt)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, metavar='N',
                        help='gradient clipping norm (default: 1.0 from memory.txt)')
    # Dataset and external-model parameters.  These are deliberately explicit:
    # the released code must not depend on the original workstation layout.
    parser.add_argument('--image-root', required=True, type=str,
                        help='MIMIC-CXR-JPG root containing pXX/p<subject>/s<study>/<dicom>.jpg')
    parser.add_argument('--metadata-csv', required=True, type=str,
                        help='MIMIC metadata CSV with dicom_id, subject_id, study_id and split')
    parser.add_argument('--report-root', required=True, type=str,
                        help='ontology JSON root containing pXX/p<subject>/s<study>.json')
    parser.add_argument('--labels-json', required=True, type=str,
                        help='ordered 12-device ontology JSON; its order is a checkpoint contract')
    parser.add_argument('--external-model-root', required=True, type=str,
                        help='directory containing Rad-DINO, CXR-BERT, and the NLI model')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--is_augmentation', action='store_true',
                        help='Use data augmentation')
    parser.set_defaults(is_augmentation=False)
    parser.add_argument('--use_counterfactual', action='store_true',
                        help='Use counterfactual text')
    parser.set_defaults(use_counterfactual=False)
    parser.add_argument('--attn_temperature', type=float, default=None, metavar='N',
                        help='attention temperature')
    parser.add_argument('--aug_degrees', type=int, default=20, metavar='N',
                        help='augmentation degrees')
    parser.add_argument('--aug_scale', type=float, nargs=2, default=[0.95, 1.05], metavar='N',
                        help='augmentation scale')
    parser.add_argument('--aug_prob', type=float, default=1.0, metavar='N',
                        help='augmentation probability')
    parser.add_argument('--pos_sample_weight', type=float, default=4.0,
                        help='sampling weight for images with >=1 positive entity (1.0 = uniform sampling)')

    # Environment parameters
    parser.add_argument('--output-dir', '--output_dir', dest='output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--from_begin', action='store_true',
                        help='train from epoch 0')
    parser.set_defaults(from_begin=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--eval_freq', default=1, type=int, help='frequency of evaluation')
    return parser


def main(args):
    misc.init_distributed_mode(args)
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    run_name = f'sage_guider_ep{args.epochs}_lr{args.lr}_bs{eff_batch_size}'
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, run_name)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    for option in ('image_root', 'metadata_csv', 'report_root', 'labels_json', 'external_model_root'):
        value = Path(getattr(args, option))
        if not value.exists():
            raise FileNotFoundError(f"--{option.replace('_', '-')} does not exist: {value}")
    dataset = pretrain_dataset_mod.PretrainDataset
    dataset_train = dataset(is_train=True, args=args)
    
    # 1. Standard Validation Dataset (for Loss)
    dataset_valid = dataset(is_train=False, args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.pos_sample_weight > 1.0:
        sampler_train = dataset_train.get_train_sampler(
            num_replicas=num_tasks, rank=global_rank, seed=args.seed,
            pos_sample_weight=args.pos_sample_weight,
        )
        print(f"Sampler_train = DistributedWeightedSampler (pos_weight={args.pos_sample_weight})")
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    
    sampler_valid = torch.utils.data.DistributedSampler(
        dataset_valid, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    print("Sampler_valid = %s" % str(sampler_valid))

    args.log_dir = os.path.join(args.output_dir, "logs") if args.output_dir else None
    if global_rank == 0 and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=dataset_train.collate_fn,
        **({'prefetch_factor': 2, 'persistent_workers': True} if args.num_workers > 0 else {})
    )
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, sampler=sampler_valid,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=dataset_valid.collate_fn,
        **({'prefetch_factor': 2, 'persistent_workers': True} if args.num_workers > 0 else {})
    )

    # define the model
    model = models.__dict__[args.model](args=args)

    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    def exclude_from_weight_decay(n, p):
        return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n

    trainable_params = [
        (n, p) for n, p in model_without_ddp.named_parameters() if p.requires_grad
    ]
    params_with_wd = [(n, p) for n, p in trainable_params if not exclude_from_weight_decay(n, p)]
    params_no_wd = [(n, p) for n, p in trainable_params if exclude_from_weight_decay(n, p)]
    
    param_groups = [
        {
            "params": [p for n, p in params_with_wd],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in params_no_wd],
            "weight_decay": 0.0,
        },
    ]
    
    try:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), fused=True)
        print("Using fused AdamW")
    except Exception as e:
        print(f"Fused AdamW unavailable ({e}); using standard AdamW")
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        
    # Calculate iterations per epoch for warmup scheduling
    args.iter_per_epoch = len(data_loader_train)
    print(f"Start training for {args.epochs} epochs")
    print(f"Iterations per epoch: {args.iter_per_epoch}")
    print(f"Warmup iterations: {args.warmup_iterations}")
    print(f"Training iterations: {args.epochs * args.iter_per_epoch}")
    print(f"LR decay iterations: {args.lr_decay_epochs * args.iter_per_epoch}")
    
    min_val_loss = float('inf')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch,
            log_writer=log_writer,
            args=args,
        )

        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, name="newest")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if epoch % args.eval_freq == 0:
            # 1. Standard Loss Validation
            val_stats = evaluate(
                data_loader_valid, model, device,
                log_writer=log_writer, epoch=epoch,
            )
            print(f"Loss of the network on the {len(dataset_valid)} validation images: {val_stats['loss']:.3f}")
            
            if val_stats["loss"] < min_val_loss:
                min_val_loss = val_stats["loss"]
                if args.output_dir:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, name="best_val_loss")
            
            # Merge stats
            log_stats.update({**{f'validation_{k}': v for k, v in val_stats.items()}})

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
