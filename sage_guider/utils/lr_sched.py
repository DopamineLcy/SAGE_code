# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math


def adjust_learning_rate(optimizer, epoch_or_iter, args):
    """Warm up by iteration, then apply half-cycle cosine decay."""
    current_iter = int(epoch_or_iter)
    if current_iter < args.warmup_iterations:
        lr = args.lr * current_iter / args.warmup_iterations
    else:
        total_iters = args.lr_decay_epochs * args.iter_per_epoch
        progress = (current_iter - args.warmup_iterations) / max(
            1, total_iters - args.warmup_iterations
        )
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
