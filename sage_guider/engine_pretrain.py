import math
from typing import Iterable

import torch

import utils.lr_sched as lr_sched
import utils.misc as misc


def _scalar(outputs, name, default=0.0):
    value = outputs.get(name, default)
    return value.item() if isinstance(value, torch.Tensor) else float(value)


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", misc.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter(
        "sampled_image_has_positive_ratio",
        misc.SmoothedValue(window_size=1, fmt="{value:.4f}"),
    )
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    for step, batch in enumerate(
        metric_logger.log_every(data_loader, 20, f"Epoch: [{epoch}]")
    ):
        if step % accum_iter == 0:
            current_iter = epoch * len(data_loader) + step
            lr_sched.adjust_learning_rate(optimizer, current_iter, args)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(batch, device=device)
            loss = outputs["loss"]

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite training loss: {loss_value}")

        (loss / accum_iter).backward()
        if (step + 1) % accum_iter == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()

        sampled_positive_ratio = float(
            batch.get("sampled_image_has_positive_ratio", 0.0)
        )
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(
            loss=loss_value,
            lr=lr,
            sampled_image_has_positive_ratio=sampled_positive_ratio,
            loss_temperature_exp=_scalar(outputs, "loss_temperature_exp"),
            attn_temperature_exp=_scalar(outputs, "attn_temperature_exp"),
            pos_avg_logits=_scalar(outputs, "pos_avg_logits"),
        )

        if log_writer is not None:
            tensorboard_step = epoch * len(data_loader) + step
            log_writer.add_scalar(
                "loss", misc.all_reduce_mean(loss_value), tensorboard_step
            )
            log_writer.add_scalar("lr", lr, tensorboard_step)
            log_writer.add_scalar(
                "data/sampled_image_has_positive_ratio",
                misc.all_reduce_mean(sampled_positive_ratio),
                tensorboard_step,
            )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, log_writer=None, epoch=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")

    for batch in metric_logger.log_every(data_loader, 10, "Validation:"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(batch, device=device)

        metric_logger.update(
            loss=outputs["loss"].item(),
            loss_temperature_exp=_scalar(outputs, "loss_temperature_exp"),
            attn_temperature_exp=_scalar(outputs, "attn_temperature_exp"),
            pos_avg_logits=_scalar(outputs, "pos_avg_logits"),
        )

    metric_logger.synchronize_between_processes()
    stats = {key: meter.global_avg for key, meter in metric_logger.meters.items()}
    print(f"* loss {stats['loss']:.3f}")

    if log_writer is not None and epoch is not None:
        log_writer.add_scalar("valid_loss", stats["loss"], epoch)
        log_writer.add_scalar(
            "valid_loss_temperature_exp", stats["loss_temperature_exp"], epoch
        )
        if stats["attn_temperature_exp"] > 0:
            log_writer.add_scalar(
                "valid_attn_temperature_exp", stats["attn_temperature_exp"], epoch
            )
        log_writer.add_scalar("valid_pos_avg_logits", stats["pos_avg_logits"], epoch)

    return stats
