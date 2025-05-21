"""
Training script for Resnet50 on ImageNet designed for use with torchrun.
"""
import time
import os
from datetime import timedelta

import torch
import torchvision
import wandb
from tqdm import tqdm

from training.scheduler import PolynomialWarmup
from training.utils import (
    Metric,
    load_checkpoint,
    save_checkpoint,
    accuracy,
)
from training.lars import create_optimizer_lars
from training.dataloader import get_dataloader, parse_wds_labels
import config


# These values are set by torchrun.
local_rank = int(os.environ["LOCAL_RANK"])
local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

run_name = config.run_name
batch_size = config.batch_size // world_size
total_batch_size = config.batch_size
epochs = config.epochs

assert total_batch_size % 256 == 0, "batch size must be divisible by 256"
batch_multiplier = total_batch_size // 256
log_every = max(1, (32768 // 256) // batch_multiplier)

# The main/leader container will log to wandb (Weights & Biases).
# Note that using Weights & Biases requires populating the WANDB_API_KEY environment variable
# using a modal.Secret passed to the modal.App or the modal.Function.
use_wandb = rank == 0 and not config.debug and "WANDB_API_KEY" in os.environ

if config.debug:
    # Only make deterministic during debugging. Otherwise, when preempted, the
    # training data will be duplicated.
    torch.manual_seed(0)


def main() -> None:
    if config.benchmark:
        checkpoint = None
        os.environ["NCCL_DEBUG"] = "INFO"
        # Output information about the NCCL initialization.
        os.environ["NCCL_DEBUG_SUBSYS"] = "INIT"
    else:
        checkpoint = load_checkpoint(run_name)

    if use_wandb:
        wandb.init(
            project=f"resnet50-training{'-benchmark' if config.benchmark else ''}",
            config={
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "warmup_epochs": config.warmup_epochs,
                "epsilon": config.epsilon,
                "momentum": config.momentum,
                "weight_decay": config.weight_decay,
                "gpus_per_node": config.gpus_per_node,
                "nodes": config.nodes,
                "benchmark": config.benchmark,
                "runtime": config.runtime,
            },
            save_code=True,
            # Modal needs this to save the code properly.
            # I think wandb doesn't recognize that `train.py` is the main training script.
            settings=wandb.Settings(code_dir="."),
            name=run_name,
            id=checkpoint["wandb_run_id"] if checkpoint else None,
            resume="must" if checkpoint else "never",
        )

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", timeout=timedelta(minutes=120))

    train_loader = get_dataloader(
        "train",
        batch_size,
        seed=0 if config.debug else None,
        device_id=local_rank,
        shard_id=rank,
        num_shards=world_size,
    )
    val_loader = get_dataloader(
        "val",
        batch_size,
        seed=0 if config.debug else None,
        device_id=local_rank,
        shard_id=rank,
        num_shards=world_size,
    )

    model = torchvision.models.resnet50().cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = create_optimizer_lars(
        model=model,
        lr=config.learning_rate,
        epsilon=config.epsilon,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        bn_bias_separately=True,
    )

    lr_scheduler = PolynomialWarmup(
        optimizer,
        decay_steps=epochs * len(train_loader),
        warmup_steps=int(config.warmup_epochs * len(train_loader)),
        end_lr=0.0,
        power=2.0,
        last_epoch=-1,
    )

    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    for epoch in range(checkpoint["epoch"] + 1 if checkpoint else 0, epochs):
        train(train_loader, model, criterion, optimizer, lr_scheduler, epoch)
        if not config.benchmark:
            validate(val_loader, model, criterion, epoch)
            if rank == 0 and not config.debug:
                save_checkpoint(
                    run_name,
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    wandb.run.id if wandb.run else None,
                )

        if config.debug and epoch == 3:
            break

    if use_wandb:
        wandb.finish()

    torch.distributed.barrier()  # Ensure all communication finished before exiting
    torch.distributed.destroy_process_group()


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch):
    batch_time = Metric("Time")  # Duration of each batch, tracked during training.
    train_loss = Metric("Loss")  # Training loss, tracked during training.
    train_acc = Metric("Acc@1")  # Top-1 accuracy metric, tracked during training.

    model.train()  # Set the model to training mode.

    end = time.time()

    with tqdm(
        total=len(train_loader),
        desc=f"Epoch {(epoch+1):3d}/{epochs:3d}",
        disable=local_rank != 0,
    ) as t:
        for i, data in enumerate(train_loader):
            inputs = data[0]["data"]
            labels = parse_wds_labels(data[0]["label"])

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if not config.benchmark:
                with torch.no_grad():
                    torch.distributed.all_reduce(
                        loss, op=torch.distributed.ReduceOp.SUM
                    )
                    loss = loss / torch.distributed.get_world_size()
                    train_loss.update(loss.item())
                    acc = accuracy(outputs, labels)
                    torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.SUM)
                    acc = acc / torch.distributed.get_world_size()
                    train_acc.update(acc.item())

            if i != 0:
                batch_time.update(time.time() - end, n=batch_multiplier)
            end = time.time()

            if not config.benchmark:
                t.set_postfix_str(
                    f"loss: {train_loss.avg:.4f}, acc: {100 * train_acc.avg:.2f}%, time: {batch_time.avg:.3f}"
                )
            else:
                t.set_postfix_str(f"time: {batch_time.avg:.3f}")
            t.update(1)

            if (i + 1) % log_every == 0 and use_wandb:
                data = {
                    "train/time": batch_time.avg,
                    "train/acc": 100 * train_acc.avg,
                    "train/loss": train_loss.avg,
                    "train/lr": lr_scheduler.get_last_lr()[0],
                }
                wandb.log(
                    data,
                    step=(i + epoch * len(train_loader)) * batch_multiplier,
                )

            if config.debug and i == 5:
                break

    if use_wandb:
        data = {
            "epoch": epoch,
            "train/time": batch_time.avg,
            "train/acc": 100 * train_acc.avg,
            "train/loss": train_loss.avg,
        }
        wandb.log(data)


def validate(val_loader, model, criterion, epoch):
    model.eval()
    val_loss = Metric("val_loss")
    val_acc = Metric("val_acc")

    start_time = time.time()

    with tqdm(
        total=len(val_loader),
        desc=f"Epoch {(epoch+1):3d}/{epochs:3d}",
        disable=local_rank != 0,
    ) as t:
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs = data[0]["data"]
                labels = parse_wds_labels(data[0]["label"])

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                loss = loss / torch.distributed.get_world_size()
                val_loss.update(loss.item())
                acc = accuracy(outputs, labels)
                torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.SUM)
                acc = acc / torch.distributed.get_world_size()
                val_acc.update(acc.item())

                t.update(1)
                if i + 1 == len(val_loader):
                    t.set_postfix_str(
                        f"val_loss: {val_loss.avg:.4f}, val_acc: {100 * val_acc.avg:.2f}%",
                        refresh=False,
                    )

                if config.debug and i == 5:
                    break

    if rank == 0 and not config.debug:
        wandb.log(
            {
                "epoch": epoch,
                "val/loss": val_loss.avg,
                "val/acc": 100 * val_acc.avg,
                "val/time": time.time() - start_time,
            }
        )


if __name__ == "__main__":
    main()
