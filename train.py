import os
import torch
import argparse
import warnings
import utils
from torch import nn
from tqdm import tqdm
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from models.guidecad import GuideCAD
from dataset.cad_dataset import CADDataset, get_dataloader
from torch.utils.tensorboard import SummaryWriter
from loss import CADLoss
from configs.parse import load_config
from dataset.cad_dataset import *

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
    module="torch.storage",
)


def setup_ddp(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "12354",
):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train_one_epoch(
    epoch: int,
    rank: int,
    model: nn.Module,
    optimizer,
    train_loader,
    loss_fn,
    scaler,
    cfg,
    writer=None,
    accumulation_steps: int = 4,
):
    model.train()
    if isinstance(model, DDP):
        train_loader.sampler.set_epoch(epoch)

    progress_bar = (
        tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Rank {rank} Training Epoch {epoch}",
            disable=rank != 0,
            ncols=160,
        )
        if rank == 0
        else train_loader
    )

    total_loss = 0.0
    total_cmd_loss = 0.0
    total_param_loss = 0.0

    for step, data in enumerate(progress_bar, 1):
        gt_command = data["command"].cuda(rank)
        gt_param = data["args"].cuda(rank)
        prefix = data["prefix"].cuda(rank)
        input_ids = data["input_ids"].cuda(rank)
        mask = data["mask"].cuda(rank)

        with torch.amp.autocast("cuda"):
            command_logits, args_logits, gpt_logit = model(input_ids=input_ids, attention_mask=mask, img_embed=prefix)
            loss_dict = loss_fn(
                command_logits,
                args_logits,
                gt_command,
                gt_param,
                text_logits=gpt_logit[:, cfg.model.prefix_length :, :],
                gt_text=input_ids,
            )

        cmd_loss = loss_dict["loss_cmd"]
        args_loss = loss_dict["loss_args"]

        if accumulation_steps > 1:
            loss = cmd_loss + args_loss
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if step % accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss = cmd_loss + args_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_cmd_loss += cmd_loss.item()
        total_param_loss += cmd_loss.item()

        if rank == 0:
            logging = {
                "total_loss": total_loss,
                "cmd_loss": cmd_loss.item(),
                "args_loss": args_loss.item(),
            }
            progress_bar.set_postfix(logging)

    return {
        "total_loss": total_loss / len(train_loader),
        "total_cmd_loss": total_cmd_loss / len(train_loader),
        "total_param_loss": total_param_loss / len(train_loader),
    }


@torch.no_grad()
def valid_one_epoch(epoch, rank, model, val_loader, loss_fn, cfg, writer=None):
    model.eval()
    if isinstance(model, DDP):
        val_loader.sampler.set_epoch(epoch)

    progress_bar = (
        tqdm(
            val_loader,
            total=len(val_loader),
            desc=f"Rank {rank} Validation Epoch {epoch}",
            disable=rank != 0,
            ncols=100,
        )
        if rank == 0
        else val_loader
    )

    total_loss = 0.0
    total_cmd_loss = 0.0
    total_param_loss = 0.0
    for data in progress_bar:
        gt_command = data["command"].cuda(rank)
        gt_param = data["args"].cuda(rank)
        prefix = data["prefix"].cuda(rank)
        input_ids = data["input_ids"].cuda(rank)
        mask = data["mask"].cuda(rank)

        command_logits, args_logits, gpt_logit = model(input_ids=input_ids, attention_mask=mask, img_embed=prefix)

        # compute the loss
        loss_dict = loss_fn(
            command_logits,
            args_logits,
            gt_command,
            gt_param,
            text_logits=gpt_logit[:, cfg.model.prefix_length :, :],
            gt_text=input_ids,
        )

        cmd_loss = loss_dict["loss_cmd"].item()
        args_loss = loss_dict["loss_args"].item()

        total_cmd_loss += cmd_loss
        total_param_loss += args_loss
        total_loss += cmd_loss + args_loss

    return {
        "total_loss": total_loss / len(val_loader),
        "total_cmd_loss": total_cmd_loss / len(val_loader),
        "total_param_loss": total_param_loss / len(val_loader),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: nn.Module,
    scheduler: nn.Module,
    cfg,
    epoch: int,
    scaler: torch.amp.GradScaler,
    file_name: str,
    save_dir="checkpoints",
):
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": {k.replace("module.", ""): v for k, v in model_state.items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "config": cfg,
    }

    save_path = os.path.join(save_dir, f"{file_name}_datasize{cfg.data.ratio}.pt")
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved for epoch {epoch} at {save_path}")


def main(rank, cfg, world_size):
    utils.set_randomness(cfg.training.seed)
    setup_ddp(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        writer = SummaryWriter(log_dir=cfg.log.project_name)
    else:
        writer = None

    tokenizer = setup_tokenizer(cfg.model.model_name)
    train_loader = get_dataloader("train", cfg, tokenizer, rank, cfg.model.prefix_mode)
    val_loader = get_dataloader("validation", cfg, tokenizer, rank, cfg.model.prefix_mode)

    model = GuideCAD(
        prefix_embed_dim=cfg.model.img_embed_dim,
        num_commands=cfg.cad_params.num_commands,
        num_params=cfg.cad_params.num_params,
        param_range=cfg.cad_params.param_range,
        mapping_num_layers=cfg.model.mapping_num_layers,
        prefix_length=cfg.model.prefix_length,
        d_model=cfg.model.d_model,
        dim_z=cfg.model.dim_z,
        adapt_embed=cfg.model.adapt_embed,
        finetune=cfg.model.finetune,
    ).to(device)
    model.gpt.resize_token_embeddings(len(tokenizer))

    scaler = torch.amp.GradScaler()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.optimizer.lr,
        eps=1e-8,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.scheduler.t_max,
        eta_min=cfg.scheduler.min_lr,
    )

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    loss_fn = CADLoss(cfg, device, tokenizer, False).to(device)

    total_params, trainable_params = utils.count_model_parameters(model)
    print("total_params, trainable_params", total_params, trainable_params)

    best_loss = 1e9
    max_epoch = -1
    for epoch in range(cfg.training.epochs):
        train_metrics = train_one_epoch(
            epoch,
            rank,
            model,
            optimizer,
            train_loader,
            loss_fn,
            scaler,
            cfg,
            writer=writer,
            accumulation_steps=cfg.model.accumulation_steps,
        )

        valid_metrics = valid_one_epoch(epoch, rank, model, val_loader, loss_fn, cfg, writer=writer)

        if writer:
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)

            for key, metrics in valid_metrics.items():
                writer.add_scalar(f"val/{key}", metrics, epoch)

        if rank == 0:  # Only the main process logs
            if best_loss > valid_metrics["total_loss"]:
                best_loss = valid_metrics["total_loss"]
                max_epoch = epoch
                print(f"best model update !!! best_loss={best_loss}")
                save_checkpoint(
                    model, optimizer, scheduler, cfg, epoch, scaler, cfg.checkpoint.best_model, cfg.checkpoint.save_dir
                )
            else:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    cfg,
                    epoch,
                    scaler,
                    cfg.checkpoint.latest_model,
                    cfg.checkpoint.save_dir,
                )

            print(
                f"Epoch {epoch}: Loss(train, val, best) = {train_metrics['total_loss']:.4f}, {valid_metrics['total_loss']:.4f}, {best_loss:.4f} | bestloss_max_epoch:{max_epoch} | "
                f"valid_metrics = {valid_metrics}"
            )

        if scheduler is not None:
            scheduler.step()

    if world_size > 1:
        destroy_process_group()

    if rank == 0 and writer:
        writer.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="configs/guidecad.yaml")
    p = args.parse_args()
    print("p", p.config)

    config = load_config(p.config)
    print("config", config)
    utils.set_randomness(config.training.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus.gpu_ids

    world_size = torch.cuda.device_count()

    if world_size > 1:
        torch.multiprocessing.spawn(main, args=(config, world_size), nprocs=world_size, join=True)

    else:
        main(0, config, 1)
