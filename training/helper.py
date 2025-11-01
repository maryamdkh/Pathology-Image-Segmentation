import torch
import numpy as np
from pathlib import Path

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_optimizer(model, config):
    opt_name = config["training"]["optimizer"].lower()
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


def build_scheduler(optimizer, config):
    sch_name = config["training"]["scheduler"].lower()
    num_epochs = config["training"]["num_epochs"]

    if sch_name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif sch_name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    else:
        raise ValueError(f"Unsupported scheduler: {sch_name}")


def resume_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, device):
    start_epoch = 1
    best_val_dice = -1.0
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt and scaler:
            scaler.load_state_dict(ckpt["scaler_state"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "val_dice" in ckpt:
            best_val_dice = ckpt["val_dice"]
        print(f"Resumed from epoch {start_epoch - 1} | Best val dice: {best_val_dice:.4f}")
    else:
        print("Starting new training run.")
    return start_epoch, best_val_dice


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_dice, patient_split, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler else None,
        "val_dice": val_dice,
        "patient_split": {
            "train": patient_split["train_patients"].tolist(),
            "val": patient_split["val_patients"].tolist()
        },
    }, path)
