import torch
from torch.data.utils import DataLoader
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast

from training.metrics import dice_coeff
from training.helper import set_seed, build_optimizer, build_scheduler,resume_checkpoint, save_checkpoint
from data.datasets import CoCaHisDataset
from data.helper import build_Cocahis_patient_split
from data.transforms import get_train_transform, get_val_transform
from training.models import build_seg_model
from utils.utils import check_system_resources
from training.losses import LossFactory

def train_one_epoch(model: torch.Module, dataloader: DataLoader, optimizer,criterion, device, scaler: GradScaler = None):

    model.train()
    running_loss = 0.0
    running_dice = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)              # [B,3,H,W]
        masks = batch["mask"].to(device).float()                # [B,H,W] or [B,1,H,W] depending
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)                 # -> [B,1,H,W]

        optimizer.zero_grad()

        with autocast(enabled=(scaler is not None)):
            logits = model(images)                      # [B,1,H,W]
            loss, loss_dict  = criterion(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # metrics - use probabilities
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            batch_dice = dice_coeff(preds, masks).item()

        running_loss += loss.item()
        running_dice += batch_dice
        n_batches += 1
        pbar.set_postfix(loss=f"{running_loss / n_batches:.4f}",
                         dice=f"{running_dice / n_batches:.4f}")

    return running_loss / n_batches, running_dice / n_batches

def validate_one_epoch(model: torch.Module, dataloader: DataLoader,criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Val", leave=False)
    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            logits = model(images)
            loss, loss_dict  = criterion(logits, masks)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            batch_dice = dice_coeff(preds, masks).item()

            running_loss += loss.item()
            running_dice += batch_dice
            n_batches += 1
            pbar.set_postfix(val_loss=f"{running_loss / n_batches:.4f}",
                             val_dice=f"{running_dice / n_batches:.4f}")

    return running_loss / n_batches, running_dice / n_batches


# ================================
# Main training loop
# ================================
def train_model(config: dict, logger=None, device=None, verbose=False):
    """
    Main training loop for pathology segmentation models.
    Fully config-driven, modular, and maintainable.
    """

    # -------------------------------------------------------------------------
    # 1. Setup & Reproducibility
    # -------------------------------------------------------------------------
    set_seed(config["training"].get("seed", 42))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_checkpoint_path = config["paths"]["checkpoint_dir"] + "/best_model.pth"

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    ds_cfg = config["dataset"]
    patient_split = build_Cocahis_patient_split(
        h5_path=ds_cfg['path'],
        val_fraction=ds_cfg["val_fraction"],
        seed=config["training"]["seed"]
    )

    train_ds = CoCaHisDataset(
        h5_path=ds_cfg['path'],
        image_type=ds_cfg["image_type"],
        split="train",
        patient_split=patient_split,
        transform=get_train_transform(),
        tile_size=ds_cfg["tile_size"],
        overlap=ds_cfg["overlap"]
    )

    val_ds = CoCaHisDataset(
        h5_path=ds_cfg['path'],
        image_type=ds_cfg["image_type"],
        split="val",
        patient_split=patient_split,
        transform=get_val_transform(),
        tile_size=ds_cfg["tile_size"],
        overlap=ds_cfg["overlap"]
    )

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"],
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    # -------------------------------------------------------------------------
    # 3. Model, Optimizer, Scheduler
    # -------------------------------------------------------------------------
    model = build_seg_model(config, device=device)

    if verbose:
        check_system_resources(model)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler() if (config["training"]["use_amp"] and torch.cuda.is_available()) else None
    criterion = LossFactory(config)

    # -------------------------------------------------------------------------
    # 4. Resume Checkpoint (if any)
    # -------------------------------------------------------------------------
    start_epoch, best_val_dice = resume_checkpoint(
        model, optimizer, scheduler, scaler,
        config["model"].get("checkpoint_path"), device
    )

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    num_epochs = config["training"]["num_epochs"]

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer,
                                                 criterion=criterion, device=device, scaler=scaler)
        val_loss, val_dice = validate_one_epoch(model, val_loader,
                                                criterion=criterion, device=device)
        scheduler.step(val_dice)

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": train_dice,
            "val_dice": val_dice,
        }
        if logger is not None:
            logger.log_metrics(metrics, step=epoch)

        print(f"Epoch {epoch} summary: train_loss={train_loss:.4f}, train_dice={train_dice:.4f}; "
              f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f}")

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_dice, patient_split, best_checkpoint_path)
            print(f"💾 Saved new best checkpoint: {best_checkpoint_path} (val_dice={val_dice:.4f})")

    if logger is not None:
        logger.finish()

    print(f"Training finished ✅ | Best val dice: {best_val_dice:.4f}")
    print("Best checkpoint saved at:", best_checkpoint_path)
    return best_checkpoint_path

