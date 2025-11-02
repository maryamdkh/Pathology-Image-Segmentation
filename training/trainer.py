import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast

from training.metrics import dice_coeff,iou_coeff,precision_recall
from training.helper import set_seed, build_optimizer, build_scheduler,resume_checkpoint, save_checkpoint, EarlyStopping
from data.helper import  create_dataloaders
from models.helper import build_seg_model
from utils.utils import check_system_resources, get_device
from training.losses import LossFactory

def train_one_epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer,criterion, device,
                    scaler: GradScaler = None, accumulation_steps: int = 1):

    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    n_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)              # [B,3,H,W]
        masks = batch["mask"].to(device).float()                # [B,H,W] or [B,1,H,W] depending
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)                 # -> [B,1,H,W]


        with autocast(enabled=(scaler is not None)):
            logits = model(images)                      # [B,1,H,W]
            loss, loss_dict  = criterion(logits, masks)

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation: step only every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # metrics - use probabilities
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            batch_dice = dice_coeff(preds, masks).item()
            batch_iou = iou_coeff(preds, masks).item() 

        running_loss += loss.item()* accumulation_steps  # Scale back
        running_dice += batch_dice
        running_iou += batch_iou
        n_batches += 1
        pbar.set_postfix(loss=f"{running_loss / n_batches:.4f}",
                         dice=f"{running_dice / n_batches:.4f}",
                         iou=f"{running_iou / n_batches:.4f}")

    return running_loss / n_batches, running_dice / n_batches,running_iou / n_batches

def validate_one_epoch(model: torch.nn.Module, dataloader: DataLoader,criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0  # Add precision to detect bias
    running_recall = 0.0     # Add recall
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
            batch_iou = iou_coeff(preds, masks).item()
            batch_precision, batch_recall = precision_recall(preds, masks)

            running_loss += loss.item()
            running_dice += batch_dice
            running_iou += batch_iou
            running_precision += batch_precision
            running_recall += batch_recall
            n_batches += 1
            pbar.set_postfix(val_loss=f"{running_loss / n_batches:.4f}",
                             val_dice=f"{running_dice / n_batches:.4f}",
                             val_iou=f"{running_iou / n_batches:.4f}")
            metrics = {
                'loss': running_loss / n_batches,
                'dice': running_dice / n_batches,
                'iou': running_iou / n_batches,
                'precision': running_precision / n_batches,
                'recall': running_recall / n_batches,
                'f1': 2 * (running_precision * running_recall) / (running_precision + running_recall + 1e-8) / n_batches
            }

    return metrics


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
    device = get_device(config["training"].get("device", 'auto'))
    best_checkpoint_path = config["training"]["checkpoint_dir"] + "/best_model.pth"

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    patient_split, dataloaders = create_dataloaders(config=config, splits=['train','val'])

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
    accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)

    

    # -------------------------------------------------------------------------
    # 4. Resume Checkpoint (if any)
    # -------------------------------------------------------------------------
    start_epoch, best_val_dice = resume_checkpoint(
        model, optimizer, scheduler, scaler,
        config["model"].get("checkpoint_path"), device
    )

    # Early stopping setup
    early_stopping = EarlyStopping(
        patience=config["training"].get("early_stopping_patience", 15),
        min_delta=config["training"].get("early_stopping_delta", 1e-4),
        mode='max',  # For dice score (higher is better)
        best_score = best_val_dice,
        verbose=True
    )

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    num_epochs = config["training"]["num_epochs"]
    train_history = {
        'train_loss': [], 'train_dice': [], 'train_iou': [],
        'val_loss': [], 'val_dice': [], 'val_iou': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'learning_rates': []
    }

    # Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 50)

        train_loss, train_dice, train_iou = train_one_epoch(model, dataloaders['train'], optimizer,
                                                 criterion=criterion, device=device, scaler=scaler,
                                                 accumulation_steps=accumulation_steps)
        val_metrics = validate_one_epoch(model, dataloaders['val'],
                                                criterion=criterion, device=device)
        
        
        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        if config["training"]["scheduler"] == "ReduceLROnPlateau":
            scheduler.step(val_metrics['dice'])
        else:
            scheduler.step()

        # Store history
        train_history['train_loss'].append(train_loss)
        train_history['train_dice'].append(train_dice)
        train_history['train_iou'].append(train_iou)
        train_history['val_loss'].append(val_metrics['loss'])
        train_history['val_dice'].append(val_metrics['dice'])
        train_history['val_iou'].append(val_metrics['iou'])
        train_history['val_precision'].append(val_metrics['precision'])
        train_history['val_recall'].append(val_metrics['recall'])
        train_history['val_f1'].append(val_metrics['f1'])
        train_history['learning_rates'].append(current_lr)

        # Logging
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_metrics['loss'],
            "train_dice": train_dice,
            "val_dice": val_metrics['dice'],
            "train_iou": train_iou,
            "val_iou": val_metrics['iou'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "val_f1": val_metrics['f1'],
            "learning_rate": current_lr,
        }

        if logger is not None:
            logger.log_metrics(metrics, step=epoch)
        
        print(f"Epoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        print(f"  Val Details - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  LR: {current_lr:.2e}")

        # Save best model (based on validation dice)
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_metrics['dice'], patient_split, best_checkpoint_path)
            print(f"Saved new best checkpoint: {best_checkpoint_path} (val_dice={val_metrics['dice']:.4f})")
        
        # Check early stopping (using validation dice)
        if early_stopping(val_metrics['dice'], model, epoch):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if logger is not None:
        logger.finish()

    print(f"\nTraining finished")
    print(f"Best val dice: {best_val_dice:.4f} at epoch {early_stopping.best_epoch}")
    print(f"Best checkpoint: {best_checkpoint_path}")

    return best_checkpoint_path,train_history

