import torch
import numpy as np
from pathlib import Path


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, mode='max',best_score =None, restore_best=True, verbose=True):
        """
        Early stopping handler
        
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics where higher is better, 'min' for loss
            best_score: best_score if available
            restore_best: Whether to restore best weights when stopping
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == 'max':
            self.compare = lambda x, y: (x - y) > min_delta
            self.best_score = -float('inf') if self.best_score == None else self.best_score
        else:  # mode == 'min'
            self.compare = lambda x, y: (y - x) > min_delta
            self.best_score = float('inf') if self.best_score == None else self.best_score
            
    def __call__(self, score, model, epoch):
        if self.mode == 'max':
            is_better = score > (self.best_score + self.min_delta)
        else:
            is_better = score < (self.best_score - self.min_delta)
            
        if is_better:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            # Save best model state
            self.best_state = {
                'model_state': model.state_dict().copy(),
                'epoch': epoch,
                'score': score
            }
            if self.verbose:
                print(f"🚀 EarlyStopping: New best score: {score:.6f} (epoch {epoch})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")
                
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best and hasattr(self, 'best_state'):
                    model.load_state_dict(self.best_state['model_state'])
                    if self.verbose:
                        print(f"🔄 EarlyStopping: Restored best model from epoch {self.best_epoch} (score: {self.best_score:.6f})")
                if self.verbose:
                    print(f"🛑 EarlyStopping: Triggered at epoch {epoch}")
                    
        return self.early_stop
    
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
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


def build_scheduler(optimizer, config):
    sch_name = config["training"]["scheduler"].lower()
    num_epochs = config["training"]["num_epochs"]

    if sch_name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif sch_name == "cosineannealingwarmrestartslr":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2,eta_min=1e-6)
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
