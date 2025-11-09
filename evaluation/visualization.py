import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

def generate_evaluation_visualizations(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 5,
    threshold: float = 0.5
):
    """
    Generate visualization comparing ground truth vs predicted masks.
    Properly reconstructs full images from tiles using tile coordinates.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    
    # Store tiles for each unique image
    image_tiles = defaultdict(list)
    
    print("🔍 Collecting tile predictions for reconstruction...")
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"]
            patient_nums = batch["patient_num"]
            image_nums = batch["image_num"]
            image_types = batch["image_type"]
            tile_coords = batch["tile_coords"]
            original_dims = batch["original_dims"]
            
            # Get predictions
            logits = model(images)
            preds = torch.sigmoid(logits) > threshold
            preds = preds.cpu().numpy().astype(np.uint8)
            
            # Store each tile with its metadata
            for i in range(len(images)):
                img_id = f"{patient_nums[i]}_{image_nums[i]}_{image_types[i]}"
                
                tile_info = {
                    'tile_img': images[i].cpu().numpy().transpose(1, 2, 0),
                    'tile_gt_mask': masks[i].numpy(),
                    'tile_pred_mask': preds[i, 0],  # Remove channel dimension
                    'patient_num': patient_nums[i],
                    'image_num': image_nums[i],
                    'image_type': image_types[i],
                    'tile_coords': tile_coords[i],
                    'original_dims': original_dims[i]
                }
                
                image_tiles[img_id].append(tile_info)
    
    # Reconstruct and visualize full images
    print("🎨 Reconstructing full images from tiles...")
    sample_count = 0
    
    for img_id, tiles in image_tiles.items():
        if sample_count >= num_samples:
            break
            
        # Reconstruct full image from tiles
        reconstructed = reconstruct_from_tiles(tiles)
        if reconstructed:
            full_image, full_gt_mask, full_pred_mask, metadata = reconstructed
            
            # Create visualization
            fig = create_comparison_visualization(
                full_image, full_gt_mask, full_pred_mask, 
                metadata['patient_num'], metadata['image_num'], metadata['image_type']
            )
            
            # Save figure
            save_path = output_dir / f"sample_{sample_count+1}_{img_id}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
            plt.close(fig)
            
            print(f"✅ Saved visualization: {save_path}")
            sample_count += 1

def reconstruct_from_tiles(tiles: List[Dict]) -> Optional[Tuple]:
    """
    Reconstruct full image from tiles using actual tile coordinates.
    
    Args:
        tiles: List of tile dictionaries with image data and coordinates
        
    Returns:
        Tuple of (full_image, full_gt_mask, full_pred_mask, metadata) or None if reconstruction fails
    """
    try:
        # Check if this is a single image (no tiling)
        if tiles[0]['tile_coords'] is None:
            # Single image - no reconstruction needed
            tile = tiles[0]
            full_image = (tile['tile_img'] * 255).astype(np.uint8)
            full_gt_mask = tile['tile_gt_mask'].astype(np.uint8)
            full_pred_mask = tile['tile_pred_mask'].astype(np.uint8)
        else:
            # Multiple tiles - reconstruct using coordinates
            # Get original dimensions from first tile (all tiles should have same dimensions)
            H, W = tiles[0]['original_dims']
            
            # Create empty canvases for reconstruction (use original dimensions)
            full_image = np.zeros((H, W, 3), dtype=np.uint8)
            full_gt_mask = np.zeros((H, W), dtype=np.uint8)
            full_pred_mask = np.zeros((H, W), dtype=np.uint8)
            
            # Place each tile in its correct position
            for tile in tiles:
                y0, y1, x0, x1 = tile['tile_coords']
                
                # Convert tile image back to uint8
                tile_img = (tile['tile_img'] * 255).astype(np.uint8)
                
                # Get tile dimensions
                tile_h, tile_w = tile_img.shape[:2]
                
                # Ensure we don't exceed original dimensions
                y1 = min(y1, H)
                x1 = min(x1, W)
                
                # Place tile in full image
                full_image[y0:y1, x0:x1] = tile_img
                full_gt_mask[y0:y1, x0:x1] = tile['tile_gt_mask'][:y1-y0, :x1-x0]
                full_pred_mask[y0:y1, x0:x1] = tile['tile_pred_mask'][:y1-y0, :x1-x0]
        
        metadata = {
            'patient_num': tiles[0]['patient_num'],
            'image_num': tiles[0]['image_num'],
            'image_type': tiles[0]['image_type'],
            'num_tiles': len(tiles)
        }
        
        return full_image, full_gt_mask, full_pred_mask, metadata
        
    except Exception as e:
        print(f"⚠️ Tile reconstruction failed: {e}")
        return None

def create_comparison_visualization(
    image: np.ndarray,
    gt_mask: np.ndarray, 
    pred_mask: np.ndarray,
    patient_num: int,
    image_num: int,
    image_type: str
):
    """
    Create a comparison visualization between ground truth and prediction.
    
    Args:
        image: Original image [H, W, 3]
        gt_mask: Ground truth mask [H, W]
        pred_mask: Predicted mask [H, W]
        patient_num: Patient number
        image_num: Image number
        image_type: Type of image (raw/sn1/sn2)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image\nPatient {patient_num}, Image {image_num}\nType: {image_type}")
    axes[0].axis('off')
    
    # Ground truth mask overlay
    axes[1].imshow(image)
    axes[1].imshow(gt_mask, alpha=0.5, cmap='Reds')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')
    
    # Predicted mask overlay
    axes[2].imshow(image)
    axes[2].imshow(pred_mask, alpha=0.5, cmap='Reds')
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig