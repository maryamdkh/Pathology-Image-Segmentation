import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py

def visualize_image_with_prediction(dataset, model, 
                                    image_type: str, image_idx: int,
                                      alpha: float = 0.6, device: str = 'cuda'):
    """
    Reconstruct and visualize an image with both real and predicted segmentation overlays.
    
    Args:
        dataset: CoCaHisDataset instance
        model: PyTorch model for prediction
        image_type: Image type ("raw", "sn1", "sn2")
        image_idx: Index of the image in original dataset
        alpha: Transparency for segmentation overlay
        device: Device to run model on ('cuda' or 'cpu')
    """
    # Get all tiles for the image
    indices = []
    for key in dataset.image_index_to_dataset_indices.keys():
        indices.append(int(key[1]))

    tiles = dataset.get_image_tiles(image_type, indices[image_idx])
    
    if not tiles:
        print(f"No tiles found for image_type '{image_type}', image_idx {image_idx}")
        return
    
    # Get original dimensions from first tile
    H, W = tiles[0]["original_dims"]
    
    # Get original unpadded dimensions by loading the image once
    with h5py.File(dataset.h5_path, 'r') as f:
        original_img = np.array(f[f"HE/{image_type}"][image_idx])
        original_h, original_w = original_img.shape[:2]
    
    # Initialize reconstruction arrays
    image_recon = np.zeros((H, W, 3), dtype=np.float32)
    mask_recon = np.zeros((H, W), dtype=np.float32)
    pred_recon = np.zeros((H, W), dtype=np.float32)
    count_recon = np.zeros((H, W), dtype=np.float32)
    
    model.eval()
    
    # Reconstruct from tiles
    with torch.no_grad():
        for tile in tiles:
            tile_coords = tile["tile_coords"]
            if tile_coords is None:
                # No tiling case - use the whole image
                img_tensor = tile["image"].unsqueeze(0).to(device)  # [1,C,H,W]
                
                # Get prediction
                pred = model(img_tensor)
                if isinstance(pred, torch.Tensor):
                    pred = torch.sigmoid(pred) if pred.dim() > 1 else pred
                    pred_mask = (pred > 0.5).squeeze().cpu().numpy()
                else:
                    pred_mask = pred.squeeze()
                
                # Convert to numpy
                img = tile["image"].permute(1, 2, 0).numpy()
                mask = tile["mask"].numpy()
                
                image_recon = img
                mask_recon = mask
                pred_recon = pred_mask
                count_recon = np.ones((H, W))
                break
            
            y0, y1, x0, x1 = tile_coords
            
            # Convert tensor to numpy for reconstruction
            img_tile = tile["image"]
            mask_tile = tile["mask"]
            
            # Convert to numpy arrays
            img_np = img_tile.permute(1, 2, 0).numpy() if isinstance(img_tile, torch.Tensor) else img_tile
            mask_np = mask_tile.numpy() if isinstance(mask_tile, torch.Tensor) else mask_tile
            
            # Get prediction for this tile
            img_tensor = img_tile.unsqueeze(0) if not isinstance(img_tile, torch.Tensor) else img_tile.unsqueeze(0)
            img_tensor = img_tensor.to(device)
            pred = model(img_tensor)
            
            if isinstance(pred, torch.Tensor):
                pred = torch.sigmoid(pred) if pred.dim() > 1 else pred
                pred_mask = (pred > 0.5).squeeze().cpu().numpy()
            else:
                pred_mask = pred.squeeze()
            
            # Add to reconstruction with averaging for overlap regions
            image_recon[y0:y1, x0:x1] += img_np
            mask_recon[y0:y1, x0:x1] += mask_np
            pred_recon[y0:y1, x0:x1] += pred_mask
            count_recon[y0:y1, x0:x1] += 1
    
    # Average overlapping regions
    valid_mask = count_recon > 0
    image_recon[valid_mask] = image_recon[valid_mask] / count_recon[valid_mask][:, None]
    mask_recon[valid_mask] = mask_recon[valid_mask] / count_recon[valid_mask]
    pred_recon[valid_mask] = pred_recon[valid_mask] / count_recon[valid_mask]
    
    # Threshold the prediction after averaging
    pred_recon = (pred_recon > 0.5).astype(np.float32)
    
    # Crop to original unpadded dimensions
    image_recon = image_recon[:original_h, :original_w]
    mask_recon = mask_recon[:original_h, :original_w]
    pred_recon = pred_recon[:original_h, :original_w]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Ground Truth
    axes[0, 0].imshow(image_recon)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_recon, cmap='Reds')
    axes[0, 1].set_title('Ground Truth Mask')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(image_recon)
    axes[0, 2].imshow(mask_recon, cmap='Reds', alpha=alpha)
    axes[0, 2].set_title('GT Overlay')
    axes[0, 2].axis('off')
    
    # Row 2: Predictions
    axes[1, 0].imshow(image_recon)
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_recon, cmap='Reds')
    axes[1, 1].set_title('Predicted Mask')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(image_recon)
    axes[1, 2].imshow(pred_recon, cmap='Reds', alpha=alpha)
    axes[1, 2].set_title('Prediction Overlay')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Image: {image_type}, Index: {image_idx}, Size: {original_h}x{original_w}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return image_recon, mask_recon, pred_recon
