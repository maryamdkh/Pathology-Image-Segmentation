"""
Functional visualization utilities for segmentation results.
"""

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def create_segmentation_overlay(
    image: np.ndarray,
    binary_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35
) -> np.ndarray:
    """
    Create segmentation overlay on original image.
    
    Args:
        image: Original image in HWC format
        binary_mask: Binary segmentation mask
        color: RGB color for overlay
        alpha: Transparency of overlay
        
    Returns:
        Image with segmentation overlay
    """
    if np.max(image) <= 1:
        image = (image * 255).astype(np.uint8)
    
    color_mask = np.zeros_like(image)
    color_mask[..., 0] = binary_mask * color[0]  # R
    color_mask[..., 1] = binary_mask * color[1]  # G  
    color_mask[..., 2] = binary_mask * color[2]  # B
    
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)


def predict_single_image(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate segmentation prediction for a single image using tiling.
    
    Returns:
        Tuple of (probability_mask, binary_mask)
    """
    model.eval()
    
    # Ensure image is in HWC format
    if image.ndim == 3 and image.shape[0] == 3:  # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))
    
    H, W, C = image.shape
    
    # Apply padding for tiling
    padded_image, pad_info = apply_tiling_padding(image, tile_size)
    tiles, positions = extract_tiles(padded_image, tile_size, overlap)
    
    # Process each tile
    probabilities = process_tiles(model, tiles, device)
    
    # Reconstruct full prediction
    probability_mask = reconstruct_from_tiles(probabilities, positions, padded_image.shape, tile_size, overlap)
    probability_mask = remove_padding(probability_mask, pad_info)
    
    return probability_mask, (probability_mask > 0.5).astype(np.uint8)


def apply_tiling_padding(
    image: np.ndarray,
    tile_size: int
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Apply padding to make image divisible by tile size."""
    H, W = image.shape[:2]
    
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    return padded, {"original_height": H, "original_width": W, "pad_h": pad_h, "pad_w": pad_w}


def extract_tiles(
    image: np.ndarray,
    tile_size: int,
    overlap: int
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Extract tiles from image with specified overlap."""
    stride = tile_size - overlap
    tiles, positions = [], []
    
    for y in range(0, image.shape[0] - tile_size + 1, stride):
        for x in range(0, image.shape[1] - tile_size + 1, stride):
            tiles.append(image[y:y + tile_size, x:x + tile_size])
            positions.append((y, x))
    
    return tiles, positions


def process_tiles(
    model: torch.nn.Module,
    tiles: List[np.ndarray],
    device: torch.device
) -> List[np.ndarray]:
    """Process tiles through model and return probabilities."""
    preprocess = transforms.ToTensor()
    probabilities = []
    
    with torch.no_grad():
        for tile in tiles:
            tile_tensor = preprocess(tile).unsqueeze(0).to(device)
            prediction = model(tile_tensor)
            
            # Handle different model output formats
            if isinstance(prediction, (list, tuple)):
                prediction = prediction[0]
            
            probability = torch.sigmoid(prediction).squeeze().cpu().numpy()
            probabilities.append(probability)
    
    return probabilities


def reconstruct_from_tiles(
    tile_predictions: List[np.ndarray],
    positions: List[Tuple[int, int]],
    image_shape: Tuple[int, int],
    tile_size: int,
    overlap: int
) -> np.ndarray:
    """Reconstruct full prediction from tiles."""
    reconstructed = np.zeros(image_shape[:2], dtype=np.float32)
    weight_map = np.zeros_like(reconstructed)
    
    for (y, x), prediction in zip(positions, tile_predictions):
        reconstructed[y:y + tile_size, x:x + tile_size] += prediction
        weight_map[y:y + tile_size, x:x + tile_size] += 1
    
    return reconstructed / np.maximum(weight_map, 1e-8)


def remove_padding(
    image: np.ndarray,
    pad_info: Dict[str, int]
) -> np.ndarray:
    """Remove padding added for tiling."""
    H = pad_info["original_height"]
    W = pad_info["original_width"]
    return image[:H, :W]


def visualize_prediction_comparison(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    ground_truth_mask: Optional[np.ndarray] = None,
    tile_size: int = 512,
    overlap: int = 64,
    threshold: float = 0.5,
    save_path: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate and visualize segmentation prediction with optional ground truth comparison.
    """
    # Generate prediction
    probability_mask, binary_mask = predict_single_image(
        model, image, device, tile_size, overlap
    )
    
    # Create visualization
    fig, axes = create_comparison_figure(ground_truth_mask is not None)
    
    # Original image
    if np.max(image) <= 1:
        display_image = (image * 255).astype(np.uint8)
    else:
        display_image = image.astype(np.uint8)
    
    # Prediction overlay
    pred_overlay = create_segmentation_overlay(display_image, binary_mask)
    
    if ground_truth_mask is not None:
        # Ground truth overlay
        gt_overlay = create_segmentation_overlay(
            display_image, ground_truth_mask, color=(255, 0, 0)
        )
        
        axes[0].imshow(cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Predicted Segmentation")
        axes[0].axis("off")
        
        axes[1].imshow(cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
    else:
        axes.imshow(cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB))
        axes.set_title("Predicted Segmentation")
        axes.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved visualization to: {save_path}")
    
    plt.show()
    
    return probability_mask, binary_mask


def create_comparison_figure(has_ground_truth: bool) -> Tuple[plt.Figure, plt.Axes]:
    """Create appropriate figure for comparison visualization."""
    if has_ground_truth:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    
    return fig, axes