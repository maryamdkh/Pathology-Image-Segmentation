"""
Functional visualization utilities for segmentation results.
"""

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Optional, Tuple, List
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
        image: Original image in HWC format (Height, Width, Channels)
        binary_mask: Binary segmentation mask (Height, Width)
        color: RGB color for overlay
        alpha: Transparency of overlay
        
    Returns:
        Image with segmentation overlay in RGB format
    """
    # Ensure image is in HWC format and uint8
    if image.ndim == 3 and image.shape[0] == 3:  # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))
    
    if np.max(image) <= 1:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Ensure mask has same spatial dimensions as image
    if binary_mask.shape != image.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
    
    # Ensure mask is binary and uint8
    binary_mask = (binary_mask > 0).astype(np.uint8)
    
    # Create color mask
    color_mask = np.zeros_like(image)
    color_mask[..., 0] = binary_mask * color[0]  # R
    color_mask[..., 1] = binary_mask * color[1]  # G  
    color_mask[..., 2] = binary_mask * color[2]  # B
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    
    return overlay


def prepare_image_for_visualization(image: np.ndarray) -> np.ndarray:
    """
    Prepare image for visualization by ensuring proper format.
    
    Returns:
        Image in HWC uint8 format
    """
    # Handle different input formats
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    
    # Convert CHW to HWC if needed
    if image.ndim == 3 and image.shape[0] in [1, 3]:  # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))
    
    # Handle single channel images
    if image.ndim == 2:  # HW -> HWC
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:  # HW1 -> HWC
        image = np.repeat(image, 3, axis=2)
    
    # Normalize to 0-255
    if np.max(image) <= 1:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    return image


def prepare_mask_for_visualization(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Prepare mask for visualization.
    
    Returns:
        Binary mask with specified shape
    """
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    
    # Handle different mask formats
    if mask.ndim == 3:
        if mask.shape[0] == 1:  # 1HW -> HW
            mask = mask[0]
        elif mask.shape[2] == 1:  # HW1 -> HW
            mask = mask[:, :, 0]
    
    # Ensure binary and resize to target shape
    mask = (mask > 0.5).astype(np.uint8)
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]))
    
    return mask


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
    
    # Prepare image for processing
    image_processed = prepare_image_for_visualization(image)
    H, W = image_processed.shape[:2]
    
    # Apply padding for tiling
    padded_image, pad_info = apply_tiling_padding(image_processed, tile_size)
    tiles, positions = extract_tiles(padded_image, tile_size, overlap)
    
    # Process each tile
    probabilities = process_tiles(model, tiles, device)
    
    # Reconstruct full prediction
    probability_mask = reconstruct_from_tiles(probabilities, positions, padded_image.shape, tile_size, overlap)
    probability_mask = remove_padding(probability_mask, pad_info)
    
    binary_mask = (probability_mask > 0.5).astype(np.uint8)
    
    return probability_mask, binary_mask


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
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
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
    
    # Prepare images for visualization
    display_image = prepare_image_for_visualization(image)
    
    # Create prediction overlay
    pred_overlay = create_segmentation_overlay(display_image, binary_mask)
    
    # Create ground truth overlay if available
    if ground_truth_mask is not None:
        gt_mask_processed = prepare_mask_for_visualization(ground_truth_mask, display_image.shape[:2])
        gt_overlay = create_segmentation_overlay(
            display_image, gt_mask_processed, color=(255, 0, 0)
        )
    
    # Create visualization
    fig, axes = create_comparison_figure(ground_truth_mask is not None)
    
    if ground_truth_mask is not None:
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
        return fig, axes
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        return fig, axes


def debug_shapes(image: np.ndarray, mask: np.ndarray):
    """Debug function to print shapes of image and mask."""
    print(f"Image shape: {image.shape}, dtype: {image.dtype}, range: [{np.min(image):.3f}, {np.max(image):.3f}]")
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, range: [{np.min(mask):.3f}, {np.max(mask):.3f}]")