import matplotlib.pyplot as plt
from typing import List, Any, Optional


def visualize_Cocahis_comparison(
    raw_images: List[Any],
    sn1_images: List[Any],
    sn2_images: List[Any],
    gt: List[Any],
    gts: List[List[Any]],
    selected_image: int = 43,
    save_paths: Optional[List[str]] = None,
    figsize_top: tuple = (15, 3),
    figsize_bottom: tuple = (15, 3)
) -> None:
    """
    Visualize image comparisons for different processing types and annotators.
    
    Args:
        raw_images: List of raw images
        sn1_images: List of standardized normalized images (target 1)
        sn2_images: List of standardized normalized images (target 2)
        gt: List of ground truth images (majority vote)
        gts: List of lists containing individual annotator ground truths
        selected_image: Index of the image to visualize
        save_paths: Optional list of paths to save figures [top_path, bottom_path]
        figsize_top: Figure size for the top comparison
        figsize_bottom: Figure size for the bottom comparison
    """
    if save_paths is None:
        save_paths = ["up.png", "down.png"]
    
    # Validate inputs
    if selected_image >= len(raw_images):
        raise ValueError(f"selected_image index {selected_image} out of range")
    
    def _create_top_comparison(
        raw_images: List[Any],
        sn1_images: List[Any],
        sn2_images: List[Any],
        gt: List[Any],
        selected_image: int,
        save_path: str,
        figsize: tuple
    ) -> None:
        """Create the top row comparison figure."""
        names = [
            "Raw",
            "Stan normalized (target 1)",
            "Stan normalized (target 2)", 
            "Majority Vote GT"
        ]
        
        images = [
            raw_images[selected_image],
            sn1_images[selected_image], 
            sn2_images[selected_image],
            gt[selected_image]
        ]
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        for ax, image, name in zip(axes, images, names):
            ax.imshow(image)
            ax.set_title(name, fontname="sans-serif", fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_bottom_comparison(
        gts: List[List[Any]],
        selected_image: int,
        save_path: str,
        figsize: tuple
    ) -> None:
        """Create the bottom row annotators comparison figure."""
        num_annotators = len(gts)
        
        fig, axes = plt.subplots(1, num_annotators, figsize=figsize)
        
        for i, ax in enumerate(axes):
            ax.set_title(f"Annotator {i+1}", fontname="sans-serif", fontweight="bold")
            ax.imshow(gts[i][selected_image])
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


    # Top row: Main image comparisons
    _create_top_comparison(
        raw_images, sn1_images, sn2_images, gt, 
        selected_image, save_paths[0], figsize_top
    )
    
    # Bottom row: Individual annotators
    _create_bottom_comparison(
        gts, selected_image, save_paths[1], figsize_bottom
    )





