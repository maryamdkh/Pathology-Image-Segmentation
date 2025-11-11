import numpy as np
from typing import Optional, Dict, List, Union, Tuple
import h5py
import torch
from torch.utils.data import Dataset
import math
import cv2

def pad_to_multiple_of_32(img, **kwargs):
    """
    Pad image or mask so that height and width are divisible by 32.
    Uses constant padding (value=0).
    """
    h, w = img.shape[:2]
    new_h = math.ceil(h / 32) * 32
    new_w = math.ceil(w / 32) * 32
    pad_h = new_h - h
    pad_w = new_w - w

    # Pad evenly on bottom/right (no need to center-pad)
    img = cv2.copyMakeBorder(
        img,
        top=0,
        bottom=pad_h,
        left=0,
        right=pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    return img

def ensure_channels_first(tensor: torch.Tensor, expected_channels: int = 3) -> torch.Tensor:
    """
    Ensure tensor is in channels-first format [C, H, W].
    
    Args:
        tensor: Input tensor that could be in [H, W, C] or [C, H, W] format
        expected_channels: Expected number of channels
    
    Returns:
        Tensor in [C, H, W] format
    """
    if tensor.dim() == 3:
        # Check if channels are last (typical case: H, W, C)
        if tensor.shape[2] == expected_channels and tensor.shape[0] != expected_channels:
            # Convert from [H, W, C] to [C, H, W]
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] == expected_channels:
            # Already in [C, H, W] format
            pass
        else:
            # Ambiguous case, try to infer
            if tensor.shape[-1] in [1, 3, 4]:  # Common channel sizes
                tensor = tensor.permute(2, 0, 1)
            # Otherwise keep as is and warn
            else:
                print(f"Warning: Ambiguous tensor shape {tensor.shape}. Expected channels-first format.")
    
    return tensor

class CoCaHisDataset(Dataset):

    def __init__(self, h5_path: str, image_type: Union[str, List[str]] = "raw", split: str = "train",
                 transform=None, patient_split: Optional[Dict[str, np.ndarray]] = None,
                 tile_size: int = None, overlap: int = 0, verbose: bool = True ,auto_fix_channels: bool = True):
        """
        PyTorch custom dataset for CoCaHis HDF5 structure.

        Args:
            h5_path (str): Path to CoCaHis.hdf5
            image_type (str or List[str]): One of ["raw", "sn1", "sn2"] or combination like ["raw", "sn1", "sn2"]
            split (str): "train", "val", or "test"
            transform (callable, optional): Optional transform to be applied on images
            patient_split (dict, optional): Dict with keys 'train_patients' and 'val_patients'
            tile_size (int): size of the tiles if tiling is requested
            overlap (int): overlap between tiles if tiling is requested

        Returns a dict with keys:
            image: Tensor float32 [C,H,W] in [0,1]
            mask: Tensor long or float [H,W] (0/1)
            patient_num: int
            image_num: int
            image_type: str (the specific type of this sample)
        """
        
        # Validate and normalize image_type parameter
        if isinstance(image_type, str):
            if image_type == "all":
                self.image_types = ["raw", "sn1", "sn2"]
            else:
                self.image_types = [image_type]
        elif isinstance(image_type, list):
            valid_types = ["raw", "sn1", "sn2"]
            for img_type in image_type:
                if img_type not in valid_types:
                    raise ValueError(f"Invalid image_type '{img_type}'. Must be one of {valid_types}")
            self.image_types = image_type
        else:
            raise ValueError("image_type must be str or list of str")
            
        assert split in ["train", "val", "test"]

        self.h5_path = h5_path
        self.split = split
        self.transform = transform
        self.tile_size = tile_size
        self.overlap = overlap
        self.verbose = verbose
        self.auto_fix_channels = auto_fix_channels

        # We'll store all samples from all requested image types
        self.samples = []  # List of tuples: (image_type, index_in_original_dataset, tile_coords, original_dims)
        
        # Create mapping from image index to dataset indices for efficient lookup
        self.image_index_to_dataset_indices = {}
        
        with h5py.File(self.h5_path, 'r') as f:
            self.train_test_split = f["HE"].attrs["train_test_split"]
            self.patient_nums = f["HE"].attrs["patient_num"]
            self.image_nums = f["HE"].attrs["image_num"]

            # Get base indices for the split
            if split == "test":
                base_indices = np.where(self.train_test_split == "test")[0]
            elif patient_split is not None:
                all_patients = self.patient_nums
                if split == "train":
                    chosen_patients = patient_split["train_patients"]
                else:
                    chosen_patients = patient_split["val_patients"]

                base_indices = np.where(
                    (np.isin(all_patients, chosen_patients)) &
                    (self.train_test_split == "train")
                )[0]
            else:
                # fallback: use original dataset split
                base_indices = np.where(self.train_test_split == split)[0]

            # Build samples list from all requested image types
            for img_type in self.image_types:
                if self.tile_size is not None:
                    # With tiling: create tiles for each image type
                    for idx in base_indices:
                        img = np.array(f[f"HE/{img_type}"][idx])
                        H, W = pad_to_multiple_of_32(img).shape[:2]
                        step = self.tile_size - self.overlap
                        
                        # Store dataset indices for this image
                        image_key = (img_type, idx)
                        self.image_index_to_dataset_indices[image_key] = []
                        
                        for y in range(0, H, step):
                            for x in range(0, W, step):
                                y1 = min(y + self.tile_size, H)
                                x1 = min(x + self.tile_size, W)
                                y0 = y1 - self.tile_size
                                x0 = x1 - self.tile_size
                                
                                dataset_idx = len(self.samples)
                                self.samples.append((img_type, idx, (y0, y1, x0, x1), (H, W)))
                                self.image_index_to_dataset_indices[image_key].append(dataset_idx)
                else:
                    # Without tiling: add each image directly
                    for idx in base_indices:
                        img = np.array(f[f"HE/{img_type}"][idx])
                        H, W = pad_to_multiple_of_32(img).shape[:2]
                        
                        dataset_idx = len(self.samples)
                        self.samples.append((img_type, idx, None, (H, W)))
                        
                        # Store dataset index for this image
                        image_key = (img_type, idx)
                        self.image_index_to_dataset_indices[image_key] = [dataset_idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.tile_size is not None:
            img_type, img_idx, tile_coords, original_dims = self.samples[idx]
            y0, y1, x0, x1 = tile_coords
            H, W = original_dims
        else:
            img_type, img_idx, tile_coords, original_dims = self.samples[idx]
            H, W = original_dims

        
        with h5py.File(self.h5_path, 'r') as f:
            if tile_coords is not None:
                # With tiling
                y0, y1, x0, x1 = tile_coords
                img = np.array(f[f"HE/{img_type}"][img_idx])
                mask = np.array(f["GT/GT_majority_vote"][img_idx])
                
                # Pad first
                img = pad_to_multiple_of_32(img)
                mask = pad_to_multiple_of_32(mask)
                
                # Crop tile
                img = img[y0:y1, x0:x1]
                mask = mask[y0:y1, x0:x1]
            else:
                # Without tiling
                img = np.array(f[f"HE/{img_type}"][img_idx])
                mask = np.array(f["GT/GT_majority_vote"][img_idx])
                img = pad_to_multiple_of_32(img)
                mask = pad_to_multiple_of_32(mask)

            patient_num = int(f["HE"].attrs["patient_num"][img_idx])
            image_num = int(f["HE"].attrs["image_num"][img_idx])

        # Apply transform
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]
            
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # Auto-fix channel dimensions if enabled
        if self.auto_fix_channels:
            original_img_shape = img.shape
            original_mask_shape = mask.shape
            
            img = ensure_channels_first(img, expected_channels=3)
            mask = ensure_channels_first(mask, expected_channels=1)
            
            if self.verbose and (img.shape != original_img_shape or mask.shape != original_mask_shape):
                print(f"Fixed tensor shapes: image {original_img_shape} -> {img.shape}, "
                      f"mask {original_mask_shape} -> {mask.shape}")

        return {
            "image": img, 
            "mask": mask, 
            "patient_num": patient_num, 
            "image_num": image_num,
            "image_type": img_type,
            "tile_coords": tile_coords,
            "original_dims": (H, W)
        }

    def get_image_tiles(self, image_type: str, image_idx: int) -> List[Dict]:
        """
        Get all tiles corresponding to a specific image index and image type.
        
        Args:
            image_type (str): The image type ("raw", "sn1", or "sn2")
            image_idx (int): The index of the image in the original dataset
            
        Returns:
            List[Dict]: A list of dictionaries, each containing the same items as __getitem__
                       for each tile belonging to the specified image
        """
        image_key = (image_type, image_idx)
        
        if image_key not in self.image_index_to_dataset_indices:
            raise ValueError(f"No tiles found for image_type '{image_type}' and image_idx {image_idx}")
        
        dataset_indices = self.image_index_to_dataset_indices[image_key]
        tiles = []
        
        for dataset_idx in dataset_indices:
            tile_data = self[dataset_idx]  # Use existing __getitem__ method
            tiles.append(tile_data)
            
        return tiles

    def get_available_image_indices(self, image_type: str = None) -> List[Tuple[str, int]]:
        """
        Get all available image indices in the dataset.
        
        Args:
            image_type (str, optional): Filter by specific image type. If None, returns all.
            
        Returns:
            List[Tuple[str, int]]: List of (image_type, image_idx) tuples
        """
        if image_type is None:
            return list(self.image_index_to_dataset_indices.keys())
        else:
            return [(img_type, idx) for img_type, idx in self.image_index_to_dataset_indices.keys() 
                    if img_type == image_type]

    def get_dataset_stats(self):
        """Get statistics about the dataset composition."""
        stats = {}
        for img_type in self.image_types:
            type_count = sum(1 for sample in self.samples if sample[0] == img_type)
            stats[img_type] = type_count
        stats["total"] = len(self.samples)
        
        # Add image-level statistics
        stats["unique_images"] = len(self.image_index_to_dataset_indices)
        stats["tiles_per_image"] = {}
        for image_key, indices in self.image_index_to_dataset_indices.items():
            img_type, img_idx = image_key
            if img_type not in stats["tiles_per_image"]:
                stats["tiles_per_image"][img_type] = []
            stats["tiles_per_image"][img_type].append(len(indices))
            
        return stats