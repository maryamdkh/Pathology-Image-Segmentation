import h5py
from typing import Tuple, Dict, Any, List
import numpy as np
import math
import cv2
from sklearn.model_selection import train_test_split
from data.datasets import CoCaHisDataset
from data.transforms import get_train_transform, get_val_transform
from torch.utils.data import DataLoader

def create_dataloaders(config,splits):
    ds_cfg = config["dataset"]
    patient_split = build_Cocahis_patient_split(
        h5_path=ds_cfg['path'],
        val_fraction=ds_cfg["val_fraction"],
        seed=config["training"]["seed"]
    )
    transform_mapping = {
        "train": get_train_transform(),
        "val": get_val_transform(),
        "test": None
    }
    train_loaders = {}
    for split in splits:
        dataset = CoCaHisDataset(
            h5_path= ds_cfg['path'],
            image_type= ds_cfg["image_type"],
            split=split,
            patient_split=None if split == "test" else patient_split,
            tile_size=ds_cfg["tile_size"],
            overlap=ds_cfg["overlap"],
            transform= transform_mapping[split]
        )
        print(f"Number of samples in {split} dataset: {len(dataset)}")

        train_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
        train_loaders[split] = train_loader

    return patient_split,train_loaders

def build_Cocahis_patient_split(h5_path: str, val_fraction: float = 0.2, seed: int = 42):
    with h5py.File(h5_path, "r") as f:
        all_splits = f["HE"].attrs["train_test_split"]
        all_patients = f["HE"].attrs["patient_num"]

    # Consider only dataset-internal "train" entries for building train/val split
    train_mask = (all_splits == "train")
    patients_in_train = all_patients[train_mask]
    unique_patients = np.unique(patients_in_train)

    train_patients, val_patients = train_test_split(unique_patients,
                                                    test_size=val_fraction,
                                                    random_state=seed,
                                                    shuffle=True)

    return {"train_patients": train_patients, "val_patients": val_patients}

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

def load_cocahis_dataset(data_path: str) -> Dict[str, Any]:
    """
    Load CoCaHis dataset from HDF5 file with structured data extraction.
    
    Args:
        data_path: Path to the HDF5 file
        
    Returns:
        Dictionary containing all dataset components with structured organization
    """

    def _load_individual_annotations(f: h5py.File) -> List[np.ndarray]:
        """
        Load individual annotator ground truth data.
        
        Args:
            f: Open HDF5 file object
            
        Returns:
            List of ground truth arrays from all annotators
        """
        individual_gts = []
        
        for i in range(1, 8):  # Annotators 1-7
            dataset_key = f"GT/GT{str(i)}"
            if dataset_key in f:
                individual_gts.append(f[dataset_key][()])
            else:
                raise KeyError(f"Annotator dataset not found: {dataset_key}")
        
        return individual_gts


    def _compute_dataset_statistics(dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute basic statistics about the loaded dataset.
        
        Args:
            dataset: Loaded dataset dictionary
            
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'num_images': len(dataset['raw_images']),
            'image_shape': dataset['raw_images'][0].shape,
            'num_annotators': len(dataset['individual_gts']),
            'train_test_split_shape': dataset['train_test_split'].shape,
            'unique_patients': len(np.unique(dataset['patient_numbers'])),
            'data_type': dataset['raw_images'].dtype
        }
        
        return stats
    
    dataset = {}
    
    try:
        with h5py.File(data_path, 'r') as f:
            # Load main image datasets
            dataset['raw_images'] = f["HE/raw"][()]
            dataset['sn1_images'] = f["HE/sn1"][()]
            dataset['sn2_images'] = f["HE/sn2"][()]
            dataset['gt_majority_vote'] = f["GT/GT_majority_vote"][()]
            
            # Load metadata attributes
            dataset['train_test_split'] = f["HE/"].attrs["train_test_split"]
            dataset['patient_numbers'] = f["HE/"].attrs["patient_num"]
            dataset['image_numbers'] = f["HE/"].attrs["image_num"]
            
            # Load individual annotator ground truths
            dataset['individual_gts'] = _load_individual_annotations(f)
            
            # Add dataset statistics
            dataset['stats'] = _compute_dataset_statistics(dataset)
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")
    except KeyError as e:
        raise KeyError(f"Required dataset field missing: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")
    
    return dataset

def print_Cocahis_summary(dataset: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the loaded dataset.
    
    Args:
        dataset: Loaded dataset dictionary
    """
    stats = dataset['stats']
    
    print("📊 CoCaHis Dataset Summary")
    print("=" * 40)
    print(f"Total images: {stats['num_images']}")
    print(f"Image shape: {stats['image_shape']}")
    print(f"Number of annotators: {stats['num_annotators']}")
    print(f"Unique patients: {stats['unique_patients']}")
    print(f"Data type: {stats['data_type']}")
    print(f"Train/test split shape: {stats['train_test_split_shape']}")
    print("=" * 40)


