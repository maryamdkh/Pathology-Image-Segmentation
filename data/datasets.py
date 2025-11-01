import numpy as np
from typing import Optional, Dict
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

class CoCaHisDataset(Dataset):

    def __init__(self, h5_path: str, image_type: str="raw", split: str="train",
                 transform=None, patient_split: Optional[Dict[str, np.ndarray]]=None,
                 tile_size: int=None, overlap: int=0):
        """
          PyTorch custom dataset for CoCaHis HDF5 structure.

          Args:
              h5_path (str): Path to CoCaHis.hdf5
              image_type (str): One of ["raw", "sn1", "sn2"]
              split (str): "train" or "test"
              transform (callable, optional): Optional transform to be applied on images
              patient_split (dict, optional): Dict with keys 'train_patients' and 'val_patients'.
              tile_size (int): size of the tiles if tiling is requested
              overlap (int): overlap between tiles is tiling is requested


          Returns a dict with keys:
              image: Tensor float32 [C,H,W] in [0,1]
              mask:  Tensor long or float [H,W] (0/1)
              patient_num: int
              image_num: int
        """

        assert image_type in ["raw", "sn1", "sn2"]
        assert split in ["train", "val", "test"]

        self.h5_path = h5_path
        self.image_type = image_type
        self.split = split
        self.transform = transform
        self.tile_size = tile_size  # e.g., 512
        self.overlap = overlap      # e.g., 64 pixels

        with h5py.File(self.h5_path, 'r') as f:
            self.train_test_split = f["HE"].attrs["train_test_split"]
            self.patient_nums = f["HE"].attrs["patient_num"]
            self.image_nums = f["HE"].attrs["image_num"]

            if split == "test":
                self.indices = np.where(self.train_test_split == "test")[0]
            elif patient_split is not None:
                all_patients = self.patient_nums
                if split == "train":
                    chosen_patients = patient_split["train_patients"]
                else:
                    chosen_patients = patient_split["val_patients"]

                self.indices = np.where(
                    (np.isin(all_patients, chosen_patients)) &
                    (self.train_test_split == "train")
                )[0]
            else:
                # fallback: use original dataset split
                self.indices = np.where(self.train_test_split == split)[0]

        # ---- Precompute tile indices if tiling is used ----
        if self.tile_size is not None:
            self.tile_mapping = []  # list of tuples: (image_idx, y0, y1, x0, x1)
            with h5py.File(self.h5_path, 'r') as f:
                for idx in self.indices:
                    img = np.array(f[f"HE/{self.image_type}"][idx])
                    H, W = pad_to_multiple_of_32(img).shape[:2]
                    step = self.tile_size - self.overlap
                    for y in range(0, H, step):
                        for x in range(0, W, step):
                            y1 = min(y + self.tile_size, H)
                            x1 = min(x + self.tile_size, W)
                            y0 = y1 - self.tile_size
                            x0 = x1 - self.tile_size
                            self.tile_mapping.append((idx, y0, y1, x0, x1))

    def __len__(self):
        if self.tile_size is not None:
            return len(self.tile_mapping)
        return len(self.indices)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            if self.tile_size is not None:
                img_idx, y0, y1, x0, x1 = self.tile_mapping[idx]
                img = np.array(f[f"HE/{self.image_type}"][img_idx])
                mask = np.array(f["GT/GT_majority_vote"][img_idx])
                patient_num = int(f["HE"].attrs["patient_num"][img_idx])
                image_num = int(f["HE"].attrs["image_num"][img_idx])

                # Pad first
                img = pad_to_multiple_of_32(img)
                mask = pad_to_multiple_of_32(mask)

                # Crop tile
                img = img[y0:y1, x0:x1]
                mask = mask[y0:y1, x0:x1]

            else:
                img_idx = self.indices[idx]
                img = np.array(f[f"HE/{self.image_type}"][img_idx])
                mask = np.array(f["GT/GT_majority_vote"][img_idx])
                patient_num = int(f["HE"].attrs["patient_num"][img_idx])
                image_num = int(f["HE"].attrs["image_num"][img_idx])
                img = pad_to_multiple_of_32(img)
                mask = pad_to_multiple_of_32(mask)

        # Apply transform
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]
        else:
            img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0

        return {"image": img, "mask": mask, "patient_num": patient_num, "image_num": image_num}
