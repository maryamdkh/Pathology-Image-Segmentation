import albumentations as A

def get_cocahis_paper_augmentation():

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
                scale=(0.85, 1.15),
                rotate=(-20, 20),
                shear=(-8, 8),
                p=0.5
            ),
      

    ])

def get_medium_augmentation():
    """
    Balanced augmentation for pathology images
    Conservative approach that preserves diagnostic color information
    """
    return A.Compose([
        # Spatial transformations 
        A.OneOf([
            A.Rotate(limit=20, p=0.7),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.05, 0.05),
                rotate=(-20, 20),
                shear=(-8, 8),
                p=0.5
            ),
        ], p=0.8),
        
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # Mild elastic deformations (realistic tissue variations)
        A.ElasticTransform(
            alpha=25,        # Conservative
            sigma=5,
            p=0.3
        ),
        
        # VERY CONSERVATIVE color variations
        A.OneOf([
            # Mild brightness/contrast (different microscope settings)
            A.RandomBrightnessContrast(
                brightness_limit=0.08,    # Very small
                contrast_limit=0.08,      # Very small
                p=0.4
            ),
            
            # Tiny gamma adjustments
            A.RandomGamma(
                gamma_limit=(92, 108),    # Minimal gamma
                p=0.3
            ),
            
            # Minimal color shifts (stain variations)
            A.HueSaturationValue(
                hue_shift_limit=3,        # Almost imperceptible hue
                sat_shift_limit=10,       # Mild saturation
                val_shift_limit=8,        # Small value changes
                p=0.3
            ),
            
            # No color changes
            A.NoOp(p=0.4)
        ], p=0.6),  # Only 60% chance of any color augmentation
        
        # Non-color destructive augmentations
        A.OneOf([
            A.GaussNoise(std_range=(0.1, 0.2), p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.MotionBlur(blur_limit=(3, 5), p=0.2),
        ], p=0.4),
        
        # Safe mask-based augmentations
        A.CoarseDropout(
            num_holes_range=(2, 4),
            hole_height_range=(10, 20),
            hole_width_range=(10, 20),
            fill="random_uniform",
            p=0.4
        ),
        
        # Normalization
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
        
        A.pytorch.transforms.ToTensorV2(),
    ])

def get_strong_augmentation():
    """
    Strong augmentations for pathology images
    Aggressive spatial transforms but still color-conservative
    Use when you need heavy regularization and have enough data
    """
    return A.Compose([
        # AGGRESSIVE spatial transformations (color-safe)
        A.OneOf([
            A.Rotate(limit=35, p=0.8),
            A.Affine(
                scale=(0.7, 1.3),
                translate_percent=(-0.1, 0.1),
                rotate=(-35, 35),
                shear=(-15, 15),
                p=0.6
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                p=0.5
            ),
        ], p=0.9),
        
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.3),  # 90 degree rotation
        
        # Strong elastic deformations
        A.OneOf([
            A.ElasticTransform(
                alpha=50,
                sigma=7,
                p=0.4
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=0.3
            ),
            A.OpticalDistortion(
                distort_limit=0.2,
                shift_limit=0.1,
                p=0.3
            ),
        ], p=0.6),
        
        # CONSERVATIVE color variations (same as medium - don't increase!)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.4
            ),
            A.RandomGamma(gamma_limit=(90, 110), p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=12,
                val_shift_limit=10,
                p=0.3
            ),
            A.NoOp(p=0.3)
        ], p=0.7),
        
        # More aggressive non-color augmentations
        A.OneOf([
            A.GaussNoise(std_range=(0.1, 0.2), p=0.4), # 10-20% of max value
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
            A.MedianBlur(blur_limit=3, p=0.2),
            A.ISONoise(
                color_shift=(0.01, 0.03),
                intensity=(0.1, 0.3),
                p=0.2
            ),
        ], p=0.7),
        
        # Strong mask-based augmentations
        A.OneOf([
            A.CoarseDropout(
                num_holes_range=(3, 6),
                hole_height_range=(10, 20),
                hole_width_range=(10, 20),
                fill="random_uniform",
                p=0.8
            ),
            A.RandomGridShuffle(grid=(3, 3), p=0.2),
            A.RandomSizedBBoxSafeCrop(
                height=512, 
                width=512, 
                p=0.2
            ),
        ], p=0.5),
        
        # # Cutout/Mixup style augmentations (advanced)
        # A.Cutout(
        #     num_holes=8,
        #     max_h_size=16,
        #     max_w_size=16,
        #     fill_value=0,
        #     p=0.3
        # ),
        
        # Normalization
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
        
        A.pytorch.transforms.ToTensorV2(),
    ])

def get_val_test_transform():
    """
    Minimal transforms for validation/testing - preserves all original information
    No augmentations, just basic preprocessing
    """
    return A.Compose([
        # Ensure consistent input size (adjust dimensions as needed)
        # A.Resize(height=512, width=512, p=1.0),
        
        # Normalization only - no color/geometric changes
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
        
        A.pytorch.transforms.ToTensorV2(),
    ])
