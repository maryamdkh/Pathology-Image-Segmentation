import albumentations as A
from data.helper import pad_to_multiple_of_32

def get_train_transform():
    return A.Compose([
        A.Rotate(limit=20, p=0.5),
        A.Affine(scale=(0.8, 1.2), shear=(-10, 10), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Pad to nearest multiple of 32
        A.Lambda(name="pad_to_multiple_of_32",image=pad_to_multiple_of_32, mask=pad_to_multiple_of_32),

        # Normalize to ImageNet mean/std
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        A.pytorch.transforms.ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Lambda(name="pad_to_multiple_of_32",image=pad_to_multiple_of_32, mask=pad_to_multiple_of_32),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        A.pytorch.transforms.ToTensorV2(),
    ])
