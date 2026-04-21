import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomResizedCrop(height=512, width=512, scale=(0.75, 1.25), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=0, val_shift_limit=0, p=0.3),
    ])


def get_val_transforms() -> A.Compose:
    return A.Compose([])
