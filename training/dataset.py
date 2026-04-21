import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CHIP_SIZE = 512

_COURSE_RE = re.compile(r'^(.+?)_\d+_')


def _course_id(stem: str) -> str:
    m = _COURSE_RE.match(stem)
    return m.group(1) if m else stem


class DanishOrthophotosDataset(Dataset):
    """
    Loads pre-chipped data produced by training/prepare_data.py.

    out_dir/
      images/  <stem>.jpg     RGB JPEG chips
      masks/   <stem>.png     uint8 single-channel class ID masks (0–5)

    Split is done per course before instantiation — pass the chip paths
    that belong to this split.
    """

    def __init__(self, chip_paths: list[Path], transform=None):
        self.chip_paths = chip_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.chip_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path  = self.chip_paths[idx]
        mask_path = img_path.parent.parent / "masks" / img_path.with_suffix(".png").name

        image      = np.array(Image.open(img_path).convert("RGB"))    # (H, W, 3) uint8
        class_mask = np.array(Image.open(mask_path))                   # (H, W) uint8 class IDs

        if self.transform is not None:
            augmented  = self.transform(image=image, mask=class_mask)
            image      = augmented["image"]
            class_mask = augmented["mask"]

        image      = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        class_mask = torch.from_numpy(class_mask.astype(np.int64)).long()
        return image, class_mask


def build_splits(chipped_dir: str, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Split chip paths by course to avoid tile-level data leakage.
    Returns (train_paths, val_paths, test_paths).

    Chip filenames are <img_stem>_<chip_idx:04d>.jpg where img_stem encodes
    the course, e.g. Blokhus_1000_03_0012.jpg → course = Blokhus.
    """
    images_dir = Path(chipped_dir) / "images"
    courses: dict[str, list[Path]] = {}

    for jpg in sorted(images_dir.glob("*.jpg")):
        # img_stem is the original image stem before chip index was appended
        # e.g. "Blokhus_1000_03_0012" — strip the last _NNNN suffix
        img_stem = "_".join(jpg.stem.rsplit("_", 1)[:-1])
        cid = _course_id(img_stem)
        courses.setdefault(cid, []).append(jpg)

    course_ids = sorted(courses.keys())
    n       = len(course_ids)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_ids = course_ids[:n_train]
    val_ids   = course_ids[n_train : n_train + n_val]
    test_ids  = course_ids[n_train + n_val :]

    def gather(ids):
        paths = []
        for cid in ids:
            paths.extend(courses[cid])
        return paths

    return gather(train_ids), gather(val_ids), gather(test_ids)
