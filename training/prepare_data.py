"""
Converts the DatasetNinja-packaged Danish Golf Courses Orthophotos dataset
into 512×512 chips for training.

Dataset layout (as shipped):
  danish-golf-courses-orthophotos-DatasetNinja/
    ds/
      img/   *.jpg                    RGB JPEG images
      ann/   *.jpg.json               DatasetNinja annotation files

Each annotation JSON contains:
  - size: { height, width }
  - objects[]: { classTitle, geometryType: "bitmap",
                 bitmap: { data: <base64(zlib(PNG))>, origin: [x, y] } }

Output layout:
  <out_dir>/
    images/  <course_id>_<chip_idx:04d>.jpg
    masks/   <course_id>_<chip_idx:04d>.png   (uint8, values = class IDs 0–5)

Usage:
  python -m training.prepare_data \\
    --dataset-dir danish-golf-courses-orthophotos-DatasetNinja \\
    --out-dir data/chipped
"""

import argparse
import base64
import io
import json
import re
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

CHIP_SIZE = 512
OVERLAP   = 64
STRIDE    = CHIP_SIZE - OVERLAP  # 448

CLASS_TITLE_TO_ID = {
    "fairway": 2,
    "green":   1,
    "tee":     3,  # dataset uses "tee"; platform schema uses "tee_box"
    "bunker":  4,
    "water":   5,  # dataset uses "water"; platform schema uses "water_hazard"
}

_COURSE_RE = re.compile(r'^(.+?)_\d+_')


def _course_id(stem: str) -> str:
    """Extract course ID from image stem, e.g. 'Blokhus_1000_03' → 'Blokhus'."""
    m = _COURSE_RE.match(stem)
    return m.group(1) if m else stem


def _decode_bitmap(data: str) -> np.ndarray:
    """
    Decode DatasetNinja bitmap.data into a 2D binary uint8 array.
    Format: base64(zlib(PNG)) where PNG pixels are 0 (background) or >0 (annotated).
    """
    raw = base64.b64decode(data)
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass  # already raw PNG bytes (defensive fallback)
    img = Image.open(io.BytesIO(raw))
    arr = np.array(img)
    return (arr > 0).astype(np.uint8)


def _build_class_mask(ann: dict) -> np.ndarray:
    """Build an H×W uint8 array of class IDs from a DatasetNinja annotation dict."""
    h = ann["size"]["height"]
    w = ann["size"]["width"]
    mask = np.zeros((h, w), dtype=np.uint8)

    for obj in ann.get("objects", []):
        class_title = obj.get("classTitle", "")
        class_id = CLASS_TITLE_TO_ID.get(class_title)
        if class_id is None:
            continue  # unknown class — skip

        if obj.get("geometryType") != "bitmap":
            continue  # only bitmap geometry supported

        bitmap_arr = _decode_bitmap(obj["bitmap"]["data"])
        ox, oy = obj["bitmap"]["origin"]  # origin is [x, y]

        bh, bw = bitmap_arr.shape
        # Clamp to image bounds
        y1, y2 = oy, min(oy + bh, h)
        x1, x2 = ox, min(ox + bw, w)
        by2, bx2 = y2 - oy, x2 - ox

        mask[y1:y2, x1:x2][bitmap_arr[:by2, :bx2] > 0] = class_id

    return mask


def _chip_and_save(
    image: np.ndarray,
    class_mask: np.ndarray,
    course_id: str,
    images_dir: Path,
    masks_dir: Path,
    skip_background: bool,
) -> int:
    h, w = class_mask.shape
    saved = 0
    chip_idx = 0

    y_starts = list(range(0, max(1, h - CHIP_SIZE + 1), STRIDE))
    x_starts = list(range(0, max(1, w - CHIP_SIZE + 1), STRIDE))

    if not y_starts or y_starts[-1] + CHIP_SIZE < h:
        y_starts.append(max(0, h - CHIP_SIZE))
    if not x_starts or x_starts[-1] + CHIP_SIZE < w:
        x_starts.append(max(0, w - CHIP_SIZE))

    # Deduplicate while preserving order
    y_starts = list(dict.fromkeys(y_starts))
    x_starts = list(dict.fromkeys(x_starts))

    for y in y_starts:
        for x in x_starts:
            img_chip  = np.zeros((CHIP_SIZE, CHIP_SIZE, 3), dtype=np.uint8)
            mask_chip = np.zeros((CHIP_SIZE, CHIP_SIZE),    dtype=np.uint8)

            y2 = min(y + CHIP_SIZE, h)
            x2 = min(x + CHIP_SIZE, w)
            ph, pw = y2 - y, x2 - x

            img_chip[:ph,  :pw]  = image[y:y2, x:x2]
            mask_chip[:ph, :pw]  = class_mask[y:y2, x:x2]

            if skip_background and mask_chip.max() == 0:
                chip_idx += 1
                continue

            stem = f"{course_id}_{chip_idx:04d}"
            Image.fromarray(img_chip).save(images_dir / f"{stem}.jpg", quality=95)
            Image.fromarray(mask_chip, mode="L").save(masks_dir / f"{stem}.png")

            saved    += 1
            chip_idx += 1

    return saved


def prepare(dataset_dir: str, out_dir: str, skip_background: bool):
    ds_path     = Path(dataset_dir) / "ds"
    ann_dir     = ds_path / "ann"
    img_dir     = ds_path / "img"
    images_out  = Path(out_dir) / "images"
    masks_out   = Path(out_dir) / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.jpg.json"))
    if not ann_files:
        raise ValueError(f"No annotation files found in {ann_dir}")

    total_chips = 0
    for ann_path in ann_files:
        img_name   = ann_path.stem          # e.g. "Blokhus_1000_03.jpg"
        img_stem   = Path(img_name).stem    # e.g. "Blokhus_1000_03"
        img_path   = img_dir / img_name
        course_id  = _course_id(img_stem)

        if not img_path.exists():
            print(f"  [skip] {img_name} — image not found")
            continue

        with open(ann_path) as f:
            ann = json.load(f)

        image      = np.array(Image.open(img_path).convert("RGB"))
        class_mask = _build_class_mask(ann)

        n = _chip_and_save(image, class_mask, img_stem,
                           images_out, masks_out, skip_background)
        total_chips += n

    print(f"Done. {total_chips} chips written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True,
                        help="Path to danish-golf-courses-orthophotos-DatasetNinja/")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for chipped images and masks")
    parser.add_argument("--keep-background", action="store_true",
                        help="Keep chips with no annotated pixels (default: skip)")
    args = parser.parse_args()
    prepare(args.dataset_dir, args.out_dir, skip_background=not args.keep_background)
