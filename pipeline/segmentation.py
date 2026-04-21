import io
import json
import os
import tempfile

import boto3
import numpy as np
import rasterio
import torch
from rasterio.transform import from_bounds
from torchvision.models.segmentation import deeplabv3_resnet50

NUM_CLASSES = 6
CHIP_SIZE = 512


def _s3_client():
    return boto3.client(
        "s3",
        region_name=os.environ["AWS_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def _checkpoint_exists(bucket: str, key: str) -> bool:
    s3 = _s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def _load_model(device: torch.device) -> torch.nn.Module:
    bucket = os.environ["S3_CHECKPOINT_BUCKET"]
    key = os.environ["MODEL_CHECKPOINT_KEY"]
    s3 = _s3_client()

    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        s3.download_file(bucket, key, tmp.name)
        model = deeplabv3_resnet50(num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(tmp.name, map_location=device))

    model.eval()
    return model.to(device)


def _run_inference(model: torch.nn.Module, chip: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(chip.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    with torch.no_grad():
        out = model(tensor)["out"]  # (1, C, H, W)
    return out.softmax(dim=1).squeeze(0).cpu().numpy()  # (C, H, W)


def _infer_with_oom_retry(model, chip, device) -> np.ndarray:
    try:
        return _run_inference(model, chip, device)
    except torch.cuda.OutOfMemoryError:
        # Halve effective batch by splitting chip in half and averaging
        torch.cuda.empty_cache()
        h = chip.shape[0] // 2
        top = np.pad(chip[:h], ((0, h), (0, 0), (0, 0)))
        bot = np.pad(chip[h:], ((h, 0), (0, 0), (0, 0)))
        top_probs = _run_inference(model, top, device)
        bot_probs = _run_inference(model, bot, device)
        return (top_probs + bot_probs) / 2


async def segment_course(course_id: str, chips_prefix: str, force: bool = False) -> str:
    bucket = os.environ["S3_CHECKPOINT_BUCKET"]
    mask_key = f"checkpoints/{course_id}/mask.tif"

    if not force and _checkpoint_exists(bucket, mask_key):
        return mask_key

    s3 = _s3_client()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device)

    # Load chip metadata
    meta_obj = s3.get_object(Bucket=bucket, Key=f"{chips_prefix}/metadata.json")
    metadata = json.loads(meta_obj["Body"].read())

    # Determine full image size from chip origins + CHIP_SIZE
    max_y = max(m["origin_y"] for m in metadata) + CHIP_SIZE
    max_x = max(m["origin_x"] for m in metadata) + CHIP_SIZE
    prob_map = np.zeros((NUM_CLASSES, max_y, max_x), dtype=np.float32)
    count_map = np.zeros((max_y, max_x), dtype=np.float32)

    for meta in metadata:
        key = meta["key"]
        buf = io.BytesIO()
        s3.download_fileobj(bucket, key, buf)
        buf.seek(0)
        chip = np.load(buf)  # (H, W, 3) uint8

        probs = _infer_with_oom_retry(model, chip, device)  # (C, H, W)

        y, x = meta["origin_y"], meta["origin_x"]
        prob_map[:, y:y + CHIP_SIZE, x:x + CHIP_SIZE] += probs
        count_map[y:y + CHIP_SIZE, x:x + CHIP_SIZE] += 1

    count_map = np.maximum(count_map, 1)
    prob_map /= count_map[np.newaxis, :, :]
    class_mask = prob_map.argmax(axis=0).astype(np.uint8)  # (H, W)

    # Write as single-band GeoTIFF and upload to S3
    manifest_obj = s3.get_object(Bucket=bucket, Key=f"jobs/{course_id}/manifest.json")
    manifest = json.loads(manifest_obj["Body"].read())
    bbox = manifest["bbox"]

    transform = from_bounds(
        bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"],
        max_x, max_y,
    )
    buf = io.BytesIO()
    with rasterio.open(buf, "w", driver="GTiff", height=max_y, width=max_x,
                       count=1, dtype=np.uint8, crs="EPSG:4326", transform=transform) as dst:
        dst.write(class_mask, 1)
    buf.seek(0)
    s3.upload_fileobj(buf, bucket, mask_key)
    return mask_key
