import io
import json
import math
import os
from dataclasses import dataclass

import boto3
import numpy as np
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

CHIP_SIZE = 512
OVERLAP = 64
STRIDE = CHIP_SIZE - OVERLAP

WGS84 = CRS.from_epsg(4326)
MAPBOX_ZOOM = 18


@dataclass
class BoundingBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


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


def _deg2tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    lat_r = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return x, y


def _fetch_mapbox_tile(x: int, y: int, zoom: int, token: str) -> bytes:
    url = f"https://api.mapbox.com/v4/mapbox.satellite/{zoom}/{x}/{y}@2x.jpg90?access_token={token}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def _download_tiles(bbox: BoundingBox, zoom: int, token: str) -> np.ndarray:
    x_min, y_min = _deg2tile(bbox.max_lat, bbox.min_lon, zoom)
    x_max, y_max = _deg2tile(bbox.min_lat, bbox.max_lon, zoom)

    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    tile_px = 512
    mosaic = np.zeros((rows * tile_px, cols * tile_px, 3), dtype=np.uint8)

    for row_i, ty in enumerate(range(y_min, y_max + 1)):
        for col_i, tx in enumerate(range(x_min, x_max + 1)):
            raw = _fetch_mapbox_tile(tx, ty, zoom, token)
            from PIL import Image
            img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
            r = row_i * tile_px
            c = col_i * tile_px
            mosaic[r:r + tile_px, c:c + tile_px] = img

    return mosaic


def _reproject_to_wgs84(image: np.ndarray, src_crs: CRS, src_bbox: BoundingBox) -> tuple[np.ndarray, object]:
    h, w, _ = image.shape
    src_transform = from_bounds(
        src_bbox.min_lon, src_bbox.min_lat, src_bbox.max_lon, src_bbox.max_lat, w, h
    )
    dst_transform, dst_w, dst_h = calculate_default_transform(src_crs, WGS84, w, h,
        left=src_bbox.min_lon, bottom=src_bbox.min_lat,
        right=src_bbox.max_lon, top=src_bbox.max_lat)

    dst = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    for band in range(3):
        reproject(
            source=image[:, :, band],
            destination=dst[:, :, band],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=WGS84,
            resampling=Resampling.bilinear,
        )
    return dst, dst_transform


def _chip_image(image: np.ndarray, transform, course_id: str) -> list[dict]:
    h, w, _ = image.shape
    chips = []
    chip_idx = 0

    for y in range(0, max(1, h - CHIP_SIZE + 1), STRIDE):
        for x in range(0, max(1, w - CHIP_SIZE + 1), STRIDE):
            y_end = min(y + CHIP_SIZE, h)
            x_end = min(x + CHIP_SIZE, w)
            chip = image[y:y_end, x:x_end]

            if chip.shape[0] < CHIP_SIZE or chip.shape[1] < CHIP_SIZE:
                pad = np.zeros((CHIP_SIZE, CHIP_SIZE, 3), dtype=np.uint8)
                pad[:chip.shape[0], :chip.shape[1]] = chip
                chip = pad

            chips.append({
                "idx": chip_idx,
                "data": chip,
                "origin_x": x,
                "origin_y": y,
            })
            chip_idx += 1

    return chips


def _upload_chips(chips: list[dict], course_id: str, bucket: str) -> str:
    s3 = _s3_client()
    prefix = f"checkpoints/{course_id}/chips"

    metadata = []
    for chip in chips:
        key = f"{prefix}/chip_{chip['idx']:04d}.npy"
        buf = io.BytesIO()
        np.save(buf, chip["data"])
        buf.seek(0)
        s3.upload_fileobj(buf, bucket, key)
        metadata.append({
            "idx": chip["idx"],
            "key": key,
            "origin_x": chip["origin_x"],
            "origin_y": chip["origin_y"],
        })

    meta_key = f"{prefix}/metadata.json"
    s3.put_object(Bucket=bucket, Key=meta_key, Body=json.dumps(metadata).encode())
    return prefix


async def preprocess_course(course_id: str, force: bool = False) -> str:
    bucket = os.environ["S3_CHECKPOINT_BUCKET"]
    prefix = f"checkpoints/{course_id}/chips"
    meta_key = f"{prefix}/metadata.json"

    if not force and _checkpoint_exists(bucket, meta_key):
        return prefix

    token = os.environ["MAPBOX_TOKEN"]

    # Fetch bounding box from DB (injected via env or passed via job payload in prod)
    # For now, read from S3 job manifest written by the API layer
    s3 = _s3_client()
    manifest = json.loads(s3.get_object(Bucket=bucket, Key=f"jobs/{course_id}/manifest.json")["Body"].read())
    bbox = BoundingBox(**manifest["bbox"])

    image = _download_tiles(bbox, MAPBOX_ZOOM, token)
    image, transform = _reproject_to_wgs84(image, WGS84, bbox)
    chips = _chip_image(image, transform, course_id)
    prefix = _upload_chips(chips, course_id, bucket)
    return prefix
