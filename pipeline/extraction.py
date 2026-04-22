import io
import json
import os
import tempfile
import uuid

import boto3
import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from shapely.geometry import mapping, shape
from shapely.ops import transform

CLASS_MAP = {
    1: "green",
    2: "fairway",
    3: "tee_box",
    4: "bunker",
    5: "water_hazard",
}

SIMPLIFY_TOLERANCE_M = 1.5
MIN_AREA_SQM = 20.0

_to_metric = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform


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


def _vectorize_class(
    mask_array: np.ndarray,
    geo_transform: tuple,
    projection: str,
    class_id: int,
) -> list:
    """
    Return a list of Shapely geometries for all connected regions of class_id.
    Uses GDAL Polygonize with the binary mask as both source and validity mask,
    so only foreground pixels (value == class_id) produce output polygons.
    """
    binary = (mask_array == class_id).astype(np.uint8)
    h, w = binary.shape

    mem_driver = gdal.GetDriverByName("MEM")
    mem_ds = mem_driver.Create("", w, h, 1, gdal.GDT_Byte)
    mem_ds.SetGeoTransform(geo_transform)
    mem_ds.SetProjection(projection)
    src_band = mem_ds.GetRasterBand(1)
    src_band.WriteArray(binary)
    src_band.FlushCache()

    ogr_mem = ogr.GetDriverByName("Memory")
    out_ds = ogr_mem.CreateDataSource("out")
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    out_layer = out_ds.CreateLayer("polys", srs=srs)
    out_layer.CreateField(ogr.FieldDefn("val", ogr.OFTInteger))

    # Passing src_band as the mask band excludes all pixels where binary == 0
    gdal.Polygonize(src_band, src_band, out_layer, 0, [], callback=None)

    geometries = []
    for feat in out_layer:
        geom_ref = feat.GetGeometryRef()
        if geom_ref is None:
            continue
        geom = shapely_wkt.loads(geom_ref.ExportToWkt())
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_valid and not geom.is_empty:
            geometries.append(geom)

    out_ds = None
    mem_ds = None
    return geometries


def _simplify_and_filter(geom) -> "shape | None":
    """
    Project WGS84 → EPSG:3857, apply Douglas-Peucker at 1.5 m, filter < 20 m²,
    then reproject back to WGS84. Returns None if the polygon is too small.
    """
    geom_m = transform(_to_metric, geom)
    geom_m = geom_m.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)
    if geom_m.area < MIN_AREA_SQM:
        return None
    return transform(_to_wgs84, geom_m)


async def extract_polygons(course_id: str, mask_key: str, force: bool = False) -> str:
    bucket = os.environ["S3_CHECKPOINT_BUCKET"]
    out_key = f"checkpoints/{course_id}/polygons_raw.geojson"

    if not force and _checkpoint_exists(bucket, out_key):
        return out_key

    s3 = _s3_client()
    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    try:
        s3.download_file(bucket, mask_key, tmp.name)
        tmp.close()

        ds = gdal.Open(tmp.name)
        band = ds.GetRasterBand(1)
        mask_array = band.ReadAsArray()
        geo_transform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        ds = None

        features = []
        for class_id, feature_type in CLASS_MAP.items():
            for geom in _vectorize_class(mask_array, geo_transform, projection, class_id):
                simplified = _simplify_and_filter(geom)
                if simplified is None:
                    continue
                features.append({
                    "type": "Feature",
                    "id": str(uuid.uuid4()),
                    "geometry": mapping(simplified),
                    "properties": {
                        "feature_type": feature_type,
                        "class_id": class_id,
                    },
                })
    finally:
        os.unlink(tmp.name)

    body = json.dumps({"type": "FeatureCollection", "features": features}).encode()
    s3.put_object(Bucket=bucket, Key=out_key, Body=body)
    return out_key
