import json
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from shapely.geometry import Polygon, shape

# Skip this entire module when GDAL (osgeo) is not installed.
# In CI the Docker image provides GDAL; on macOS dev boxes it may be absent.
pytest.importorskip("osgeo", reason="GDAL (osgeo) not installed")

from pipeline.extraction import (
    CLASS_MAP,
    extract_polygons,
    MIN_AREA_SQM,
    SIMPLIFY_TOLERANCE_M,
    _simplify_and_filter,
    _vectorize_class,
)

# A minimal WGS84 geo_transform covering a small area near Copenhagen.
# Format: (x_min, pixel_width, 0, y_max, 0, -pixel_height)
# Pixel width/height in degrees — ~1 m per pixel at 55°N latitude.
_GEO_TRANSFORM = (12.0, 9e-6, 0.0, 55.01, 0.0, -9e-6)
_PROJECTION_WGS84 = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)


def _make_mask(height: int, width: int, class_id: int, fill_rect=None) -> np.ndarray:
    """Return a uint8 mask of given size. fill_rect=(r0, r1, c0, c1) sets a rectangle to class_id."""
    arr = np.zeros((height, width), dtype=np.uint8)
    if fill_rect:
        r0, r1, c0, c1 = fill_rect
        arr[r0:r1, c0:c1] = class_id
    return arr


class TestVectorizeClass:
    def test_empty_mask_returns_no_geometries(self):
        mask = _make_mask(50, 50, 1)  # all zeros
        result = _vectorize_class(mask, _GEO_TRANSFORM, _PROJECTION_WGS84, class_id=1)
        assert result == []

    def test_single_filled_block_returns_one_polygon(self):
        mask = _make_mask(50, 50, 1, fill_rect=(10, 40, 10, 40))
        result = _vectorize_class(mask, _GEO_TRANSFORM, _PROJECTION_WGS84, class_id=1)
        assert len(result) == 1
        poly = result[0]
        assert poly.geom_type in ("Polygon", "MultiPolygon")
        assert poly.area > 0

    def test_different_class_returns_nothing(self):
        # Mask has class 2 pixels; querying class 1 should return nothing
        mask = _make_mask(50, 50, 2, fill_rect=(10, 40, 10, 40))
        result = _vectorize_class(mask, _GEO_TRANSFORM, _PROJECTION_WGS84, class_id=1)
        assert result == []

    def test_two_separate_regions_return_two_polygons(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[5:20, 5:20] = 1    # top-left block
        mask[70:90, 70:90] = 1  # bottom-right block
        result = _vectorize_class(mask, _GEO_TRANSFORM, _PROJECTION_WGS84, class_id=1)
        assert len(result) == 2

    def test_all_classes_in_class_map_are_vectorizable(self):
        for class_id in CLASS_MAP:
            mask = _make_mask(30, 30, class_id, fill_rect=(5, 25, 5, 25))
            result = _vectorize_class(mask, _GEO_TRANSFORM, _PROJECTION_WGS84, class_id=class_id)
            assert len(result) >= 1, f"Class {class_id} produced no polygons"


class TestSimplifyAndFilter:
    def _large_wgs84_polygon(self):
        """A ~60 m × 60 m square near Copenhagen — well above 20 m² threshold."""
        # 0.0006 degrees ≈ 66 m longitude at 55°N; 0.0005 degrees ≈ 55 m latitude
        return Polygon([
            (12.00, 55.00),
            (12.00060, 55.00),
            (12.00060, 55.00050),
            (12.00, 55.00050),
            (12.00, 55.00),
        ])

    def _tiny_wgs84_polygon(self):
        """A ~1 m × 1 m square — well below 20 m² threshold."""
        delta = 0.000009  # ~1 m in degrees at 55°N
        return Polygon([
            (12.00, 55.00),
            (12.00 + delta, 55.00),
            (12.00 + delta, 55.00 + delta),
            (12.00, 55.00 + delta),
            (12.00, 55.00),
        ])

    def test_large_polygon_passes_filter(self):
        geom = self._large_wgs84_polygon()
        result = _simplify_and_filter(geom)
        assert result is not None
        assert result.geom_type in ("Polygon", "MultiPolygon")

    def test_tiny_polygon_filtered_out(self):
        geom = self._tiny_wgs84_polygon()
        result = _simplify_and_filter(geom)
        assert result is None

    def test_output_is_in_wgs84_range(self):
        geom = self._large_wgs84_polygon()
        result = _simplify_and_filter(geom)
        assert result is not None
        b = result.bounds  # (min_lon, min_lat, max_lon, max_lat)
        assert -180 <= b[0] <= 180
        assert -90 <= b[1] <= 90


# ── extract_polygons ─────────────────────────────────────────────────────────

class TestExtractPolygons:
    """End-to-end tests for extract_polygons with S3 and GDAL mocked."""

    _GEO_TRANSFORM = (12.0, 9e-6, 0.0, 55.01, 0.0, -9e-6)
    _PROJECTION = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    )

    def _mock_gdal_ds(self, mask_array):
        from osgeo import gdal as real_gdal
        band = MagicMock()
        band.ReadAsArray = MagicMock(return_value=mask_array)
        ds = MagicMock()
        ds.GetRasterBand = MagicMock(return_value=band)
        ds.GetGeoTransform = MagicMock(return_value=self._GEO_TRANSFORM)
        ds.GetProjection = MagicMock(return_value=self._PROJECTION)
        return ds

    def _s3_and_env(self, monkeypatch):
        for k, v in [
            ("S3_CHECKPOINT_BUCKET", "test-bucket"),
            ("AWS_REGION", "ap-northeast-2"),
            ("AWS_ACCESS_KEY_ID", "testing"),
            ("AWS_SECRET_ACCESS_KEY", "testing"),
        ]:
            monkeypatch.setenv(k, v)
        s3 = MagicMock()
        s3.download_file = MagicMock()
        s3.put_object = MagicMock()
        return s3

    @pytest.mark.asyncio
    async def test_produces_geojson_and_uploads_to_s3(self, monkeypatch):
        s3 = self._s3_and_env(monkeypatch)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1  # large green region

        with (
            patch("pipeline.extraction._s3_client", return_value=s3),
            patch("pipeline.extraction._checkpoint_exists", return_value=False),
            patch("pipeline.extraction.gdal.Open", return_value=self._mock_gdal_ds(mask)),
        ):
            result = await extract_polygons("course-1", "checkpoints/c/mask.tif", False)

        assert result == "checkpoints/course-1/polygons_raw.geojson"
        s3.put_object.assert_called_once()
        body = s3.put_object.call_args.kwargs["Body"]
        fc = json.loads(body)
        assert fc["type"] == "FeatureCollection"
        # Should have at least one polygon (the green region)
        assert len(fc["features"]) >= 1

    @pytest.mark.asyncio
    async def test_checkpoint_hit_returns_without_s3_download(self, monkeypatch):
        s3 = self._s3_and_env(monkeypatch)
        with (
            patch("pipeline.extraction._s3_client", return_value=s3),
            patch("pipeline.extraction._checkpoint_exists", return_value=True),
        ):
            result = await extract_polygons("course-2", "mask.tif", False)

        assert result == "checkpoints/course-2/polygons_raw.geojson"
        s3.download_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_flag_bypasses_checkpoint(self, monkeypatch):
        s3 = self._s3_and_env(monkeypatch)
        mask = np.zeros((50, 50), dtype=np.uint8)

        with (
            patch("pipeline.extraction._s3_client", return_value=s3),
            patch("pipeline.extraction._checkpoint_exists", return_value=True),
            patch("pipeline.extraction.gdal.Open", return_value=self._mock_gdal_ds(mask)),
        ):
            await extract_polygons("course-3", "mask.tif", force=True)

        # put_object should be called even though checkpoint exists
        s3.put_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_mask_produces_empty_feature_collection(self, monkeypatch):
        s3 = self._s3_and_env(monkeypatch)
        mask = np.zeros((50, 50), dtype=np.uint8)  # all background

        with (
            patch("pipeline.extraction._s3_client", return_value=s3),
            patch("pipeline.extraction._checkpoint_exists", return_value=False),
            patch("pipeline.extraction.gdal.Open", return_value=self._mock_gdal_ds(mask)),
        ):
            await extract_polygons("course-4", "mask.tif", False)

        fc = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert fc["features"] == []
