import numpy as np
import pytest

from pipeline.preprocessing import BoundingBox, _chip_image, _deg2tile


def test_deg2tile_known_value():
    # Copenhagen approx: lat=55.67, lon=12.57, zoom=18
    x, y = _deg2tile(55.67, 12.57, 18)
    assert isinstance(x, int)
    assert isinstance(y, int)
    assert x > 0 and y > 0


def test_deg2tile_zoom_increases_tile_count():
    x1, y1 = _deg2tile(55.67, 12.57, 16)
    x2, y2 = _deg2tile(55.67, 12.57, 18)
    # Higher zoom = larger tile coordinates
    assert x2 > x1
    assert y2 > y1


def test_chip_image_produces_correct_count():
    # 1024×1024 image, STRIDE=448, should produce a grid of chips
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    chips = _chip_image(image, transform=None, course_id="test-course")
    assert len(chips) > 0
    for chip in chips:
        assert chip["data"].shape == (512, 512, 3)


def test_chip_image_small_image_pads_to_chip_size():
    # Image smaller than one chip — should produce exactly one padded chip
    image = np.ones((300, 400, 3), dtype=np.uint8) * 128
    chips = _chip_image(image, transform=None, course_id="test-course")
    assert len(chips) == 1
    assert chips[0]["data"].shape == (512, 512, 3)
    # Original pixels preserved
    assert np.all(chips[0]["data"][:300, :400] == 128)
    # Padded area is zero
    assert np.all(chips[0]["data"][300:, :] == 0)


def test_chip_image_origin_metadata():
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    chips = _chip_image(image, transform=None, course_id="test-course")
    first = chips[0]
    assert first["origin_x"] == 0
    assert first["origin_y"] == 0
    assert first["idx"] == 0


def test_bounding_box_fields():
    bbox = BoundingBox(min_lon=12.0, min_lat=55.0, max_lon=12.1, max_lat=55.1)
    assert bbox.max_lon > bbox.min_lon
    assert bbox.max_lat > bbox.min_lat
