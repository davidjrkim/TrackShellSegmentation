import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.db_write import write_to_db


def _feature(hole_number, ftype, lon=12.0, lat=55.0, confidence=0.85, needs_review=False):
    fid = str(uuid.uuid4())
    return {
        "type": "Feature",
        "id": fid,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lon - 0.001, lat - 0.001],
                [lon + 0.001, lat - 0.001],
                [lon + 0.001, lat + 0.001],
                [lon - 0.001, lat + 0.001],
                [lon - 0.001, lat - 0.001],
            ]],
        },
        "properties": {
            "feature_type": ftype,
            "class_id": 1,
            "hole_number": hole_number,
            "confidence": confidence,
            "needs_review": needs_review,
        },
    }


def _make_geojson(*features):
    return json.dumps({"type": "FeatureCollection", "features": list(features)}).encode()


class _AsyncCM:
    """Minimal async context manager that returns a given value on __aenter__."""
    def __init__(self, value=None):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *_):
        return False


def _make_pool(course_status="pending"):
    conn = AsyncMock()
    conn.execute = AsyncMock()
    # conn.transaction() must return an async context manager (not a coroutine)
    conn.transaction = MagicMock(return_value=_AsyncCM(None))

    pool = AsyncMock()
    pool.fetchrow = AsyncMock(return_value={"status": course_status} if course_status else None)
    # pool.acquire() must return an async context manager that yields conn
    pool.acquire = MagicMock(return_value=_AsyncCM(conn))
    return pool, conn


@pytest.fixture
def s3_mock(monkeypatch):
    monkeypatch.setenv("S3_CHECKPOINT_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_REGION", "ap-northeast-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")

    body = MagicMock()
    body.read = MagicMock(return_value=_make_geojson(
        _feature(1, "tee_box", lon=12.000),
        _feature(1, "green",   lon=12.001),
        _feature(1, "fairway", lon=12.0005),
    ))
    s3 = MagicMock()
    s3.get_object = MagicMock(return_value={"Body": body})
    with patch("pipeline.db_write._s3_client", return_value=s3):
        yield s3


class TestWriteToDb:
    @pytest.mark.asyncio
    async def test_normal_run_inserts_hole_and_features(self, s3_mock):
        pool, conn = _make_pool("pending")
        await write_to_db("course-1", "key/assigned.geojson", False, pool)
        # Should have inserted 1 hole + N feature rows + updated course
        calls = [str(c) for c in conn.execute.call_args_list]
        assert any("INSERT INTO holes" in c for c in calls)
        assert any("INSERT INTO features" in c for c in calls)
        assert any("UPDATE courses" in c for c in calls)

    @pytest.mark.asyncio
    async def test_reviewed_course_raises_without_force(self, s3_mock):
        pool, _ = _make_pool("reviewed")
        with pytest.raises(RuntimeError, match="reviewed"):
            await write_to_db("course-1", "key/assigned.geojson", False, pool)

    @pytest.mark.asyncio
    async def test_published_course_raises_without_force(self, s3_mock):
        pool, _ = _make_pool("published")
        with pytest.raises(RuntimeError, match="published"):
            await write_to_db("course-1", "key/assigned.geojson", False, pool)

    @pytest.mark.asyncio
    async def test_reviewed_course_proceeds_with_force(self, s3_mock):
        pool, conn = _make_pool("reviewed")
        await write_to_db("course-1", "key/assigned.geojson", True, pool)
        calls = [str(c) for c in conn.execute.call_args_list]
        assert any("UPDATE courses" in c for c in calls)

    @pytest.mark.asyncio
    async def test_unknown_course_proceeds(self, s3_mock):
        pool, conn = _make_pool(None)
        await write_to_db("course-new", "key/assigned.geojson", False, pool)
        calls = [str(c) for c in conn.execute.call_args_list]
        assert any("INSERT INTO holes" in c for c in calls)

    @pytest.mark.asyncio
    async def test_unassigned_features_are_skipped(self, monkeypatch):
        monkeypatch.setenv("S3_CHECKPOINT_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_REGION", "ap-northeast-2")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        body = MagicMock()
        body.read = MagicMock(return_value=_make_geojson(
            _feature(None, "fairway"),  # no hole_number → should be skipped
        ))
        s3 = MagicMock()
        s3.get_object = MagicMock(return_value={"Body": body})
        with patch("pipeline.db_write._s3_client", return_value=s3):
            pool, conn = _make_pool("pending")
            await write_to_db("course-1", "key/assigned.geojson", False, pool)
            calls = [str(c) for c in conn.execute.call_args_list]
            assert not any("INSERT INTO holes" in c for c in calls)

    @pytest.mark.asyncio
    async def test_multiple_same_type_features_merged_into_multipolygon(self, monkeypatch):
        monkeypatch.setenv("S3_CHECKPOINT_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_REGION", "ap-northeast-2")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        # Two fairways in hole 1 → should produce a single MULTIPOLYGON insert
        body = MagicMock()
        body.read = MagicMock(return_value=_make_geojson(
            _feature(1, "tee_box", lon=12.000),
            _feature(1, "green",   lon=12.003),
            _feature(1, "fairway", lon=12.001),
            _feature(1, "fairway", lon=12.002),  # second fairway → MULTIPOLYGON
        ))
        s3 = MagicMock()
        s3.get_object = MagicMock(return_value={"Body": body})
        with patch("pipeline.db_write._s3_client", return_value=s3):
            pool, conn = _make_pool("pending")
            await write_to_db("course-1", "key/assigned.geojson", False, pool)
            feature_inserts = [
                c for c in conn.execute.call_args_list
                if "INSERT INTO features" in str(c)
            ]
            # tee_box + green + 1 merged fairway = 3 feature rows
            assert len(feature_inserts) == 3
