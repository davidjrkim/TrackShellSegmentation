import base64
import io
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from pipeline.assignment import (
    ADJACENCY_THRESHOLD_M,
    REVIEW_THRESHOLD,
    _build_spatial_graph,
    _call_llm_with_retry,
    _graph_to_json,
    _render_composite,
    _score_and_flag,
    assign_holes,
)


def _square_feature(
    center_lon: float,
    center_lat: float,
    half: float = 0.0005,
    ftype: str = "fairway",
    feat_id: str | None = None,
) -> dict:
    """Create a GeoJSON square feature centred at (center_lon, center_lat)."""
    return {
        "type": "Feature",
        "id": feat_id or str(uuid.uuid4()),
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [center_lon - half, center_lat - half],
                [center_lon + half, center_lat - half],
                [center_lon + half, center_lat + half],
                [center_lon - half, center_lat + half],
                [center_lon - half, center_lat - half],
            ]],
        },
        "properties": {"feature_type": ftype, "class_id": 2},
    }


class TestBuildSpatialGraph:
    def test_adjacent_polygons_share_edge(self):
        # Two squares that share a border → distance == 0 → edge expected
        fa = _square_feature(12.0, 55.0, half=0.0005)
        fb = _square_feature(12.001, 55.0, half=0.0005)  # touching edge at 12.0005
        G = _build_spatial_graph([fa, fb])
        assert G.has_edge(fa["id"], fb["id"])

    def test_distant_polygons_have_no_edge(self):
        # 1 degree of longitude ≈ 63 km at 55°N — well above 10 m threshold
        fa = _square_feature(12.0, 55.0)
        fb = _square_feature(13.0, 55.0)
        G = _build_spatial_graph([fa, fb])
        assert not G.has_edge(fa["id"], fb["id"])

    def test_all_features_become_nodes(self):
        features = [_square_feature(12.0 + i * 0.01, 55.0) for i in range(5)]
        G = _build_spatial_graph(features)
        assert G.number_of_nodes() == 5

    def test_node_carries_feature_type(self):
        fa = _square_feature(12.0, 55.0, ftype="green")
        G = _build_spatial_graph([fa])
        assert G.nodes[fa["id"]]["feature_type"] == "green"

    def test_empty_feature_list_produces_empty_graph(self):
        G = _build_spatial_graph([])
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0


class TestScoreAndFlag:
    def _make_index(self, *features):
        return {f["id"]: f for f in features}

    def test_full_hole_preserves_llm_confidence(self):
        tee = _square_feature(12.0, 55.0, ftype="tee_box")
        green = _square_feature(12.001, 55.0, ftype="green")  # ~60 m away
        fairway = _square_feature(12.0005, 55.0, ftype="fairway")
        hole = {
            "confidence": 0.90,
            "tee_box_id": tee["id"],
            "green_id": green["id"],
            "fairway_ids": [fairway["id"]],
        }
        score, needs_review = _score_and_flag(hole, self._make_index(tee, green, fairway))
        assert score == 0.90
        assert needs_review is False

    def test_missing_tee_penalises_by_040(self):
        green = _square_feature(12.0, 55.0, ftype="green")
        hole = {"confidence": 0.80, "tee_box_id": None, "green_id": green["id"]}
        score, _ = _score_and_flag(hole, self._make_index(green))
        assert abs(score - 0.40) < 1e-9

    def test_missing_green_penalises_by_040(self):
        tee = _square_feature(12.0, 55.0, ftype="tee_box")
        hole = {"confidence": 0.80, "tee_box_id": tee["id"], "green_id": None}
        score, _ = _score_and_flag(hole, self._make_index(tee))
        assert abs(score - 0.40) < 1e-9

    def test_missing_both_clamps_to_zero(self):
        hole = {"confidence": 0.70, "tee_box_id": None, "green_id": None}
        score, needs_review = _score_and_flag(hole, {})
        assert score == 0.0
        assert needs_review is True

    def test_large_gap_without_fairway_penalises(self):
        # Place tee and green ~200 m apart (≫ 150 m threshold)
        # At 55°N: 0.002 degrees longitude ≈ 0.002 × 111319 = 222 m in EPSG:3857
        tee = _square_feature(12.00, 55.0, ftype="tee_box")
        green = _square_feature(12.002, 55.0, ftype="green")
        hole = {
            "confidence": 0.85,
            "tee_box_id": tee["id"],
            "green_id": green["id"],
            "fairway_ids": [],
        }
        score, _ = _score_and_flag(hole, self._make_index(tee, green))
        assert score < 0.85  # penalised

    def test_below_threshold_sets_needs_review(self):
        hole = {"confidence": 0.65, "tee_box_id": None, "green_id": None}
        _, needs_review = _score_and_flag(hole, {})
        assert needs_review is True

    def test_above_threshold_clears_needs_review(self):
        tee = _square_feature(12.0, 55.0, ftype="tee_box")
        green = _square_feature(12.001, 55.0, ftype="green")
        fairway = _square_feature(12.0005, 55.0, ftype="fairway")
        hole = {
            "confidence": 0.95,
            "tee_box_id": tee["id"],
            "green_id": green["id"],
            "fairway_ids": [fairway["id"]],
        }
        _, needs_review = _score_and_flag(hole, self._make_index(tee, green, fairway))
        assert needs_review is False


class TestRenderComposite:
    def _decode_png(self, b64: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(b64)))

    def test_empty_features_returns_valid_png(self):
        b64 = _render_composite([])
        img = self._decode_png(b64)
        assert img.size == (1024, 1024)

    def test_nonempty_features_returns_1024_png(self):
        features = [
            _square_feature(12.0, 55.0, ftype="green"),
            _square_feature(12.001, 55.0, ftype="fairway"),
            _square_feature(11.999, 55.0, ftype="tee_box"),
        ]
        b64 = _render_composite(features)
        img = self._decode_png(b64)
        assert img.size == (1024, 1024)
        assert img.mode == "RGB"

    def test_output_is_valid_base64(self):
        b64 = _render_composite([])
        # Should not raise
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_all_feature_types_render_without_error(self):
        ftypes = ["green", "fairway", "tee_box", "bunker", "water_hazard"]
        features = [
            _square_feature(12.0 + i * 0.002, 55.0, ftype=ft)
            for i, ft in enumerate(ftypes)
        ]
        b64 = _render_composite(features)
        img = self._decode_png(b64)
        assert img.size == (1024, 1024)


# ── _graph_to_json ──────────────────────────────────────────────────────────

class TestGraphToJson:
    def test_output_has_nodes_and_edges_keys(self):
        fa = _square_feature(12.0, 55.0, ftype="green")
        fb = _square_feature(12.001, 55.0, ftype="fairway")
        G = _build_spatial_graph([fa, fb])
        data = json.loads(_graph_to_json(G))
        assert "nodes" in data
        assert "edges" in data

    def test_node_ids_match_features(self):
        fa = _square_feature(12.0, 55.0)
        fb = _square_feature(13.0, 55.0)
        G = _build_spatial_graph([fa, fb])
        data = json.loads(_graph_to_json(G))
        node_ids = {n["id"] for n in data["nodes"]}
        assert fa["id"] in node_ids
        assert fb["id"] in node_ids

    def test_adjacent_pair_has_one_edge(self):
        fa = _square_feature(12.0, 55.0)
        fb = _square_feature(12.001, 55.0)  # touching
        G = _build_spatial_graph([fa, fb])
        data = json.loads(_graph_to_json(G))
        assert len(data["edges"]) == 1


# ── _call_llm_with_retry ────────────────────────────────────────────────────

_VALID_LLM_RESPONSE = json.dumps({"holes": [{"hole_number": 1, "tee_box_id": "a", "green_id": "b", "fairway_ids": [], "other_ids": [], "confidence": 0.9}]})
_FAKE_IMG_B64 = base64.b64encode(b"fake-png-bytes").decode()


class TestCallLlmWithRetry:
    def _mock_model(self, response_text=_VALID_LLM_RESPONSE):
        resp = MagicMock()
        resp.text = response_text
        model = MagicMock()
        model.generate_content_async = AsyncMock(return_value=resp)
        return model

    @pytest.mark.asyncio
    async def test_success_returns_parsed_json(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mock_model = self._mock_model()
        with patch("pipeline.assignment.genai.GenerativeModel", return_value=mock_model):
            result = await _call_llm_with_retry(_FAKE_IMG_B64, "{}", 1)
            assert "holes" in result

    @pytest.mark.asyncio
    async def test_passes_n_holes_in_system_prompt(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mock_model = self._mock_model()
        with patch("pipeline.assignment.genai.GenerativeModel", return_value=mock_model) as mock_cls:
            await _call_llm_with_retry(_FAKE_IMG_B64, "{}", 9)
        call_kwargs = mock_cls.call_args.kwargs
        assert "9" in call_kwargs["system_instruction"]

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from google.api_core import exceptions as gapi_exceptions
        timeout_err = gapi_exceptions.DeadlineExceeded("timeout")
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(side_effect=timeout_err)
        with (
            patch("pipeline.assignment.genai.GenerativeModel", return_value=mock_model),
            patch("pipeline.assignment.asyncio.sleep", AsyncMock()),
        ):
            with pytest.raises(gapi_exceptions.DeadlineExceeded):
                await _call_llm_with_retry(_FAKE_IMG_B64, "{}", 1)
        assert mock_model.generate_content_async.call_count == 3


# ── assign_holes ────────────────────────────────────────────────────────────

class TestAssignHoles:
    def _raw_geojson(self, tee_id, green_id, fw_id):
        def _feat(fid, ftype, lon):
            return {
                "type": "Feature", "id": fid,
                "geometry": {"type": "Polygon", "coordinates": [[
                    [lon - 0.0005, 54.9995], [lon + 0.0005, 54.9995],
                    [lon + 0.0005, 55.0005], [lon - 0.0005, 55.0005],
                    [lon - 0.0005, 54.9995],
                ]]},
                "properties": {"feature_type": ftype, "class_id": 1},
            }
        return {"type": "FeatureCollection", "features": [
            _feat(tee_id, "tee_box", 12.000),
            _feat(green_id, "green",   12.002),
            _feat(fw_id,   "fairway", 12.001),
        ]}

    @pytest.mark.asyncio
    async def test_returns_checkpoint_key(self, monkeypatch):
        monkeypatch.setenv("S3_CHECKPOINT_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_REGION", "ap-northeast-2")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")

        tee_id, green_id, fw_id = (str(uuid.uuid4()) for _ in range(3))
        llm_result = {"holes": [{"hole_number": 1, "tee_box_id": tee_id,
                                  "green_id": green_id, "fairway_ids": [fw_id],
                                  "other_ids": [], "confidence": 0.88}]}

        body = MagicMock()
        body.read = MagicMock(return_value=json.dumps(self._raw_geojson(tee_id, green_id, fw_id)).encode())
        s3 = MagicMock()
        s3.get_object = MagicMock(return_value={"Body": body})
        s3.put_object = MagicMock()

        with (
            patch("pipeline.assignment._s3_client", return_value=s3),
            patch("pipeline.assignment._checkpoint_exists", return_value=False),
            patch("pipeline.assignment._call_llm_with_retry", AsyncMock(return_value=llm_result)),
        ):
            result = await assign_holes("course-x", "raw.geojson", False)

        assert result == "checkpoints/course-x/polygons_assigned.geojson"
        s3.put_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_checkpoint_hit_returns_early(self, monkeypatch):
        monkeypatch.setenv("S3_CHECKPOINT_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_REGION", "ap-northeast-2")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")

        s3 = MagicMock()
        with (
            patch("pipeline.assignment._s3_client", return_value=s3),
            patch("pipeline.assignment._checkpoint_exists", return_value=True),
        ):
            result = await assign_holes("course-y", "raw.geojson", False)

        assert result == "checkpoints/course-y/polygons_assigned.geojson"
        s3.get_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_unassigned_features_get_null_hole_number(self, monkeypatch):
        monkeypatch.setenv("S3_CHECKPOINT_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_REGION", "ap-northeast-2")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")

        tee_id, green_id, fw_id = (str(uuid.uuid4()) for _ in range(3))
        llm_result = {"holes": []}  # LLM assigns nothing

        body = MagicMock()
        body.read = MagicMock(return_value=json.dumps(self._raw_geojson(tee_id, green_id, fw_id)).encode())
        s3 = MagicMock()
        s3.get_object = MagicMock(return_value={"Body": body})
        s3.put_object = MagicMock()

        with (
            patch("pipeline.assignment._s3_client", return_value=s3),
            patch("pipeline.assignment._checkpoint_exists", return_value=False),
            patch("pipeline.assignment._call_llm_with_retry", AsyncMock(return_value=llm_result)),
        ):
            await assign_holes("course-z", "raw.geojson", False)

        saved = json.loads(s3.put_object.call_args.kwargs["Body"])
        for f in saved["features"]:
            assert f["properties"]["hole_number"] is None
