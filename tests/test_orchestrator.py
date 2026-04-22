from unittest.mock import AsyncMock, patch

import pytest

from pipeline.orchestrator import run_pipeline


def _make_pool():
    pool = AsyncMock()
    pool.execute = AsyncMock()
    return pool


class TestRunPipeline:
    @pytest.mark.asyncio
    async def test_happy_path_runs_all_stages_in_order(self):
        pool = _make_pool()
        calls = []

        async def fake_preprocess(course_id, force):
            calls.append("preprocessing")
            return "checkpoints/c/chips"

        async def fake_segment(course_id, chips, force):
            calls.append("segmentation")
            return "checkpoints/c/mask.tif"

        async def fake_extract(course_id, mask_key, force):
            calls.append("extraction")
            return "checkpoints/c/polygons_raw.geojson"

        async def fake_assign(course_id, raw_key, force):
            calls.append("assignment")
            return "checkpoints/c/polygons_assigned.geojson"

        async def fake_write(course_id, assigned_key, force, pool):
            calls.append("db_write")

        # orchestrator imports pipeline functions locally inside run_pipeline,
        # so we patch at their source modules (not on the orchestrator namespace)
        with (
            patch("pipeline.preprocessing.preprocess_course", fake_preprocess),
            patch("pipeline.segmentation.segment_course", fake_segment),
            patch("pipeline.extraction.extract_polygons", fake_extract),
            patch("pipeline.assignment.assign_holes", fake_assign),
            patch("pipeline.db_write.write_to_db", fake_write),
        ):
            await run_pipeline("job-1", "course-1", False, pool)

        assert calls == ["preprocessing", "segmentation", "extraction", "assignment", "db_write"]

    @pytest.mark.asyncio
    async def test_completed_status_set_on_success(self):
        pool = _make_pool()

        with (
            patch("pipeline.preprocessing.preprocess_course", AsyncMock(return_value="chips")),
            patch("pipeline.segmentation.segment_course", AsyncMock(return_value="mask")),
            patch("pipeline.extraction.extract_polygons", AsyncMock(return_value="raw")),
            patch("pipeline.assignment.assign_holes", AsyncMock(return_value="assigned")),
            patch("pipeline.db_write.write_to_db", AsyncMock()),
        ):
            await run_pipeline("job-1", "course-1", False, pool)

        status_calls = [str(c) for c in pool.execute.call_args_list]
        assert any("completed" in c for c in status_calls)

    @pytest.mark.asyncio
    async def test_failed_status_set_on_exception(self):
        pool = _make_pool()

        with patch("pipeline.preprocessing.preprocess_course", AsyncMock(side_effect=RuntimeError("boom"))):
            with pytest.raises(RuntimeError, match="boom"):
                await run_pipeline("job-1", "course-1", False, pool)

        status_calls = [str(c) for c in pool.execute.call_args_list]
        assert any("failed" in c for c in status_calls)

    @pytest.mark.asyncio
    async def test_stage_name_updated_before_each_stage(self):
        pool = _make_pool()

        with (
            patch("pipeline.preprocessing.preprocess_course", AsyncMock(return_value="chips")),
            patch("pipeline.segmentation.segment_course", AsyncMock(return_value="mask")),
            patch("pipeline.extraction.extract_polygons", AsyncMock(return_value="raw")),
            patch("pipeline.assignment.assign_holes", AsyncMock(return_value="assigned")),
            patch("pipeline.db_write.write_to_db", AsyncMock()),
        ):
            await run_pipeline("job-1", "course-1", False, pool)

        # _set_stage calls pool.execute(sql, status, stage, job_id)
        # so stage is at positional index 2
        stage_args = [c.args[2] for c in pool.execute.call_args_list if len(c.args) > 2]
        for stage in ("preprocessing", "segmentation", "extraction", "assignment", "db_write"):
            assert stage in stage_args, f"Stage '{stage}' never set on job"
