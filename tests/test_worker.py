import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from pipeline.worker import (
    _claim_next_job,
    _sleep_or_stop,
    reclaim_orphaned_jobs,
    run_worker,
)


def _pool(execute_return="UPDATE 0", claim_rows=None):
    pool = AsyncMock()
    pool.execute = AsyncMock(return_value=execute_return)

    rows = list(claim_rows or [])

    async def fetchrow(_sql, *args, **kwargs):
        return rows.pop(0) if rows else None

    pool.fetchrow = AsyncMock(side_effect=fetchrow)
    return pool


class TestReclaimOrphanedJobs:
    @pytest.mark.asyncio
    async def test_returns_zero_when_nothing_to_reclaim(self):
        pool = _pool(execute_return="UPDATE 0")
        assert await reclaim_orphaned_jobs(pool) == 0

    @pytest.mark.asyncio
    async def test_parses_count_from_asyncpg_status_string(self):
        pool = _pool(execute_return="UPDATE 3")
        assert await reclaim_orphaned_jobs(pool) == 3

    @pytest.mark.asyncio
    async def test_handles_empty_status_string(self):
        pool = _pool(execute_return="")
        assert await reclaim_orphaned_jobs(pool) == 0

    @pytest.mark.asyncio
    async def test_only_targets_running_status(self):
        pool = _pool(execute_return="UPDATE 1")
        await reclaim_orphaned_jobs(pool)
        sql = pool.execute.call_args.args[0]
        assert "status = 'running'" in sql
        assert "status = 'queued'" in sql  # the SET clause


class TestClaimNextJob:
    @pytest.mark.asyncio
    async def test_returns_row_from_fetchrow(self):
        row = {"id": "abc", "course_id": "c1", "force": False}
        pool = _pool(claim_rows=[row])
        assert await _claim_next_job(pool) == row

    @pytest.mark.asyncio
    async def test_uses_for_update_skip_locked(self):
        pool = _pool()
        await _claim_next_job(pool)
        sql = pool.fetchrow.call_args.args[0]
        assert "FOR UPDATE SKIP LOCKED" in sql
        assert "status = 'queued'" in sql


class TestSleepOrStop:
    @pytest.mark.asyncio
    async def test_returns_after_timeout_when_event_unset(self):
        stop = asyncio.Event()
        await _sleep_or_stop(stop, 0.01)  # should just time out

    @pytest.mark.asyncio
    async def test_returns_immediately_when_event_set(self):
        stop = asyncio.Event()
        stop.set()
        await asyncio.wait_for(_sleep_or_stop(stop, 5.0), timeout=0.5)


class TestRunWorker:
    @pytest.mark.asyncio
    async def test_reclaims_then_processes_one_queued_job_then_stops(self):
        pool = _pool(
            execute_return="UPDATE 1",
            claim_rows=[{"id": "j1", "course_id": "c1", "force": False}],
        )
        stop = asyncio.Event()
        ran = []

        async def fake_run_pipeline(job_id, course_id, force, p):
            ran.append((job_id, course_id, force))
            stop.set()

        with patch("pipeline.worker.run_pipeline", fake_run_pipeline):
            await asyncio.wait_for(run_worker(pool, stop), timeout=2.0)

        assert ran == [("j1", "c1", False)]
        # reclaim was called exactly once at startup
        reclaim_calls = [
            c for c in pool.execute.call_args_list
            if "status = 'queued'" in c.args[0] and "WHERE status = 'running'" in c.args[0]
        ]
        assert len(reclaim_calls) == 1

    @pytest.mark.asyncio
    async def test_swallows_pipeline_exceptions_and_keeps_running(self):
        pool = _pool(
            execute_return="UPDATE 0",
            claim_rows=[
                {"id": "j1", "course_id": "c1", "force": False},
                {"id": "j2", "course_id": "c2", "force": True},
            ],
        )
        stop = asyncio.Event()
        ran = []

        async def fake_run_pipeline(job_id, course_id, force, p):
            ran.append(job_id)
            if job_id == "j1":
                raise RuntimeError("boom")
            stop.set()

        with patch("pipeline.worker.run_pipeline", fake_run_pipeline):
            await asyncio.wait_for(run_worker(pool, stop), timeout=2.0)

        assert ran == ["j1", "j2"]

    @pytest.mark.asyncio
    async def test_swallows_polling_exceptions_and_retries(self):
        pool = AsyncMock()
        pool.execute = AsyncMock(return_value="UPDATE 0")
        calls = {"n": 0}

        async def fetchrow(_sql, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("db blip")
            return {"id": "j1", "course_id": "c1", "force": False}

        pool.fetchrow = AsyncMock(side_effect=fetchrow)
        stop = asyncio.Event()

        async def fake_run_pipeline(*_a, **_kw):
            stop.set()

        with (
            patch("pipeline.worker.run_pipeline", fake_run_pipeline),
            patch("pipeline.worker.POLL_INTERVAL_SECONDS", 0.01),
        ):
            await asyncio.wait_for(run_worker(pool, stop), timeout=2.0)

        assert calls["n"] >= 2

    @pytest.mark.asyncio
    async def test_exits_promptly_when_no_jobs_and_stop_set(self):
        pool = _pool(execute_return="UPDATE 0")
        stop = asyncio.Event()
        stop.set()  # already stopped before loop begins

        with patch("pipeline.worker.run_pipeline", AsyncMock()):
            await asyncio.wait_for(run_worker(pool, stop), timeout=1.0)
