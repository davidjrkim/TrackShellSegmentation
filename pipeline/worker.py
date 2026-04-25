import asyncio
import logging

import asyncpg

from pipeline.orchestrator import run_pipeline

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 2


async def reclaim_orphaned_jobs(pool: asyncpg.Pool) -> int:
    """
    Requeue any jobs left in 'running' status — they belong to a previous
    container that was killed mid-pipeline. Stages are idempotent via S3
    checkpoints, so a resumed job skips work it already finished.

    Single-worker assumption: if you scale to multiple worker processes per
    environment, this needs a lease/heartbeat column to avoid stomping on a
    peer's in-flight job.
    """
    result = await pool.execute(
        "UPDATE pipeline_jobs SET status = 'queued', stage = NULL WHERE status = 'running'"
    )
    n = int(result.split()[-1]) if result else 0
    if n:
        logger.warning("Reclaimed %d orphaned running job(s)", n)
    return n


async def _claim_next_job(pool: asyncpg.Pool):
    return await pool.fetchrow(
        """
        UPDATE pipeline_jobs
        SET status = 'running'
        WHERE id = (
            SELECT id FROM pipeline_jobs
            WHERE status = 'queued'
            ORDER BY created_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING id::text AS id, course_id::text AS course_id, force
        """
    )


async def run_worker(pool: asyncpg.Pool, stop_event: asyncio.Event) -> None:
    await reclaim_orphaned_jobs(pool)

    while not stop_event.is_set():
        try:
            row = await _claim_next_job(pool)
        except Exception:
            logger.exception("Failed to poll for jobs")
            await _sleep_or_stop(stop_event, POLL_INTERVAL_SECONDS)
            continue

        if row is None:
            await _sleep_or_stop(stop_event, POLL_INTERVAL_SECONDS)
            continue

        try:
            await run_pipeline(row["id"], row["course_id"], row["force"], pool)
        except Exception:
            logger.exception("Pipeline failed for job %s", row["id"])


async def _sleep_or_stop(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
