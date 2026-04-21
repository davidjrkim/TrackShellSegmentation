import traceback

import asyncpg


async def _set_stage(pool: asyncpg.Pool, job_id: str, stage: str, status: str = "running"):
    await pool.execute(
        "UPDATE pipeline_jobs SET status = $1, stage = $2 WHERE id = $3",
        status, stage, job_id,
    )


async def run_pipeline(job_id: str, course_id: str, force: bool, pool: asyncpg.Pool):
    try:
        from pipeline.preprocessing import preprocess_course
        from pipeline.segmentation import segment_course
        from pipeline.extraction import extract_polygons
        from pipeline.assignment import assign_holes
        from pipeline.db_write import write_to_db

        await _set_stage(pool, job_id, "preprocessing")
        chips_prefix = await preprocess_course(course_id, force)

        await _set_stage(pool, job_id, "segmentation")
        mask_key = await segment_course(course_id, chips_prefix, force)

        await _set_stage(pool, job_id, "extraction")
        raw_geojson_key = await extract_polygons(course_id, mask_key, force)

        await _set_stage(pool, job_id, "assignment")
        assigned_geojson_key = await assign_holes(course_id, raw_geojson_key, force)

        await _set_stage(pool, job_id, "db_write")
        await write_to_db(course_id, assigned_geojson_key, force, pool)

        await pool.execute(
            "UPDATE pipeline_jobs SET status = 'completed', stage = 'done' WHERE id = $1",
            job_id,
        )

    except Exception:
        error = traceback.format_exc()
        await pool.execute(
            "UPDATE pipeline_jobs SET status = 'failed', error_message = $1 WHERE id = $2",
            error, job_id,
        )
        raise
