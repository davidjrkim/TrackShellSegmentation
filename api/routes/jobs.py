import asyncio
import json
import uuid
from enum import Enum
from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import UUID4, BaseModel, field_validator

router = APIRouter()


class JobType(str, Enum):
    full = "full"


class RunJobRequest(BaseModel):
    course_id: UUID4
    job_type: JobType
    force: bool = False

    @field_validator("force")
    @classmethod
    def force_must_be_bool(cls, v):
        if not isinstance(v, bool):
            raise ValueError("force must be a boolean")
        return v


class JobResponse(BaseModel):
    job_id: str
    status: str


@router.post("/run", response_model=JobResponse)
async def run_job(body: RunJobRequest, background_tasks: BackgroundTasks, request: Request):
    job_id = str(uuid.uuid4())
    pool = request.app.state.db_pool

    await pool.execute(
        """
        INSERT INTO pipeline_jobs (id, course_id, status, job_type, force)
        VALUES ($1, $2, 'queued', $3, $4)
        """,
        job_id, str(body.course_id), body.job_type.value, body.force,
    )

    from pipeline.orchestrator import run_pipeline
    background_tasks.add_task(run_pipeline, job_id, str(body.course_id), body.force, pool)

    return JobResponse(job_id=job_id, status="queued")


@router.get("/{job_id}/status")
async def job_status(job_id: str, request: Request):
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id")

    pool = request.app.state.db_pool
    row = await pool.fetchrow("SELECT status, error_message FROM pipeline_jobs WHERE id = $1", job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": row["status"], "error_message": row["error_message"]}


@router.get("/{job_id}/stream")
async def job_stream(job_id: str, request: Request):
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id")

    pool = request.app.state.db_pool

    async def event_generator() -> AsyncGenerator[str, None]:
        last_status = None
        while True:
            row = await pool.fetchrow(
                "SELECT status, stage, error_message FROM pipeline_jobs WHERE id = $1", job_id
            )
            if row is None:
                yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
                return

            current = {"status": row["status"], "stage": row["stage"]}
            if current != last_status:
                yield f"data: {json.dumps(current)}\n\n"
                last_status = current

            if row["status"] in ("completed", "failed"):
                return

            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
