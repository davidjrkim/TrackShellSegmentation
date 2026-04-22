import os
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI

from api.routes.jobs import router as jobs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_pool = await asyncpg.create_pool(
        os.environ["DATABASE_URL"],
        min_size=1,
        max_size=10,
    )
    yield
    await app.state.db_pool.close()


app = FastAPI(title="TrackShell Segmentation API", lifespan=lifespan)
app.include_router(jobs_router, prefix="/jobs")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
