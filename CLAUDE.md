# TrackShellSegmentation — CLAUDE.md

This file is read automatically by Claude Code. It tells you everything
you need to work in this codebase without asking questions.

**Also read:** `RULES.md` — security protocols, quality standards, and behavioral
rules that govern all code changes in this repo.

---

## What this project does

ML pipeline that takes satellite imagery of a golf course and produces
numbered, typed GeoJSON polygon features (greens, fairways, bunkers,
tee boxes, water hazards) saved to PostgreSQL/PostGIS.

Two ML stages:
1. DeepLabv3+ (ResNet-50) — semantic segmentation, pixel classification
2. Vision LLM (Claude claude-sonnet-4-20250514) — hole number assignment (1–18)

Runs as a FastAPI job runner triggered by the golf-mapping-platform dashboard.

---

## Stack

- Python 3.11
- PyTorch 2.x + torchvision (DeepLabv3+)
- GDAL 3.7+ (vectorization)
- rasterio (GeoTIFF / CRS handling)
- Shapely 2.x (polygon simplification)
- NetworkX (spatial graph for hole assignment)
- FastAPI (job runner HTTP server)
- asyncpg (PostgreSQL driver)
- Anthropic SDK (vision LLM hole assignment)
- Pillow (composite image rendering for LLM)
- pyproj 3.x (CRS projection for accurate metric distance/area calculations)
- AWS S3 (checkpoint storage)

---

## Project structure

```
TrackShellSegmentation/
  api/
    main.py           ← FastAPI app entry point
    routes/
      jobs.py         ← POST /jobs/run, GET /jobs/{id}/status, GET /jobs/{id}/stream
  pipeline/
    preprocessing.py  ← tile download, CRS reprojection, 512×512 chipping
    segmentation.py   ← DeepLabv3+ inference
    extraction.py     ← mask stitching, GDAL vectorization, Douglas-Peucker
    assignment.py     ← spatial graph, topology rules, LLM hole assignment
    db_write.py       ← write holes + features to PostGIS
  training/
    train.py          ← training script (run on cloud GPU, not in prod)
    dataset.py        ← Danish Orthophotos dataset loader
    loss.py           ← combined Dice + CrossEntropy loss
    augmentation.py   ← training augmentations
  checkpoints/        ← local checkpoint cache (gitignored)
  tests/
    test_preprocessing.py
    test_extraction.py
    test_assignment.py
  Dockerfile
  requirements.txt
  .env.example
```

---

## Run locally

```bash
# 1. Copy and fill in environment variables
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the job runner API
uvicorn api.main:app --reload --port 8001

# API is now available at http://localhost:8001
# Trigger a job: POST /jobs/run { "course_id": "uuid", "job_type": "full" }
```

---

## Run tests

```bash
pytest tests/ -v

# Run a specific test file
pytest tests/test_extraction.py -v

# Run with coverage
pytest tests/ --cov=pipeline --cov-report=term-missing
```

Tests must pass before any PR can be merged.

---

## Training (cloud GPU only — do not run locally)

```bash
# Run on Lambda Labs A100 instance
python training/train.py \
  --data-dir /path/to/danish-orthophotos \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --output-dir s3://trackshell-models/checkpoints/

# Monitor training
tensorboard --logdir runs/
```

Training is NOT part of the FastAPI job runner. It is a separate offline
process. The job runner loads a pretrained model checkpoint from S3.

---

## Environment variables

```
# Database
DATABASE_URL          PostgreSQL connection string (PostGIS enabled)
                      Format: postgresql+asyncpg://user:pass@host:5432/dbname

# AWS
AWS_ACCESS_KEY_ID     AWS credentials
AWS_SECRET_ACCESS_KEY AWS credentials
AWS_REGION            e.g. ap-northeast-2 (Seoul)
S3_CHECKPOINT_BUCKET  S3 bucket for model checkpoints and stage outputs
                      e.g. golf-mapping-checkpoints

# LLM
ANTHROPIC_API_KEY     For vision LLM hole assignment calls

# Satellite tiles
MAPBOX_TOKEN          For fetching satellite imagery base layer
                      Used when rendering composite image for LLM

# Model
MODEL_CHECKPOINT_KEY  S3 key for the active model checkpoint
                      e.g. models/deeplabv3plus_v1.2.pth
```

Never commit .env to git. Use AWS Parameter Store for production secrets.

---

## Pipeline stages

Each stage checkpoints its output to S3 before the next stage starts.
A failed stage can be retried without rerunning earlier stages.

```
Stage 1 — Preprocessing
  Input:  GPS bounding box
  Output: s3://{bucket}/checkpoints/{course_id}/chips/
  Key fn: pipeline/preprocessing.py → preprocess_course()

Stage 2 — Segmentation
  Input:  chips from Stage 1
  Output: s3://{bucket}/checkpoints/{course_id}/mask.tif
  Key fn: pipeline/segmentation.py → segment_course()

Stage 3 — Polygon extraction
  Input:  mask.tif from Stage 2
  Output: s3://{bucket}/checkpoints/{course_id}/polygons_raw.geojson
  Key fn: pipeline/extraction.py → extract_polygons()

Stage 4 — Hole assignment
  Input:  polygons_raw.geojson from Stage 3
  Output: s3://{bucket}/checkpoints/{course_id}/polygons_assigned.geojson
  Key fn: pipeline/assignment.py → assign_holes()

Stage 5 — DB write
  Input:  polygons_assigned.geojson from Stage 4
  Output: rows in holes + features tables
  Key fn: pipeline/db_write.py → write_to_db()
```

To force a full rerun (ignore checkpoints):
```bash
POST /jobs/run { "course_id": "uuid", "job_type": "full", "force": true }
```

---

## Class definitions (6 classes in v1)

```python
CLASS_MAP = {
    0: 'background',
    1: 'green',
    2: 'fairway',
    3: 'tee_box',
    4: 'bunker',
    5: 'water_hazard',
}
# Note: rough (class 6) is deferred to v2
```

The Danish Orthophotos dataset encodes classes as RGB colours in the mask GeoTIFF.
Preprocessing must convert RGB → integer class ID (exact match, no tolerance):

```python
RGB_TO_CLASS = {
    (142, 243, 122): 1,  # green
    ( 77, 156,  77): 2,  # fairway
    (250,  36,   0): 3,  # tee_box  (dataset label: "tee")
    (246, 246, 158): 4,  # bunker
    ( 46, 200, 231): 5,  # water_hazard  (dataset label: "water")
}
# Unmatched pixels → class 0 (background)
```

---

## Geometry decisions (already made — do not change)

- **Geometry type**: MULTIPOLYGON (handles split fairways)
- **CRS**: WGS84 SRID 4326 for all stored geometry
- **Simplification tolerance**: 1.5 metres (Douglas-Peucker)
- **Minimum polygon area**: 20 m² (noise threshold)
- **Chip size**: 512×512 pixels with 64px overlap
- **LLM image size**: 1024×1024 PNG (satellite + polygon overlay composite)
- **Review threshold**: holes with confidence < 0.70 flagged for human review

---

## Branch strategy

```
main        → production (auto-deploys via GitHub Actions + AWS ECS)
dev         → staging
feature/*   → PR to dev, all tests must pass
```

Never push directly to main. Always PR through dev first.

---

## Deployment

Deployed as a Docker container on AWS ECS Fargate.
GitHub Actions builds and pushes to ECR on merge to main.
ECS pulls the new image and performs a rolling deployment.

```bash
# Build Docker image locally for testing
docker build -t TrackShellSegmentation .
docker run --env-file .env -p 8001:8001 TrackShellSegmentation
```

---

## Do NOT

- Modify database migrations directly — migrations live in golf-mapping-platform
- Add write endpoints outside of db_write.py — all DB writes go through one place
- Commit model checkpoint files (.pth) to git — store on S3 only
- Change the S3 checkpoint key schema without updating all pipeline stages
- Run training on the production ECS instance — training is offline only
- Remove the force=False checkpoint check — it prevents expensive reruns
- Change the CLASS_MAP without updating the DB feature_type enum and API
- Overwrite a course whose status is `reviewed` or `published` — the pipeline must abort unless an explicit operator-triggered force flag is set

---

## Architecture decisions reference

Full PRDs are in the /docs folder of golf-mapping-platform repo.
Key decisions already locked:

- MULTIPOLYGON not POLYGON (handles non-contiguous fairways)
- Single-tenant platform
- Soft delete for polygons in review UI
- 6 classes in v1 (rough deferred to v2)
- Cloud GPU training on Lambda Labs A100
- AWS ECS Fargate hosting
- Per-stage S3 checkpoints
- Single 1024×1024 LLM image (adaptive tiling if >20% holes flagged on large courses)
