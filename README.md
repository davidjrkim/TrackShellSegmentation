# TrackShellSegmentation

ML pipeline for automated golf course mapping. Takes satellite imagery
and produces numbered, typed polygon features (greens, fairways, bunkers,
tee boxes, water hazards) for each of the 18 holes on a course.

Part of the [Golf Course Mapping Platform](#related-repositories).

---

## How it works

```
Satellite imagery (GPS bounding box)
          ↓
Stage 1 — Preprocessing
  Tile download → reproject to WGS84 → chip to 512×512px
          ↓
Stage 2 — Semantic Segmentation  (DeepLabv3+ / ResNet-50)
  Classifies every pixel as: green, fairway, tee box,
  bunker, water hazard, or background
          ↓
Stage 3 — Polygon Extraction
  Stitches chips → GDAL vectorization → Douglas-Peucker simplification
  Output: typed but unlabeled GeoJSON polygons
          ↓
Stage 4 — Hole Assignment  (Vision LLM)
  Spatial graph + topology rules + Gemini vision model
  assigns hole numbers 1–18 to each polygon
          ↓
Stage 5 — Database Write
  Saves to PostgreSQL/PostGIS
  Flags low-confidence holes for human review
```

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Segmentation model | DeepLabv3+ (ResNet-50, PyTorch) |
| Hole assignment | Gemini 2.0 Flash (vision LLM) |
| Geospatial processing | GDAL, rasterio, Shapely |
| Job runner API | FastAPI |
| Database | PostgreSQL + PostGIS |
| Checkpoint storage | AWS S3 |
| Hosting | AWS ECS Fargate |

---

## Requirements

- Python 3.11+
- GDAL 3.7+
- pyproj 3.x
- CUDA-capable GPU (for training only — inference runs on CPU)
- AWS account with S3 and ECS access
- Anthropic API key

---

## Quick start

```bash
# Clone
git clone https://github.com/davidjrkim/TrackShellSegmentation
cd TrackShellSegmentation

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Start job runner
uvicorn api.main:app --reload --port 8001

# Trigger a test job
curl -X POST http://localhost:8001/jobs/run \
  -H "Content-Type: application/json" \
  -d '{"course_id": "your-course-uuid", "job_type": "full"}'
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/jobs/run` | Trigger a new pipeline job |
| `GET` | `/jobs/{id}/status` | Poll job status |
| `GET` | `/jobs/{id}/stream` | SSE live progress stream |
| `DELETE` | `/jobs/{id}` | Cancel a running job |
| `GET` | `/health` | Health check |

---

## Model classes

The segmentation model classifies pixels into 6 classes (v1):

| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | background | Everything outside golf features |
| 1 | green | Putting surface |
| 2 | fairway | Main playing corridor |
| 3 | tee_box | Starting platform |
| 4 | bunker | Sand trap |
| 5 | water_hazard | Ponds, streams, lakes |

---

## Training

Training uses the Danish Golf Courses Orthophotos dataset.
See `training/` for training scripts.

```bash
# Train on cloud GPU (Lambda Labs A100 recommended)
python training/train.py \
  --data-dir /path/to/danish-orthophotos \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --output-dir s3://trackshell-models/checkpoints/
```

Training is a separate offline process — not part of the job runner.

---

## Related repositories

- [golf-mapping-platform](https://github.com/yourusername/golf-mapping-platform) — Web dashboard for managing courses and reviewing ML output
- [golf-course-api](https://github.com/yourusername/golf-course-api) — Consumer API serving course data to the rangefinder app

---

## Documentation

Full technical specification: see PRD 1 (ML Pipeline) in the
golf-mapping-platform repo under `/docs`.

For AI agents and new developers: read `CLAUDE.md`.
For CI/CD and deployment: read `CICD.md`.

---

## License

Private — all rights reserved.
