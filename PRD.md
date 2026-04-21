# 🛰️ TrackShell Course Mapping Platform — PRD 1: ML Pipeline

**Version:** 1.0 | **April 2026**

| Field | Value |
|---|---|
| Status | 🟡 Draft |
| Owner | David |
| Core Stack | Python 3.11 · PyTorch 2 · GDAL · rasterio · FastAPI |
| Model | DeepLabv3+ (ResNet-50 backbone) |
| Hole Assignment | Vision LLM (`claude-sonnet-4-20250514`) |
| Depends On | PRD 2c (Database Schema) |
| Last Updated | April 2026 |

---

## 1. Purpose & Scope

This PRD defines the end-to-end ML pipeline that takes raw satellite imagery of a golf course and produces a set of numbered, typed, geospatially accurate polygon features saved to the platform database. It covers data preparation, model architecture, training strategy, inference, hole assignment, post-processing, and the job runner that orchestrates all steps.

**Scope:**
- ✅ Covers: data pipeline, model spec, training config, inference API, hole assignment logic, output contract.
- ❌ Does not cover: the web dashboard UI, the consumer-facing API, or database migrations (see PRD 2c).

---

## 2. Pipeline Overview

The pipeline runs as an async background job triggered per course. It has two distinct ML stages — semantic segmentation and hole assignment — plus pre/post-processing steps around each.

```
INPUT:  GPS bounding box  →  course_id  →  satellite tile URL(s)

┌─────────────────────────────────────────────────────────┐
│  STAGE 1 — Preprocessing                                │
│  Download tiles → reproject to WGS84 → chip to 512×512 │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 2 — Semantic Segmentation  (DeepLabv3+)          │
│  Input:  512×512 RGB chips                              │
│  Output: 512×512 class mask (7 classes)                 │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 3 — Polygon Extraction                           │
│  Stitch masks → vectorize → Douglas-Peucker simplify    │
│  Output: unlabeled GeoJSON polygons                     │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 4 — Hole Assignment  (Vision LLM)                │
│  Build spatial graph → apply topology rules →           │
│  LLM assigns hole 1–18 → confidence scoring             │
│  Output: labeled GeoJSON with hole numbers              │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 5 — DB Write                                     │
│  Normalize → insert features → update course status     │
│  Flag low-confidence holes for human review             │
└─────────────────────────────────────────────────────────┘

OUTPUT: Populated holes + features tables in PostgreSQL/PostGIS
```

---

## 3. Training Data

### 3.1 Primary Dataset — Danish Golf Courses Orthophotos

The Danish Orthophotos dataset is the foundation of training. It provides high-resolution aerial imagery of Danish golf courses with pixel-level semantic annotations.

| Property | Detail |
|---|---|
| Source | Danish Geodata Agency (Styrelsen for Dataforsyning og Infrastruktur) |
| Resolution | ~10–20 cm/pixel GSD (ground sample distance) |
| Format | GeoTIFF + accompanying mask GeoTIFF |
| Mask encoding | **RGB colour values** — not integer class IDs. Each annotated pixel has a specific RGB colour per class (see §3.2). Preprocessing must convert RGB → integer class ID before training. |
| CRS | EPSG:25832 (UTM Zone 32N) — must reproject to WGS84 |
| Annotation | Pixel-level semantic masks per feature class |
| Coverage | Multiple Danish golf courses — exact count to be confirmed in dataset audit |
| License | **ODbL v1.0** — Aalborg University, 2023. Commercial use permitted. Share-alike applies to the database only; trained model weights are a produced work and may remain proprietary. |

### 3.2 Class Definitions

The v1 model outputs one of **6 classes** per pixel (Decision 5 — `rough` deferred to v2). Class 0 is background.

The dataset mask encodes classes as **RGB colour values**. The table below documents the audited mask colours alongside the integer class IDs assigned during preprocessing. The dataset uses the label `tee` — this maps to `tee_box` in the platform schema.

| Class ID | Platform Label | Dataset Label | Mask RGB | Hex | Notes |
|---|---|---|---|---|---|
| 0 | `background` | *(unlabeled)* | — | — | All pixels not matching a class colour. Paths, car parks, trees, buildings, everything outside course features. |
| 1 | `green` | `green` | (142, 243, 122) | `#8EF37A` | Putting surface — typically smallest, most manicured polygon |
| 2 | `fairway` | `fairway` | (77, 156, 77) | `#4D9C4D` | Main playing corridor — largest polygon per hole |
| 3 | `tee_box` | `tee` | (250, 36, 0) | `#FA2400` | Elevated starting platform — small, often rectangular. Dataset uses `tee`; mapped to `tee_box`. |
| 4 | `bunker` | `bunker` | (246, 246, 158) | `#F6F69E` | Sand trap — visually distinct (bright/pale colour) |
| 5 | `water_hazard` | `water` | (46, 200, 231) | `#2EC8E7` | Ponds, streams, lakes — may be irregular shaped. Dataset uses `water`; mapped to `water_hazard`. |
| ~~6~~ | ~~`rough`~~ | *(absent)* | — | — | **Deferred to v2** — not present in dataset; adds training complexity for low rangefinder value. Model output channels = 6 in v1. |

### 3.2.1 RGB → Class ID Conversion

The preprocessing stage must decode the RGB mask GeoTIFF into an integer class ID mask before training. Exact RGB matching is used (no tolerance) — the dataset colours are distinct enough that fuzzy matching is unnecessary.

```python
import numpy as np

# Audited mask colour → class ID mapping
RGB_TO_CLASS = {
    (142, 243, 122): 1,  # green
    ( 77, 156,  77): 2,  # fairway
    (250,  36,   0): 3,  # tee_box  (dataset label: "tee")
    (246, 246, 158): 4,  # bunker
    ( 46, 200, 231): 5,  # water_hazard  (dataset label: "water")
}

def rgb_mask_to_class_ids(rgb_mask: np.ndarray) -> np.ndarray:
    """Convert H×W×3 uint8 RGB mask to H×W int64 class ID mask.
    Unmatched pixels (background) map to class 0.
    """
    h, w, _ = rgb_mask.shape
    class_mask = np.zeros((h, w), dtype=np.int64)  # default = 0 (background)
    for rgb, class_id in RGB_TO_CLASS.items():
        match = np.all(rgb_mask == np.array(rgb, dtype=np.uint8), axis=-1)
        class_mask[match] = class_id
    return class_mask
```

### 3.3 Train / Val / Test Split

Split must be done **per course**, not per tile. Splitting by tile causes data leakage — adjacent tiles from the same course share visual context and will inflate validation metrics artificially.

| Split | Proportion | Notes |
|---|---|---|
| Train | 70% | Used for gradient updates |
| Validation | 15% | Used for hyperparameter tuning and early stopping |
| Test | 15% | Held out — evaluated once after final model selection only |

### 3.4 Tiling Strategy

Full orthophoto tiles are too large to feed directly into the model. Each image is sliced into fixed 512×512 pixel chips with a 64-pixel overlap to avoid boundary artefacts. During inference, overlapping predictions are averaged before vectorization.

```python
# Pseudocode — tiling with overlap
CHIP_SIZE    = 512   # pixels
OVERLAP      = 64    # pixels
STRIDE       = CHIP_SIZE - OVERLAP  # = 448

for y in range(0, image_height - CHIP_SIZE, STRIDE):
    for x in range(0, image_width - CHIP_SIZE, STRIDE):
        chip = image[y:y+CHIP_SIZE, x:x+CHIP_SIZE]
        mask = label[y:y+CHIP_SIZE, x:x+CHIP_SIZE]
        save(chip, mask, metadata={x, y, crs_transform})
```

### 3.5 Class Imbalance Handling

Fairways dominate pixel count per course; tee boxes and greens are tiny. Without correction the model learns to predict fairway everywhere. Two strategies are combined:

- **Weighted loss:** per-class weights inversely proportional to pixel frequency computed from training set
- **Oversampling:** tiles containing underrepresented classes (`green`, `tee_box`, `bunker`) are sampled 2–3× more frequently per epoch

---

## 4. Model Architecture — DeepLabv3+

### 4.1 Why DeepLabv3+

DeepLabv3+ is chosen over U-Net and plain DeepLabv3 for this task for three reasons:
1. Its atrous (dilated) convolutions capture multi-scale context simultaneously — essential when a single forward pass must handle a wide fairway and a small tee box.
2. Its encoder-decoder with skip connections preserves fine boundary detail.
3. Its ImageNet-pretrained ResNet-50 backbone provides strong feature extraction with minimal training data, which matters given the limited size of the Danish dataset.

### 4.2 Architecture Specification

| Component | Configuration |
|---|---|
| Backbone | ResNet-50 pretrained on ImageNet (freeze first 2 stages during warm-up) |
| ASPP rates | 6, 12, 18  (atrous spatial pyramid pooling) |
| Output stride | 16 (balance between receptive field and spatial resolution) |
| Decoder | Low-level features from backbone layer 1 + ASPP output, upsampled 4× |
| Final upsample | 4× bilinear interpolation to match input resolution |
| Output channels | 6 (one per class including background; `rough` deferred to v2 — Decision 5) |
| Input size | 512 × 512 × 3 (RGB) |
| Framework | PyTorch 2.x via `torchvision.models.segmentation` |

### 4.3 Loss Function

Combined Dice Loss and Cross-Entropy Loss. Dice Loss handles class imbalance by optimising the overlap ratio directly. Cross-Entropy provides stable gradients early in training. The combination is weighted 0.5/0.5 and can be tuned during validation.

```python
def combined_loss(pred, target, weights, alpha=0.5):
    ce   = F.cross_entropy(pred, target, weight=weights)
    dice = dice_loss(pred.softmax(dim=1), F.one_hot(target, 6).permute(0,3,1,2).float())
    return alpha * ce + (1 - alpha) * dice
```

### 4.4 Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Batch size | 8 | Fits in 16GB VRAM; larger batches improve BN stability |
| Learning rate | 1e-4 | Conservative start; backbone LR = 1e-5 (10× lower) |
| LR schedule | PolyLR | Smooth decay common in segmentation literature |
| Epochs | 50 max | Early stopping on val mIoU with patience=8 |
| Optimizer | AdamW | Weight decay 1e-4 |
| Augmentation | See §4.5 | |
| Mixed precision | Yes (fp16) | Speed + memory via `torch.cuda.amp` |
| Gradient clip | 1.0 | Prevent exploding gradients |

### 4.5 Data Augmentation

Applied randomly during training only. Validation and test sets use no augmentation.

- Random horizontal + vertical flip (p=0.5 each)
- Random rotation ±30°
- Random crop and resize (scale 0.75–1.25)
- Color jitter: brightness ±0.2, contrast ±0.2, saturation ±0.1
- Gaussian blur (p=0.2) — simulates lower resolution imagery
- Seasonal colour shift: hue rotation ±15° — simulates different grass conditions

### 4.6 Evaluation Metrics

| Metric | Target | Notes |
|---|---|---|
| mIoU (mean Intersection over Union) | ≥ 0.72 | Primary metric. Averaged across all 6 v1 classes (no `rough`) |
| IoU — green | ≥ 0.80 | Critical for rangefinder accuracy |
| IoU — fairway | ≥ 0.78 | Largest class, should be easiest |
| IoU — tee_box | ≥ 0.70 | Small class, hardest to segment |
| IoU — bunker | ≥ 0.72 | Visually distinct but irregular shape |
| IoU — water | ≥ 0.75 | Usually clear boundary |
| Inference time | < 8s per course tile | On GPU; acceptable for batch processing |

---

## 5. Polygon Extraction (Stage 3)

### 5.1 Mask Stitching

After inference, the overlapping 512×512 prediction chips are stitched back into a full-resolution mask. In overlap zones, per-pixel class probabilities are averaged across contributing chips before taking the argmax. This smooths boundary artefacts at chip edges.

### 5.2 Vectorization

The stitched integer class mask is vectorized using GDAL's `Polygonize` function. This converts runs of same-class pixels into vector polygons with pixel-accurate boundaries. Output is a GeoJSON `FeatureCollection` in WGS84.

```python
# Vectorize a single class from the full mask
from osgeo import gdal, ogr

def vectorize_class(mask_path, class_id, out_geojson):
    ds   = gdal.Open(mask_path)
    band = ds.GetRasterBand(1)

    # Create binary mask for this class
    binary = (band.ReadAsArray() == class_id).astype("uint8")
    mem_ds = gdal.GetDriverByName("MEM").Create("", ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
    mem_ds.SetGeoTransform(ds.GetGeoTransform())
    mem_ds.SetProjection(ds.GetProjection())
    mem_ds.GetRasterBand(1).WriteArray(binary)

    out_ds  = ogr.GetDriverByName("GeoJSON").CreateDataSource(out_geojson)
    out_lyr = out_ds.CreateLayer("features")
    gdal.Polygonize(mem_ds.GetRasterBand(1), None, out_lyr, -1)
```

### 5.3 Simplification — Douglas-Peucker

Raw GDAL polygons have jagged pixel-aligned boundaries unsuitable for a rangefinder app. Douglas-Peucker simplification reduces vertex count while preserving overall shape. Epsilon (tolerance) is set to 1.5 metres.

```python
from shapely.geometry import shape
from shapely.ops import unary_union, transform
from pyproj import Transformer

SIMPLIFY_TOLERANCE_M = 1.5  # metres
MIN_AREA_SQM         = 20   # drop polygons smaller than 20 m² (noise)

# Project WGS84 → local metric CRS for accurate distance/area math.
# 111320 m/degree is only valid at the equator; Danish (~56°N) and Korean
# (~37°N) courses have significant east-west scale compression without this.
_to_metric   = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
_to_wgs84    = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform

def simplify_features(geojson_features):
    cleaned = []
    for feat in geojson_features:
        geom = shape(feat["geometry"])
        geom_m = transform(_to_metric, geom)               # project to metres
        geom_m = geom_m.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)
        if geom_m.area < MIN_AREA_SQM:
            continue                                        # discard noise polygons
        geom = transform(_to_wgs84, geom_m)                # back to WGS84 for storage
        cleaned.append({**feat, "geometry": geom.__geo_interface__})
    return cleaned
```

### 5.4 Polygon Geometry Type

> **DECISION:** Store as `MULTIPOLYGON` in PostGIS. Some fairways wrap around obstacles and produce non-contiguous shapes. `MULTIPOLYGON` handles both simple and complex cases without schema changes later.

---

## 6. Hole Assignment (Stage 4)

### 6.1 Overview

After extraction, polygons are typed (`green`, `fairway`, etc.) but carry no hole number. Stage 4 assigns hole numbers 1–18 using a two-step approach: first, hard topology rules eliminate most impossible assignments; second, a vision LLM resolves ambiguous cases and assigns the full routing.

### 6.2 Step 1 — Spatial Graph Construction

All polygons are loaded as nodes. Edges connect polygons that are spatially adjacent (share a boundary or are within 10 metres of each other).

```python
import networkx as nx
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer

ADJACENCY_THRESHOLD_M = 10  # metres

# Distance must be computed in a metric CRS — degree-based distance is
# not isotropic and varies by ~20% between equator and 56°N (Denmark).
_to_metric = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

def build_spatial_graph(features):
    G = nx.Graph()
    geoms = [
        (f["id"], transform(_to_metric, shape(f["geometry"])), f["feature_type"])
        for f in features
    ]
    for i, (id_a, geom_a, type_a) in enumerate(geoms):
        G.add_node(id_a, geom=geom_a, feature_type=type_a)
        for id_b, geom_b, type_b in geoms[i+1:]:
            if geom_a.distance(geom_b) < ADJACENCY_THRESHOLD_M:
                G.add_edge(id_a, id_b)
    return G
```

### 6.3 Step 2 — Golf Topology Rules (Hard Constraints)

Before calling the LLM, deterministic rules prune impossible assignments.

| Rule | Logic |
|---|---|
| 1 green per hole | If a candidate hole group already has a green, reject any additional green polygon |
| 1 tee box per hole | Same as above for tee boxes |
| Greens are small | Discard green candidates with area > 1,000 m² |
| Tee boxes are small | Discard tee box candidates with area > 500 m² |
| Fairway connects tee→green | Fairway polygon must be spatially between tee and green centroids |
| No shared fairways | A fairway polygon cannot belong to two different holes |
| Sequential routing | Hole N green centroid should be near hole N+1 tee centroid (within 150 m) |

### 6.4 Step 3 — Vision LLM Assignment

The full-course polygon overlay is rendered as a 1024×1024 PNG with colour-coded feature types and polygon ID labels. This image plus the spatial graph data is sent to the vision LLM for final hole number assignment.

```python
SYSTEM_PROMPT = """
You are a golf course cartographer. You will receive:
  1. A satellite-derived overhead map of a golf course with colour-coded
     feature polygons (green=greens, yellow=fairways, red=tee boxes,
     tan=bunkers, blue=water hazards).
  2. A JSON spatial graph of polygon adjacency.

Your task: assign hole numbers 1 through {n_holes} to the polygons.

Rules:
  - Each hole must have exactly 1 tee box and 1 green.
  - Routing order should follow a logical course flow.
  - Hole 1 tee should be closest to the approximate clubhouse location.
  - Return ONLY valid JSON. No preamble.

Output format:
{
  "holes": [
    {
      "hole_number": 1,
      "tee_box_id": "uuid",
      "green_id": "uuid",
      "fairway_ids": ["uuid", ...],
      "other_ids": ["uuid", ...],
      "confidence": 0.95
    }, ...
  ]
}
"""
```

### 6.5 Confidence Scoring & Review Flagging

Each hole receives a confidence score from the LLM response. Additional programmatic checks downgrade the score if topology rules are violated. Holes below 0.70 are flagged for human review.

```python
REVIEW_THRESHOLD = 0.70

def score_and_flag(hole_assignment, spatial_graph):
    score = hole_assignment["confidence"]   # from LLM

    # Penalise rule violations in LLM output
    if not has_tee(hole_assignment):   score -= 0.40
    if not has_green(hole_assignment): score -= 0.40
    if fairway_gap(hole_assignment, spatial_graph) > 150:
        score -= 0.20

    needs_review = score < REVIEW_THRESHOLD
    return round(max(score, 0.0), 3), needs_review
```

---

## 7. Job Runner

### 7.1 Architecture

The pipeline runs as a FastAPI background task, triggered via HTTP POST from the web dashboard. Jobs are tracked in the `pipeline_jobs` table (see PRD 2c). Long-running jobs stream status updates via server-sent events (SSE) to the dashboard.

```
POST /api/jobs/run
Body:     { "course_id": "uuid", "job_type": "full" }
Response: { "job_id": "uuid", "status": "queued" }

GET  /api/jobs/{job_id}/status     # polling fallback
GET  /api/jobs/{job_id}/stream     # SSE live updates
```

### 7.2 Job Stages & Status Transitions

```
queued
  → running (preprocessing started)
  → running (segmentation in progress)
  → running (polygon extraction)
  → running (hole assignment)
  → running (db write)
  → completed  (course.status = "assigned")
  → failed     (course.status = "failed", error_message logged)
```

### 7.3 Error Handling

- Any unhandled exception sets job status to `failed` and logs full traceback to `pipeline_jobs.error_message`
- Failed jobs can be re-triggered from the dashboard without re-running already-completed stages — per-stage checkpoints are stored on **AWS S3** at `checkpoints/{course_id}/{stage_name}/` (Decision 16). Stages 1–4 each check for an existing checkpoint before running. A `force=True` flag triggers a full rerun regardless. Checkpoints are auto-deleted when the course reaches `published` status.
- GPU OOM errors trigger automatic batch size halving and retry once before failing
- LLM API errors (rate limit, timeout) retry with exponential backoff up to 3 attempts

---

## 8. Output Contract

The pipeline's final output is a set of database writes. The consumer API and dashboard read from the DB, not directly from the pipeline.

| Table | What gets written |
|---|---|
| `pipeline_jobs` | Status updated to `completed` or `failed`; timing and stats recorded |
| `holes` | 18 rows inserted (or updated) with `hole_number`, tee/green centroids, confidence score, `needs_review` flag |
| `features` | N rows inserted — one per polygon — with `hole_id`, `feature_type`, geometry (PostGIS), `confidence_score` |
| `courses` | `status` updated to `assigned` (or `failed`) |

> **NOTE:** The pipeline never updates a course whose status is already `"reviewed"` or `"published"` without an explicit operator-triggered re-run flag. This protects human-reviewed data from being overwritten.

---

## 9. Training Environment

Training runs on **Lambda Labs cloud GPU** (Decision 6). Lambda Labs provides on-demand NVIDIA A100/H100 instances with no minimum commitment. The training job is packaged as a Docker container (built and pushed via GitHub Actions to AWS ECR) and launched on Lambda. Model checkpoints are written to AWS S3 during training.

| Resource | Choice |
|---|---|
| GPU provider | Lambda Labs (cloud on-demand) |
| Instance type | 1× A100 (80GB) for initial run; scale to 8× if epoch time unacceptable |
| Checkpoint storage | AWS S3 — `s3://trackshell-models/checkpoints/` |
| Model artefact storage | AWS S3 — `s3://trackshell-models/releases/` |

---

## 10. Environment & Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.11 | Runtime |
| PyTorch | 2.x | Model training and inference |
| torchvision | 0.16+ | DeepLabv3+ model definition |
| GDAL | 3.7+ | Raster I/O and vectorization |
| rasterio | 1.3+ | GeoTIFF reading, CRS handling |
| Shapely | 2.x | Polygon simplification and spatial ops |
| NetworkX | 3.x | Spatial graph construction |
| FastAPI | 0.110+ | Job runner HTTP server |
| asyncpg | latest | Async PostgreSQL driver |
| anthropic SDK | latest | Vision LLM API calls |
| numpy | 1.26+ | Array operations |
| Pillow | 10+ | Overlay image rendering for LLM input |
| pyproj | 3.x | CRS projection for accurate metric distance/area calculations |

---

## 11. Decisions Applied

All open questions from this PRD are resolved. Decisions are recorded in the Decisions Tracker.

| # | Question | Decision | Tracker |
|---|---|---|---|
| 1 | Class annotation schema + license for the Danish dataset | Dataset confirmed: `rough` (class 6) is the only missing label. License confirmed: ODbL v1.0 (Aalborg University, 2023) — commercial use permitted, model weights may be proprietary. Training gate is clear. | Decision 4 |
| 2 | GPU availability: cloud or local? | **Lambda Labs cloud GPU** for v1. See §9. | Decision 6 |
| 3 | Include `rough` in v1 or defer to v2? | **Deferred to v2.** V1 model is 6-class. `rough` absent from training data and low rangefinder value. **Product consequence:** players in rough will return `located_on: null` from the `/locate` endpoint — consumer apps must handle this as a normal game state. See PRD 3 §5.5. | Decision 5 |
| 4 | Checkpoint resume: which stages? | **Per-stage S3 checkpoints for stages 1–4.** See §7.3. | Decision 16 |
| 5 | LLM vision input: single 1024×1024 or tiled for large courses? | **Single 1024×1024 for v1.** Test on a real large course; tile if detail is insufficient. | Decision 17 |

---

*TrackShell Course Mapping Platform · PRD 1: ML Pipeline · v1.0*
