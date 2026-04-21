# TrackShellSegmentation — Agent Rules

Rules for AI coding agents working in this repository.
These rules override general defaults. Read before writing any code.

---

## 1. Secrets & credentials

- **Never** read, print, log, or surface any value from `.env`, AWS Parameter Store,
  or environment variables in code output, comments, or test fixtures.
- **Never** hardcode credentials, API keys, bucket names, or connection strings in
  source files. All secrets are injected at runtime via environment variables.
- **Never** commit `.env`. The only allowed secrets file is `.env.example`
  (contains key names only — no values).
- AWS credentials in code must only be read from the environment
  (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`). Never construct boto3/S3 clients
  with literal credentials.
- The `ANTHROPIC_API_KEY` is consumed by the Anthropic SDK only. Never pass it
  as a function argument or include it in LLM prompt construction.

---

## 2. Database writes

- **All** database writes go through `pipeline/db_write.py → write_to_db()`.
  Do not add DB write logic to any other file.
- **Always** use parameterized queries via asyncpg. Never use string interpolation
  or f-strings to build SQL. This is a hard SQL-injection rule.
- **Never** modify database migrations. Migrations live in the `golf-mapping-platform`
  repo. If a schema change is needed, flag it — do not implement it here.
- **Never** run `write_to_db()` on a course whose `status` is `"reviewed"` or
  `"published"` unless `force=True` is explicitly set by the operator. This protects
  human-reviewed data. If you encounter this guard, do not remove or bypass it.

---

## 3. Pipeline integrity

- **Never** remove or weaken the `force=False` checkpoint guard in any pipeline
  stage. This guard prevents expensive S3 re-downloads and re-inference.
- **Never** change the S3 checkpoint key schema (`checkpoints/{course_id}/{stage}/`)
  without updating every stage that reads or writes those paths. A partial change
  breaks resume logic silently.
- **Never** change `CLASS_MAP` (the 6-class integer mapping) without a corresponding
  update to the `feature_type` enum in the DB schema and the consumer API contract.
  These three must stay in sync.
- **Never** change the geometry type from `MULTIPOLYGON`, the CRS from WGS84
  SRID 4326, the simplification tolerance from 1.5 m, or the minimum polygon area
  from 20 m². These are locked decisions.
- **Never** change the confidence review threshold from `0.70` without explicit
  instruction. It governs which holes are flagged for human review.

---

## 4. S3 bucket separation

Two distinct S3 contexts exist. Do not mix them:

| Context | Bucket | What lives there |
|---|---|---|
| Per-course inference checkpoints | `S3_CHECKPOINT_BUCKET` env var | Stage outputs for active pipeline jobs |
| Training model storage | `s3://trackshell-models/` | Checkpoints (`/checkpoints/`) and releases (`/releases/`) from training runs |

- **Never** write training artefacts to the inference checkpoint bucket or vice versa.
- **Never** commit `.pth` model checkpoint files to git. All model files live on S3.

---

## 5. Geospatial correctness

- **Always** use a metric CRS (EPSG:3857) when computing distances or areas.
  Degree-based distance (WGS84 EPSG:4326) is not isotropic — it introduces
  significant error at Danish (~56°N) and Korean (~37°N) latitudes.
  Use `pyproj.Transformer` to project before any distance/area calculation.
- **Never** replace the `pyproj`-based projection with a flat-earth approximation
  (e.g., `111320 * degrees`). This was explicitly rejected.
- The `ADJACENCY_THRESHOLD_M = 10` metres and sequential routing threshold of
  `150 m` in `assignment.py` are tuned constants — do not change without testing
  on real course data.

---

## 6. API & input validation

- Validate all external inputs at the API boundary (`api/routes/jobs.py`):
  - `course_id` must be a valid UUID.
  - `job_type` must be an allowed enum value.
  - `force` must be a boolean, defaulting to `False`.
- **Never** pass raw user input to shell commands, subprocess calls, or file paths
  without sanitization. GDAL and rasterio calls that accept file paths are
  particularly sensitive.
- The LLM system prompt in `assignment.py` must request JSON-only output and
  must not be modified to accept free-form text. The response is `json.loads()`d
  directly; malformed output is caught and retried, not eval'd.

---

## 7. Training isolation

- Training code lives in `training/` and is **never** imported by or executed from
  the FastAPI job runner (`api/`, `pipeline/`).
- **Never** trigger a training run from the ECS production instance. Training is
  a manual offline process on Lambda Labs GPU instances.
- Training scripts write checkpoints to `s3://trackshell-models/checkpoints/` and
  final model artefacts to `s3://trackshell-models/releases/`. New model versions
  are activated by updating `MODEL_CHECKPOINT_KEY` in AWS Parameter Store —
  not by changing code.

---

## 8. Code quality

- Tests in `tests/` must pass before any change is considered complete.
  Run: `pytest tests/ -v --cov=pipeline`. Coverage must stay ≥ 70%.
- Do not add new runtime dependencies without adding them to `requirements.txt`.
- Do not add dependencies that duplicate existing ones (e.g., do not add `requests`
  when `httpx` is already available via FastAPI).
- Error handling covers only what can actually fail at system boundaries
  (S3 I/O, DB connections, LLM API calls, GDAL file reads). Do not add defensive
  handling for internal invariants.
- LLM API calls must implement exponential backoff with a maximum of 3 retries
  before raising. Do not swallow LLM errors silently.
- GPU OOM during inference triggers one automatic retry with halved batch size.
  If it fails again, raise — do not retry indefinitely.

---

## 9. Branch & deploy rules

- **Never** push directly to `main` or `dev`.
- All work must go through a `feature/*` branch → PR to `dev` → PR to `main`.
- **Never** use `--no-verify` to skip pre-commit hooks or CI checks.
- **Never** force-push to `main`.
- Docker images are built and pushed by GitHub Actions on merge. Do not build
  and push images manually to the production ECR repository.

---

## 10. What to do when uncertain

- If a change could affect the S3 checkpoint schema, the CLASS_MAP, geometry
  constants, or the DB write contract — **stop and ask** before implementing.
- If a course's `status` is `"reviewed"` or `"published"` and a pipeline run is
  requested without `force=True` — **abort the job and log a clear error message**.
  Do not silently skip or overwrite.
- If the Danish Orthophotos dataset license status is still unconfirmed, do not
  use that data for any training run. This is an open legal gate (see PRD §3.1).
