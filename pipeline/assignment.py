import asyncio
import base64
import io
import json
import os

import boto3
import google.generativeai as genai
from google.api_core import exceptions as gapi_exceptions
import networkx as nx
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform

ADJACENCY_THRESHOLD_M = 10
REVIEW_THRESHOLD = 0.70

FEATURE_COLORS = {
    "green": (0, 200, 50),
    "fairway": (200, 200, 0),
    "tee_box": (220, 50, 50),
    "bunker": (210, 180, 140),
    "water_hazard": (30, 100, 220),
}
IMG_SIZE = 1024
MARGIN = 40

_to_metric = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

# Double-brace the JSON output template to escape .format() interpolation
SYSTEM_PROMPT_TEMPLATE = """\
You are a golf course cartographer. You will receive:
  1. A satellite-derived overhead map of a golf course with colour-coded \
feature polygons (green=greens, yellow=fairways, red=tee boxes, \
tan=bunkers, blue=water hazards). Each polygon is labelled with the first \
6 characters of its UUID.
  2. A JSON object listing polygon types and adjacency edges.

Your task: assign hole numbers 1 through {n_holes} to the polygons.

Rules:
  - Each hole must have exactly 1 tee box and 1 green.
  - Routing order should follow a logical course flow.
  - Hole 1 tee should be nearest the approximate clubhouse location.
  - Return ONLY valid JSON. No preamble, no commentary.

Output format:
{{
  "holes": [
    {{
      "hole_number": 1,
      "tee_box_id": "full-uuid",
      "green_id": "full-uuid",
      "fairway_ids": ["full-uuid"],
      "other_ids": ["full-uuid"],
      "confidence": 0.95
    }}
  ]
}}"""


def _s3_client():
    return boto3.client(
        "s3",
        region_name=os.environ["AWS_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def _checkpoint_exists(bucket: str, key: str) -> bool:
    s3 = _s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def _build_spatial_graph(features: list) -> nx.Graph:
    G = nx.Graph()
    nodes = [
        (
            f["id"],
            transform(_to_metric, shape(f["geometry"])),
            f["properties"]["feature_type"],
        )
        for f in features
    ]
    for feat_id, geom, ftype in nodes:
        G.add_node(feat_id, geom=geom, feature_type=ftype)
    for i, (id_a, geom_a, _) in enumerate(nodes):
        for id_b, geom_b, _ in nodes[i + 1:]:
            if geom_a.distance(geom_b) < ADJACENCY_THRESHOLD_M:
                G.add_edge(id_a, id_b)
    return G


def _graph_to_json(G: nx.Graph) -> str:
    return json.dumps({
        "nodes": [
            {"id": n, "feature_type": G.nodes[n]["feature_type"]}
            for n in G.nodes
        ],
        "edges": [{"from": u, "to": v} for u, v in G.edges],
    })


def _render_composite(features: list) -> str:
    """Render all polygon features onto a 1024×1024 PNG, return base64-encoded string."""
    if not features:
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (50, 50, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    all_lons, all_lats = [], []
    for f in features:
        geom = shape(f["geometry"])
        b = geom.bounds  # (min_lon, min_lat, max_lon, max_lat)
        all_lons.extend([b[0], b[2]])
        all_lats.extend([b[1], b[3]])

    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    lon_range = max_lon - min_lon or 1e-6
    lat_range = max_lat - min_lat or 1e-6
    draw_w = IMG_SIZE - 2 * MARGIN
    draw_h = IMG_SIZE - 2 * MARGIN

    def to_px(lon: float, lat: float) -> tuple[int, int]:
        x = MARGIN + (lon - min_lon) / lon_range * draw_w
        y = MARGIN + (max_lat - lat) / lat_range * draw_h
        return (int(x), int(y))

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    for f in features:
        ftype = f["properties"]["feature_type"]
        color = FEATURE_COLORS.get(ftype, (128, 128, 128))
        geom = shape(f["geometry"])
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            coords = [to_px(lon, lat) for lon, lat in poly.exterior.coords]
            if len(coords) >= 3:
                draw.polygon(coords, fill=color, outline=(255, 255, 255))
        centroid = geom.centroid
        draw.text(to_px(centroid.x, centroid.y), f["id"][:6], fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _score_and_flag(hole: dict, feature_index: dict) -> tuple[float, bool]:
    """
    Compute a per-hole confidence score and review flag.
    Penalises missing tee (-0.40), missing green (-0.40), and a tee-to-green
    distance > 150 m with no fairway (-0.20).
    """
    score = float(hole.get("confidence", 0.5))

    tee_id = hole.get("tee_box_id")
    green_id = hole.get("green_id")
    fairway_ids = hole.get("fairway_ids") or []
    has_tee = bool(tee_id and tee_id in feature_index)
    has_green = bool(green_id and green_id in feature_index)

    if not has_tee:
        score -= 0.40
    if not has_green:
        score -= 0.40

    if has_tee and has_green and not fairway_ids:
        tee_m = transform(_to_metric, shape(feature_index[tee_id]["geometry"]))
        green_m = transform(_to_metric, shape(feature_index[green_id]["geometry"]))
        if tee_m.centroid.distance(green_m.centroid) > 150:
            score -= 0.20

    score = round(max(score, 0.0), 3)
    return score, score < REVIEW_THRESHOLD


async def _call_llm_with_retry(image_b64: str, graph_json: str, n_holes: int) -> dict:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT_TEMPLATE.format(n_holes=n_holes),
    )
    image_bytes = base64.b64decode(image_b64)
    last_exc: Exception = RuntimeError("LLM call failed with no attempts")

    for attempt in range(3):
        try:
            resp = await model.generate_content_async([
                {"mime_type": "image/png", "data": image_bytes},
                f"Spatial graph:\n{graph_json}",
            ])
            return json.loads(resp.text)
        except (
            gapi_exceptions.ResourceExhausted,
            gapi_exceptions.DeadlineExceeded,
            gapi_exceptions.InternalServerError,
        ) as exc:
            last_exc = exc
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)

    raise last_exc


async def assign_holes(course_id: str, raw_geojson_key: str, force: bool = False) -> str:
    bucket = os.environ["S3_CHECKPOINT_BUCKET"]
    out_key = f"checkpoints/{course_id}/polygons_assigned.geojson"

    if not force and _checkpoint_exists(bucket, out_key):
        return out_key

    s3 = _s3_client()
    obj = s3.get_object(Bucket=bucket, Key=raw_geojson_key)
    geojson = json.loads(obj["Body"].read())
    features = geojson["features"]

    G = _build_spatial_graph(features)
    graph_json = _graph_to_json(G)
    image_b64 = _render_composite(features)

    n_holes = sum(1 for f in features if f["properties"]["feature_type"] == "green")
    n_holes = max(1, min(n_holes, 18))

    llm_result = await _call_llm_with_retry(image_b64, graph_json, n_holes)

    feature_index = {f["id"]: f for f in features}
    assigned: dict[str, dict] = {}

    for hole in llm_result.get("holes", []):
        confidence, needs_review = _score_and_flag(hole, feature_index)
        hole_num = hole.get("hole_number")
        candidate_ids = (
            [hole.get("tee_box_id"), hole.get("green_id")]
            + (hole.get("fairway_ids") or [])
            + (hole.get("other_ids") or [])
        )
        for fid in candidate_ids:
            if fid and fid in feature_index:
                assigned[fid] = {
                    "hole_number": hole_num,
                    "confidence": confidence,
                    "needs_review": needs_review,
                }

    for f in features:
        annotation = assigned.get(
            f["id"],
            {"hole_number": None, "confidence": 0.0, "needs_review": True},
        )
        f["properties"].update(annotation)

    body = json.dumps({"type": "FeatureCollection", "features": features}).encode()
    s3.put_object(Bucket=bucket, Key=out_key, Body=body)
    return out_key
