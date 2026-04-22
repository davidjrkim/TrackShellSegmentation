import json
import os
import uuid
from collections import defaultdict

import boto3
from shapely.geometry import MultiPolygon, shape
from shapely.ops import unary_union


def _s3_client():
    return boto3.client(
        "s3",
        region_name=os.environ["AWS_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


async def write_to_db(
    course_id: str,
    assigned_key: str,
    force: bool,
    pool,
) -> None:
    row = await pool.fetchrow(
        "SELECT status FROM courses WHERE id = $1", course_id
    )
    if row and row["status"] in ("reviewed", "published") and not force:
        raise RuntimeError(
            f"Course {course_id} has status '{row['status']}' — "
            "pipeline aborted to protect human-reviewed data. "
            "Set force=True to override."
        )

    s3 = _s3_client()
    bucket = os.environ["S3_CHECKPOINT_BUCKET"]
    obj = s3.get_object(Bucket=bucket, Key=assigned_key)
    features = json.loads(obj["Body"].read())["features"]

    by_hole: dict[int, list] = defaultdict(list)
    for f in features:
        hole_num = f["properties"].get("hole_number")
        if hole_num is not None:
            by_hole[hole_num].append(f)

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Idempotent: clear existing data for this course before re-inserting
            await conn.execute(
                "DELETE FROM features WHERE hole_id IN "
                "(SELECT id FROM holes WHERE course_id = $1)",
                course_id,
            )
            await conn.execute(
                "DELETE FROM holes WHERE course_id = $1", course_id
            )

            for hole_number in sorted(by_hole):
                hole_feats = by_hole[hole_number]
                hole_id = str(uuid.uuid4())

                tees = [f for f in hole_feats if f["properties"]["feature_type"] == "tee_box"]
                greens = [f for f in hole_feats if f["properties"]["feature_type"] == "green"]
                confidence = hole_feats[0]["properties"].get("confidence", 0.0)
                needs_review = hole_feats[0]["properties"].get("needs_review", True)

                tee_ewkt = None
                if tees:
                    c = shape(tees[0]["geometry"]).centroid
                    tee_ewkt = f"SRID=4326;POINT({c.x} {c.y})"

                green_ewkt = None
                if greens:
                    c = shape(greens[0]["geometry"]).centroid
                    green_ewkt = f"SRID=4326;POINT({c.x} {c.y})"

                await conn.execute(
                    """
                    INSERT INTO holes
                        (id, course_id, hole_number, confidence, needs_review,
                         tee_centroid, green_centroid)
                    VALUES ($1, $2, $3, $4, $5,
                            ST_GeomFromEWKT($6), ST_GeomFromEWKT($7))
                    """,
                    hole_id,
                    course_id,
                    hole_number,
                    confidence,
                    needs_review,
                    tee_ewkt,
                    green_ewkt,
                )

                by_type: dict[str, list] = defaultdict(list)
                for f in hole_feats:
                    by_type[f["properties"]["feature_type"]].append(f)

                for ftype, type_feats in by_type.items():
                    geoms = [shape(f["geometry"]) for f in type_feats]
                    merged = unary_union(geoms)
                    if merged.geom_type == "Polygon":
                        merged = MultiPolygon([merged])
                    geom_ewkt = f"SRID=4326;{merged.wkt}"
                    feat_confidence = max(
                        f["properties"].get("confidence", 0.0) for f in type_feats
                    )

                    await conn.execute(
                        """
                        INSERT INTO features
                            (id, hole_id, feature_type, geometry, confidence)
                        VALUES ($1, $2, $3, ST_GeomFromEWKT($4), $5)
                        """,
                        str(uuid.uuid4()),
                        hole_id,
                        ftype,
                        geom_ewkt,
                        feat_confidence,
                    )

            await conn.execute(
                "UPDATE courses SET status = 'assigned' WHERE id = $1",
                course_id,
            )
