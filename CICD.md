# TrackShellSegmentation — CI/CD

## Overview

```
Push to feature/* or dev
        ↓
GitHub Actions — test.yml
  - pip install
  - pytest (must pass to merge)
        ↓
Merge to dev → deploy to staging (ECS Fargate staging cluster)
        ↓
Merge dev to main → deploy to production (ECS Fargate prod cluster)
```

---

## Branch strategy

| Branch | Purpose | Auto-deploys to |
|--------|---------|----------------|
| `main` | Production | AWS ECS Fargate (prod) |
| `dev` | Staging | AWS ECS Fargate (staging) |
| `feature/*` | Development | No auto-deploy — tests only |

**Rules:**
- Never push directly to `main` or `dev`
- All feature branches must PR to `dev` first
- `dev` → `main` PRs require at least one passing test run
- PRs are blocked from merging if tests fail

---

## GitHub Actions workflows

### test.yml — runs on every PR
```
Trigger: pull_request to dev or main
Steps:
  1. Checkout code
  2. Set up Python 3.11
  3. pip install -r requirements.txt
  4. pytest tests/ -v --cov=pipeline
  5. Fail PR if any test fails or coverage < 70%
```

### deploy.yml — runs on merge to dev or main
```
Trigger: push to dev or main
Steps:
  1. Checkout code
  2. Configure AWS credentials (from GitHub Secrets)
  3. Login to Amazon ECR
  4. Build Docker image
  5. Tag image with git SHA
  6. Push image to ECR
  7. Update ECS task definition with new image
  8. Deploy to ECS Fargate (rolling update)
  9. Wait for deployment to stabilise
  10. Run smoke test: GET /health → expect 200
```

---

## Environment variables

### GitHub Secrets (set in repo Settings → Secrets)

```
AWS_ACCESS_KEY_ID         AWS deploy user credentials
AWS_SECRET_ACCESS_KEY     AWS deploy user credentials
AWS_REGION                ap-northeast-2
ECR_REGISTRY              123456789.dkr.ecr.ap-northeast-2.amazonaws.com
ECR_REPOSITORY            TrackShellSegmentation
ECS_CLUSTER_PROD          golf-mapping-prod
ECS_CLUSTER_STAGING       golf-mapping-staging
ECS_SERVICE_PROD          TrackShellSegmentation-prod
ECS_SERVICE_STAGING       TrackShellSegmentation-staging
```

### Application secrets (stored in AWS Parameter Store)
Production secrets are never stored in GitHub. The ECS task definition
pulls them from AWS Parameter Store at runtime:

```
/golf-mapping/prod/DATABASE_URL
/golf-mapping/prod/GEMINI_API_KEY
/golf-mapping/prod/AWS_S3_BUCKET
/golf-mapping/prod/MAPBOX_TOKEN
/golf-mapping/prod/MODEL_CHECKPOINT_KEY
```

---

## Docker

```dockerfile
# Build
docker build -t TrackShellSegmentation .

# Run locally with env file
docker run --env-file .env -p 8001:8001 TrackShellSegmentation

# Check image size (keep under 4GB — PyTorch is large)
docker images TrackShellSegmentation
```

Image is based on `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`.
GDAL is installed via apt in the Dockerfile.

---

## Deployment

### Automatic (normal workflow)
Merge to `main` → GitHub Actions handles everything automatically.
Deployment takes approximately 3–5 minutes.
Monitor progress in the GitHub Actions tab.

### Manual (emergency)
```bash
# Force a new deployment without a code change
aws ecs update-service \
  --cluster golf-mapping-prod \
  --service TrackShellSegmentation-prod \
  --force-new-deployment
```

---

## Rollback

### Via AWS Console
1. Go to ECS → Clusters → golf-mapping-prod
2. Click TrackShellSegmentation-prod service
3. Click Update service
4. Select previous task definition revision
5. Click Update

### Via CLI
```bash
# List recent task definition revisions
aws ecs list-task-definitions \
  --family-prefix TrackShellSegmentation \
  --sort DESC

# Roll back to a specific revision
aws ecs update-service \
  --cluster golf-mapping-prod \
  --service TrackShellSegmentation-prod \
  --task-definition TrackShellSegmentation:42
```

---

## Monitoring

- **Logs**: AWS CloudWatch → Log groups → /ecs/TrackShellSegmentation
- **Errors**: Sentry (DSN in Parameter Store)
- **Alerts**: CloudWatch alarm on error rate > 1% or p95 latency > 10s
- **Health check**: GET https://api-internal.golfmap.io/health

---

## Smoke test after deployment

```bash
# Verify job runner is up
curl https://api-internal.golfmap.io/health

# Expected response
{ "status": "ok", "version": "x.x.x" }
```

If smoke test fails, GitHub Actions marks the deployment as failed
and triggers a Slack/email alert. Rollback manually if needed.

---

## Training deployments (separate process)

Model training is NOT part of this CI/CD pipeline.
Training runs manually on Lambda Labs A100 instances.

When a new model checkpoint is ready:
1. Upload .pth file to S3: `s3://trackshell-models/releases/`
2. Update `MODEL_CHECKPOINT_KEY` in AWS Parameter Store
3. Force a new ECS deployment to pick up the new checkpoint key
4. Test on a known course before processing new courses

---

## Contact

If the pipeline is failing and you cannot fix it:
- Check CloudWatch logs first
- Check Sentry for the full stack trace
- Check if the issue is a failed LLM API call (GEMINI_API_KEY expiry)
- Check if the issue is S3 permissions (AWS IAM role)
