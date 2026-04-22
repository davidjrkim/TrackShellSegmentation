FROM python:3.11-slim-bookworm

# GDAL system library — Python wheel must be built against the exact same version.
# See also: .github/workflows/test.yml which uses the same gdal-config approach.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_CONFIG=/usr/bin/gdal-config

WORKDIR /app

# PyTorch CPU-only — ECS Fargate has no GPU.
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# numpy must be present before building the GDAL Python extension (gdal_array).
# GDAL and torch/torchvision are handled above; skip them here.
RUN pip install --no-cache-dir numpy==1.26.4 && \
    GDAL_VER=$(gdal-config --version) && \
    pip install --no-cache-dir "GDAL==${GDAL_VER}" --no-binary :gdal: --no-build-isolation && \
    grep -vE '^(GDAL|torch|torchvision)==' requirements.txt | pip install --no-cache-dir -r /dev/stdin

COPY . .

EXPOSE 8001

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
