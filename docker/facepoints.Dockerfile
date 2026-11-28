# Separate inference runtime: Facepoints 1.0.0 declares numpy<2.
FROM python:3.12-slim-bookworm
ARG TORCH_VARIANT=cu128
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    HF_HUB_OFFLINE=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=/app/models
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs ca-certificates libglib2.0-0 libgomp1 \
    && git lfs install --system && rm -rf /var/lib/apt/lists/*
RUN case "$TORCH_VARIANT" in cpu|cu128) ;; *) exit 1 ;; esac \
    && python -m pip install --no-cache-dir --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
        "torch==2.7.1+${TORCH_VARIANT}" "torchvision==0.22.1+${TORCH_VARIANT}"
RUN python -m pip install --no-cache-dir numpy==1.26.4 opencv-python-headless==4.11.0.86 \
        timm==1.0.15 safetensors==0.5.3 Pillow==11.2.1 && python -m pip check
WORKDIR /app
COPY persona_lab/facepoints_inference.py /app/facepoints_inference.py
