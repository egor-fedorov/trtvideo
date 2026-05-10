FROM nvcr.io/nvidia/tensorrt:26.03-py3

ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY upscaler/ upscaler/
COPY benchmark.py inference.py inference_gpu.py ./
COPY tools/ tools/

ARG INSTALL_DEV=0
RUN if [ "$INSTALL_DEV" = "1" ]; then \
        pip install --no-cache-dir ".[docker,dev]"; \
    else \
        pip install --no-cache-dir ".[docker]"; \
    fi

ENTRYPOINT []
