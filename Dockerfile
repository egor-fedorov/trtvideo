FROM nvcr.io/nvidia/tensorrt:26.03-py3

ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY upscaler/ upscaler/
COPY inference.py inference_gpu.py ./
COPY tools/ tools/

RUN pip install --no-cache-dir ".[docker]"

ENTRYPOINT []
