# syntax=docker/dockerfile:1.7

ARG UV_IMAGE=ghcr.io/astral-sh/uv@sha256:78a7ff97cd27b7124a5f3c2aefe146170793c56a1e03321dd31a289f6d82a04f
FROM ${UV_IMAGE} AS uv
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ARG TAS_REVISION=ac259ddf13c191230a4c65a4b251c9fb28884104
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300 \
    PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git ffmpeg libglib2.0-0 libgomp1
RUN git init /opt/tas && git -C /opt/tas remote add origin https://github.com/NevermindNilas/TheAnimeScripter && \
    git -C /opt/tas fetch --depth 1 origin "${TAS_REVISION}" && \
    git -C /opt/tas checkout --detach FETCH_HEAD && \
    test "$(git -C /opt/tas rev-parse HEAD)" = "${TAS_REVISION}"
COPY benchmarks/docker/tas-requirements.txt /tmp/tas-requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv python install 3.14.2 && uv venv /opt/tas-runtime --python 3.14.2 && \
    uv pip install --python /opt/tas-runtime/bin/python -r /tmp/tas-requirements.txt
ENV PATH=/opt/tas-runtime/bin:${PATH} PYTHONPATH=/app:/app/src TRTVIDEO_TAS_ROOT=/opt/tas
# Install TAS's own hash-verified FFmpeg during the build, never during a run.
RUN cd /opt/tas && python -c "import src.constants as c; c.SYSTEM='Linux'; c.WHEREAMIRUNFROM='/opt/tas'; c.FFMPEGPATH='/opt/tas/ffmpeg_shared/ffmpeg'; from src.infra.getFFMPEG import getFFMPEG; getFFMPEG()" && \
    python -c "import torch, nelux, tensorrt; print(torch.__version__, nelux.__version__, tensorrt.__version__)" && \
    touch /opt/tas/TAS-Log.log /opt/tas/metadata.json && \
    chmod 666 /opt/tas/TAS-Log.log /opt/tas/metadata.json && \
    mkdir -p /opt/tas/output /opt/tas/weights && chmod 777 /opt/tas/output /opt/tas/weights
WORKDIR /app
COPY src/ src/
COPY benchmarks/ benchmarks/
ARG VCS_REF=unknown
ARG VCS_DIRTY=unknown
ENV TRTVIDEO_TAS_REVISION=${TAS_REVISION} \
    TRTVIDEO_BASE_IMAGE=ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 \
    TRTVIDEO_BUILD_REVISION=${VCS_REF} TRTVIDEO_BUILD_DIRTY=${VCS_DIRTY}
LABEL org.opencontainers.image.source="https://github.com/NevermindNilas/TheAnimeScripter" \
    org.opencontainers.image.revision=${TAS_REVISION}
ENTRYPOINT []
