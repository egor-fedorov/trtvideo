# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=ai-media-enhancer:benchmark
FROM ${BASE_IMAGE}

ARG VSTR_REVISION=885e8bb827fc431fce8e3109e7d60b0c38aa2035
ARG VAPOURSYNTH_VERSION=77
ARG VAPOURSYNTH_HEADERS_REVISION=325756ed04588b31840fdb74479537cddcba4bf7
ARG BESTSOURCE_VERSION=19.0

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        g++ \
        git \
        ninja-build

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python "${VIRTUAL_ENV}" \
        "vapoursynth==${VAPOURSYNTH_VERSION}" \
        "vapoursynth-bestsource==${BESTSOURCE_VERSION}"

RUN git init /tmp/vs-mlrt && \
    git -C /tmp/vs-mlrt remote add origin https://github.com/AmusementClub/vs-mlrt.git && \
    git -C /tmp/vs-mlrt fetch --depth 1 origin "${VSTR_REVISION}" && \
    git -C /tmp/vs-mlrt checkout --detach FETCH_HEAD && \
    git init /tmp/vapoursynth-headers && \
    git -C /tmp/vapoursynth-headers remote add origin \
        https://github.com/vapoursynth/vapoursynth.git && \
    git -C /tmp/vapoursynth-headers fetch --depth 1 origin \
        "${VAPOURSYNTH_HEADERS_REVISION}" && \
    git -C /tmp/vapoursynth-headers checkout --detach FETCH_HEAD && \
    VS_INCLUDE=/tmp/vapoursynth-headers/include && \
    test -f "${VS_INCLUDE}/VapourSynth.h" && \
    cmake -S /tmp/vs-mlrt/vstrt -B /tmp/vs-mlrt/vstrt/build -G Ninja \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/opt/vstrt \
        -D TENSORRT_HOME=/usr \
        -D VAPOURSYNTH_INCLUDE_DIRECTORY="${VS_INCLUDE}" && \
    cmake --build /tmp/vs-mlrt/vstrt/build --parallel && \
    cmake --install /tmp/vs-mlrt/vstrt/build && \
    mkdir -p /opt/vstrt/plugins && \
    find /opt/vstrt -type f -name 'libvstrt.so' -exec cp '{}' /opt/vstrt/plugins/libvstrt.so \; && \
    test -f /opt/vstrt/plugins/libvstrt.so && \
    rm -rf /tmp/vs-mlrt /tmp/vapoursynth-headers

ENV VAPOURSYNTH_EXTRA_PLUGIN_PATH=/opt/vstrt/plugins \
    VSTR_REVISION=${VSTR_REVISION} \
    VAPOURSYNTH_VERSION=${VAPOURSYNTH_VERSION} \
    VAPOURSYNTH_HEADERS_REVISION=${VAPOURSYNTH_HEADERS_REVISION} \
    BESTSOURCE_VERSION=${BESTSOURCE_VERSION}

LABEL org.opencontainers.image.source="https://github.com/AmusementClub/vs-mlrt" \
      org.opencontainers.image.revision="${VSTR_REVISION}"
