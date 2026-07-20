ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

IMAGE ?= ai-media-enhancer:latest
BENCHMARK_IMAGE ?= ai-media-enhancer:benchmark
VSTRT_IMAGE ?= ai-media-enhancer:benchmark-vstrt
VIDEO2X_IMAGE ?= ai-media-enhancer:benchmark-video2x
VIDEO2X_UPSTREAM_IMAGE ?= ghcr.io/k4yt3x/video2x@sha256:e21b6893269b4cb6f5603802726fd7537be241f6b39217b73530478861acbca1

MANIFEST ?= benchmarks/workloads/realesrgan_x2plus_sintel.json
COMPETITORS ?= benchmarks/competitors.json
VARIANT ?= 1080p
ENGINE ?=
PLAN_ENGINE ?= models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_$(VARIANT).engine
OUTPUT_DIR ?= artefacts/benchmarks/ai-media-$(VARIANT)
GPU_ID ?= 0
ARGS ?=
TRTEXEC_ARGS ?=
VSTRT_ARGS ?=
VIDEO2X_ARGS ?=

.PHONY: help build build-vstrt build-video2x prepare verify run-ai-media
.PHONY: plan-trtexec plan-vstrt plan-video2x dry-run
.PHONY: run-trtexec run-vstrt run-video2x

help:
	@printf '%s\n' \
		'build          Build the project benchmark image' \
		'build-vstrt    Build the pinned vs-mlrt/vstrt image' \
		'build-video2x  Build the pinned Video2X image' \
		'prepare        Download and prepare canonical workload assets' \
		'verify         Verify canonical workload assets' \
		'dry-run        Print plans for trtexec, vstrt and Video2X' \
		'run-ai-media   Run the project benchmark suite (ENGINE required)' \
		'run-trtexec    Run the TensorRT ceiling suite (ENGINE required)' \
		'run-vstrt      Run the vs-mlrt/vstrt suite (ENGINE required)' \
		'run-video2x    Run the Video2X product-level suite'

build:
	DOCKER_BUILDKIT=1 docker build \
		--build-arg VCS_REF="$$(git -C "$(ROOT)" rev-parse HEAD)" \
		--build-arg VCS_DIRTY="$$(if test -z "$$(git -C "$(ROOT)" status --porcelain)"; then echo 0; else echo 1; fi)" \
		--target benchmark \
		-t $(BENCHMARK_IMAGE) "$(ROOT)"

build-vstrt: build
	DOCKER_BUILDKIT=1 docker build \
		--build-arg BASE_IMAGE="$(BENCHMARK_IMAGE)" \
		-f "$(ROOT)/benchmarks/docker/vstrt.Dockerfile" \
		-t $(VSTRT_IMAGE) "$(ROOT)"

build-video2x:
	DOCKER_BUILDKIT=1 docker build \
		--build-arg BASE_IMAGE="$(VIDEO2X_UPSTREAM_IMAGE)" \
		--build-arg VCS_REF="$$(git -C "$(ROOT)" rev-parse HEAD)" \
		--build-arg VCS_DIRTY="$$(if test -z "$$(git -C "$(ROOT)" status --porcelain)"; then echo 0; else echo 1; fi)" \
		-f "$(ROOT)/benchmarks/docker/video2x.Dockerfile" \
		-t $(VIDEO2X_IMAGE) "$(ROOT)"

prepare:
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos"
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/models:/app/models" \
		-v "$(ROOT)/videos:/app/videos" \
		-v "$(ROOT)/benchmarks:/app/benchmarks:ro" \
		$(IMAGE) python3 -m benchmarks.scripts.prepare_workload prepare \
			--manifest /app/$(MANIFEST) --root /app $(ARGS)

verify:
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos"
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/models:/app/models" \
		-v "$(ROOT)/videos:/app/videos" \
		-v "$(ROOT)/benchmarks:/app/benchmarks:ro" \
		$(IMAGE) python3 -m benchmarks.scripts.prepare_workload verify \
			--manifest /app/$(MANIFEST) --root /app $(ARGS)

run-ai-media:
	@test -n "$(ENGINE)" || (echo "ERROR: ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos" "$(ROOT)/$(OUTPUT_DIR)"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e AI_MEDIA_IMAGE_REF="$(BENCHMARK_IMAGE)" \
		-e AI_MEDIA_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(BENCHMARK_IMAGE))" \
		-v "$(ROOT)/models:/app/models" \
		-v "$(ROOT)/videos:/app/videos" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.run_ai_media \
			--manifest /app/$(MANIFEST) \
			--variant $(VARIANT) \
			--engine /app/$(ENGINE) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/$(OUTPUT_DIR) \
			$(ARGS)

plan-trtexec:
	mkdir -p "$(ROOT)/artefacts/benchmarks"
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/benchmarks:/app/benchmarks:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.run_trtexec \
			--manifest /app/$(MANIFEST) \
			--competitors /app/$(COMPETITORS) \
			--variant $(VARIANT) \
			--engine /app/$(or $(ENGINE),$(PLAN_ENGINE)) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/artefacts/benchmarks/trtexec-$(VARIANT) \
			--dry-run $(ARGS) $(TRTEXEC_ARGS)

plan-vstrt:
	mkdir -p "$(ROOT)/artefacts/benchmarks"
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/benchmarks:/app/benchmarks:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(VSTRT_IMAGE) python3 -m benchmarks.scripts.run_vstrt \
			--manifest /app/$(MANIFEST) \
			--competitors /app/$(COMPETITORS) \
			--variant $(VARIANT) \
			--engine /app/$(or $(ENGINE),$(PLAN_ENGINE)) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/artefacts/benchmarks/vstrt-$(VARIANT) \
			--dry-run $(ARGS) $(VSTRT_ARGS)

plan-video2x:
	mkdir -p "$(ROOT)/artefacts/benchmarks"
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/benchmarks:/app/benchmarks:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(VIDEO2X_IMAGE) python3 -m benchmarks.scripts.run_video2x \
			--manifest /app/$(MANIFEST) \
			--competitors /app/$(COMPETITORS) \
			--variant $(VARIANT) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/artefacts/benchmarks/video2x-$(VARIANT) \
			--dry-run $(ARGS) $(VIDEO2X_ARGS)

dry-run: plan-trtexec plan-vstrt plan-video2x

run-trtexec:
	@test -n "$(ENGINE)" || (echo "ERROR: ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/artefacts/benchmarks/trtexec-$(VARIANT)"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e AI_MEDIA_IMAGE_REF="$(BENCHMARK_IMAGE)" \
		-e AI_MEDIA_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(BENCHMARK_IMAGE))" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.run_trtexec \
			--manifest /app/$(MANIFEST) \
			--competitors /app/$(COMPETITORS) \
			--variant $(VARIANT) \
			--engine /app/$(ENGINE) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/artefacts/benchmarks/trtexec-$(VARIANT) \
			$(ARGS) $(TRTEXEC_ARGS)

run-vstrt:
	@test -n "$(ENGINE)" || (echo "ERROR: ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos" "$(ROOT)/artefacts/benchmarks/vstrt-$(VARIANT)"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e AI_MEDIA_IMAGE_REF="$(VSTRT_IMAGE)" \
		-e AI_MEDIA_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(VSTRT_IMAGE))" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(VSTRT_IMAGE) python3 -m benchmarks.scripts.run_vstrt \
			--manifest /app/$(MANIFEST) \
			--competitors /app/$(COMPETITORS) \
			--variant $(VARIANT) \
			--engine /app/$(ENGINE) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/artefacts/benchmarks/vstrt-$(VARIANT) \
			$(ARGS) $(VSTRT_ARGS)

run-video2x:
	mkdir -p "$(ROOT)/videos" "$(ROOT)/artefacts/benchmarks/video2x-$(VARIANT)"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e AI_MEDIA_IMAGE_REF="$(VIDEO2X_IMAGE)" \
		-e AI_MEDIA_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(VIDEO2X_IMAGE))" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(VIDEO2X_IMAGE) python3 -m benchmarks.scripts.run_video2x \
			--manifest /app/$(MANIFEST) \
			--competitors /app/$(COMPETITORS) \
			--variant $(VARIANT) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/artefacts/benchmarks/video2x-$(VARIANT) \
			$(ARGS) $(VIDEO2X_ARGS)
