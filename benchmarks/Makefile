ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

IMAGE ?= ai-media-enhancer:latest
BENCHMARK_IMAGE ?= ai-media-enhancer:benchmark
VSTRT_IMAGE ?= ai-media-enhancer:benchmark-vstrt
VSGAN_IMAGE ?= ai-media-enhancer:benchmark-vsgan
VSGAN_UPSTREAM_IMAGE ?= docker.io/styler00dollar/vsgan_tensorrt@sha256:1f23b8b43864021fb5a9e795c72e0a51b2bba568e6bbc24be175f924b828aaef

MANIFEST ?= benchmarks/workloads/realesrgan_x2plus_sintel.json
IMPLEMENTATIONS ?= benchmarks/implementations.json
VARIANT ?= 1080p
ENGINE ?=
PLAN_ENGINE ?= models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_$(VARIANT).engine
VSGAN_ENGINE ?= models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_$(VARIANT).engine
ONNX ?= models/benchmarks/realesrgan-x2plus/onnx/realesrgan_x2plus_$(VARIANT)_fp16.onnx
OUTPUT_DIR ?= artefacts/benchmarks/ai-media-$(VARIANT)
GPU_ID ?= 0
ARGS ?=
TRTEXEC_ARGS ?=
VSTRT_ARGS ?=
VSGAN_ARGS ?=

.PHONY: help build build-vstrt build-vsgan build-vsgan-engine prepare verify
.PHONY: run-ai-media plan-trtexec plan-vstrt plan-vsgan dry-run
.PHONY: run-trtexec run-vstrt run-vsgan

help:
	@printf '%s\n' \
		'build               Build the project benchmark image' \
		'build-vstrt         Build the TensorRT 11 VapourSynth parity image' \
		'build-vsgan         Build the pinned stock VSGAN benchmark wrapper' \
		'build-vsgan-engine  Build a stock TensorRT 10.16 engine (ONNX/VSGAN_ENGINE)' \
		'prepare             Download and prepare the selected workload assets' \
		'verify              Verify the selected workload assets' \
		'dry-run             Print diagnostic, parity and VSGAN product plans' \
		'run-ai-media        Run the project benchmark suite (ENGINE required)' \
		'run-vstrt           Run strict TensorRT 11 parity (ENGINE required)' \
		'run-vsgan           Run stock VSGAN (VSGAN_ENGINE required)' \
		'run-trtexec         Run the internal TensorRT diagnostic (ENGINE required)'

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

build-vsgan:
	DOCKER_BUILDKIT=1 docker build \
		--build-arg BASE_IMAGE="$(VSGAN_UPSTREAM_IMAGE)" \
		--build-arg VCS_REF="$$(git -C "$(ROOT)" rev-parse HEAD)" \
		--build-arg VCS_DIRTY="$$(if test -z "$$(git -C "$(ROOT)" status --porcelain)"; then echo 0; else echo 1; fi)" \
		-f "$(ROOT)/benchmarks/docker/vsgan.Dockerfile" \
		-t $(VSGAN_IMAGE) "$(ROOT)"

build-vsgan-engine:
	mkdir -p "$(ROOT)/models" "$(ROOT)/models/cache"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/models:/app/models" \
		$(VSGAN_IMAGE) /opt/vsgan-benchmark/bin/python3 -m benchmarks.scripts.build_vsgan_engine \
			--manifest /app/$(MANIFEST) \
			--variant $(VARIANT) \
			--onnx /app/$(ONNX) \
			--output /app/$(VSGAN_ENGINE) \
			--timing-cache /app/models/cache/vsgan-trt10.cache \
			$(ARGS)

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
			--implementations /app/$(IMPLEMENTATIONS) \
			--variant $(VARIANT) \
			--engine /app/$(or $(ENGINE),$(PLAN_ENGINE)) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/artefacts/benchmarks/diagnostic-trtexec-$(VARIANT) \
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
			--implementations /app/$(IMPLEMENTATIONS) \
			--variant $(VARIANT) \
			--engine /app/$(or $(ENGINE),$(PLAN_ENGINE)) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/artefacts/benchmarks/parity-vstrt-$(VARIANT) \
			--dry-run $(ARGS) $(VSTRT_ARGS)

plan-vsgan:
	mkdir -p "$(ROOT)/artefacts/benchmarks"
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(VSGAN_IMAGE) /opt/vsgan-benchmark/bin/python3 -m benchmarks.scripts.run_vsgan \
			--manifest /app/$(MANIFEST) \
			--implementations /app/$(IMPLEMENTATIONS) \
			--variant $(VARIANT) \
			--engine /app/$(VSGAN_ENGINE) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/artefacts/benchmarks/product-vsgan-$(VARIANT) \
			--dry-run $(ARGS) $(VSGAN_ARGS)

dry-run: plan-trtexec plan-vstrt plan-vsgan

run-trtexec:
	@test -n "$(ENGINE)" || (echo "ERROR: ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/artefacts/benchmarks/diagnostic-trtexec-$(VARIANT)"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e AI_MEDIA_IMAGE_REF="$(BENCHMARK_IMAGE)" \
		-e AI_MEDIA_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(BENCHMARK_IMAGE))" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.run_trtexec \
			--manifest /app/$(MANIFEST) \
			--implementations /app/$(IMPLEMENTATIONS) \
			--variant $(VARIANT) \
			--engine /app/$(ENGINE) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/artefacts/benchmarks/diagnostic-trtexec-$(VARIANT) \
			$(ARGS) $(TRTEXEC_ARGS)

run-vstrt:
	@test -n "$(ENGINE)" || (echo "ERROR: ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos" "$(ROOT)/artefacts/benchmarks/parity-vstrt-$(VARIANT)"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e AI_MEDIA_IMAGE_REF="$(VSTRT_IMAGE)" \
		-e AI_MEDIA_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(VSTRT_IMAGE))" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(VSTRT_IMAGE) python3 -m benchmarks.scripts.run_vstrt \
			--manifest /app/$(MANIFEST) \
			--implementations /app/$(IMPLEMENTATIONS) \
			--variant $(VARIANT) \
			--engine /app/$(ENGINE) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/artefacts/benchmarks/parity-vstrt-$(VARIANT) \
			$(ARGS) $(VSTRT_ARGS)

run-vsgan:
	@test -n "$(VSGAN_ENGINE)" || (echo "ERROR: VSGAN_ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos" "$(ROOT)/artefacts/benchmarks/product-vsgan-$(VARIANT)"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e AI_MEDIA_IMAGE_REF="$(VSGAN_IMAGE)" \
		-e AI_MEDIA_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(VSGAN_IMAGE))" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(VSGAN_IMAGE) /opt/vsgan-benchmark/bin/python3 -m benchmarks.scripts.run_vsgan \
			--manifest /app/$(MANIFEST) \
			--implementations /app/$(IMPLEMENTATIONS) \
			--variant $(VARIANT) \
			--engine /app/$(VSGAN_ENGINE) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/artefacts/benchmarks/product-vsgan-$(VARIANT) \
			$(ARGS) $(VSGAN_ARGS)
