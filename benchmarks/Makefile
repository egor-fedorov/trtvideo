ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

HOST_PYTHON ?= python3
IMAGE ?= trtvideo:latest
BENCHMARK_IMAGE ?= trtvideo:benchmark
VSTRT_IMAGE ?= trtvideo:benchmark-vstrt
VSGAN_IMAGE ?= trtvideo:benchmark-vsgan
VSGAN_UPSTREAM_IMAGE ?= docker.io/styler00dollar/vsgan_tensorrt@sha256:1f23b8b43864021fb5a9e795c72e0a51b2bba568e6bbc24be175f924b828aaef

MANIFEST ?= benchmarks/workloads/realesrgan_x2plus_sintel.json
IMPLEMENTATIONS ?= benchmarks/implementations.json
VARIANT ?= 1080p
ENGINE ?=
PLAN_ENGINE ?= models/benchmarks/realesrgan-x2plus/engines/realesrgan_x2plus_$(VARIANT).engine
VSGAN_ENGINE ?= models/benchmarks/realesrgan-x2plus/engines/vsgan/realesrgan_x2plus_$(VARIANT).engine
ONNX ?= models/benchmarks/realesrgan-x2plus/onnx/realesrgan_x2plus_$(VARIANT)_fp16.onnx
TRT_TIMING_CACHE ?= models/cache/benchmark-trt11.cache
GPU_ID ?= 0
ARGS ?=
TRTEXEC_ARGS ?=
VSTRT_ARGS ?=
VSGAN_ARGS ?=
NSIGHT_ARGS ?=
EXECUTION_PROFILE ?= upstream-default
WORKLOAD_NAME := $(basename $(notdir $(MANIFEST)))
PROJECT_OUTPUT_DIR ?= artefacts/benchmarks/project/$(WORKLOAD_NAME)-$(VARIANT)
TRTVIDEO_OUTPUT_DIR ?= $(PROJECT_OUTPUT_DIR)
VSTRT_OUTPUT_DIR ?= artefacts/benchmarks/comparative/per-implementation/$(EXECUTION_PROFILE)/vstrt/$(WORKLOAD_NAME)-$(VARIANT)
VSGAN_OUTPUT_DIR ?= artefacts/benchmarks/comparative/per-implementation/$(EXECUTION_PROFILE)/vsgan/$(WORKLOAD_NAME)-$(VARIANT)
TRTEXEC_OUTPUT_DIR ?= artefacts/benchmarks/diagnostics/trtexec/$(WORKLOAD_NAME)-$(VARIANT)
NSIGHT_OUTPUT_DIR ?= artefacts/benchmarks/diagnostics/nsight/$(WORKLOAD_NAME)-$(VARIANT)
NSIGHT_FRAMES ?= 120
REPORT_IMAGE ?= trtvideo:dev
RESULTS_DIR ?= benchmarks/results/rtx-3090
FIGURES_DIR ?= $(RESULTS_DIR)/figures
TUNED_PUBLICATION_SOURCE ?=
TUNED_PUBLICATION_OUTPUT ?= $(RESULTS_DIR)/tuned.json
CAMPAIGN_NAME ?= $(WORKLOAD_NAME)-$(VARIANT)
CAMPAIGN_DIR ?= artefacts/benchmarks/comparative/campaigns/$(EXECUTION_PROFILE)/$(CAMPAIGN_NAME)
CAMPAIGN_IDLE_SECONDS ?= 10
CAMPAIGN_RUN_ARGS = --runs 1 --extra-runs 0 --idle-seconds 0
MODEL_SPACE_DIR ?= artefacts/benchmarks/comparative/quality/model-space/$(EXECUTION_PROFILE)/$(WORKLOAD_NAME)-$(VARIANT)
MODEL_SPACE_ARGS ?=
PRODUCT_OUTPUT_DIR ?= artefacts/benchmarks/comparative/quality/product-output/$(EXECUTION_PROFILE)/$(WORKLOAD_NAME)-$(VARIANT)
QUALITY_FRAMES ?= 1000
QUALITY_RUN_ARGS = --frames $(QUALITY_FRAMES) --runs 1 --extra-runs 0 --idle-seconds 0 --keep-outputs
TUNING_CONTRACT ?= benchmarks/tuning/candidates.json
TUNING_DIR ?= artefacts/benchmarks/comparative/tuning/$(WORKLOAD_NAME)-$(VARIANT)
TUNING_RESUME ?= 0
TUNING_720_DIR ?= artefacts/benchmarks/comparative/tuning/$(WORKLOAD_NAME)-720p
TUNING_1080_DIR ?= artefacts/benchmarks/comparative/tuning/$(WORKLOAD_NAME)-1080p
TUNING_MATRIX_OUTPUT ?= artefacts/benchmarks/comparative/tuning/$(WORKLOAD_NAME)-matrix.json
MODEL_SPACE_REFERENCE ?=
MODEL_SPACE_CANDIDATE ?=
MODEL_SPACE_REPORT ?=
RESUME ?= 0

.PHONY: help build build-vstrt build-vsgan build-project-engine build-vsgan-engine prepare verify
.PHONY: run-project run-trtvideo plan-trtexec plan-vstrt plan-vsgan plan-nsight dry-run
.PHONY: run-trtexec run-vstrt run-vsgan
.PHONY: profile-nsight
.PHONY: capture-model-trtvideo capture-model-vstrt capture-model-vsgan
.PHONY: compare-model-space model-space-parity
.PHONY: compare-model-space-candidate
.PHONY: capture-output-trtvideo capture-output-vstrt capture-output-vsgan
.PHONY: compare-product-output product-output-parity quality-gates
.PHONY: campaign-preflight campaign-trtvideo campaign-vstrt campaign-vsgan
.PHONY: aggregate-campaign run-campaign run-comparative
.PHONY: run-tuned-sweep rank-tuned run-tuned-quality run-tuned-campaign
.PHONY: verify-tuned-matrix
.PHONY: publish-tuned figures figures-check

help:
	@printf '%s\n' \
		'build               Build the project benchmark image' \
		'build-vstrt         Build the TensorRT 11 VapourSynth image' \
		'build-vsgan         Build the pinned upstream VSGAN benchmark wrapper' \
		'build-project-engine  Build one project TRT11 engine (ONNX/ENGINE)' \
		'build-vsgan-engine  Build a pinned VSGAN TRT10.16 engine (ONNX/VSGAN_ENGINE)' \
		'prepare             Download and prepare the selected workload assets' \
		'verify              Verify the selected workload assets' \
		'dry-run             Print diagnostic and selected-profile plans' \
		'run-project         Run the project-only regression benchmark' \
		'run-comparative     Run/resume the selected-profile rotation' \
		'run-trtvideo        Run the low-level project implementation suite' \
		'run-vstrt           Run vstrt with EXECUTION_PROFILE (ENGINE required)' \
		'run-vsgan           Run pinned VSGAN (VSGAN_ENGINE required)' \
		'run-trtexec         Run the internal TensorRT diagnostic (ENGINE required)' \
		'plan-nsight         Print the project Nsight diagnostic plan' \
		'profile-nsight      Capture one project Nsight Systems trace' \
		'model-space-parity  Compare pre/post-TensorRT RGB float tensors' \
		'product-output-parity  Compare retained MP4 outputs with PSNR/SSIM' \
		'quality-gates       Run model-space and product-output parity gates' \
		'run-tuned-sweep     Evaluate every declared tuned scheduling candidate' \
		'rank-tuned          Rebuild the tuned candidate selection report' \
		'run-tuned-quality   Fully validate winners; promote on candidate failure' \
		'run-tuned-campaign  Run the rotated campaign for validated tuned winners' \
		'verify-tuned-matrix Require valid tuned evidence for 720p and 1080p' \
		'publish-tuned       Build publication JSON from a copied tuned evidence root' \
		'figures             Regenerate published SVG figures from result JSON' \
		'figures-check       Require committed SVG figures to match result JSON' \
		'run-campaign        Low-level comparative campaign coordinator' \
		'aggregate-campaign  Rebuild campaign.json and results.md from raw runs'

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

build-project-engine:
	@test -n "$(ONNX)" || (echo "ERROR: ONNX is required" >&2; exit 2)
	@test -n "$(ENGINE)" || (echo "ERROR: ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/$(dir $(ENGINE))" "$(ROOT)/models/cache"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/models:/app/models" \
		$(IMAGE) build-engine \
			/app/$(ONNX) \
			-o /app/$(ENGINE) \
			--timing-cache /app/$(TRT_TIMING_CACHE)

build-vsgan-engine:
	mkdir -p "$(ROOT)/models" "$(ROOT)/models/cache"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e TRTVIDEO_IMAGE_REF="$(VSGAN_IMAGE)" \
		-e TRTVIDEO_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(VSGAN_IMAGE))" \
		-v "$(ROOT)/models:/app/models" \
		$(VSGAN_IMAGE) /opt/vsgan-benchmark/bin/python3 -m benchmarks.scripts.workloads.build_vsgan_engine \
			--manifest /app/$(MANIFEST) \
			--variant $(VARIANT) \
			--onnx /app/$(ONNX) \
			--output /app/$(VSGAN_ENGINE) \
			--timing-cache /app/models/cache/vsgan-trt10.16.0.72.cache \
			$(ARGS)

prepare:
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos"
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/models:/app/models" \
		-v "$(ROOT)/videos:/app/videos" \
		-v "$(ROOT)/benchmarks:/app/benchmarks:ro" \
		$(IMAGE) python3 -m benchmarks.scripts.workloads.prepare prepare \
			--manifest /app/$(MANIFEST) --root /app $(ARGS)

verify:
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos"
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/models:/app/models" \
		-v "$(ROOT)/videos:/app/videos" \
		-v "$(ROOT)/benchmarks:/app/benchmarks:ro" \
		$(IMAGE) python3 -m benchmarks.scripts.workloads.prepare verify \
			--manifest /app/$(MANIFEST) --root /app $(ARGS)

run-project:
	$(MAKE) run-trtvideo TRTVIDEO_OUTPUT_DIR="$(PROJECT_OUTPUT_DIR)"

run-trtvideo:
	@test -n "$(ENGINE)" || (echo "ERROR: ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos" "$(ROOT)/$(TRTVIDEO_OUTPUT_DIR)"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e TRTVIDEO_IMAGE_REF="$(BENCHMARK_IMAGE)" \
		-e TRTVIDEO_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(BENCHMARK_IMAGE))" \
		-v "$(ROOT)/models:/app/models" \
		-v "$(ROOT)/videos:/app/videos" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.runners.trtvideo \
			--manifest /app/$(MANIFEST) \
			--variant $(VARIANT) \
			--engine /app/$(ENGINE) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/$(TRTVIDEO_OUTPUT_DIR) \
			$(ARGS)

plan-trtexec:
	mkdir -p "$(ROOT)/artefacts/benchmarks"
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/benchmarks:/app/benchmarks:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.runners.trtexec \
			--manifest /app/$(MANIFEST) \
			--implementations /app/$(IMPLEMENTATIONS) \
			--variant $(VARIANT) \
			--engine /app/$(or $(ENGINE),$(PLAN_ENGINE)) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/$(TRTEXEC_OUTPUT_DIR) \
			--dry-run $(ARGS) $(TRTEXEC_ARGS)

plan-vstrt:
	mkdir -p "$(ROOT)/artefacts/benchmarks"
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/benchmarks:/app/benchmarks:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(VSTRT_IMAGE) python3 -m benchmarks.scripts.runners.vstrt \
			--manifest /app/$(MANIFEST) \
			--implementations /app/$(IMPLEMENTATIONS) \
			--variant $(VARIANT) \
			--engine /app/$(or $(ENGINE),$(PLAN_ENGINE)) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/$(VSTRT_OUTPUT_DIR) \
			--execution-profile $(EXECUTION_PROFILE) \
			--dry-run $(ARGS) $(VSTRT_ARGS)

plan-vsgan:
	mkdir -p "$(ROOT)/artefacts/benchmarks"
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(VSGAN_IMAGE) /opt/vsgan-benchmark/bin/python3 -m benchmarks.scripts.runners.vsgan \
			--manifest /app/$(MANIFEST) \
			--implementations /app/$(IMPLEMENTATIONS) \
			--variant $(VARIANT) \
			--engine /app/$(VSGAN_ENGINE) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/$(VSGAN_OUTPUT_DIR) \
			--execution-profile $(EXECUTION_PROFILE) \
			--dry-run $(ARGS) $(VSGAN_ARGS)

dry-run: plan-trtexec plan-vstrt plan-vsgan

plan-nsight:
	mkdir -p "$(ROOT)/artefacts/benchmarks"
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-e TRTVIDEO_IMAGE_REF="$(BENCHMARK_IMAGE)" \
		-e TRTVIDEO_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(BENCHMARK_IMAGE))" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.diagnostics.nsight \
			--manifest /app/$(MANIFEST) \
			--variant $(VARIANT) \
			--engine /app/$(or $(ENGINE),$(PLAN_ENGINE)) \
			--gpu-id $(GPU_ID) \
			--frames $(NSIGHT_FRAMES) \
			--output-dir /app/$(NSIGHT_OUTPUT_DIR) \
			--dry-run $(NSIGHT_ARGS)

run-trtexec:
	@test -n "$(ENGINE)" || (echo "ERROR: ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/$(TRTEXEC_OUTPUT_DIR)"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e TRTVIDEO_IMAGE_REF="$(BENCHMARK_IMAGE)" \
		-e TRTVIDEO_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(BENCHMARK_IMAGE))" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.runners.trtexec \
			--manifest /app/$(MANIFEST) \
			--implementations /app/$(IMPLEMENTATIONS) \
			--variant $(VARIANT) \
			--engine /app/$(ENGINE) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/$(TRTEXEC_OUTPUT_DIR) \
			$(ARGS) $(TRTEXEC_ARGS)

run-vstrt:
	@test -n "$(ENGINE)" || (echo "ERROR: ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos" "$(ROOT)/$(VSTRT_OUTPUT_DIR)"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e TRTVIDEO_IMAGE_REF="$(VSTRT_IMAGE)" \
		-e TRTVIDEO_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(VSTRT_IMAGE))" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(VSTRT_IMAGE) python3 -m benchmarks.scripts.runners.vstrt \
			--manifest /app/$(MANIFEST) \
			--implementations /app/$(IMPLEMENTATIONS) \
			--variant $(VARIANT) \
			--engine /app/$(ENGINE) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/$(VSTRT_OUTPUT_DIR) \
			--execution-profile $(EXECUTION_PROFILE) \
			$(ARGS) $(VSTRT_ARGS)

run-vsgan:
	@test -n "$(VSGAN_ENGINE)" || (echo "ERROR: VSGAN_ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos" "$(ROOT)/$(VSGAN_OUTPUT_DIR)"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e TRTVIDEO_IMAGE_REF="$(VSGAN_IMAGE)" \
		-e TRTVIDEO_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(VSGAN_IMAGE))" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(VSGAN_IMAGE) /opt/vsgan-benchmark/bin/python3 -m benchmarks.scripts.runners.vsgan \
			--manifest /app/$(MANIFEST) \
			--implementations /app/$(IMPLEMENTATIONS) \
			--variant $(VARIANT) \
			--engine /app/$(VSGAN_ENGINE) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/$(VSGAN_OUTPUT_DIR) \
			--execution-profile $(EXECUTION_PROFILE) \
			$(ARGS) $(VSGAN_ARGS)

profile-nsight:
	@test -n "$(ENGINE)" || (echo "ERROR: ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos" "$(ROOT)/$(NSIGHT_OUTPUT_DIR)"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e TRTVIDEO_IMAGE_REF="$(BENCHMARK_IMAGE)" \
		-e TRTVIDEO_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(BENCHMARK_IMAGE))" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.diagnostics.nsight \
			--manifest /app/$(MANIFEST) \
			--variant $(VARIANT) \
			--engine /app/$(ENGINE) \
			--gpu-id $(GPU_ID) \
			--frames $(NSIGHT_FRAMES) \
			--output-dir /app/$(NSIGHT_OUTPUT_DIR) \
			$(NSIGHT_ARGS)

capture-model-trtvideo:
	@test -n "$(ENGINE)" || (echo "ERROR: ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos" "$(ROOT)/$(MODEL_SPACE_DIR)/trtvideo"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e TRTVIDEO_IMAGE_REF="$(BENCHMARK_IMAGE)" \
		-e TRTVIDEO_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(BENCHMARK_IMAGE))" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.quality.capture_trtvideo \
			--manifest /app/$(MANIFEST) \
			--variant $(VARIANT) \
			--engine /app/$(ENGINE) \
			--gpu-id $(GPU_ID) \
			--execution-profile $(EXECUTION_PROFILE) \
			--output-dir /app/$(MODEL_SPACE_DIR)/trtvideo \
			$(MODEL_SPACE_ARGS)

capture-model-vstrt:
	@test -n "$(ENGINE)" || (echo "ERROR: ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos" "$(ROOT)/$(MODEL_SPACE_DIR)/vstrt"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e TRTVIDEO_IMAGE_REF="$(VSTRT_IMAGE)" \
		-e TRTVIDEO_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(VSTRT_IMAGE))" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(VSTRT_IMAGE) python3 -m benchmarks.scripts.quality.capture_vspipe \
			--implementation vstrt \
			--manifest /app/$(MANIFEST) \
			--implementations /app/$(IMPLEMENTATIONS) \
			--variant $(VARIANT) \
			--engine /app/$(ENGINE) \
			--script /app/benchmarks/vstrt/upscale.vpy \
			--gpu-id $(GPU_ID) \
			--execution-profile $(EXECUTION_PROFILE) \
			--output-dir /app/$(MODEL_SPACE_DIR)/vstrt \
			$(MODEL_SPACE_ARGS) $(VSTRT_ARGS)

capture-model-vsgan:
	@test -n "$(VSGAN_ENGINE)" || (echo "ERROR: VSGAN_ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos" "$(ROOT)/$(MODEL_SPACE_DIR)/vsgan"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e TRTVIDEO_IMAGE_REF="$(VSGAN_IMAGE)" \
		-e TRTVIDEO_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(VSGAN_IMAGE))" \
		-v "$(ROOT)/models:/app/models:ro" \
		-v "$(ROOT)/videos:/app/videos:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(VSGAN_IMAGE) /opt/vsgan-benchmark/bin/python3 \
			-m benchmarks.scripts.quality.capture_vspipe \
			--implementation vsgan \
			--manifest /app/$(MANIFEST) \
			--implementations /app/$(IMPLEMENTATIONS) \
			--variant $(VARIANT) \
			--engine /app/$(VSGAN_ENGINE) \
			--script /app/benchmarks/vsgan/upscale.vpy \
			--gpu-id $(GPU_ID) \
			--execution-profile $(EXECUTION_PROFILE) \
			--output-dir /app/$(MODEL_SPACE_DIR)/vsgan \
			$(MODEL_SPACE_ARGS) $(VSGAN_ARGS)

compare-model-space:
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.quality.compare_model_space \
			--manifest /app/$(MANIFEST) \
			--variant $(VARIANT) \
			--reference /app/$(MODEL_SPACE_DIR)/trtvideo/manifest.json \
			--candidate /app/$(MODEL_SPACE_DIR)/vstrt/manifest.json \
			--candidate /app/$(MODEL_SPACE_DIR)/vsgan/manifest.json \
			--output /app/$(MODEL_SPACE_DIR)/model-space-parity.json

compare-model-space-candidate:
	@test -n "$(MODEL_SPACE_REFERENCE)" || (echo "ERROR: MODEL_SPACE_REFERENCE is required" >&2; exit 2)
	@test -n "$(MODEL_SPACE_CANDIDATE)" || (echo "ERROR: MODEL_SPACE_CANDIDATE is required" >&2; exit 2)
	@test -n "$(MODEL_SPACE_REPORT)" || (echo "ERROR: MODEL_SPACE_REPORT is required" >&2; exit 2)
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.quality.compare_model_space \
			--manifest /app/$(MANIFEST) \
			--variant $(VARIANT) \
			--reference /app/$(MODEL_SPACE_REFERENCE) \
			--candidate /app/$(MODEL_SPACE_CANDIDATE) \
			--output /app/$(MODEL_SPACE_REPORT)

model-space-parity:
	@printf '[quality:model-space 1/4] Capture trtvideo tensors\n'
	$(MAKE) capture-model-trtvideo
	@printf '[quality:model-space 2/4] Capture vs-mlrt tensors\n'
	$(MAKE) capture-model-vstrt
	@printf '[quality:model-space 3/4] Capture VSGAN tensors\n'
	$(MAKE) capture-model-vsgan
	@printf '[quality:model-space 4/4] Compare captured tensors\n'
	$(MAKE) compare-model-space

capture-output-trtvideo:
	$(MAKE) run-trtvideo \
		TRTVIDEO_OUTPUT_DIR="$(PRODUCT_OUTPUT_DIR)/trtvideo" \
		ARGS="$(QUALITY_RUN_ARGS)"

capture-output-vstrt:
	$(MAKE) run-vstrt \
		VSTRT_OUTPUT_DIR="$(PRODUCT_OUTPUT_DIR)/vstrt" \
		ARGS="$(QUALITY_RUN_ARGS)"

capture-output-vsgan:
	$(MAKE) run-vsgan \
		VSGAN_OUTPUT_DIR="$(PRODUCT_OUTPUT_DIR)/vsgan" \
		ARGS="$(QUALITY_RUN_ARGS)"

compare-product-output:
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.quality.compare_product_output \
			--manifest /app/$(MANIFEST) \
			--variant $(VARIANT) \
			--reference /app/$(PRODUCT_OUTPUT_DIR)/trtvideo/run-01/manifest.json \
			--candidate /app/$(PRODUCT_OUTPUT_DIR)/vstrt/run-01/manifest.json \
			--candidate /app/$(PRODUCT_OUTPUT_DIR)/vsgan/run-01/manifest.json \
			--output /app/$(PRODUCT_OUTPUT_DIR)/product-output-parity.json \
			--root /app

product-output-parity:
	@printf '[quality:product-output 1/4] Run trtvideo quality output\n'
	$(MAKE) capture-output-trtvideo
	@printf '[quality:product-output 2/4] Run vs-mlrt quality output\n'
	$(MAKE) capture-output-vstrt
	@printf '[quality:product-output 3/4] Run VSGAN quality output\n'
	$(MAKE) capture-output-vsgan
	@printf '[quality:product-output 4/4] Compare decoded outputs\n'
	$(MAKE) compare-product-output

quality-gates: model-space-parity product-output-parity

run-tuned-sweep:
	cd "$(ROOT)" && "$(HOST_PYTHON)" -m benchmarks.scripts.tuning.workflow sweep \
		--contract "$(TUNING_CONTRACT)" \
		--manifest "$(MANIFEST)" \
		--variant "$(VARIANT)" \
		--engine "$(ENGINE)" \
		--vsgan-engine "$(VSGAN_ENGINE)" \
		--gpu-id "$(GPU_ID)" \
		--sweep-dir "$(TUNING_DIR)" \
		$(if $(filter 1,$(TUNING_RESUME)),--resume,)

rank-tuned:
	cd "$(ROOT)" && "$(HOST_PYTHON)" -m benchmarks.scripts.tuning.rank \
		--contract "$(TUNING_CONTRACT)" \
		--manifest "$(MANIFEST)" \
		--variant "$(VARIANT)" \
		--sweep-dir "$(TUNING_DIR)" \
		--root "$(ROOT)" \
		--output "$(TUNING_DIR)/selection.json" \
		--disqualifications "$(TUNING_DIR)/disqualifications.json"

run-tuned-quality:
	cd "$(ROOT)" && "$(HOST_PYTHON)" -m benchmarks.scripts.tuning.workflow quality \
		--contract "$(TUNING_CONTRACT)" \
		--manifest "$(MANIFEST)" \
		--variant "$(VARIANT)" \
		--engine "$(ENGINE)" \
		--vsgan-engine "$(VSGAN_ENGINE)" \
		--gpu-id "$(GPU_ID)" \
		--sweep-dir "$(TUNING_DIR)"

run-tuned-campaign:
	cd "$(ROOT)" && "$(HOST_PYTHON)" -m benchmarks.scripts.tuning.workflow campaign \
		--contract "$(TUNING_CONTRACT)" \
		--manifest "$(MANIFEST)" \
		--variant "$(VARIANT)" \
		--engine "$(ENGINE)" \
		--vsgan-engine "$(VSGAN_ENGINE)" \
		--gpu-id "$(GPU_ID)" \
		--sweep-dir "$(TUNING_DIR)" \
		$(if $(filter 1,$(RESUME)),--resume,)

verify-tuned-matrix:
	cd "$(ROOT)" && "$(HOST_PYTHON)" -m benchmarks.scripts.tuning.matrix \
		--root "$(ROOT)" \
		--campaign-720p "$(TUNING_720_DIR)/final-campaign.json" \
		--campaign-1080p "$(TUNING_1080_DIR)/final-campaign.json" \
		--output "$(TUNING_MATRIX_OUTPUT)"

publish-tuned:
	@test -n "$(TUNED_PUBLICATION_SOURCE)" || \
		{ echo 'ERROR: set TUNED_PUBLICATION_SOURCE relative to the repository root' >&2; exit 2; }
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-e PYTHONPATH=/app \
		-v "$(ROOT):/app" \
		$(REPORT_IMAGE) python3 -m benchmarks.scripts.report.publish_tuned \
			--source-dir /app/$(TUNED_PUBLICATION_SOURCE) \
			--output /app/$(TUNED_PUBLICATION_OUTPUT)

figures:
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-e MPLCONFIGDIR=/tmp/matplotlib \
		-e XDG_CACHE_HOME=/tmp/cache \
		-e PYTHONPATH=/app \
		-v "$(ROOT):/app" \
		$(REPORT_IMAGE) python3 -m benchmarks.scripts.report.figures \
			--results-dir /app/$(RESULTS_DIR) \
			--output-dir /app/$(FIGURES_DIR)

figures-check:
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-e MPLCONFIGDIR=/tmp/matplotlib \
		-e XDG_CACHE_HOME=/tmp/cache \
		-e PYTHONPATH=/app \
		-v "$(ROOT):/app:ro" \
		$(REPORT_IMAGE) python3 -m benchmarks.scripts.report.figures \
			--results-dir /app/$(RESULTS_DIR) \
			--output-dir /app/$(FIGURES_DIR) \
			--check

campaign-preflight:
	@case "$(EXECUTION_PROFILE)" in \
		upstream-default|tuned) ;; \
		*) echo "ERROR: unknown EXECUTION_PROFILE: $(EXECUTION_PROFILE)" >&2; exit 2 ;; \
	esac
	@command -v "$(HOST_PYTHON)" >/dev/null 2>&1 || \
		(echo "ERROR: comparative campaign requires host Python >= 3.10,<3.13; not found: $(HOST_PYTHON)" >&2; exit 2)
	@"$(HOST_PYTHON)" -c \
		'import sys; version = sys.version_info[:2]; raise SystemExit(0 if (3, 10) <= version < (3, 13) else 1)' || \
		(echo "ERROR: comparative campaign requires host Python >= 3.10,<3.13 ($(HOST_PYTHON))" >&2; exit 2)
	@test -n "$(ENGINE)" || (echo "ERROR: ENGINE is required" >&2; exit 2)
	@test -n "$(VSGAN_ENGINE)" || (echo "ERROR: VSGAN_ENGINE is required" >&2; exit 2)
	@test -f "$(ROOT)/$(ENGINE)" || (echo "ERROR: ENGINE not found: $(ENGINE)" >&2; exit 2)
	@test -f "$(ROOT)/$(VSGAN_ENGINE)" || (echo "ERROR: VSGAN_ENGINE not found: $(VSGAN_ENGINE)" >&2; exit 2)
	@test -z "$$(git -C "$(ROOT)" status --porcelain)" || \
		(echo "ERROR: campaign requires a clean worktree" >&2; exit 2)
	@if test -e "$(ROOT)/$(CAMPAIGN_DIR)" && test "$(RESUME)" != "1"; then \
		echo "ERROR: campaign directory exists; use RESUME=1 or choose CAMPAIGN_NAME" >&2; \
		exit 2; \
	fi
	@revision="$$(git -C "$(ROOT)" rev-parse HEAD)"; \
	for image in "$(BENCHMARK_IMAGE)" "$(VSTRT_IMAGE)" "$(VSGAN_IMAGE)"; do \
		docker run --rm -e EXPECTED_REVISION="$$revision" "$$image" sh -c \
			'test "$$TRTVIDEO_BUILD_REVISION" = "$$EXPECTED_REVISION" && test "$$TRTVIDEO_BUILD_DIRTY" = "0"' || { \
			echo "ERROR: $$image was not built from clean revision $$revision" >&2; \
			exit 2; \
		}; \
	done

campaign-trtvideo:
	@output="$(ROOT)/$(CAMPAIGN_DIR)/trtvideo/round-$(ROUND)"; \
	manifest="$$output/run-01/manifest.json"; \
	if test -f "$$manifest"; then \
		test "$(RESUME)" = "1" || exit 2; \
		grep -q '"status": "valid"' "$$manifest" || { echo "ERROR: invalid existing $$manifest" >&2; exit 2; }; \
		echo "Resume: trtvideo round $(ROUND) is already valid"; \
	elif test -e "$$output"; then \
		echo "ERROR: partial trtvideo round $(ROUND); remove $$output before resume" >&2; exit 2; \
	else \
		$(MAKE) run-trtvideo TRTVIDEO_OUTPUT_DIR="$(CAMPAIGN_DIR)/trtvideo/round-$(ROUND)" \
			ARGS="$(CAMPAIGN_RUN_ARGS)"; \
	fi

campaign-vstrt:
	@output="$(ROOT)/$(CAMPAIGN_DIR)/vstrt/round-$(ROUND)"; \
	manifest="$$output/run-01/manifest.json"; \
	if test -f "$$manifest"; then \
		test "$(RESUME)" = "1" || exit 2; \
		grep -q '"status": "valid"' "$$manifest" || { echo "ERROR: invalid existing $$manifest" >&2; exit 2; }; \
		echo "Resume: vstrt round $(ROUND) is already valid"; \
	elif test -e "$$output"; then \
		echo "ERROR: partial vstrt round $(ROUND); remove $$output before resume" >&2; exit 2; \
	else \
		$(MAKE) run-vstrt VSTRT_OUTPUT_DIR="$(CAMPAIGN_DIR)/vstrt/round-$(ROUND)" \
			ARGS="$(CAMPAIGN_RUN_ARGS)"; \
	fi

campaign-vsgan:
	@output="$(ROOT)/$(CAMPAIGN_DIR)/vsgan/round-$(ROUND)"; \
	manifest="$$output/run-01/manifest.json"; \
	if test -f "$$manifest"; then \
		test "$(RESUME)" = "1" || exit 2; \
		grep -q '"status": "valid"' "$$manifest" || { echo "ERROR: invalid existing $$manifest" >&2; exit 2; }; \
		echo "Resume: VSGAN round $(ROUND) is already valid"; \
	elif test -e "$$output"; then \
		echo "ERROR: partial VSGAN round $(ROUND); remove $$output before resume" >&2; exit 2; \
	else \
		$(MAKE) run-vsgan VSGAN_OUTPUT_DIR="$(CAMPAIGN_DIR)/vsgan/round-$(ROUND)" \
			ARGS="$(CAMPAIGN_RUN_ARGS)"; \
	fi

aggregate-campaign:
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(ROOT)/benchmarks:/app/benchmarks:ro" \
		-v "$(ROOT)/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.campaign.aggregate \
			--campaign-dir /app/$(CAMPAIGN_DIR) \
			--root /app \
			--idle-seconds $(CAMPAIGN_IDLE_SECONDS) \
			--execution-profile $(EXECUTION_PROFILE) \
			$(if $(wildcard $(ROOT)/$(MODEL_SPACE_DIR)/model-space-parity.json),--model-space-report /app/$(MODEL_SPACE_DIR)/model-space-parity.json) \
			$(if $(wildcard $(ROOT)/$(PRODUCT_OUTPUT_DIR)/product-output-parity.json),--product-output-report /app/$(PRODUCT_OUTPUT_DIR)/product-output-parity.json) \
			$(ARGS)

run-campaign: campaign-preflight
	cd "$(ROOT)" && "$(HOST_PYTHON)" -m benchmarks.scripts.campaign.run \
		--campaign-dir "$(ROOT)/$(CAMPAIGN_DIR)" \
		--make-campaign-dir "$(CAMPAIGN_DIR)" \
		--benchmarks-dir "$(ROOT)/benchmarks" \
		--idle-seconds "$(CAMPAIGN_IDLE_SECONDS)" \
		--execution-profile "$(EXECUTION_PROFILE)" \
		--vstrt-arguments="$(VSTRT_ARGS)" \
		--vsgan-arguments="$(VSGAN_ARGS)" \
		$(if $(filter 1,$(RESUME)),--resume,)

run-comparative: run-campaign
