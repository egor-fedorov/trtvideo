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
VSTRT_OUTPUT_DIR ?= artefacts/benchmarks/parity-vstrt-$(VARIANT)
VSGAN_OUTPUT_DIR ?= artefacts/benchmarks/product-vsgan-$(VARIANT)
GPU_ID ?= 0
ARGS ?=
TRTEXEC_ARGS ?=
VSTRT_ARGS ?=
VSGAN_ARGS ?=
WORKLOAD_NAME := $(basename $(notdir $(MANIFEST)))
CAMPAIGN_NAME ?= $(WORKLOAD_NAME)-$(VARIANT)
CAMPAIGN_DIR ?= artefacts/benchmarks/campaigns/$(CAMPAIGN_NAME)
CAMPAIGN_IDLE_SECONDS ?= 10
CAMPAIGN_RUN_ARGS = --runs 1 --extra-runs 0 --idle-seconds 0
RESUME ?= 0

.PHONY: help build build-vstrt build-vsgan build-vsgan-engine prepare verify
.PHONY: run-ai-media plan-trtexec plan-vstrt plan-vsgan dry-run
.PHONY: run-trtexec run-vstrt run-vsgan
.PHONY: campaign-preflight campaign-ai-media campaign-vstrt campaign-vsgan
.PHONY: aggregate-campaign run-campaign

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
		'run-trtexec         Run the internal TensorRT diagnostic (ENGINE required)' \
		'run-campaign        Run/resume the canonical rotated parity campaign' \
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

build-vsgan-engine:
	mkdir -p "$(ROOT)/models" "$(ROOT)/models/cache"
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e AI_MEDIA_IMAGE_REF="$(VSGAN_IMAGE)" \
		-e AI_MEDIA_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(VSGAN_IMAGE))" \
		-v "$(ROOT)/models:/app/models" \
		$(VSGAN_IMAGE) /opt/vsgan-benchmark/bin/python3 -m benchmarks.scripts.build_vsgan_engine \
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
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos" "$(ROOT)/$(VSTRT_OUTPUT_DIR)"
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
			--output-dir /app/$(VSTRT_OUTPUT_DIR) \
			$(ARGS) $(VSTRT_ARGS)

run-vsgan:
	@test -n "$(VSGAN_ENGINE)" || (echo "ERROR: VSGAN_ENGINE is required" >&2; exit 2)
	mkdir -p "$(ROOT)/models" "$(ROOT)/videos" "$(ROOT)/$(VSGAN_OUTPUT_DIR)"
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
			--output-dir /app/$(VSGAN_OUTPUT_DIR) \
			$(ARGS) $(VSGAN_ARGS)

campaign-preflight:
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
			'test "$$AI_MEDIA_BUILD_REVISION" = "$$EXPECTED_REVISION" && test "$$AI_MEDIA_BUILD_DIRTY" = "0"' || { \
			echo "ERROR: $$image was not built from clean revision $$revision" >&2; \
			exit 2; \
		}; \
	done

campaign-ai-media:
	@output="$(ROOT)/$(CAMPAIGN_DIR)/ai-media/round-$(ROUND)"; \
	manifest="$$output/run-01/manifest.json"; \
	if test -f "$$manifest"; then \
		test "$(RESUME)" = "1" || exit 2; \
		grep -q '"status": "valid"' "$$manifest" || { echo "ERROR: invalid existing $$manifest" >&2; exit 2; }; \
		echo "Resume: ai-media round $(ROUND) is already valid"; \
	elif test -e "$$output"; then \
		echo "ERROR: partial ai-media round $(ROUND); remove $$output before resume" >&2; exit 2; \
	else \
		$(MAKE) run-ai-media OUTPUT_DIR="$(CAMPAIGN_DIR)/ai-media/round-$(ROUND)" \
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
		$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.aggregate_campaign \
			--campaign-dir /app/$(CAMPAIGN_DIR) \
			--root /app \
			--idle-seconds $(CAMPAIGN_IDLE_SECONDS) \
			$(ARGS)

run-campaign: campaign-preflight
	@set -eu; \
	first=1; \
	run_sequence() { \
		for entry in "$$@"; do \
			if test "$$first" = "0"; then sleep "$(CAMPAIGN_IDLE_SECONDS)"; fi; \
			round="$${entry%%:*}"; implementation="$${entry#*:}"; \
			$(MAKE) "campaign-$$implementation" ROUND="$$round"; \
			first=0; \
		done; \
	}; \
	run_extra_sequence() { \
		run_sequence \
			04:vsgan 04:vstrt 04:ai-media \
			05:ai-media 05:vsgan 05:vstrt; \
	}; \
	aggregate() { \
		docker run --rm --user "$$(id -u):$$(id -g)" \
			-v "$(ROOT)/benchmarks:/app/benchmarks:ro" \
			-v "$(ROOT)/artefacts:/app/artefacts" \
			$(BENCHMARK_IMAGE) python3 -m benchmarks.scripts.aggregate_campaign \
				--campaign-dir /app/$(CAMPAIGN_DIR) \
				--root /app \
				--idle-seconds $(CAMPAIGN_IDLE_SECONDS) "$$@"; \
	}; \
	run_sequence \
		01:ai-media 01:vstrt 01:vsgan \
		02:vstrt 02:vsgan 02:ai-media \
		03:vsgan 03:ai-media 03:vstrt; \
	if test -d "$(ROOT)/$(CAMPAIGN_DIR)/vsgan/round-04" || \
		test -d "$(ROOT)/$(CAMPAIGN_DIR)/ai-media/round-05"; then \
		run_extra_sequence; \
		aggregate; \
		exit 0; \
	fi; \
	status=0; aggregate --request-extra-exit-code || status=$$?; \
	if test "$$status" = "3"; then \
		run_extra_sequence; \
		aggregate; \
	elif test "$$status" != "0"; then \
		exit "$$status"; \
	fi
