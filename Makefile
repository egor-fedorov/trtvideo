IMAGE ?= ai-media-enhancer:latest
DEV_IMAGE ?= ai-media-enhancer:dev
BENCHMARK_IMAGE ?= ai-media-enhancer:benchmark
DOCKER_RUN ?= docker run --rm -v "$$PWD:/app"
BENCHMARK_MANIFEST ?= benchmarks/workloads/realesrgan_x2plus_sintel.json
BENCHMARK_ARGS ?=
BENCHMARK_VARIANT ?= 1080p
BENCHMARK_ENGINE ?=
BENCHMARK_OUTPUT_DIR ?= artefacts/benchmarks/ai-media-$(BENCHMARK_VARIANT)
GPU_ID ?= 0

.PHONY: build build-dev build-benchmark lint typecheck compile test-unit check shell
.PHONY: benchmark-prepare benchmark-verify benchmark-ai-media

build:
	DOCKER_BUILDKIT=1 docker build \
		--build-arg VCS_REF="$$(git rev-parse HEAD)" \
		--build-arg VCS_DIRTY="$$(if test -z "$$(git status --porcelain)"; then echo 0; else echo 1; fi)" \
		--target production \
		-t $(IMAGE) .

build-dev:
	DOCKER_BUILDKIT=1 docker build \
		--build-arg INSTALL_DEV=1 \
		--build-arg VCS_REF="$$(git rev-parse HEAD)" \
		--build-arg VCS_DIRTY="$$(if test -z "$$(git status --porcelain)"; then echo 0; else echo 1; fi)" \
		--target production \
		-t $(DEV_IMAGE) .

build-benchmark:
	DOCKER_BUILDKIT=1 docker build \
		--build-arg VCS_REF="$$(git rev-parse HEAD)" \
		--build-arg VCS_DIRTY="$$(if test -z "$$(git status --porcelain)"; then echo 0; else echo 1; fi)" \
		--target benchmark \
		-t $(BENCHMARK_IMAGE) .

lint:
	$(DOCKER_RUN) $(DEV_IMAGE) ruff check .

typecheck:
	$(DOCKER_RUN) $(DEV_IMAGE) mypy .

compile:
	$(DOCKER_RUN) $(DEV_IMAGE) python3 -m compileall -q ai_media benchmarks tests/unit

test-unit:
	$(DOCKER_RUN) $(DEV_IMAGE) python3 -m pytest -q tests/unit

check: lint typecheck compile test-unit

shell:
	$(DOCKER_RUN) -it $(DEV_IMAGE) bash

benchmark-prepare:
	mkdir -p models videos
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$$PWD/models:/app/models" \
		-v "$$PWD/videos:/app/videos" \
		-v "$$PWD/benchmarks:/app/benchmarks:ro" \
		$(IMAGE) python3 benchmarks/scripts/prepare_workload.py prepare \
			--manifest /app/$(BENCHMARK_MANIFEST) --root /app $(BENCHMARK_ARGS)

benchmark-verify:
	mkdir -p models videos
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$$PWD/models:/app/models" \
		-v "$$PWD/videos:/app/videos" \
		-v "$$PWD/benchmarks:/app/benchmarks:ro" \
		$(IMAGE) python3 benchmarks/scripts/prepare_workload.py verify \
			--manifest /app/$(BENCHMARK_MANIFEST) --root /app $(BENCHMARK_ARGS)

benchmark-ai-media:
	@test -n "$(BENCHMARK_ENGINE)" || (echo "ERROR: BENCHMARK_ENGINE is required" >&2; exit 2)
	mkdir -p models videos $(BENCHMARK_OUTPUT_DIR)
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" \
		-e AI_MEDIA_IMAGE_REF="$(BENCHMARK_IMAGE)" \
		-e AI_MEDIA_IMAGE_ID="$$(docker image inspect --format '{{.Id}}' $(BENCHMARK_IMAGE))" \
		-v "$$PWD/models:/app/models" \
		-v "$$PWD/videos:/app/videos" \
		-v "$$PWD/artefacts:/app/artefacts" \
		$(BENCHMARK_IMAGE) python3 benchmarks/scripts/run_ai_media.py \
			--manifest /app/$(BENCHMARK_MANIFEST) \
			--variant $(BENCHMARK_VARIANT) \
			--engine /app/$(BENCHMARK_ENGINE) \
			--gpu-id $(GPU_ID) \
			--output-dir /app/$(BENCHMARK_OUTPUT_DIR) \
			$(BENCHMARK_ARGS)
