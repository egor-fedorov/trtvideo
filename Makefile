IMAGE ?= ai-media-enhancer:latest
DEV_IMAGE ?= ai-media-enhancer:dev
DOCKER_RUN ?= docker run --rm -v "$$PWD:/app"
BENCHMARK_MANIFEST ?= benchmarks/workloads/realesrgan_x2plus_sintel.json
BENCHMARK_ARGS ?=

.PHONY: build build-dev lint typecheck compile test-unit check shell
.PHONY: benchmark-prepare benchmark-verify

build:
	DOCKER_BUILDKIT=1 docker build -t $(IMAGE) .

build-dev:
	DOCKER_BUILDKIT=1 docker build --build-arg INSTALL_DEV=1 -t $(DEV_IMAGE) .

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
