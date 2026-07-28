IMAGE ?= trtvideo:latest
DEV_IMAGE ?= trtvideo:dev
DOCKER_RUN ?= docker run --rm -e PYTHONPATH=/app/src -v "$$PWD:/app"
DEMO_DIR := $(CURDIR)/.demo
DEMO_GPU_ID ?= 0
DEMO_FORCE ?= 0
DEMO_FORCE_ARG = $(if $(filter 1 true yes,$(DEMO_FORCE)),--force,)

.PHONY: build build-dev demo demo-clean lint typecheck compile test-unit test-media-integration check cli-smoke shell

build:
	DOCKER_BUILDKIT=1 docker build \
		--build-arg VCS_REF="$$(git rev-parse HEAD)" \
		--build-arg VCS_DIRTY="$$(if test -z "$$(git status --porcelain)"; then echo 0; else echo 1; fi)" \
		--target production \
		-t $(IMAGE) .

build-dev:
	DOCKER_BUILDKIT=1 docker build \
		-f docker/checks.Dockerfile \
		-t $(DEV_IMAGE) .

demo: build
	mkdir -p "$(DEMO_DIR)"
	docker run --rm --gpus all \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(DEMO_DIR):/app/.demo" \
		$(IMAGE) python3 -m trtvideo.cli.demo \
			--root /app/.demo \
			--gpu-id "$(DEMO_GPU_ID)" $(DEMO_FORCE_ARG)
	@printf 'Validated output: %s/output/demo_1440p.mkv\n' "$(DEMO_DIR)"

demo-clean:
	@test "$(abspath $(DEMO_DIR))" = "$(CURDIR)/.demo" || \
		{ printf 'Refusing to remove unexpected DEMO_DIR: %s\n' "$(DEMO_DIR)"; exit 2; }
	rm -rf "$(DEMO_DIR)"

lint:
	$(DOCKER_RUN) $(DEV_IMAGE) ruff check .

typecheck:
	$(DOCKER_RUN) $(DEV_IMAGE) mypy

compile:
	$(DOCKER_RUN) $(DEV_IMAGE) python3 -m compileall -q src/trtvideo benchmarks tests
	$(DOCKER_RUN) $(DEV_IMAGE) python3 -m py_compile \
		benchmarks/vstrt/upscale.vpy benchmarks/vsgan/upscale.vpy

test-unit:
	$(DOCKER_RUN) $(DEV_IMAGE) python3 -m pytest -q tests/unit

test-media-integration:
	$(DOCKER_RUN) $(DEV_IMAGE) python3 -m pytest -q -m docker tests/integration

cli-smoke:
	$(DOCKER_RUN) $(DEV_IMAGE) upscale --help
	$(DOCKER_RUN) $(DEV_IMAGE) benchmark-upscale --help
	$(DOCKER_RUN) $(DEV_IMAGE) python3 -m trtvideo.cli.demo --help
	$(DOCKER_RUN) $(DEV_IMAGE) export-onnx --help
	$(DOCKER_RUN) $(DEV_IMAGE) prepare-onnx --help
	$(DOCKER_RUN) $(DEV_IMAGE) build-engine --help

check: lint typecheck compile test-unit test-media-integration cli-smoke

shell:
	$(DOCKER_RUN) -it $(DEV_IMAGE) bash
