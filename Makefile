IMAGE ?= trtvideo:latest
MODEL_TOOLS_IMAGE ?= trtvideo:model-tools
DEV_IMAGE ?= trtvideo:dev
DOCKER_RUN ?= docker run --rm -e PYTHONPATH=/app/src -v "$$PWD:/app"
PROJECT_VERSION := $(shell sed -n 's/^version = "\([^"]*\)"$$/\1/p' pyproject.toml)
DEMO_DIR := $(CURDIR)/.demo
DEMO_GPU_ID ?= 0
DEMO_FORCE ?= 0
DEMO_FORCE_ARG = $(if $(filter 1 true yes,$(DEMO_FORCE)),--force,)

ifeq ($(strip $(PROJECT_VERSION)),)
$(error Could not read the project version from pyproject.toml)
endif

.PHONY: build build-model-tools build-dev demo demo-clean figures figures-check format format-check lint typecheck typecheck-mypy typecheck-ty compile test-unit test-media-integration check cli-smoke shell

build:
	DOCKER_BUILDKIT=1 docker build \
		--build-arg BUILD_DATE="$$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
		--build-arg VERSION="$(PROJECT_VERSION)" \
		--build-arg VCS_REF="$$(git rev-parse HEAD)" \
		--build-arg VCS_DIRTY="$$(if test -z "$$(git status --porcelain)"; then echo 0; else echo 1; fi)" \
		--build-arg IMAGE_REFERENCE="$(IMAGE)" \
		--target production \
		-t $(IMAGE) .

build-model-tools:
	DOCKER_BUILDKIT=1 docker build \
		--build-arg BUILD_DATE="$$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
		--build-arg VERSION="$(PROJECT_VERSION)" \
		--build-arg VCS_REF="$$(git rev-parse HEAD)" \
		--build-arg VCS_DIRTY="$$(if test -z "$$(git status --porcelain)"; then echo 0; else echo 1; fi)" \
		--build-arg IMAGE_REFERENCE="$(MODEL_TOOLS_IMAGE)" \
		--target model-tools \
		-t $(MODEL_TOOLS_IMAGE) .

build-dev:
	DOCKER_BUILDKIT=1 docker build \
		-f docker/checks.Dockerfile \
		-t $(DEV_IMAGE) .

demo: build-model-tools
	mkdir -p "$(DEMO_DIR)"
	docker run --rm --gpus all \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-v "$(DEMO_DIR):/app/.demo" \
		$(MODEL_TOOLS_IMAGE) python3 -m trtvideo.cli.demo \
			--root /app/.demo \
			--gpu-id "$(DEMO_GPU_ID)" $(DEMO_FORCE_ARG)
	@printf 'Validated output: %s/output/demo_1440p.mp4\n' "$(DEMO_DIR)"

demo-clean:
	@test "$(abspath $(DEMO_DIR))" = "$(CURDIR)/.demo" || \
		{ printf 'Refusing to remove unexpected DEMO_DIR: %s\n' "$(DEMO_DIR)"; exit 2; }
	rm -rf "$(DEMO_DIR)"

figures:
	$(MAKE) -C benchmarks figures

figures-check:
	$(MAKE) -C benchmarks figures-check

format:
	$(DOCKER_RUN) $(DEV_IMAGE) ruff check --select I --fix .
	$(DOCKER_RUN) $(DEV_IMAGE) ruff format .

format-check:
	$(DOCKER_RUN) $(DEV_IMAGE) ruff format --check .

lint: format-check
	$(DOCKER_RUN) $(DEV_IMAGE) ruff check .

typecheck: typecheck-mypy typecheck-ty

typecheck-mypy:
	$(DOCKER_RUN) $(DEV_IMAGE) mypy

typecheck-ty:
	$(DOCKER_RUN) $(DEV_IMAGE) ty check

compile:
	$(DOCKER_RUN) $(DEV_IMAGE) python3 -m compileall -q src/trtvideo benchmarks tests
	$(DOCKER_RUN) $(DEV_IMAGE) python3 -m py_compile \
		benchmarks/vstrt/upscale.vpy

test-unit:
	$(DOCKER_RUN) $(DEV_IMAGE) python3 -m pytest -q tests/unit

test-media-integration:
	$(DOCKER_RUN) $(DEV_IMAGE) python3 -m pytest -q -m docker tests/integration

define cli_help_check
	@if output="$$( $(DOCKER_RUN) $(DEV_IMAGE) $(1) --help 2>&1 )"; then \
		printf 'PASS %s\n' '$(1) --help'; \
	else \
		status=$$?; \
		printf 'FAIL %s (exit %s)\n' '$(1) --help' "$$status" >&2; \
		printf '%s\n' "$$output" >&2; \
		exit "$$status"; \
	fi
endef

cli-smoke:
	$(call cli_help_check,trtvideo)
	$(call cli_help_check,trtvideo doctor)
	$(call cli_help_check,trtvideo compatibility-check)
	$(call cli_help_check,trtvideo compatibility-report)
	$(call cli_help_check,benchmark-trtvideo)
	$(call cli_help_check,python3 -m trtvideo.cli.demo)
	$(call cli_help_check,export-onnx)
	$(call cli_help_check,prepare-onnx)
	$(call cli_help_check,prepare-compatibility-input)
	$(call cli_help_check,build-engine)

check: lint typecheck compile test-unit test-media-integration cli-smoke figures-check

shell:
	$(DOCKER_RUN) -it $(DEV_IMAGE) bash
