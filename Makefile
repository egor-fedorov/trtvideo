IMAGE ?= ai-media-enhancer:latest
DEV_IMAGE ?= ai-media-enhancer:dev
DOCKER_RUN ?= docker run --rm -v "$$PWD:/app"

.PHONY: build build-dev lint typecheck compile test-unit check shell

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

lint:
	$(DOCKER_RUN) $(DEV_IMAGE) ruff check .

typecheck:
	$(DOCKER_RUN) $(DEV_IMAGE) mypy

compile:
	$(DOCKER_RUN) $(DEV_IMAGE) python3 -m compileall -q ai_media benchmarks tests/unit

test-unit:
	$(DOCKER_RUN) $(DEV_IMAGE) python3 -m pytest -q tests/unit

check: lint typecheck compile test-unit

shell:
	$(DOCKER_RUN) -it $(DEV_IMAGE) bash
