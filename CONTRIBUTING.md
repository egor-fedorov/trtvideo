# Contributing

`trtvideo` is Docker-first. Local Python installations are not expected to
provide TensorRT, CV-CUDA, PyNvVideoCodec, or the project's pinned Python 3.12
runtime.

## Report An Issue

Use the structured GitHub issue forms and include the exact command, complete
error output, repository revision or release, Docker image identity, GPU model,
NVIDIA driver, and TensorRT version. For output-contract problems, include
relevant `ffprobe` output. Do not upload private media, proprietary model
weights, credentials, hostnames, or GPU UUIDs.

Security vulnerabilities must not be filed as public issues. Follow
[`SECURITY.md`](SECURITY.md) instead.

## Report Model Compatibility

Use the dedicated
[model compatibility form](https://github.com/egor-fedorov/trtvideo/issues/new?template=model_compatibility.yml)
for a successful or failed model experiment. Follow
[`docs/MODEL_CONTRACT.md`](docs/MODEL_CONTRACT.md) and include the immutable
model source and SHA256, exact conversion and runtime commands, engine sidecar,
`trtvideo doctor` output, and complete smoke-test result. Do not upload model
weights unless their license permits redistribution.

An issue does not automatically make a model `validated`. A reproducible
successful report may be listed as `community-reported`; `validated` requires
published inference and product-output quality evidence.

## Development Workflow

Build the lightweight checks image and run the complete gate before submitting
Python changes:

```bash
make build-dev
make check
```

`make check` covers Ruff, formatting, mypy, compileall, unit tests, GPU-free
media integration, CLI smoke tests, and generated benchmark-figure drift. Test
layers and GPU acceptance criteria are documented in
[`docs/TESTING.md`](docs/TESTING.md).

Use `make format` to apply repository formatting. Unit tests must remain pure
Python and must not import TensorRT, CV-CUDA, or PyNvVideoCodec.

## GPU And Benchmark Changes

- Run a short `--max-frames` or benchmark smoke test before a full GPU run.
- For color, encoding, mux, or media-preservation changes, validate frame count,
  timestamps, full decode, pixel format, color tags, bitrate, duration, and
  preserved streams.
- State the GPU, driver, image revision, commands, and checks used in the pull
  request. State explicitly when GPU verification was unavailable.
- Do not turn ad hoc measurements into performance claims. Comparative evidence
  must follow [`benchmarks/methodology.md`](benchmarks/methodology.md) from a
  clean revision and pass its quality gates.
- Do not commit models, videos, TensorRT engines, or large raw benchmark
  artifacts. Only compact privacy-reviewed publications belong under
  `benchmarks/results/`.

## Documentation And Pull Requests

Keep each pull request focused and explain the behavior change, test coverage,
and remaining risks. Use short-lived topic branches; GitHub automatically
deletes their remote head branches after merge. Update the canonical document
when behavior changes:

- `README.md` for public workflows and CLI usage;
- `docs/ARCHITECTURE.md` for runtime architecture;
- `docs/MODEL_CONTRACT.md` for supported model and compatibility evidence;
- `docs/TESTING.md` for test contracts;
- `docs/LICENSING.md` and `THIRD_PARTY_NOTICES.md` when the distributed image or
  its dependency boundary changes;
- `docs/CHANGES.md` for notable release-facing changes;
- `docs/PERFORMANCE_LOG.md` only when a comparable measurement exists.

The `Unreleased` changelog describes the net change from the latest release,
not intermediate implementation history.
