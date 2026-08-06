# Security Policy

## Supported Versions

Before the first public release, security fixes are applied only to `main`.
After release, `main` and the latest published version are supported. Older
versions may be asked to reproduce the issue against one of those revisions.

## Reporting A Vulnerability

Use [GitHub private vulnerability reporting](https://github.com/egor-fedorov/trtvideo/security/advisories/new).
Do not open a public issue, discussion, or pull request for a suspected
vulnerability.

Include:

- the affected release, commit, and Docker image reference or digest;
- impact and the conditions required to trigger it;
- minimal reproduction steps or a proof of concept;
- relevant host, GPU, driver, Docker, CUDA, and TensorRT versions;
- any known workaround.

Remove credentials, private media, proprietary model data, absolute host paths,
hostnames, and GPU UUIDs from the report. Reports are handled on a best-effort
basis; this project does not currently promise a response or remediation SLA.

Vulnerabilities originating in TensorRT, CUDA, CV-CUDA, FFmpeg,
PyNvVideoCodec, Docker, or another dependency should also be reported to the
appropriate upstream project. Report them here as well when the project's
default configuration or integration materially changes their impact.
