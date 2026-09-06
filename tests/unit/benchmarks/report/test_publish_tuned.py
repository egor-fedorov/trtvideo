import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.scripts.report.publish_tuned import (
    CANONICAL_WORKLOADS,
    EvidenceSource,
    PublicationError,
    _identity_interpretation,
    _intra_session_reproducibility,
    _run_observations,
    _tensor_set_digest,
    _workloads_for_source,
    build_document,
)
from benchmarks.scripts.tuning.matrix import verify_matrix


def test_tensor_set_digest_is_order_independent() -> None:
    artifacts = [
        {"stage": "output", "frame_index": 1, "sha256": "b" * 64},
        {"stage": "input", "frame_index": 0, "sha256": "a" * 64},
    ]
    payload = f"input\t0\t{'a' * 64}\noutput\t1\t{'b' * 64}\n"
    expected = hashlib.sha256(payload.encode()).hexdigest()

    assert _tensor_set_digest({"artifacts": artifacts}) == expected
    assert _tensor_set_digest({"artifacts": list(reversed(artifacts))}) == expected


def test_evidence_source_maps_only_canonical_paths(tmp_path: Path) -> None:
    source = EvidenceSource(tmp_path)
    evidence = tmp_path / "workload" / "selection.json"

    assert source.canonical(evidence) == (
        "artefacts/benchmarks/comparative/tuning/workload/selection.json"
    )
    assert source.resolve(source.canonical(evidence)) == evidence

    with pytest.raises(PublicationError, match="escapes tuned evidence"):
        source.resolve("artefacts/benchmarks/project/suite.json")


def test_publication_requires_complete_canonical_media_contract(tmp_path: Path) -> None:
    for base in {item[0] for item in CANONICAL_WORKLOADS}:
        (tmp_path / f"{base}-matrix.json").write_text("{}\n", encoding="utf-8")

    assert _workloads_for_source(EvidenceSource(tmp_path)) == CANONICAL_WORKLOADS


def test_publication_rejects_incomplete_canonical_media_contract(tmp_path: Path) -> None:
    (tmp_path / "realesrgan_x2plus_madrid-matrix.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(PublicationError, match="complete canonical Madrid matrix"):
        _workloads_for_source(EvidenceSource(tmp_path))


def test_identity_interpretation_does_not_overclaim_mixed_outputs() -> None:
    identities = [
        {
            "candidate_tensor_sha256_sets_identical": True,
            "candidate_mp4_sha256_identical": True,
        },
        {
            "candidate_tensor_sha256_sets_identical": False,
            "candidate_mp4_sha256_identical": False,
        },
    ]

    interpretation = _identity_interpretation(identities)

    assert "reported per workload" in interpretation
    assert "quality gates pass" in interpretation


def test_run_observations_summarize_all_campaign_rounds(tmp_path: Path) -> None:
    source = EvidenceSource(tmp_path)
    manifest_paths = []
    for index, (temperature, capped, reasons) in enumerate(
        ((54, False, ["gpu_idle"]), (58, True, ["sw_power_cap"])),
        start=1,
    ):
        path = tmp_path / f"round-{index:02d}" / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "measured": {
                        "metrics": {
                            "nvml": {
                                "temperature": {"peak_c": temperature},
                                "power": {"power_cap_observed": capped},
                                "throttle_reasons": reasons,
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        manifest_paths.append(source.canonical(path))

    campaign = {
        "rounds": [{"manifests": {"trtvideo": manifest_path}} for manifest_path in manifest_paths]
    }

    assert _run_observations(source, campaign, "trtvideo") == {
        "peak_temperature_c": 58,
        "power_cap_observed": True,
        "throttle_reasons": ["gpu_idle", "sw_power_cap"],
    }


@pytest.mark.parametrize("external", ("vsgan", "tas"))
def test_intra_session_reproducibility_compares_selected_external_profiles(external: str) -> None:
    selection = {
        "winners": {
            "vstrt": {"median_fps": 10.0},
            external: {"median_fps": 20.0},
        }
    }
    campaign = {
        "implementations": {
            "vstrt": {"statistics": {"median_fps": 10.04}},
            external: {"statistics": {"median_fps": 19.96}},
        }
    }

    result = _intra_session_reproducibility(selection, campaign)

    assert result["max_absolute_delta_percent"] == pytest.approx(0.4)
    assert [item["implementation"] for item in result["comparisons"]] == ["vstrt", external]


def test_evidence_source_rejects_relative_traversal(tmp_path: Path) -> None:
    with pytest.raises(PublicationError, match="escapes tuned evidence"):
        EvidenceSource(tmp_path).resolve("artefacts/benchmarks/comparative/tuning/../private.json")


def write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def artifact(source: EvidenceSource, path: Path) -> dict[str, str]:
    return {
        "path": source.canonical(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def refresh_campaign_hash(source: EvidenceSource, path: Path) -> None:
    for matrix_path in source.root.glob("*-matrix.json"):
        matrix = json.loads(matrix_path.read_text())
        for entry in matrix["variants"].values():
            if entry["campaign"]["path"] == source.canonical(path):
                entry["campaign"] = artifact(source, path)
                write_json(matrix_path, matrix)


@pytest.fixture
def tas_evidence(tmp_path: Path) -> EvidenceSource:
    """Small synthetic full publication: no GPU imports or real TAS claims."""
    source = EvidenceSource(tmp_path / "artefacts/benchmarks/comparative/tuning")
    implementations = json.loads(Path("benchmarks/implementations.json").read_text())[
        "implementations"
    ]
    revision = "a" * 40
    hardware = {
        "cpu": {"model": "Synthetic CPU", "logical_cores": 12},
        "gpu": {
            "name": "Synthetic GPU",
            "driver_version": "test-driver",
            "power_limit_w": 350,
        },
    }
    profile = {
        "execution_profile": "tuned",
        "decode_method": "nvdec",
        "writer": "nelux",
        "cuda_graph": True,
    }
    tas = {
        "implementation": "tas",
        "candidate_id": "tas-nvdec-nelux",
        "status": "eligible",
        "execution_profile": profile,
        "runner_arguments": ["--decode-method", "nvdec", "--writer", "nelux"],
        "median_fps": 10.0,
        "median_cpu_cores": 1.5,
        "median_peak_vram_mib": 3000.0,
    }
    vstrt = {
        "implementation": "vstrt",
        "candidate_id": "vstrt-s1-g0",
        "status": "eligible",
        "execution_profile": {
            "execution_profile": "tuned",
            "num_streams": 1,
            "vapoursynth_threads": "auto",
            "vspipe_requests": "auto",
            "cuda_graph": False,
        },
        "runner_arguments": [],
        "median_fps": 9.0,
    }
    matrices: dict[str, Any] = {}
    for base, _, variant in CANONICAL_WORKLOADS:
        directory = source.root / f"{base}-{variant}"
        quality_root = directory / "winner-quality/tas-nvdec-nelux__vstrt-s1-g0"
        comparisons = []
        inference_comparisons = []
        for name in ("vstrt", "tas"):
            capture_path = write_json(
                quality_root / f"tensor-quality/inference/{name}/manifest.json",
                {
                    "implementation": name,
                    "artifacts": [{"stage": "output", "frame_index": 0, "sha256": name * 16}],
                },
            )
            inference_comparisons.append(
                {
                    "implementation": name,
                    "status": "valid",
                    "capture_manifest_sha256": artifact(source, capture_path)["sha256"],
                    "tensors": [
                        {
                            "stage": "output",
                            "metrics": {"p99_abs": 0.001, "rmse": 0.0001, "psnr_db": 60},
                        }
                    ],
                }
            )
            comparisons.append(
                {
                    "implementation": name,
                    "status": "valid",
                    "output_sha256": name * 16,
                    "run_manifest_sha256": name * 16,
                    "metrics": {"psnr": {"average_db": 45, "frames": 1000}, "ssim": {"all": 0.99}},
                }
            )
        product_path = write_json(
            quality_root / "product-output/product-output-parity.json",
            {
                "status": "valid",
                "publishable": True,
                "workload_id": f"{base}-v1",
                "variant": variant,
                "comparisons": comparisons,
            },
        )
        tensor_paths = {}
        for kind, status in (
            ("inference-parity", "valid"),
            ("preprocessing-diagnostic", "complete"),
        ):
            tensor_paths[kind] = write_json(
                quality_root / f"tensor-quality/{kind}.json",
                {
                    "status": status,
                    "publishable": True,
                    "workload_id": f"{base}-v1",
                    "variant": variant,
                    "acceptance_gate": kind == "inference-parity",
                    "comparisons": inference_comparisons if kind == "inference-parity" else [],
                    "contract_version": 3,
                    "frame_indices": [0, 499, 999],
                    "thresholds": {},
                },
            )
        selection = {
            "status": "valid",
            "workload_id": f"{base}-v1",
            "benchmark_contract_version": 5,
            "selection_policy": {"equivalence_margin": 0.01},
            "search": {
                "completion": {"vstrt": "range-exhausted", "tas": "grid-exhausted"},
                "resource_limits": {"vstrt": None, "tas": []},
            },
            "winners": {"vstrt": vstrt, "tas": tas},
            "reconnaissance": [vstrt, tas],
            "candidates": [vstrt, tas],
            "disqualifications": {},
        }
        preflight_path = write_json(
            directory / "tas-preflight/preflight.json",
            {
                "status": "complete",
                "candidates": [
                    {
                        "candidate_id": tas["candidate_id"],
                        "execution_profile": profile,
                        "status": "valid",
                        "quality": {},
                    }
                ],
            },
        )
        selection["tas_preflight"] = {
            "path": source.canonical(preflight_path),
            "sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        }
        write_json(directory / "selection.json", selection)
        write_json(directory / "search-state.json", {"status": "complete"})
        campaign: dict[str, Any] = {
            "status": "valid",
            "publishable": True,
            "execution_profile": "tuned",
            "workload_id": f"{base}-v1",
            "benchmark_contract_version": 5,
            "variant": variant,
            "environment": {"repository_revision": revision, **hardware},
            "parameters": {
                "execution_profiles": {"vstrt": vstrt["execution_profile"], "tas": profile},
                "tensor_quality_contract_version": 3,
            },
            "assets": {"onnx_sha256": "o" * 64},
            "implementations": {},
            "rounds": [],
        }
        manifests = {}
        for name, fps in (("trtvideo", 11), ("vstrt", 9), ("tas", 10)):
            # Same engine bytes are legitimate; image/run/capture provenance remains separate.
            environment = {
                "image": {"repository_revision": revision, "id": name, "source_dirty": "0"},
                "software": {"python": "3.12"},
                "implementation": implementations.get(name, {}),
                **hardware,
            }
            run: dict[str, Any] = {
                "status": "valid",
                "started_at_utc": "2026-09-06T00:00:00Z",
                "environment": environment,
                "assets": {"engine_manifest": {"sha256": "m" * 64}},
                "measured": {
                    "validation": {},
                    "metrics": {
                        "nvml": {
                            "temperature": {"peak_c": 50},
                            "power": {"power_cap_observed": False},
                            "throttle_reasons": [],
                        }
                    },
                },
            }
            if name == "tas":
                run["measured"]["validation"]["runtime_evidence"] = {
                    "status": "valid",
                    "errors": [],
                    "source_revision": implementations["tas"]["source_revision"],
                    "adapter_sha256": "c" * 64,
                    "decode_method": "nvdec",
                    "writer": "nelux",
                    "cuda_graph": True,
                    "engine_reused": True,
                    "engine_sha256": "e" * 64,
                    "onnx_sha256": "o" * 64,
                    "runtime": {
                        **{
                            key: implementations["tas"][f"{key}_version"]
                            for key in ("python", "torch", "tensorrt", "nelux")
                        },
                        "ffmpeg": "8",
                    },
                }
            run_path = write_json(
                directory / f"winner-campaign/test/{name}/round-01/run-01/manifest.json", run
            )
            manifests[name] = source.canonical(run_path)
            statistics: dict[str, Any] = {
                f"median_{metric}": 1.0
                for metric in (
                    "wall_time_sec",
                    "startup_sec",
                    "steady_state_frame_loop_sec",
                    "finalize_mux_sec",
                    "cpu_cores",
                    "cpu_capacity_percent",
                    "gpu_utilization_percent",
                    "power_w",
                    "joules_per_frame",
                    "peak_vram_mib",
                    "output_bitrate_mbps",
                    "output_size_mib",
                )
            }
            statistics.update(
                median_fps=fps,
                values_fps=[fps],
                relative_spread=0,
                median_lifecycle_intervals_sec={},
            )
            campaign["implementations"][name] = {
                "product": name,
                "image_id": name,
                "engine_sha256": "e" * 64,
                "relative_to_trtvideo_percent": 0,
                "statistics": statistics,
                "stability": {},
            }
        campaign["rounds"] = [{"manifests": manifests}]
        campaign_path = write_json(directory / "winner-campaign/test/campaign.json", campaign)
        matrix = matrices.setdefault(
            base,
            {
                "status": "valid",
                "publishable": True,
                "environment": {"repository_revision": revision, "gpu": hardware["gpu"]},
                "variants": {},
            },
        )
        matrix["variants"][variant] = {
            "campaign": artifact(source, campaign_path),
            "quality": {
                "inference_parity": artifact(source, tensor_paths["inference-parity"]),
                "preprocessing_diagnostic": artifact(
                    source, tensor_paths["preprocessing-diagnostic"]
                ),
                "product_output": artifact(source, product_path),
            },
        }
    for base, matrix in matrices.items():
        write_json(source.root / f"{base}-matrix.json", matrix)
    return source


def publish(source: EvidenceSource) -> dict[str, Any]:
    return build_document(
        source, Path("benchmarks/implementations.json"), Path("benchmarks/tuning/candidates.json")
    )


def test_active_publication_retains_native_tas_metadata_and_selection(
    tas_evidence: EvidenceSource,
) -> None:
    document = publish(tas_evidence)
    assert document["schema_version"] == 6
    assert document["scope"]["participants"] == ["trtvideo", "vstrt", "tas"]
    assert document["environment"]["implementations"]["tas"]["product"] == "TheAnimeScripter"
    workload = document["workloads"][0]
    tas = workload["final_campaign"]["results"][2]
    metadata = document["environment"]["implementations"]["tas"]
    for name in ("python", "torch", "tensorrt", "nelux"):
        assert tas["runtime"]["software"][name] == metadata[f"{name}_version"]
    assert tas["runtime"]["native_execution"]["adapter_sha256"] == "c" * 64
    assert tas["execution_profile"]["writer"] == "nelux"
    assert tas["run_evidence"][0]["sha256"]
    assert tas["engine_manifest_sha256"] == "m" * 64
    assert workload["selection"]["candidates"][1]["median_cpu_cores"] == 1.5
    assert workload["selection"]["search"]["completion"]["tas"] == "grid-exhausted"
    assert workload["selection"]["tas_preflight"]["candidates"][0]["status"] == "valid"
    assert (
        document["external_output_identity"]["independent_provenance"]["engine_sha256_differ"]
        is False
    )
    assert document["external_output_identity"]["all_candidate_tensor_sets_identical"] is False


def test_publication_consumes_matrix_producer_evidence_contract(
    tas_evidence: EvidenceSource, tmp_path: Path
) -> None:
    for base in dict.fromkeys(item[0] for item in CANONICAL_WORKLOADS):
        matrix_path = tas_evidence.root / f"{base}-matrix.json"
        matrix = json.loads(matrix_path.read_text())
        wrappers = {}
        for variant, entry in matrix["variants"].items():
            directory = tas_evidence.root / f"{base}-{variant}"
            selection = json.loads((directory / "selection.json").read_text())
            wrappers[variant] = write_json(
                directory / "final-campaign.json",
                {
                    "schema_version": 3,
                    "document_type": "tuned-winner-campaign",
                    "status": "valid",
                    "variant": variant,
                    "winners": selection["winners"],
                    "campaign": entry["campaign"],
                    "quality": entry["quality"],
                },
            )
        write_json(matrix_path, verify_matrix(root=tmp_path, campaign_reports=wrappers))

    document = publish(tas_evidence)

    assert document["status"] == "valid"
    assert document["publishable"] is True
    for item in document["publication_matrices"]:
        assert item["evidence"]["document_type"] == "tuned-publication-matrix"
    for workload in document["workloads"]:
        base = workload["workload_id"].removesuffix("-v1")
        matrix = json.loads((tas_evidence.root / f"{base}-matrix.json").read_text())
        entry = matrix["variants"][workload["variant"]]
        final = workload["final_campaign"]
        assert final["campaign"] == entry["campaign"]
        for kind, record in entry["quality"].items():
            assert final["quality"][kind]["sha256"] == record["sha256"]
            assert final["quality"][kind]["path"] == record["path"]


def test_active_publication_rejects_legacy_participants(tas_evidence: EvidenceSource) -> None:
    path = next(tas_evidence.root.glob("*/winner-campaign/test/campaign.json"))
    campaign = json.loads(path.read_text())
    campaign["implementations"]["vsgan"] = campaign["implementations"].pop("tas")
    write_json(path, campaign)
    refresh_campaign_hash(tas_evidence, path)
    with pytest.raises(PublicationError, match="trtvideo, vstrt, and tas"):
        publish(tas_evidence)


def test_publication_rejects_tas_backend_fallback(tas_evidence: EvidenceSource) -> None:
    path = next(tas_evidence.root.glob("*/winner-campaign/test/tas/round-01/run-01/manifest.json"))
    run = json.loads(path.read_text())
    run["measured"]["validation"]["runtime_evidence"]["writer"] = "ffmpeg"
    write_json(path, run)
    with pytest.raises(PublicationError, match="native execution differs"):
        publish(tas_evidence)


def test_publication_rejects_changed_tas_preflight(tas_evidence: EvidenceSource) -> None:
    path = next(tas_evidence.root.glob("*/tas-preflight/preflight.json"))
    preflight = json.loads(path.read_text())
    preflight["candidates"][0]["status"] = "invalid"
    write_json(path, preflight)
    with pytest.raises(PublicationError, match="preflight evidence changed"):
        publish(tas_evidence)


def test_publication_rejects_changed_tas_runtime_between_rounds(
    tas_evidence: EvidenceSource,
) -> None:
    path = next(tas_evidence.root.glob("*/winner-campaign/test/campaign.json"))
    campaign = json.loads(path.read_text())
    run_path = tas_evidence.resolve(campaign["rounds"][0]["manifests"]["tas"])
    run = json.loads(run_path.read_text())
    run["measured"]["validation"]["runtime_evidence"]["runtime"]["nelux"] = "different"
    second = write_json(run_path.with_name("changed.json"), run)
    campaign["rounds"].append(
        {
            "manifests": {
                **campaign["rounds"][0]["manifests"],
                "tas": tas_evidence.canonical(second),
            }
        }
    )
    write_json(path, campaign)
    refresh_campaign_hash(tas_evidence, path)
    with pytest.raises(PublicationError, match="between campaign rounds"):
        publish(tas_evidence)


@pytest.mark.parametrize(
    "kind", ("campaign", "inference_parity", "preprocessing_diagnostic", "product_output")
)
@pytest.mark.parametrize("change", ("content", "missing_hash"))
def test_publication_requires_matrix_artifact_hashes(
    tas_evidence: EvidenceSource, kind: str, change: str
) -> None:
    path = next(tas_evidence.root.glob("*-matrix.json"))
    matrix = json.loads(path.read_text())
    entry = matrix["variants"]["720p"]
    record = entry["campaign"] if kind == "campaign" else entry["quality"][kind]
    if change == "content":
        artifact_path = tas_evidence.resolve(record["path"])
        report = json.loads(artifact_path.read_text())
        report["status"] = "invalid"
        write_json(artifact_path, report)
    else:
        del record["sha256"]
        write_json(path, matrix)

    with pytest.raises(PublicationError, match="SHA256 changed or is missing"):
        publish(tas_evidence)


@pytest.mark.parametrize("kind", ("inference_parity", "preprocessing_diagnostic", "product_output"))
@pytest.mark.parametrize(
    ("field", "value"),
    (("status", "invalid"), ("publishable", False), ("variant", "1080p"), ("workload_id", "other")),
)
def test_publication_rechecks_quality_contract_with_matching_hash(
    tas_evidence: EvidenceSource, kind: str, field: str, value: Any
) -> None:
    matrix_path = next(tas_evidence.root.glob("*-matrix.json"))
    matrix = json.loads(matrix_path.read_text())
    entry = matrix["variants"]["720p"]["quality"]
    path = tas_evidence.resolve(entry[kind]["path"])
    report = json.loads(path.read_text())
    report[field] = value
    write_json(path, report)
    entry[kind] = artifact(tas_evidence, path)
    write_json(matrix_path, matrix)

    with pytest.raises(PublicationError, match=f"{kind} quality evidence is not publishable"):
        publish(tas_evidence)


@pytest.mark.parametrize("kind", ("inference_parity", "preprocessing_diagnostic"))
@pytest.mark.parametrize("field", ("acceptance_gate", "contract_version"))
def test_publication_rechecks_tensor_quality_role(
    tas_evidence: EvidenceSource, kind: str, field: str
) -> None:
    matrix_path = next(tas_evidence.root.glob("*-matrix.json"))
    matrix = json.loads(matrix_path.read_text())
    entry = matrix["variants"]["720p"]["quality"]
    path = tas_evidence.resolve(entry[kind]["path"])
    report = json.loads(path.read_text())
    report[field] = not report[field] if field == "acceptance_gate" else 999
    write_json(path, report)
    entry[kind] = artifact(tas_evidence, path)
    write_json(matrix_path, matrix)

    with pytest.raises(
        PublicationError, match=f"{kind} (acceptance role|contract version) changed"
    ):
        publish(tas_evidence)


def test_publication_binds_capture_hashes_to_verified_inference_report(
    tas_evidence: EvidenceSource,
) -> None:
    path = next(
        tas_evidence.root.glob("*/winner-quality/*/tensor-quality/inference/tas/manifest.json")
    )
    capture = json.loads(path.read_text())
    capture["artifacts"][0]["sha256"] = "0" * 64
    write_json(path, capture)

    with pytest.raises(PublicationError, match="capture SHA256 differs"):
        publish(tas_evidence)


@pytest.mark.parametrize(
    ("component", "field", "value"),
    (
        ("gpu", "name", "Other GPU"),
        ("gpu", "driver_version", "Other driver"),
        ("gpu", "power_limit_w", 450),
        ("cpu", "model", "Other CPU"),
        ("cpu", "logical_cores", 16),
    ),
)
def test_publication_rejects_internally_consistent_matrix_from_another_host(
    tas_evidence: EvidenceSource, component: str, field: str, value: Any
) -> None:
    # Each resolution and its raw runs are consistent. Only cross-model comparison catches this.
    matrix_path = tas_evidence.root / "liveaction_span_madrid-matrix.json"
    matrix = json.loads(matrix_path.read_text())
    if component == "gpu":
        matrix["environment"]["gpu"][field] = value
    for entry in matrix["variants"].values():
        campaign_path = tas_evidence.resolve(entry["campaign"]["path"])
        campaign = json.loads(campaign_path.read_text())
        campaign["environment"][component][field] = value
        for round_value in campaign["rounds"]:
            for run_record in round_value["manifests"].values():
                path = tas_evidence.resolve(run_record)
                run = json.loads(path.read_text())
                run["environment"][component][field] = value
                write_json(path, run)
        write_json(campaign_path, campaign)
        entry["campaign"] = artifact(tas_evidence, campaign_path)
    write_json(matrix_path, matrix)

    with pytest.raises(PublicationError, match=f"session {component} differs"):
        publish(tas_evidence)


def test_publication_rejects_matrix_environment_different_from_campaign(
    tas_evidence: EvidenceSource,
) -> None:
    path = next(tas_evidence.root.glob("*-matrix.json"))
    matrix = json.loads(path.read_text())
    matrix["environment"]["gpu"]["power_limit_w"] = 450
    write_json(path, matrix)
    with pytest.raises(PublicationError, match="matrix session environment differs"):
        publish(tas_evidence)


@pytest.mark.parametrize("implementation", ("trtvideo", "vstrt", "tas"))
def test_publication_rejects_raw_run_environment_different_from_campaign(
    tas_evidence: EvidenceSource, implementation: str
) -> None:
    path = next(
        tas_evidence.root.glob(
            f"*/winner-campaign/test/{implementation}/round-01/run-01/manifest.json"
        )
    )
    run = json.loads(path.read_text())
    run["environment"]["gpu"]["name"] = "Other GPU"
    write_json(path, run)
    with pytest.raises(PublicationError, match="session gpu differs"):
        publish(tas_evidence)


@pytest.mark.parametrize(
    "field",
    ("source_revision", "python_version", "torch_version", "tensorrt_version", "nelux_version"),
)
def test_publication_rejects_current_metadata_different_from_measurements(
    tas_evidence: EvidenceSource, tmp_path: Path, field: str
) -> None:
    metadata = json.loads(Path("benchmarks/implementations.json").read_text())
    metadata["implementations"]["tas"][field] = "not-measured"
    path = write_json(tmp_path / "changed-implementations.json", metadata)

    with pytest.raises(PublicationError, match=f"publication metadata {field} differs"):
        build_document(tas_evidence, path, Path("benchmarks/tuning/candidates.json"))


@pytest.mark.parametrize("field", ("source_revision", "python", "torch", "tensorrt", "nelux"))
def test_publication_rejects_metadata_different_from_native_runtime(
    tas_evidence: EvidenceSource, field: str
) -> None:
    path = next(tas_evidence.root.glob("*/winner-campaign/test/tas/round-01/run-01/manifest.json"))
    run = json.loads(path.read_text())
    native = run["measured"]["validation"]["runtime_evidence"]
    if field == "source_revision":
        native[field] = "f" * 40
    else:
        native["runtime"][field] = "not-measured"
    write_json(path, run)

    with pytest.raises(PublicationError, match="differs from native runtime evidence"):
        publish(tas_evidence)
