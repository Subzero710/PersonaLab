"""Audit published native Facepoints correspondences and import measured rasters."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .facepoints_inference import (CLASSES, REVISION, WEIGHTS_SHA256, digest, load_selection,
                                   point_rows, validate_predictions)


from .facepoints_contract import (FACEPOINTS_WEIGHT_ATOL, FACEPOINTS_POINTS_SHA256,
                                  FACEPOINTS_GROUPS, FACEPOINTS_LENGTHS, location_key)


def native_rows(rows, triangles, groups, vertex_count):
    """Validate published indices without changing their order or coefficients."""
    lookup = {tuple(sorted(map(int, tri))) for tri in triangles}
    output, seen = [], {}
    for row in rows:
        info = {key: row[key] for key in ("index", "class", "region", "vertices", "weights")}
        issues = []
        vertices, weights = row["vertices"], row["weights"]
        count = FACEPOINTS_LENGTHS.get(row["class"])
        if (not isinstance(vertices, list) or len(vertices) != count
                or any(type(i) is not int or not 0 <= i < vertex_count for i in vertices)
                or len(set(vertices)) != len(vertices)
                or not isinstance(weights, list) or len(weights) != count
                or any(type(w) not in (int, float) or not np.isfinite(w) or not 0 <= w <= 1 for w in weights)):
            info["issues"] = ["Invalid native indices or barycentric weights"]
            output.append(info)
            continue
        v, w = np.asarray(vertices), np.asarray(weights, dtype=np.float64)
        active = v[w > 0]
        info["weight_sum_error"] = float(abs(w.sum()-1))
        if info["weight_sum_error"] > FACEPOINTS_WEIGHT_ATOL:
            issues.append("Weight sum outside pinned release tolerance; no renormalization")
        memberships = sorted(name for name, mask in groups.items() if len(active) and (mask[active] > 0).all())
        info["native_groups"] = memberships
        expected = FACEPOINTS_GROUPS[row["class"]]
        if expected not in memberships:
            issues.append(f"Expected native group {expected} absent")
        if count == 3 and tuple(sorted(vertices)) not in lookup:
            issues.append("Published triangle is absent from native topology")
        if count == 1 and weights != [1.0]:
            issues.append("Published vertex must have exact unit weight")
        if row["class"] in ("lid", "iris"):
            side = {"L": "left", "R": "right"}.get(row["region"].split("_")[0])
            if side is None or side not in memberships:
                issues.append("Published eye side disagrees with native groups")
        key = location_key(vertices, weights)
        info["duplicate_of"] = seen.get(key)
        if key not in seen:
            seen[key] = f"fp:{row['index']}"
        info["location"] = {"kind": "weighted_vertices", "index": None,
                            "weights": weights, "vertex_indices": vertices}
        info["issues"] = issues
        output.append(info)
    return output


def geometry_validation(model, definition, rows):
    """Full native FP64 projection Jacobian and multi-view geometry checks.

    The synthetic states are only numerical probes, not detector accuracy tests.
    """
    from .dense_reference import oracle_state
    from .dense_observability import ProjectionProtocol, projection_jacobian

    bound = definition.bind(model)
    sparse = bound.sparse_geometry(model)
    head = model.head
    state = oracle_state(model, 42, 3)
    head_joint = tuple(head.joint_names).index("head")

    def full(identity, expression, rotations, translation):
        return bound.evaluate(head(identity=identity, expression=expression,
                                   rotations=rotations, translation=translation))

    with torch.no_grad():
        vertices = head(identity=state.identity.expand(len(state.expressions), -1), expression=state.expressions,
                        rotations=state.rotations, translation=state.translations)
        # Independent evaluation directly from the publisher's unpadded rows.
        raw = vertices.detach().cpu().numpy()
        expected = np.stack([np.einsum("...vc,v->...c", raw[..., r["vertices"], :], r["weights"])
                             for r in rows], axis=-2)
        actual = sparse(state.identity, state.expressions, state.rotations, state.translations)
        full_points = bound.evaluate(vertices)
        sequential = torch.stack([sparse(state.identity, e, r, t) for e, r, t in zip(
            state.expressions, state.rotations, state.translations, strict=True)])
        report = {
            "seed": 42, "views": 3, "point_count": len(rows),
            "max_full_vs_published_m": float(np.max(np.abs(full_points.cpu().numpy() - expected))),
            "max_sparse_vs_published_m": float(np.max(np.abs(actual.cpu().numpy() - expected))),
            "max_batch_vs_single_m": float((actual - sequential).abs().max()),
        }
    protocol = ProjectionProtocol(state, 0, head_joint)
    px_sparse, j_sparse = projection_jacobian(protocol, sparse, 16)
    px_full, j_full = projection_jacobian(protocol, full, 16)
    report.update({"jacobian_shape": list(j_full.shape),
                   "max_sparse_vs_full_px": float((px_sparse - px_full).abs().max()),
                   "max_elementwise_jacobian_error_px": float((j_sparse - j_full).abs().max())})
    limits = {"max_full_vs_published_m": 2e-12, "max_sparse_vs_published_m": 2e-12,
              "max_batch_vs_single_m": 2e-12, "max_sparse_vs_full_px": 2e-10,
              "max_elementwise_jacobian_error_px": 2e-9}
    report["tolerances"] = limits
    report["passed"] = all(np.isfinite(report[k]) and report[k] <= v for k, v in limits.items())
    return report


def main():
    from .benchmark.model import load_benchmark_model
    from .benchmark.storage import publish_directory, read_json, write_json, digest_json
    from .dense_dataset import ExtractionAttempt, _write_rows
    from .dense_observations import DenseObservation, ImageProvenance, ModelAsset, ObservationPoint, ObservationSchema
    from .dfl import DFLFaceSample, iter_dataset
    from .observations import LandmarkObservation
    from .surface_correspondences import NativeLocation, SurfaceCorrespondence, SurfaceCorrespondences, load_correspondences

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "selection", "release", "inference", "model-file", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    ids = load_selection(a.selection, a.input)
    if digest(a.release / "points_gnm.json") != FACEPOINTS_POINTS_SHA256:
        raise ValueError("Facepoints point definition differs from the audited release")
    rows = point_rows(a.release)
    raw = read_json(a.inference / "report.json")
    expected = {"format": "personalab-facepoints-inference-v1", "revision": REVISION,
                "weights_sha256": WEIGHTS_SHA256, "points_sha256": digest(a.release / "points_gnm.json"),
                "config_sha256": digest(a.release / "model/config.json"), "selection_sha256": digest(a.selection),
                "dataset_sha256": digest(a.input)}
    if any(raw.get(k) != v for k, v in expected.items()) or [r["observation_id"] for r in raw["attempts"]] != ids:
        raise ValueError("Facepoints inference provenance or observation order differs")
    model = load_benchmark_model(a.model_file, dtype="float64", device="cpu")
    groups = dict(zip(model.head.vertex_group_names, model.head.vertex_groups.detach().cpu().numpy(), strict=True))
    native = native_rows(rows, model.triangles, groups, model.head.num_vertices)
    schema = ObservationSchema("facepoints-620-dfl", 1, "aligned_pixels", tuple(
        ObservationPoint(f"fp:{r['index']}", r["class"]) for r in rows))
    mapping = None
    mapping_error = None
    geometry_check = None
    if not any(r["issues"] for r in native):
        try:
            candidate = SurfaceCorrespondences(schema, model.metadata["topology_sha256"], tuple(
                SurfaceCorrespondence(f"fp:{r['index']}", NativeLocation(r["location"]["kind"],
                    r["location"]["index"], tuple(r["location"]["weights"]), tuple(r["location"]["vertex_indices"])),
                    tuple(r["native_groups"]), f"Myned AI points_gnm.json revision {REVISION}, index {r['index']}",
                    "Published detector target; structure checked, independent anatomical accuracy and visibility unverified",
                    duplicate_of=r["duplicate_of"])
                for r in native), source_sha256=FACEPOINTS_POINTS_SHA256)
            bound = candidate.bind(model)
            if len(bound.slots) != 620:
                raise ValueError("Require all 620 published locations")
            # Compare actual published evaluation with binder evaluation on the pinned mesh.
            verts = model.reference.detach().cpu().numpy()
            expected_xyz = np.stack([np.asarray(r["weights"]) @ verts[r["vertices"]] for r in rows])
            actual = bound.evaluate(model.reference).detach().cpu().numpy()
            if not np.allclose(expected_xyz, actual, atol=1e-12, rtol=0):
                raise ValueError("Native binder changed published barycentric positions")
            if sum(r["duplicate_of"] is not None for r in native) != 11:
                raise ValueError("Published release must preserve all 11 duplicate aliases")
            geometry_check = geometry_validation(model, candidate, rows)
            if not geometry_check["passed"]:
                raise ValueError("Full/sparse numerical equivalence failed")
            mapping = candidate
        except (ValueError, RuntimeError, KeyError) as exc:
            mapping_error = str(exc)
    else:
        mapping_error = "One or more native slots failed validation; no partial mapping emitted"
    samples = {}
    for sample in iter_dataset(a.input):
        if sample.sample_id in ids:
            if not isinstance(sample, DFLFaceSample) or sample.sample_id in samples:
                raise ValueError(f"Invalid selected DFL sample {sample.sample_id}")
            samples[sample.sample_id] = sample
    if set(samples) != set(ids):
        raise ValueError("Missing selected DFL sample")
    asset = ModelAsset("1.0.0", f"https://huggingface.co/myned-ai/facepoints/tree/{REVISION}", WEIGHTS_SHA256)
    configuration = expected | {"inference_device": raw["device"], "passes": raw["passes"],
                                "initial_box_protocol": raw["initial_box_protocol"],
                                "uncertainty": "raw predicted standard deviation in pixels; no fitting weights assigned"}
    attempts, stds = [], []
    for i, r in enumerate(raw["attempts"]):
        s = samples[r["observation_id"]]
        if r["error"] is not None:
            attempts.append(ExtractionAttempt(s.sample_id, s.image_ref, None, r["error"]))
            continue
        from hashlib import sha256
        dfl = LandmarkObservation.from_dfl(s, dtype=torch.float64)
        if (r["image_ref"] != s.image_ref or r["image_size"] != list(s.image_size)
                or r["image_sha256"] != sha256(s.read_image_bytes()).hexdigest()):
            raise ValueError(f"Source raster mismatch: {s.sample_id}")
        file = a.inference / f"observation_{i:05d}.npz"
        if r["arrays"] != file.name or digest(file) != r["arrays_sha256"]:
            raise ValueError("Prediction archive integrity mismatch")
        with np.load(file, allow_pickle=False) as stored:
            if set(stored.files) != {"points", "std", "300w", "wflw", "validity"}:
                raise ValueError("Unexpected prediction arrays")
            values = validate_predictions(stored, *s.image_size)
            if not np.array_equal(values["validity"], stored["validity"]):
                raise ValueError("Prediction validity mismatch")
        obs = DenseObservation(observation_id=s.sample_id, schema=schema, provider="facepoints",
            provider_version="1.0.0", asset=asset, source=ImageProvenance(s.image_ref, r["image_sha256"], None),
            extraction=configuration, image_size=dfl.image_size, alignment=dfl.alignment,
            landmarks=torch.from_numpy(values["points"]), validity=torch.from_numpy(values["validity"]))
        attempts.append(ExtractionAttempt(s.sample_id, s.image_ref, obs, image_size=dfl.image_size,
                                          image_sha256=r["image_sha256"]))
        stds.append((s.sample_id, values["std"]))
    with publish_directory(a.output, completion="manifest.json") as stage:
        _write_rows(stage, schema, "facepoints", "1.0.0", asset, configuration, attempts)
        # Preserve the full 620-element uncertainty separately; never encode it as confidence or visibility.
        np.savez_compressed(stage / "uncertainty_px.npz", std=np.stack([s for _, s in stds]) if stds else np.empty((0, 620)))
        write_json(stage / "uncertainty.json", {"observation_ids": [i for i, _ in stds],
                    "unit": "pixels", "sha256": digest(stage / "uncertainty_px.npz")})
        if mapping is not None:
            path = stage / "published_correspondences.json"
            write_json(path, mapping.to_record())
            loaded = load_correspondences(path, schema)
            if loaded != mapping:
                raise ValueError("Published mapping roundtrip changed its coefficients or aliases")
            loaded.bind(model)

        write_json(stage / "native_audit.json", {"model": model.metadata, "provider": expected,
            "class_counts": CLASSES, "slots": native, "structurally_admissible": sum(not r["issues"] for r in native),
            "binder_passed": mapping is not None, "mapping_error": mapping_error,
            "duplicate_count": sum(r.get("duplicate_of") is not None for r in native),
            "geometry_validation": geometry_check,
            "mapping_sha256": None if mapping is None else digest_json(mapping.to_record()),
            "anatomical_review": "pending", "visibility": "unavailable", "successes": len(stds),
            "failure_count": len(attempts)-len(stds)})
    print(f"{len(stds)}/50 imported; binder {'passed' if mapping is not None else 'failed'}", flush=True)
    if mapping is None or len(stds) != 50:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
