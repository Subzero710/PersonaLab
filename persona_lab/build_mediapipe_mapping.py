"""Generate reviewable MediaPipe-to-GNM proposals, never an implicit fitting map."""
import argparse
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .benchmark.model import load_benchmark_model, MODEL_SHA256
from .benchmark.storage import digest_json, publish_directory, read_json, require_keys, write_json
from .correspondences import DFL_TO_GNM
from .dense_dataset import load_dense_dataset
from .dfl import DFLFaceSample, iter_dataset
from .gnm_geometry import load_sparse_landmarks
from .mapping_geometry import (closest_surface, register_template, mesh_edges, curve_guides,
                               refine_surface_graph, distortion_report, polyline_distances)
from .mediapipe_face import SCHEMAS, ASSET, PROVIDER, PROVIDER_VERSION
from .surface_correspondences import NativeLocation, SurfaceCorrespondence, SurfaceCorrespondences, required_groups

CANONICAL_REVISION = "8317ba78778738ba90a521e7e4580a2ba0129c81"
CANONICAL_URL = (f"https://raw.githubusercontent.com/google-ai-edge/mediapipe/{CANONICAL_REVISION}/"
                 "mediapipe/modules/face_geometry/data/canonical_face_model.obj")
CANONICAL_SHA256 = "8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618"
# Anatomical hypotheses from the published MediaPipe connection loops and
# PersonaLab's DFL_TO_GNM descriptions. They are NOT an official crosswalk.
# Exact semantic candidates use the native landmark resource, not a snap.
SEMANTIC = {33:36, 133:39, 362:42, 263:45, 61:48, 291:54, 0:51, 17:57,
            78:60, 308:64, 13:62, 14:66}
# Additional registration guides only: eyebrow endpoints, nasal root/tip/base,
# alar edges and chin. Their association is an explicit hypothesis.
GUIDES = {70:17, 107:21, 336:22, 300:26, 168:27, 4:30, 129:31,
          2:33, 358:35, 152:8}
# Keep the previous nasal hypotheses in paired detector comparisons. Canonical
# geometry supports testing the more anterior tip and wider alar slots, but
# neither set is an independently verified detector crosswalk.
PREVIOUS_NOSE = {1:30, 98:31, 327:35}
# Ordered upper/lower feature paths: source canonical loop, native sparse knots.
# These are surface-curve hypotheses, not measured native rim definitions.
CURVES = (
    ("right_upper_eyelid", (33,246,161,160,159,158,157,173,133), (36,37,38,39)),
    ("right_lower_eyelid", (33,7,163,144,145,153,154,155,133), (36,41,40,39)),
    ("left_upper_eyelid", (362,398,384,385,386,387,388,466,263), (42,43,44,45)),
    ("left_lower_eyelid", (362,382,381,380,374,373,390,249,263), (42,47,46,45)),
    ("upper_outer_lip", (61,185,40,39,37,0,267,269,270,409,291), (48,49,50,51,52,53,54)),
    ("lower_outer_lip", (61,146,91,181,84,17,314,405,321,375,291), (48,59,58,57,56,55,54)),
    ("upper_inner_lip", (78,191,80,81,82,13,312,311,310,415,308), (60,61,62,63,64)),
    ("lower_inner_lip", (78,95,88,178,87,14,317,402,318,324,308), (60,67,66,65,64)),
)
PROPOSAL_FORMAT = "personalab-mediapipe-mapping-proposals-v2"
REVIEW_FORMAT = "personalab-mediapipe-mapping-review-v1"


def canonical_mesh(path):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != CANONICAL_SHA256:
        raise ValueError("MediaPipe canonical OBJ SHA-256 mismatch")
    vertices = np.asarray([[float(x) for x in line.split()[1:4]]
                           for line in raw.decode("utf-8").splitlines() if line.startswith("v ")])
    if vertices.shape != (468, 3) or not np.isfinite(vertices).all():
        raise ValueError("Expected the pinned 468-vertex canonical MediaPipe mesh")
    faces = np.asarray([[int(token.split("/")[0])-1 for token in line.split()[1:]]
                        for line in raw.decode("utf-8").splitlines() if line.startswith("f ")], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or (faces < 0).any() or (faces >= 468).any():
        raise ValueError("Expected native canonical triangular connectivity")
    return vertices / 100, faces  # official canonical unit: centimetres


def _options(point, groups, *, location=None, lip_branch=None):
    options = required_groups(point, location)
    if lip_branch is not None and location is None:
        options = ((lip_branch,),)
    # Do not silently accept missing native anatomy definitions.
    for option in options:
        for name in option:
            if name not in groups:
                raise ValueError(f"Native anatomical group {name!r} is missing; apply the Issue #14 head fix")
    if point.side in ("left", "right"):
        options = tuple(option + (point.side,) for option in options)
    return options


def _anatomy(point, options, groups, mirrors, indices):
    """Use the same subject-side and anatomical requirements as bind()."""
    good = np.ones(len(indices), dtype=bool)
    if point.side in ("left", "right"):
        opposite = "right" if point.side == "left" else "left"
        good &= (groups[point.side][indices] > 0).all(axis=1)
        good &= ~(groups[opposite][indices] > 0).all(axis=1)
        good &= (groups[opposite][mirrors[indices]] > 0).all(axis=1)
    masks = [good & np.logical_and.reduce([(groups[n][indices] > 0).all(axis=1) for n in option])
             for option in options]
    return masks


def _plot(path, canonical, vertices, rows, *, region, angle):
    """Two orthographic views in native axes; numbered proposals, no renderer claims."""
    image = Image.new("RGB", (1440, 760), (18, 22, 29))
    draw = ImageDraw.Draw(image)
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    selected = [row for row in rows if row["region"] == region and row["location"] is not None]
    for x0, points, title, key in ((0, canonical, "MediaPipe canonical", "canonical_m"),
                                   (720, vertices, "GNM neutral", "native_m")):
        xy = points @ rotation.T
        lo, hi = xy[:, :2].min(0), xy[:, :2].max(0)
        scale = min(660 / (hi[0] - lo[0]), 650 / (hi[1] - lo[1]))
        def pixel(v):
            p = np.asarray(v) @ rotation.T
            return x0 + 360 + (p[0] - (lo[0]+hi[0])/2)*scale, 400 - (p[1] - (lo[1]+hi[1])/2)*scale
        for p in points[np.argsort(xy[:, 2])]:
            x, y = pixel(p)
            draw.point((x, y), fill=(95, 106, 122))
        for row in selected:
            x, y = pixel(row[key])
            draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 190, 80))
            draw.text((x+3, y-8), row["point_id"].split(":")[1], fill="white")
        draw.text((x0+20, 15), f"{title} | {region} | {angle:+d} deg | PROPOSALS ONLY", fill="white")
    image.save(path)


def compare_dfl(dense_input, dfl_input, schema):
    """Compare anchor hypotheses on exactly the same images; DFL is not truth."""
    if schema.coordinate_convention != "aligned_pixels":
        raise ValueError("DFL anchor comparison requires the aligned MediaPipe schema")
    dataset = load_dense_dataset(dense_input, schema=schema, provider=PROVIDER,
                                 provider_version=PROVIDER_VERSION, asset=ASSET, device="cpu")
    observations = dataset.select()
    by_id = {obs.observation_id: obs for obs in observations}
    samples = {}
    for sample in iter_dataset(dfl_input):
        if isinstance(sample, DFLFaceSample) and sample.sample_id in by_id:
            if sample.sample_id in samples:
                raise ValueError("Duplicate matched DFL observation")
            obs = by_id[sample.sample_id]
            if sha256(sample.read_image_bytes()).hexdigest() != obs.source.sha256:
                raise ValueError(f"DFL source digest mismatch for {sample.sample_id}")
            if tuple(sample.image_size) != (obs.image_size.width, obs.image_size.height):
                raise ValueError(f"DFL raster mismatch for {sample.sample_id}")
            samples[sample.sample_id] = sample.landmarks
    if set(samples) != set(by_id):
        raise ValueError("DFL input does not contain every dense observation")
    rows = []
    for mp, gnm in (SEMANTIC | GUIDES | PREVIOUS_NOSE).items():
        values = []
        for obs in observations:
            valid = bool(obs.validity[mp])
            delta = obs.landmarks[mp].cpu().numpy() - samples[obs.observation_id][gnm]
            values.append({"observation_id": obs.observation_id, "valid": valid,
                           "delta_px": delta.tolist() if valid else None,
                           "distance_px": float(np.linalg.norm(delta)) if valid else None})
        distances = np.asarray([row["distance_px"] for row in values if row["valid"]])
        rows.append({"mp": mp, "dfl": gnm, "semantic": mp in SEMANTIC,
                     "registration_guide": mp in GUIDES, "previous_nose_hypothesis": mp in PREVIOUS_NOSE,
                     "valid_count": len(distances), "mean_px": float(distances.mean()) if len(distances) else None,
                     "rms_px": float(np.sqrt(np.mean(distances**2))) if len(distances) else None,
                     "max_px": float(distances.max()) if len(distances) else None, "observations": values})
    return {"limitation": "DFL agreement is a detector comparison, not anatomical ground truth or an approval",
            "observation_ids": list(by_id), "anchors": rows}



def _curve_diagnostics(mapped, curves):
    result = []
    for curve in curves:
        slots = np.asarray(curve["mp_slots"])
        guide = np.asarray(curve["guide_m"])
        p = mapped[slots]
        segments = []
        for i, (a, b) in enumerate(zip(slots[:-1], slots[1:], strict=True)):
            if not np.isfinite(p[i:i+2]).all():
                segments.append({"slots": [int(a), int(b)], "status": "unsupported"})
                continue
            u, v = guide[i+1] - guide[i], p[i+1] - p[i]
            denom = np.linalg.norm(u) * np.linalg.norm(v)
            cosine = float(np.clip(np.dot(u, v) / denom, -1, 1)) if denom > 0 else None
            segments.append({"slots": [int(a), int(b)], "status": "measured", "direction_cosine": cosine,
                             "length_m": float(np.linalg.norm(v))})
        result.append(curve | {"segments": segments})
    return result


@torch.no_grad()
def _mapping_diagnostics(model, schema, canonical, faces, initial, refined, rows, graph, curves):
    """Evaluate the exact proposed native locations on named expression states."""
    head = model.head
    # Exercise the same native binder as fitting without authorizing a map.
    entries = []
    for row in rows:
        location = row["location"]
        native = None if location is None else NativeLocation(
            location["kind"], location["index"], None if location["weights"] is None else tuple(location["weights"]))
        entries.append(SurfaceCorrespondence(row["point_id"], native, tuple(row["native_groups"]),
                                             "Unapproved geometry diagnostic only", row["reason"]))
    definition = SurfaceCorrespondences(schema, model.metadata["topology_sha256"], tuple(entries))
    binder = {"status": "passed"}
    try:
        definition.bind(model)
    except ValueError as error:
        # Retain failed candidate sets as diagnostic evidence, never publish a map.
        binder = {"status": "failed", "error": str(error)}
    official = load_sparse_landmarks(head)
    slots, indices, weights = [], [], []
    for slot, row in enumerate(rows):
        loc = row["location"]
        if loc is None:
            continue
        slots.append(slot)
        if loc["kind"] == "landmark":
            indices.append(official.indices[loc["index"]])
            weights.append(official.weights[loc["index"]])
        elif loc["kind"] == "triangle":
            indices.append(model.triangles[loc["index"]])
            weights.append(loc["weights"])
        else:
            raise ValueError("Unexpected proposal location type")
    indices, weights = np.asarray(indices), np.asarray(weights)
    names = tuple(head.expression_names)
    selected = ("lower_face_region_000", "left_eye_region_000", "right_eye_region_000")
    if not set(selected) <= set(names):
        raise ValueError("Native expression probe names do not match the pinned GNM model")
    states = [("neutral", None, 0.)] + [(f"{name}_{value:+g}", name, value)
              for name in selected for value in (-1., 1.)]
    posed = []
    neutral = None
    for label, name, value in states:
        expression = model.reference.new_zeros(head.expression_dim)
        if name is not None:
            expression[names.index(name)] = value
        vertices = head(identity=model.reference.new_zeros(head.identity_dim), expression=expression,
                        rotations=model.reference.new_zeros((head.num_joints, 3)),
                        translation=model.reference.new_zeros(3)).detach().cpu().numpy()
        if vertices.shape != model.reference.shape or not np.isfinite(vertices).all():
            raise ValueError("Invalid native expression geometry")
        current = np.full_like(canonical, np.nan)
        current[slots] = np.einsum("ni,nij->nj", weights, vertices[indices])
        if neutral is None:
            neutral = current.copy()
            neutral_template_delta = float(np.max(np.linalg.norm(neutral[slots]-refined[slots], axis=1)))
        posed.append({"state": label, "expression_name": name, "coefficient": value,
                      "native_vs_neutral": distortion_report(neutral, current, faces),
                      "mapped_points_m": [p.tolist() if np.isfinite(p).all() else None for p in current]})
    return {"format": "personalab-mediapipe-graph-diagnostics-v2", "binder": binder, "graph": graph,
            "initial_projection_vs_canonical": distortion_report(canonical, initial, faces),
            "refined_vs_canonical": distortion_report(canonical, refined, faces),
            "curve_guides": _curve_diagnostics(refined, curves), "expression_probes": posed,
            "native_neutral_vs_template_max_m": neutral_template_delta,
            "limitations": ["Sparse native landmark guides and centroid branch tests are proxies, not verified anatomical rims.",
                            "Normal disagreements are orientation diagnostics, not a proof of a surface fold.",
                            "Expression probes test native transport, not the detector on expressive images.",
                            "All results are proposals; no numerical threshold approves or removes a slot."]}

def propose(model, canonical, faces, schema, output, *, paired=None):
    vertices = model.reference.detach().cpu().numpy()
    triangles = model.triangles
    head = model.head
    groups = dict(zip(head.vertex_group_names, head.vertex_groups.detach().cpu().numpy(), strict=True))
    mirrors = head.mirror_indices.detach().cpu().numpy()
    official = load_sparse_landmarks(head)
    native_anchors = (vertices[official.indices] * official.weights[..., None]).sum(1)
    anchors = SEMANTIC | GUIDES
    source_ids = list(anchors)
    native_ids = list(anchors.values())
    warped, extrapolated, errors = register_template(canonical[source_ids], native_anchors[native_ids], canonical)
    targets = warped.copy()
    branches, curve_slots = {}, {}
    curve_records = []
    triangle_centers = vertices[triangles].mean(axis=1)
    curve_distances = {name: polyline_distances(triangle_centers, native_anchors[list(knots)])
                       for name, _, knots in CURVES}
    opposite = {a: b for a, b in (
        ("right_upper_eyelid", "right_lower_eyelid"), ("right_lower_eyelid", "right_upper_eyelid"),
        ("left_upper_eyelid", "left_lower_eyelid"), ("left_lower_eyelid", "left_upper_eyelid"),
        ("upper_outer_lip", "upper_inner_lip"), ("upper_inner_lip", "upper_outer_lip"),
        ("lower_outer_lip", "lower_inner_lip"), ("lower_inner_lip", "lower_outer_lip"),
    )}
    # Preserve ordering via distinct upper/lower native landmark guides.
    # Interior guide samples are projected on allowed native triangles, never
    # interpreted as exact rim vertices. Midline semantic anchors stay exact.
    for name, slots, knots in CURVES:
        guides = curve_guides(canonical, slots, native_anchors[list(knots)])
        curve_records.append({"name": name, "mp_slots": list(slots), "gnm_knots": list(knots),
                              "guide_m": guides.tolist()})
        for slot, xyz in zip(slots[1:-1], guides[1:-1], strict=True):
            if slot not in SEMANTIC:
                targets[slot] = xyz
                curve_slots[slot] = name
            if name.endswith("lip"):
                branches[slot] = "upper_lip" if name.startswith("upper") else "lower_lip"
    rows, candidates, candidate_options, fixed = [], {}, {}, set()
    for slot, point in enumerate(schema.points):
        row = {"point_id": point.identifier, "region": point.region, "side": point.side,
               "location": None, "native_groups": [], "native_m": None, "distance_m": None,
               "canonical_m": canonical[slot].tolist() if slot < 468 else None,
               "warped_m": warped[slot].tolist() if slot < 468 else None,
               "outside_anchor_hull": bool(extrapolated[slot]) if slot < 468 else None}
        if slot >= 468:
            row["reason"] = "No iris geometry in pinned 468-vertex canonical mesh; eye rotation/visibility unvalidated"
        elif point.region == "facial_oval":
            row["reason"] = "View-dependent contour not supported by current fitting representation"
        else:
            if slot in SEMANTIC:
                index = SEMANTIC[slot]
                options = _options(point, groups, location=NativeLocation("landmark", index))
                active = official.indices[index][official.weights[index] > 0][None]
                masks = _anatomy(point, options, groups, mirrors, active)
                valid = [option for option, mask in zip(options, masks, strict=True) if mask[0]]
                if valid:
                    row.update(location=asdict(NativeLocation("landmark", index)), native_groups=list(valid[0]),
                               native_m=native_anchors[index].tolist(), distance_m=float(np.linalg.norm(warped[slot]-native_anchors[index])),
                               reason="Semantic anchor hypothesis; independently review anatomical association")
                    fixed.add(slot)
                else:
                    row["reason"] = "Semantic anchor fails native anatomy; no substitution"
            else:
                options = _options(point, groups, lip_branch=branches.get(slot))
                masks = _anatomy(point, options, groups, mirrors, triangles)
                eligible_mask = np.logical_or.reduce(masks)
                if slot in curve_slots:
                    name = curve_slots[slot]
                    # Explicit branch restriction: a native triangle centroid
                    # must be closer to the intended sparse guide than the
                    # opposite guide. This is not a native rim classification.
                    eligible_mask &= curve_distances[name] <= curve_distances[opposite[name]]
                    row["curve_branch"] = name
                eligible = np.flatnonzero(eligible_mask)
                if not len(eligible):
                    row["reason"] = "No native triangle satisfies declared region/side/curve branch; no substitution"
                else:
                    local, weights, xyz, distance = closest_surface(targets[slot], vertices[triangles[eligible]])
                    index = int(eligible[local])
                    option = next(option for option, mask in zip(options, masks, strict=True) if mask[index])
                    row.update(location=asdict(NativeLocation("triangle", index, tuple(weights.tolist()))),
                               native_groups=list(option), native_m=xyz.tolist(), distance_m=distance,
                               reason="Initial curve/TPS target projected to compatible native surface; unreviewed")
                    candidates[slot] = eligible
                    candidate_options[slot] = (options, masks)
        if slot < len(canonical):
            row["guide_m"] = targets[slot].tolist()
        rows.append(row)
    initial = np.full_like(canonical, np.nan)
    for slot, row in enumerate(rows[:468]):
        if row["native_m"] is not None:
            initial[slot] = row["native_m"]
    refined, locations, graph = refine_surface_graph(canonical, targets, initial, mesh_edges(faces),
                                                      candidates, fixed, vertices, triangles)
    for slot, (index, weights) in locations.items():
        options, masks = candidate_options[slot]
        option = next(option for option, mask in zip(options, masks, strict=True) if mask[index])
        rows[slot].update(location=asdict(NativeLocation("triangle", index, tuple(weights.tolist()))),
                          native_groups=list(option), native_m=refined[slot].tolist(),
                          distance_m=float(np.linalg.norm(refined[slot]-warped[slot])),
                          guide_distance_m=float(np.linalg.norm(refined[slot]-targets[slot])),
                          reason="Graph-regularized surface proposal; anatomical review required")
    diagnostics = _mapping_diagnostics(model, schema, canonical, faces, initial, refined, rows, graph, curve_records)
    proposal = {"format": PROPOSAL_FORMAT, "schema": schema.identifier, "schema_version": schema.version,
                "provider_asset_sha256": ASSET.sha256, "canonical_sha256": CANONICAL_SHA256,
                "canonical_revision": CANONICAL_REVISION, "model_sha256": MODEL_SHA256,
                "topology_sha256": model.metadata["topology_sha256"], "landmarks_sha256": model.metadata["landmarks_sha256"],
                "paired_dfl": paired, "geometry_diagnostics_sha256": digest_json(diagnostics),
                "anchors": [{"mp": i, "gnm": anchors[i], "anatomy": DFL_TO_GNM.entries[anchors[i]].description,
                             "semantic": i in SEMANTIC, "leave_one_out_error_m": float(error)}
                            for i, error in zip(source_ids, errors, strict=True)], "rows": rows}
    review = {"format": REVIEW_FORMAT, "proposal_sha256": digest_json(proposal), "reviewer": "",
              "decisions": [{"point_id": row["point_id"], "decision": "pending" if row["location"] else "unsupported",
                             "evidence": ""} for row in rows]}
    with publish_directory(output, completion="proposals.json") as stage:
        write_json(stage / "review.json", review)
        write_json(stage / "geometry_diagnostics.json", diagnostics)
        if paired is not None:
            write_json(stage / "paired_dfl.json", paired)
        for region in sorted({row["region"] for row in rows if row["location"] is not None}):
            for angle in (0, -45, 45):
                _plot(stage / f"{region}_{angle:+d}.png", canonical, vertices, rows, region=region, angle=angle)
        write_json(stage / "proposals.json", proposal)
    print(f"{sum(row['location'] is not None for row in rows)} proposals; zero approved by this command")
    print(f"Geometry binder: {diagnostics['binder']['status']}; inspect geometry_diagnostics.json")


def approve(model, schema, proposal, review, output):
    if (proposal["format"], proposal["schema"], proposal["schema_version"], proposal["model_sha256"],
        proposal["topology_sha256"], proposal["landmarks_sha256"], proposal["canonical_sha256"], proposal["provider_asset_sha256"]) != (
        PROPOSAL_FORMAT, schema.identifier, schema.version, MODEL_SHA256, model.metadata["topology_sha256"],
        model.metadata["landmarks_sha256"], CANONICAL_SHA256, ASSET.sha256):
        raise ValueError("Proposal provenance mismatch")
    require_keys(review, {"format", "proposal_sha256", "reviewer", "decisions"}, "Mapping review")
    if review["format"] != REVIEW_FORMAT or review["proposal_sha256"] != digest_json(proposal):
        raise ValueError("Review must bind the exact proposal digest")
    reviewer = review["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Explicit reviewer required")
    rows, decisions = proposal["rows"], review["decisions"]
    if tuple(row["point_id"] for row in rows) != schema.point_ids or tuple(row["point_id"] for row in decisions) != schema.point_ids:
        raise ValueError("Proposal and review must preserve all schema slots")
    entries = []
    for row, decision in zip(rows, decisions, strict=True):
        require_keys(decision, {"point_id", "decision", "evidence"}, "Review decision")
        status, evidence = decision["decision"], decision["evidence"]
        if not isinstance(evidence, str) or status not in ("approved", "rejected", "pending", "unsupported"):
            raise ValueError("Invalid review decision")
        location = row["location"]
        if (location is None) != (status == "unsupported"):
            raise ValueError("Unsupported status must match proposal availability")
        if status in ("approved", "rejected") and not evidence.strip():
            raise ValueError("Explicit anatomical evidence required for approval or rejection")
        if status == "approved":
            native = NativeLocation(location["kind"], location["index"],
                                    None if location["weights"] is None else tuple(location["weights"]))
            entries.append(SurfaceCorrespondence(row["point_id"], native, tuple(row["native_groups"]),
                           f"Reviewer: {reviewer}; proposal {review['proposal_sha256']}; {evidence}",
                           "Cross-template correspondence; detector visibility and subject-specific accuracy not established"))
        else:
            entries.append(SurfaceCorrespondence(row["point_id"], None, (), f"Review status: {status}", row["reason"] + "; " + evidence))
    definition = SurfaceCorrespondences(schema, model.metadata["topology_sha256"], tuple(entries))
    bound = definition.bind(model)
    if not bound.slots:
        raise ValueError("No approved locations; refusing to create fitting mapping")
    with publish_directory(output, completion="mapping.json") as stage:
        write_json(stage / "proposals.json", proposal)
        write_json(stage / "review.json", review)
        write_json(stage / "mapping.json", definition.to_record())
    print(f"{len(bound.slots)}/{len(entries)} explicitly approved locations; all other slots retained as unsupported")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("propose", "approve"):
        p = sub.add_parser(command)
        p.add_argument("--model-file", type=Path, required=True)
        p.add_argument("--schema", choices=SCHEMAS, required=True)
        p.add_argument("--output", type=Path, required=True)
        if command == "propose":
            p.add_argument("--canonical", type=Path, required=True)
            p.add_argument("--dense-input", type=Path)
            p.add_argument("--dfl-input", type=Path)
        else:
            p.add_argument("--proposals", type=Path, required=True)
            p.add_argument("--review", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        model = load_benchmark_model(args.model_file, dtype="float64", device="cpu")
        schema = SCHEMAS[args.schema]
        if args.command == "propose":
            if (args.dense_input is None) != (args.dfl_input is None):
                raise ValueError("Supply both dense and DFL input paths for paired anchor diagnostics")
            paired = None if args.dense_input is None else compare_dfl(args.dense_input, args.dfl_input, schema)
            canonical, faces = canonical_mesh(args.canonical)
            propose(model, canonical, faces, schema, args.output, paired=paired)
        else:
            approve(model, schema, read_json(args.proposals), read_json(args.review), args.output)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
