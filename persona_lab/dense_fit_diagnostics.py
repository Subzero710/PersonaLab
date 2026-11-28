"""Inspect explicitly mapped dense fit projections in the stored input raster."""

from collections import Counter
from hashlib import sha256
from pathlib import Path

import torch

from .benchmark.storage import write_json
from .camera import project
from .coordinates import gnm_to_frontal_axes
from .dfl import DFLFaceSample, iter_dataset
from .extract_dense import _decode, _ordinary_sources, observation_statistics, save_dense_overlay
from .fitting.diagnostics import _overlay_indices
from .fitting.parameters import copy_camera


def _images(path, observations, selected):
    needed = {observations[index].source.reference for index in selected}
    by_reference = {}
    path = Path(path).absolute()
    if observations[0].coordinate_convention == "aligned_pixels":
        sources = ((item.image_ref, item.read_image_bytes) for item in iter_dataset(path)
                   if isinstance(item, DFLFaceSample))
    else:
        sources = ((item.reference, item.read_bytes) for item in _ordinary_sources(path))
    for reference, read_bytes in sources:
        if reference in needed:
            by_reference[reference] = read_bytes()
            if len(by_reference) == len(needed):
                break
    if missing := needed - by_reference.keys():
        raise ValueError(f"Missing dense diagnostic rasters: {', '.join(sorted(missing))}")
    return by_reference


@torch.no_grad()
def write_dense_projections(directory, model, observations, bound, states, selected, image_input):
    """Pair native projections and ordered source measurements on exact raster bytes."""
    geometry = bound.sparse_geometry(model)
    images = _images(image_input, observations, selected)
    diagnostics = [observation_statistics(obs, bound.slots) | {"observation_id": obs.observation_id}
                   for obs in observations]
    regions = {}
    for row in diagnostics:
        for region, counts in row["regions"].items():
            regions.setdefault(region, Counter()).update(counts)
    for index in selected:
        obs = observations[index]
        raster = images[obs.source.reference]
        if sha256(raster).hexdigest() != obs.source.sha256:
            raise ValueError(f"Diagnostic raster changed for {obs.observation_id}")
        image = _decode(raster, obs.source.reference)
        record = diagnostics[index]
        for name, state in states:
            camera = copy_camera(state.cameras[index], device=model.reference.device)
            points = geometry(
                model.tensor(state.identity), model.tensor(state.expression[index]),
                model.tensor(state.rotations[index]), model.tensor(state.translation[index]),
            )
            projected = project(gnm_to_frontal_axes(points), camera).cpu().tolist()
            location_pixels = dict(zip(bound.slots, projected, strict=True))
            file = f"dense_overlay_{index + 1:04d}_{name}.png"
            save_dense_overlay(directory / file, image, obs, projected=location_pixels,
                               name=f"{name.title()} projection")
            record[name] = {
                "overlay": file,
                "projection_by_point_id": {
                    obs.schema.points[slot].identifier: point for slot, point in location_pixels.items()
                },
            }
    write_json(directory / "dense_mapping_diagnostics.json", {
        "correspondences": bound.definition.to_record(), "observations": diagnostics, "regions": regions,
        "visibility": "Unavailable slots are distinct from validity and correspondence support",
    })


def write_dense_fit_overlays(directory, model, result, observations, correspondences, count, image_input):
    write_dense_projections(
        directory, model, observations, correspondences.bind(model),
        (("initial", result.initial), ("final", result.final)), tuple(_overlay_indices(result, count)),
        image_input,
    )
