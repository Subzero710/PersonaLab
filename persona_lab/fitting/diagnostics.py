"""Optional local mesh and overlay output for completed fitting runs."""

from io import BytesIO

import numpy as np
from PIL import Image
import torch

from ..camera import PerspectiveCamera, project, world_to_camera
from ..coordinates import gnm_to_frontal_axes
from ..gnm_geometry import extract_landmarks
from ..gnm_mesh import generate_mesh
from ..landmark_diagnostics import save_landmark_overlay
from .parameters import copy_camera


def write_neutral_mesh(path, model, result):
    if result.final is None:
        raise ValueError("Neutral mesh export requires a completed fit")
    mesh = generate_mesh(model.head, identity=model.tensor(result.final.identity))
    mesh.export(path, file_type="obj")


def _overlay_indices(result, count):
    count = min(count, len(result.observation_ids))
    if count == 0:
        return ()
    stats = result.final_reprojection.observations
    highest = max(range(len(stats)), key=lambda index: stats[index].rms_px)
    head_index = result.model["joint_names"].index("head")
    # Sample evenly across fitted head yaw and include the highest RMS image.
    by_yaw = sorted(
        (index for index in range(len(stats)) if index != highest),
        key=lambda index: result.final.rotations[index, head_index, 1],
    )
    positions = np.linspace(0, len(by_yaw) - 1, count - 1, dtype=int)
    return (highest, *(by_yaw[position] for position in positions))


@torch.no_grad()
def _overlay(path, model, result, state, observation, index, phase, image):
    vertices, landmarks = extract_landmarks(
        model.head, identity=model.tensor(state.identity),
        expression=model.tensor(state.expression[index]),
        rotations=model.tensor(state.rotations[index]), translation=model.tensor(state.translation[index]),
    )
    camera = copy_camera(state.cameras[index], device=model.reference.device)
    axes = gnm_to_frontal_axes(vertices)
    quads = model.head.quads
    if isinstance(camera, PerspectiveCamera):
        # Draw only complete quads in front of the lens. Landmark projection
        # itself retains the optimizer's strict depth checks.
        positive = world_to_camera(axes, camera.rotation, camera.translation)[:, 2] > 0
        quads = quads[positive[quads.long()].all(dim=1)]
        axes = axes[positive]
        quads = (positive.cumsum(0) - 1)[quads.long()]
    mesh = (project(axes, camera), quads) if quads.numel() else None
    save_landmark_overlay(
        path, project(gnm_to_frontal_axes(landmarks), camera), observation,
        mesh=mesh, image=image, subset=result.config.landmark_subset,
        title=f"{phase.capitalize()} fit, observation {index + 1}",
    )


def write_overlays(directory, model, result, observations, count, samples=()):
    """Save before/after overlays for the highest error and representative views."""
    if result.final is None:
        raise ValueError("Overlay export requires a completed fit")
    for index in _overlay_indices(result, count):
        image = None
        if samples:
            with Image.open(BytesIO(samples[index].read_image_bytes())) as source:
                image = source.convert("RGB")
        for phase in ("initial", "final"):
            _overlay(
                directory / f"overlay_{index + 1:04d}_{phase}.png", model, result,
                getattr(result, phase), observations[index], index, phase, image,
            )
