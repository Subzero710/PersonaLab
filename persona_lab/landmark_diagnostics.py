"""Write headless PNG diagnostics for projected GNM and observed landmarks."""

import argparse
from collections.abc import Sequence
from math import radians
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor

from .camera import PerspectiveCamera, project
from .coordinates import ImageSize, _validate_tensor, gnm_to_frontal_axes
from .correspondences import Region
from .gnm_geometry import GNM_LANDMARK_COUNT
from .observations import LandmarkObservation


GNM_COLOR = "#007799"
OBSERVED_COLOR = "#1c693e"
UNSUPPORTED_COLOR = "#981361"
CONTOUR_COLOR = "#a65900"
EXCLUDED_COLOR = "#626a74"
RESIDUAL_COLOR = "#a02e36"
PADDING = 18


def save_landmark_overlay(
    path: str | Path,
    projected_gnm: Tensor,
    observation: LandmarkObservation,
    *,
    mesh: tuple[Tensor, Tensor] | None = None,
    image: Image.Image | None = None,
    subset: Sequence[int] | None = None,
    pose_validity: Tensor | None = None,
    title: str = "Landmark inspection",
) -> Path:
    """Save side-by-side raster panels with mesh, indices and reprojection residuals.

    Projected landmarks are in official GNM order. Optional mesh data are
    projected (V, 2) pixel vertices and (Q, 4) GNM quads. The right panel
    overlays the observed DeepFaceLab points on an optional aligned image.
    Exclusion and residual colors refer to the requested fitting mask.
    """
    _validate_tensor(
        projected_gnm, "Projected GNM landmarks", (GNM_LANDMARK_COUNT, 2),
        like=observation.landmarks,
    )
    if projected_gnm.ndim != 2:
        raise ValueError("Diagnostics require unbatched projected landmarks")
    active = observation.fitting_mask(
        subset=subset, pose_validity=pose_validity,
    ).cpu().tolist()
    width, height = observation.image_size.width, observation.image_size.height
    raster_size = (width, height)
    if image is not None and image.size != raster_size:
        raise ValueError("Diagnostic background must match the observation image size")
    left = Image.new("RGB", raster_size, "#f4f8fc")
    right = Image.new("RGB", raster_size, "#f8faf9") if image is None else image.convert("RGB")
    mesh_plot, overlay = ImageDraw.Draw(left), ImageDraw.Draw(right)

    if mesh is not None:
        vertices, quads = mesh
        _validate_tensor(vertices, "Mesh pixel vertices", (None, 2))
        if (
            vertices.ndim != 2 or not isinstance(quads, Tensor)
            or quads.ndim != 2 or quads.shape[1] != 4
            or quads.dtype not in (torch.int32, torch.int64)
            or quads.numel() == 0 or (quads < 0).any() or (quads >= len(vertices)).any()
        ):
            raise ValueError("Mesh requires (V, 2) vertices and valid (Q, 4) quadrilateral indices")
        pixels = vertices.detach().cpu().tolist()
        for quad in quads.detach().cpu().tolist():
            loop = [tuple(pixels[index]) for index in (*quad, quad[0])]
            mesh_plot.line(loop, fill="#b9c7d4")

    predicted = projected_gnm.detach().cpu().tolist()
    observed = observation.aligned_pixels().detach().cpu().tolist()

    # Draw text first and markers last, so neighboring index labels cannot
    # erase a point's status color. Draw observed markers above predictions.
    mesh_markers = []
    overlay_markers = []

    def mark(draw, markers, position, label, color, dy):
        x, y = position
        draw.text((x + 3, y + dy), label, fill=color, stroke_width=1, stroke_fill="white")
        markers.append((position, color))

    def paint_markers(draw, markers):
        for (x, y), color in markers:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color, outline="white")

    for index, position in enumerate(predicted):
        mark(mesh_plot, mesh_markers, position, str(index), GNM_COLOR, -12)

    observed_markers = []
    squared_errors = []
    for entry in observation.correspondences.entries:
        index, target = entry.dfl_index, entry.gnm_index
        if active[index]:
            a, b = observed[index], predicted[target]
            overlay.line((tuple(a), tuple(b)), fill=RESIDUAL_COLOR, width=2)
            squared_errors.append((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    for entry in observation.correspondences.entries:
        index, target = entry.dfl_index, entry.gnm_index
        if target is None:
            color = UNSUPPORTED_COLOR
        elif active[index]:
            color = OBSERVED_COLOR
            mark(overlay, overlay_markers, predicted[target], str(target), GNM_COLOR, -12)
        else:
            color = CONTOUR_COLOR if entry.region == Region.CONTOUR else EXCLUDED_COLOR
        label = str(index) if target in (None, index) else f"{index}->{target}"
        mark(overlay, observed_markers, observed[index], label, color, 1)

    paint_markers(mesh_plot, mesh_markers)
    paint_markers(overlay, overlay_markers)
    paint_markers(overlay, observed_markers)

    canvas = Image.new("RGB", (2 * max(width, 260) + 3 * PADDING, height + 150), "white")
    right_origin = 2 * PADDING + max(width, 260)
    canvas.paste(left, (PADDING, 56))
    canvas.paste(right, (right_origin, 56))
    draw = ImageDraw.Draw(canvas)
    draw.text((PADDING, 10), title, fill="#192838")
    draw.text((PADDING, 33), "GNM mesh and official indices", fill=GNM_COLOR)
    draw.text((right_origin, 33), "DeepFaceLab indices and reprojection", fill=OBSERVED_COLOR)
    legend = (
        ("GNM landmark", GNM_COLOR), ("Active observation", OBSERVED_COLOR),
        ("Unsupported observation", UNSUPPORTED_COLOR), ("Excluded contour", CONTOUR_COLOR),
        ("Other excluded observation", EXCLUDED_COLOR), ("Reprojection residual", RESIDUAL_COLOR),
    )
    for index, (label, color) in enumerate(legend):
        x = PADDING if index % 2 == 0 else right_origin
        y = height + 64 + 17 * (index // 2)
        draw.text((x, y), label, fill=color)
    summary = f"{len(squared_errors)} active pairs"
    if squared_errors:
        summary += f"  |  RMS pixel residual: {np.sqrt(np.mean(squared_errors)):.3f}"
    draw.text((PADDING, height + 120), summary, fill="#192838")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    return path


_DIAGNOSTIC_VIEWS = (
    ("frontal-neutral", "Frontal neutral", 0, 0.0),
    ("three-quarter-image-left", "Three-quarter, nose toward image left", -38, 0.0),
    ("three-quarter-image-right", "Three-quarter, nose toward image right", 38, 0.0),
    ("profile", "Profile-oriented", 78, 0.0),
    ("lower-face-expression", "Lower-face expression", 0, 2.0),
)


@torch.no_grad()
def write_synthetic_diagnostics(model, output: Path) -> tuple[Path, ...]:
    """Evaluate five reproducible views with a known synthetic camera."""
    from .gnm_geometry import extract_landmarks, load_sparse_landmarks

    load_sparse_landmarks(model)
    template = model.template_vertex_positions
    head = model.joint_names.index("head")
    lower_face = model.expression_names.index("lower_face_region_000")
    size = ImageSize(560, 560)
    camera = PerspectiveCamera(
        image_size=size,
        rotation=template.new_zeros(3),
        translation=template.new_tensor((0, 0.265, 0.62)),
        focal_length=template.new_tensor((980, 980)),
        principal_point=template.new_tensor((279.5, 279.5)),
    )
    offset = template.new_tensor((3.5, -2.0))
    paths = []
    for name, title, yaw, expression_value in _DIAGNOSTIC_VIEWS:
        rotations = template.new_zeros((model.num_joints, 3))
        rotations[head, 1] = radians(yaw)
        expression = template.new_zeros(model.expression_dim)
        expression[lower_face] = expression_value
        vertices, landmarks = extract_landmarks(model, expression=expression, rotations=rotations)
        projected = project(gnm_to_frontal_axes(landmarks), camera)
        projected_mesh = project(gnm_to_frontal_axes(vertices), camera)
        observation = LandmarkObservation(
            observation_id=name, image_size=size, landmarks=projected + offset, known_camera=camera,
        )
        paths.append(save_landmark_overlay(
            output / f"{name}.png", projected, observation,
            mesh=(projected_mesh, model.quads), title=f"{title}  |  Synthetic offset demonstration",
        ))
    return tuple(paths)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/landmark-diagnostics"))
    parser.add_argument("--model-file", type=Path, help="local official GNM v3.0 Head NPZ")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    from .gnm_mesh import load_head_model

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested and is unavailable")
    try:
        model = load_head_model(args.model_file).to(args.device)
        paths = write_synthetic_diagnostics(model, args.output)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
