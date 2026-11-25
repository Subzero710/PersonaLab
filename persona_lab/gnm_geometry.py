"""Official GNM v3.0 Head landmarks and shared-identity batching."""

from gnm.shape import gnm_landmarks, gnm_pytorch
import numpy as np
import torch
from torch import Tensor

from .coordinates import _validate_tensor


SPARSE_HEAD = gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68
GNM_LANDMARK_COUNT = 68
# The file stores weights to three decimal places. Allow float32 roundoff too.
BARYCENTRIC_ATOL = 0.00101


def _require_head(model: gnm_pytorch.GNM) -> None:
    if (
        not isinstance(model, gnm_pytorch.GNM)
        or model.version != gnm_pytorch.GNMVersion.V3_0
        or model.variant != gnm_pytorch.GNMVariant.HEAD
    ):
        raise ValueError("Expected the PyTorch GNM v3.0 Head model")


def load_sparse_landmarks(
    model: gnm_pytorch.GNM,
) -> gnm_landmarks.LandmarksConfiguration:
    """Load and validate the official resource against the v3.0 Head topology."""
    _require_head(model)
    config = gnm_landmarks.load_landmarks(SPARSE_HEAD)
    indices, weights = config.indices, config.weights
    expected = (GNM_LANDMARK_COUNT, 3)
    if indices.shape != expected or weights.shape != expected:
        raise ValueError(f"GNM sparse head indices and weights must have shape {expected}")
    if not np.issubdtype(indices.dtype, np.integer) or (
        (indices < 0) | (indices >= model.num_vertices)
    ).any():
        raise ValueError("GNM sparse head indices fall outside the mesh topology")
    if not np.issubdtype(weights.dtype, np.floating) or not (
        np.isfinite(weights) & (weights >= 0) & (weights <= 1)
    ).all():
        raise ValueError("GNM sparse head weights must be finite and in [0, 1]")
    if not np.allclose(weights.sum(axis=-1), 1, atol=BARYCENTRIC_ATOL, rtol=0):
        raise ValueError(f"GNM sparse head weights must sum to one within {BARYCENTRIC_ATOL}")
    return config


def extract_landmarks(
    model: gnm_pytorch.GNM,
    *,
    identity: Tensor | None = None,
    expression: Tensor | None = None,
    rotations: Tensor | None = None,
    translation: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return (..., V, 3) vertices and (..., 68, 3) official GNM landmarks.

    Model and inputs share a float32/float64 dtype and a CPU/CUDA device.
    Expression, joint axis-angle rotations and root translation must share
    leading dimensions. A single (I,) identity expands over those dimensions.
    An already batched identity must match them exactly. Missing parameters
    use GNM's native neutral defaults.
    """
    _require_head(model)
    reference = model.template_vertex_positions
    if reference.dtype not in (torch.float32, torch.float64) or (
        reference.device.type not in {"cpu", "cuda"}
    ):
        raise ValueError("GNM must be float32 or float64 on CPU or CUDA")
    if identity is not None:
        _validate_tensor(identity, "Identity", (model.identity_dim,), like=reference)

    batch_shape = None
    for name, values, suffix in (
        ("Expression", expression, (model.expression_dim,)),
        ("Joint rotations", rotations, (model.num_joints, 3)),
        ("Root translation", translation, (3,)),
    ):
        if values is None:
            continue
        _validate_tensor(values, name, suffix, like=reference)
        shape = values.shape[:-len(suffix)]
        if batch_shape is not None and shape != batch_shape:
            raise ValueError("GNM expression, rotations and translation need matching batches")
        batch_shape = shape

    if identity is not None and batch_shape is not None:
        if identity.ndim == 1:
            identity = identity.expand(*batch_shape, model.identity_dim)
        elif identity.shape[:-1] != batch_shape:
            raise ValueError("Batched GNM identity must match the observation batch dimensions")

    return model.vertices_and_landmarks(
        landmarks_type=SPARSE_HEAD,
        identity=identity,
        expression=expression,
        rotations=rotations,
        translation=translation,
    )
