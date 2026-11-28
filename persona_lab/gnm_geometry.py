"""Official GNM v3.0 Head landmarks and shared-identity batching."""

from gnm.shape import gnm_common, gnm_landmarks, gnm_pytorch
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


class SparseHeadLandmarks:
    """Evaluate weighted native surface locations with the existing GNM operations."""

    def __init__(self, head, *, indices=None, weights=None):
        if indices is None and weights is None:
            config = load_sparse_landmarks(head)
            indices, weights = config.indices, config.weights
        elif indices is None or weights is None:
            raise ValueError("Surface geometry requires both native indices and weights")
        template = head.template_vertex_positions
        vertices, triangles = np.unique(indices, return_inverse=True)
        native = torch.as_tensor(vertices, dtype=torch.long, device=template.device)
        self.triangles = torch.as_tensor(
            triangles.reshape(indices.shape), dtype=torch.long, device=template.device,
        )
        self.weights = torch.as_tensor(weights, dtype=template.dtype, device=template.device)
        self.template = template.index_select(0, native)
        self.identity_basis = head.vertex_identity_basis.index_select(1, native)
        self.expression_basis = head.expression_basis.index_select(1, native)
        self.skinning_weights = head.skinning_weights.index_select(1, native)
        self.pose_regressor = head.pose_correctives_regressor
        if self.pose_regressor is not None:
            self.pose_regressor = self.pose_regressor.reshape(-1, head.num_vertices, 3).index_select(
                1, native,
            ).flatten(1)
        self.head = head

    def __call__(self, identity, expression, rotations, translation):
        identity = identity.expand(*expression.shape[:-1], self.head.identity_dim)
        vertices = gnm_common.vertex_positions_bind_pose(
            identity, expression, self.template, self.identity_basis, self.expression_basis,
        )
        joints = self.head.joint_positions_bind_pose(identity)
        matrices = gnm_common.axis_angle_to_rotation_matrix(rotations)
        corrective = gnm_common.compute_pose_correctives_from_rotation_matrices(
            matrices, self.pose_regressor, self.template, self.head.num_joints, len(self.template),
        )
        posed = gnm_common.linear_blend_skinning_from_rotation_matrices(
            vertices + corrective, joints, matrices, translation,
            self.skinning_weights, self.head.joint_parent_indices,
        )
        return (posed[..., self.triangles, :] * self.weights[..., None]).sum(dim=-2)


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
