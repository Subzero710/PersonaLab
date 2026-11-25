"""Pinned model provenance and the shared benchmark projection path."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

from gnm.shape import gnm_data_loader, gnm_pytorch
import numpy as np
import torch

from ..camera import project
from ..coordinates import gnm_to_frontal_axes, pixels_to_normalized
from ..correspondences import DFL_LANDMARK_COUNT
from ..gnm_geometry import SPARSE_HEAD, extract_landmarks, load_sparse_landmarks
from ..gnm_mesh import load_head_model
from .configuration import torch_options
from .storage import digest_arrays


MODEL_REPOSITORY = "google/gnm-v3"
MODEL_REVISION = "c01e90d298d82301f9fd18f54806be751775cb7c"
MODEL_PATH = "v3_0/gnm_head.npz"
MODEL_SHA256 = "61d78bbfb4ad8e0b38495804a4caef3214d3df00f8c3f68761e63b41ce3747eb"
MODEL_URL = f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{MODEL_PATH}"


def _asset_checksum(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _verify_asset(path):
    checksum = _asset_checksum(path)
    if checksum != MODEL_SHA256:
        raise ValueError(f"GNM v3.0 Head checksum mismatch for {path}: {checksum}")
    return checksum


def _cached_asset():
    cache_path = Path(gnm_data_loader.get_default_gnm_cache_dir()) / MODEL_PATH
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".benchmark-head-", dir=cache_path.parent) as temp:
        download = Path(temp) / cache_path.name
        with urlopen(MODEL_URL, timeout=120) as source, download.open("wb") as target:
            shutil.copyfileobj(source, target)
        _verify_asset(download)
        try:
            os.link(download, cache_path)
        except FileExistsError:
            # A concurrent resolver published the file; the caller verifies it.
            pass
    return cache_path


@dataclass(frozen=True, slots=True)
class BenchmarkModel:
    head: gnm_pytorch.GNM
    triangles: np.ndarray
    metadata: dict

    @property
    def reference(self):
        return self.head.template_vertex_positions

    @property
    def numpy_dtype(self):
        return np.dtype("float32" if self.reference.dtype == torch.float32 else "float64")

    def tensor(self, values):
        return torch.tensor(values, dtype=self.reference.dtype, device=self.reference.device)

    @torch.no_grad()
    def neutral(self, identity):
        head = self.head
        vertices = head(
            identity=self.tensor(identity),
            expression=self.reference.new_zeros(head.expression_dim),
            rotations=self.reference.new_zeros((head.num_joints, 3)),
            translation=self.reference.new_zeros(3),
        )
        if vertices.shape != (head.num_vertices, 3) or not torch.isfinite(vertices).all():
            raise ValueError("GNM returned invalid neutral vertices")
        return vertices.cpu().numpy()

    @torch.no_grad()
    def landmarks(self, identity, expression, rotations, translation):
        vertices, landmarks = extract_landmarks(
            self.head, identity=self.tensor(identity), expression=self.tensor(expression),
            rotations=self.tensor(rotations), translation=self.tensor(translation),
        )
        if not torch.isfinite(vertices).all() or not torch.isfinite(landmarks).all():
            raise ValueError("GNM returned nonfinite posed geometry")
        return landmarks


def load_benchmark_model(model_file=None, *, dtype="float64", device="cpu"):
    """Load the verified GNM asset through PersonaLab's existing PyTorch loader."""
    dtype, device = torch_options(dtype, device)
    path = Path(model_file) if model_file is not None else _cached_asset()
    checksum = _verify_asset(path)
    head = load_head_model(path).to(device=device, dtype=dtype)
    landmarks = load_sparse_landmarks(head)
    triangles = head.triangles.detach().cpu().numpy().copy()
    metadata = {
        "version": head.version.value,
        "variant": head.variant.value,
        "asset": MODEL_URL,
        "asset_sha256": checksum,
        "identity_count": head.identity_dim,
        "identity_names": list(head.identity_names),
        "expression_count": head.expression_dim,
        "expression_names": list(head.expression_names),
        "joint_count": head.num_joints,
        "joint_names": list(head.joint_names),
        "vertex_count": head.num_vertices,
        "triangle_count": len(triangles),
        "topology_sha256": digest_arrays({"triangles": triangles}),
        "landmark_configuration": SPARSE_HEAD.name,
        "landmarks_sha256": digest_arrays({
            "indices": landmarks.indices, "weights": landmarks.weights,
        }),
    }
    return BenchmarkModel(head, triangles, metadata)


def project_dfl(landmarks, camera, correspondences):
    """Project official GNM points and gather supported rows in 68-slot DFL order."""
    projected = project(gnm_to_frontal_axes(landmarks), camera)
    supported = [entry for entry in correspondences.entries if entry.supported]
    dfl_indices = [entry.dfl_index for entry in supported]
    gnm_indices = [entry.gnm_index for entry in supported]
    dfl = projected.new_zeros((*projected.shape[:-2], DFL_LANDMARK_COUNT, 2))
    dfl[..., dfl_indices, :] = projected[..., gnm_indices, :]
    return dfl


def in_frame(pixels, image_size, supported):
    """Use half-open raster edges; unsupported slots have no frame status."""
    normalized = pixels_to_normalized(pixels, image_size)
    return ((normalized >= 0) & (normalized < 1)).all(dim=-1) & supported
