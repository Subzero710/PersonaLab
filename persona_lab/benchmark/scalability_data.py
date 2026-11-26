"""Create and verify the immutable CPU float32 reference datasets for V6."""

from collections import Counter
import hashlib
from itertools import cycle, islice

import numpy as np
import torch

from ..correspondences import DFL_TO_GNM
from .configuration import resolve_config
from .dataset import load_dataset, observation_id
from .focal_estimation import HETEROGENEOUS_FACTORS
from .generate import generate_dataset
from .model import MODEL_SHA256
from .scalability_cases import PARENT_SIZE, STUDY
from .storage import digest_arrays, digest_json, json_bytes


REFERENCE_SCHEMA = f"{STUDY}-reference-v2-fp32"
PRECISION_REFERENCE_SCHEMA = f"{STUDY}-reference-v3-fp32-high-low"
SCHEDULE = tuple(islice(cycle(HETEROGENEOUS_FACTORS), PARENT_SIZE))
INPUT_FIELDS = {"observed": "landmarks", "validity": "validity", "weights": "weights"}
REFERENCE_FILES = ("manifest.json", "topology.npz", "subject_0000.npz")


def generation_config(seed):
    return resolve_config(
        "C", seed=seed, subjects=1, observations_per_subject=PARENT_SIZE,
        camera="perspective", identity_selection="head", device="cpu", dtype="float32",
        perspective_focal_factors=SCHEDULE,
    )


def file_digests(path):
    result = {}
    for name in REFERENCE_FILES:
        with (path / name).open("rb") as source:
            result[name] = hashlib.file_digest(source, "sha256").hexdigest()
    return result


def generate_parent(path, seed, model_file):
    dataset = generate_dataset(path, generation_config(seed), model_file=model_file)
    return {"dataset_id": dataset.dataset_id, "files_sha256": file_digests(path)}


def load_parent(path, seed, reference):
    files = file_digests(path)
    if files != reference["files_sha256"]:
        raise ValueError(f"Reference archive fingerprint mismatch for seed {seed}")
    dataset = load_dataset(path)
    if (reference.get("pixel_representation") not in (None, "raster-fp32", "compensated-fp32-high-low-v1")
            or dataset.has_precision_pixels != (reference.get("pixel_representation") == "compensated-fp32-high-low-v1")
            or dataset.has_precision_pixels and dataset.precision_source_dataset_id != reference.get("source_dataset_id")):
        raise ValueError("Reference pixel representation differs from dataset schema")
    if (
        dataset.dataset_id != reference["dataset_id"]
        or json_bytes(dataset.config.metadata()) != json_bytes(generation_config(seed).metadata())
        or dataset.model_metadata["asset_sha256"] != MODEL_SHA256
        or dataset.correspondences != DFL_TO_GNM
    ):
        raise ValueError(f"Invalid V6 reference dataset for seed {seed}")
    return dataset


def _input_digest(arrays, ids, image_sizes, dataset_id):
    return digest_json({
        "arrays_sha256": digest_arrays(arrays),
        "observation_ids": ids, "image_sizes": image_sizes, "dataset_id": dataset_id,
    })


def verify_fitter_input(observations, provenance, *, device):
    """Verify the transferred observation values without exposing truth to the fitter."""
    precise = provenance.get("pixel_representation") == "compensated-fp32-high-low-v1"
    fields = INPUT_FIELDS | ({"pixel_low": "pixel_low"} if precise else {})
    for item in observations:
        if (item.pixel_low is not None) != precise:
            raise ValueError("Transferred pixel representation differs from the reference")
        if item.known_camera is not None or item.camera_initialization is not None or any(
            getattr(item, field).device != device for field in fields.values()
        ):
            raise ValueError("V6 fitting requires estimated cameras and observations on the requested device")
        if item.correspondences != DFL_TO_GNM or item.coordinate_convention != "aligned_pixels":
            raise ValueError("V6 fitting requires the original correspondences and aligned-pixel coordinates")
        if item.landmarks.dtype != torch.float32 or item.weights.dtype != torch.float32:
            raise ValueError("V6 fitting requires float32 landmarks and weights")
    arrays = {name: torch.stack([getattr(obs, field) for obs in observations]).cpu().numpy()
              for name, field in fields.items()}
    ids = tuple(obs.observation_id for obs in observations)
    sizes = tuple((obs.image_size.width, obs.image_size.height) for obs in observations)
    if _input_digest(arrays, ids, sizes, provenance["dataset_id"]) != provenance["fitter_input_sha256"]:
        raise ValueError("Transferred fitting observations differ from the serialized reference subset")


def verified_subset(parent, count):
    subject, = parent.subject_ids
    ids = parent.observation_ids(subject)[:count]
    if ids != tuple(observation_id(subject, index) for index in range(count)):
        raise ValueError("V6 requires a prefix of the original ordered parent observations")
    dataset = parent.select_observations(subject, ids)
    truth = dataset.ground_truth(subject)
    if tuple(row["id"] for row in truth.observations) != ids:
        raise ValueError("Subset observation metadata does not match the selected order")
    sizes = (parent.config.image_size,) * count
    factors = [row["camera"]["focal_length"][0] / min(parent.config.image_size)
               for row in truth.observations]
    if not np.allclose(factors, SCHEDULE[:count], atol=0, rtol=2 * np.finfo("float32").eps):
        raise ValueError("Reference focal factors differ from the requested V6 schedule")
    expression_norms = np.linalg.norm(truth.arrays["expression"], axis=1)
    precise = dataset.has_precision_pixels
    fields = INPUT_FIELDS | ({"pixel_low": "pixel_low"} if precise else {})
    provenance = {
        "pixel_representation": "compensated-fp32-high-low-v1" if precise else "raster-fp32",
        "precision_source_dataset_id": dataset.precision_source_dataset_id,
        "source_dataset_id": parent.dataset_id, "dataset_id": dataset.dataset_id,
        "observation_ids": ids, "actual_observations": len(ids),
        "content_sha256": digest_json({
            "arrays_sha256": digest_arrays(truth.arrays), "observations": truth.observations,
            "model": parent.model_metadata, "image_sizes": sizes, "parent": parent.dataset_id,
        }),
        "identity_sha256": digest_arrays({"identity": truth.arrays["identity"]}),
        "fitter_input_sha256": _input_digest(
            {name: truth.arrays[name] for name in fields}, ids, sizes, dataset.dataset_id,
        ),
        "distribution": {
            "viewpoint_counts": dict(Counter(row["view"] for row in truth.observations)),
            "viewpoint_degrees": parent.config.view_degrees,
            "expression_preset_counts": dict(Counter(row["expression"] for row in truth.observations)),
            "expression_l2": {
                "min": float(expression_norms.min()), "median": float(np.median(expression_norms)),
                "max": float(expression_norms.max()),
            },
            "focal_factors": factors, "focal_factor_counts": dict(Counter(map(str, factors))),
            "viewpoint_focal_factors": {row["view"]: factor
                                        for row, factor in zip(truth.observations, factors, strict=True)},
        },
    }
    return dataset, subject, provenance
