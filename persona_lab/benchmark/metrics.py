"""Independent reductions for raw coefficients, native geometry and raster error."""

import numpy as np
from scipy.spatial.transform import Rotation

from ..correspondences import DFL_LANDMARK_COUNT


REGION_THRESHOLD = 1e-4
NEUTRAL_REGIONS = {
    "exterior_skin": "skin_exterior", "facial": "hockey_mask", "eyes": "eyes", "teeth": "teeth",
}
METRIC_UNITS = {"coefficients": "native", "geometry": "m", "reprojection": "px", "rotation": "deg"}


def _finite_float(values, name):
    array = np.asarray(values)
    if array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite floating-point values")
    return array.astype(np.float64, copy=False)


def _difference(estimated, ground_truth):
    a = _finite_float(estimated, "Estimate")
    b = _finite_float(ground_truth, "Ground truth")
    if a.shape != b.shape:
        raise ValueError("Estimate and ground truth must have the same shape")
    with np.errstate(over="raise", invalid="raise"):
        return a - b


def _distance(estimated, ground_truth):
    with np.errstate(over="raise", invalid="raise"):
        return np.linalg.norm(_difference(estimated, ground_truth), axis=-1)


def _rms(values):
    with np.errstate(over="raise", invalid="raise"):
        return float(np.sqrt(np.mean(np.square(values))))


def coefficient_errors(estimated, ground_truth, *, estimated_names=None, ground_truth_names=None):
    """Raw MAE, RMSE and maximum absolute error in the model's ordered basis."""
    error = np.abs(_difference(estimated, ground_truth))
    if error.ndim != 1 or not len(error):
        raise ValueError("Expected nonempty coefficient vectors")
    if estimated_names is not None or ground_truth_names is not None:
        if (
            estimated_names is None or ground_truth_names is None
            or tuple(estimated_names) != tuple(ground_truth_names)
            or len(estimated_names) != len(error)
        ):
            raise ValueError("Coefficient names and ordering must match the vectors")
    return {
        "mae": float(np.mean(error)), "rmse": _rms(error),
        "max_absolute": float(np.max(error)), "coefficient_count": len(error),
    }


def grouped_coefficient_errors(estimated, ground_truth, names, *, trainable=None, regions=None):
    """Score native coefficient groups, with empty selections represented by null."""
    estimated, ground_truth = np.asarray(estimated), np.asarray(ground_truth)
    metrics = {"global": coefficient_errors(
        estimated, ground_truth, estimated_names=names, ground_truth_names=names,
    )}
    groups = dict(regions or {})
    if trainable is not None:
        if len(set(trainable)) != len(trainable) or not set(trainable) <= set(names):
            raise ValueError("Trainable coefficients must be unique native names")
        mask = np.isin(names, trainable)
        groups.update(trainable=np.flatnonzero(mask), fixed=np.flatnonzero(~mask))
    for name, indices in groups.items():
        metrics[name] = coefficient_errors(estimated[indices], ground_truth[indices]) if len(indices) else None
    return metrics


def pool_coefficient_errors(metrics):
    """Pool MAE and squared error using the actual number of coefficients."""
    metrics = [item for item in metrics if item is not None]
    count = sum(item["coefficient_count"] for item in metrics)
    if not count:
        return None
    return {
        "coefficient_count": count,
        "mae": sum(item["coefficient_count"] * item["mae"] for item in metrics) / count,
        "rmse": float(np.sqrt(sum(item["coefficient_count"] * item["rmse"] ** 2 for item in metrics) / count)),
        "max_absolute": max(item["max_absolute"] for item in metrics),
    }


def fixed_coefficient_error(initial, truth, names, trainable):
    """Return mismatched fixed coefficient names and their full-native RMSE floor."""
    delta = _difference(initial, truth)
    if delta.shape != (len(names),):
        raise ValueError("Fixed-coefficient analysis requires a full native vector")
    fixed = ~np.isin(names, trainable)
    return {
        "differing_coefficients": [name for name, value in zip(names, delta * fixed, strict=True) if value != 0],
        "full_native_rmse_floor": _rms(delta * fixed),
    }


def vertex_errors(estimated, ground_truth):
    """Compare matching (V, 3) positions in meters without any alignment."""
    if np.ndim(estimated) != 2 or np.shape(estimated)[-1] != 3 or len(estimated) == 0:
        raise ValueError("Expected a nonempty (V, 3) array of corresponding vertices")
    distances = _distance(estimated, ground_truth)
    return {
        "mean_m": float(np.mean(distances)), "rms_m": _rms(distances),
        "max_m": float(np.max(distances)), "vertex_count": len(distances),
    }


def neutral_geometry_errors(model, identity, truth):
    """Score unaligned meshes in meters and expose the native anatomical regions."""
    vertices = model.neutral(identity)
    weights = model.head.vertex_groups.detach().cpu().numpy()
    groups = {}
    for name, values in zip(model.head.vertex_group_names, weights, strict=True):
        mask = values > REGION_THRESHOLD
        if mask.any():
            groups[name] = vertex_errors(vertices[mask], truth[mask])
    return {
        "neutral_geometry": vertex_errors(vertices, truth),
        "neutral_geometry_regions": {name: groups[region] for name, region in NEUTRAL_REGIONS.items()},
        "vertex_groups": groups,
        "vertex_group_weight_threshold": REGION_THRESHOLD,
    }


def summarize_pixels(distances, total):
    """Pool raw valid pixel distances before calculating summary statistics."""
    count = len(distances)
    return {
        "mean_px": float(np.mean(distances)) if count else None,
        "median_px": float(np.median(distances)) if count else None,
        "rms_px": _rms(distances) if count else None,
        "valid_count": count,
        "excluded_count": total - count,
    }


def reprojection_errors(estimated, ground_truth, valid):
    """Score valid raster slots and preserve the original 68-slot order."""
    estimate = _finite_float(estimated, "Estimated pixels")
    reference = _finite_float(ground_truth, "Reference pixels")
    if estimate.shape != (DFL_LANDMARK_COUNT, 2) or reference.shape != estimate.shape:
        raise ValueError("Expected (68, 2) projected pixels")
    valid = np.asarray(valid)
    if valid.shape != (DFL_LANDMARK_COUNT,) or valid.dtype != np.bool_:
        raise ValueError("Expected a boolean (68,) validity mask")
    distances = _distance(estimate[valid], reference[valid])
    errors = [None] * len(valid)
    for index, distance in zip(np.flatnonzero(valid), distances, strict=True):
        errors[index] = float(distance)
    return summarize_pixels(distances, len(valid)) | {"per_landmark_px": errors}


def rotation_errors_deg(estimated, ground_truth):
    """Shortest relative SO(3) angle per axis-angle vector, in degrees."""
    a, b = _finite_float(estimated, "Estimated rotation"), _finite_float(ground_truth, "True rotation")
    if a.ndim not in (1, 2) or a.shape[-1] != 3 or a.shape != b.shape or not a.size:
        raise ValueError("Rotations must share shape (3,) or (J, 3)")
    result = np.rad2deg((Rotation.from_rotvec(a).inv() * Rotation.from_rotvec(b)).magnitude())
    if not np.isfinite(result).all():
        raise ValueError("Rotation error is nonfinite")
    return result


def translation_error(estimated, ground_truth):
    return float(_distance(estimated, ground_truth))
