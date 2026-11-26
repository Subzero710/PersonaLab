"""Independent per-view derivatives, evaluated in observation batches.

The independent copies of shared identity have equal values but separate
leaves. The derivative of the sum of corresponding residual rows is then
exactly the per-view derivative, without cross-view Jacobian blocks.
"""
from dataclasses import replace

import torch

from ..camera import PerspectiveCamera, project
from ..coordinates import gnm_to_frontal_axes
from .losses import _landmark_residuals
from .parameters import InvalidGeometry, stack_cameras


def _require_supported(parameters, layout):
    config = parameters.config
    if (config.lm.arithmetic_dtype != "float32" or config.camera_mode != "estimated"
        or config.camera_model != "perspective" or config.focal_mode != "per-observation"
        or not layout.identity or layout.shared_focal or set(layout.names) != {"expression", "pose", "camera_delta", "local_log_focal"}
        or any(getattr(config, f"{group}_weight") != 0 for group in ("identity", "expression", "pose", "camera"))):
        raise ValueError("FP32 square-root fitting currently requires the explicit zero-prior joint per-observation perspective protocol")


def project_fp32_batch(parameters, start, stop, values):
    """Same sparse residual forward for FP32 objectives and Jacobians."""
    count = stop - start
    initial = parameters.initial
    if (values.identity is None or values.expression is None or values.pose is None
        or values.camera_delta is None or values.local_log_focal is None
        or values.root is not None or values.shared_log_focal is not None):
        raise ValueError("FP32 square-root projection requires joint per-view perspective fitting")
    identity = initial.identity.expand(count, -1).index_copy(1, parameters.identity_indices,
                                                            values.identity.expand(count, -1))
    expression = initial.expression[start:stop].index_copy(1, parameters.expression_indices, values.expression)
    rotations = initial.rotations[start:stop].index_copy(1, parameters.head_indices, values.pose[:, None, :])
    cameras = initial.cameras[start:stop]
    if any(not isinstance(camera, PerspectiveCamera) or camera.image_size != cameras[0].image_size for camera in cameras):
        raise ValueError("Batched FP32 Jacobian requires one perspective raster size")
    camera = stack_cameras(cameras)
    offsets = values.camera_delta
    initial_z = camera.translation[:, 2:]
    translation = torch.cat((camera.translation[:, :2] + offsets[:, :2] * initial_z,
                             initial_z * offsets[:, 2:].exp()), dim=1)
    focal = camera.focal_length * values.local_log_focal.exp()
    camera = replace(camera, translation=translation, focal_length=focal)
    landmarks = parameters.landmark_geometry(identity, expression, rotations, initial.translation[start:stop])
    try:
        return project(gnm_to_frontal_axes(landmarks), camera)
    except ValueError as error:
        raise InvalidGeometry(str(error)) from error


@torch.enable_grad()
def batch_jacobian(data, parameters, layout, position, start, stop):
    _require_supported(parameters, layout)
    count = stop - start
    width = position.identity.numel()
    point = torch.cat((position.identity.expand(count, -1), position.local[start:stop]), dim=1).detach().requires_grad_()
    values = replace(parameters.values().batch(start, stop), identity=point[:, :width], **{
        name: point[:, width:][:, columns] for name, columns in layout.columns()
    })
    projected = project_fp32_batch(parameters, start, stop, values)
    normalized, _ = _landmark_residuals(projected, data, start, stop)
    weights = (data.weights[start:stop] / data.weight_sum).sqrt()
    residual = (normalized * weights[..., None]).flatten(1)
    summed = residual.sum(0)
    rows = parameters.config.lm.jacobian_rows
    m = summed.numel()
    jacobian = residual.new_empty((count, m, point.shape[-1]))
    for begin in range(0, m, rows):
        end = min(begin + rows, m)
        seeds = residual.new_zeros((end - begin, m))
        seeds[:, begin:end].diagonal().fill_(1)
        gradient, = torch.autograd.grad(summed, point, grad_outputs=seeds,
                                        is_grads_batched=True, retain_graph=end < m)
        jacobian[:, begin:end] = gradient.transpose(0, 1)
    if not torch.isfinite(residual).all() or not torch.isfinite(jacobian).all():
        raise FloatingPointError("Nonfinite batched residual or Jacobian")
    return residual.detach(), jacobian.detach()
