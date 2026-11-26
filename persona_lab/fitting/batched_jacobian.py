"""Independent per-view derivatives, evaluated in observation batches.

The independent copies of shared identity have equal values but separate
leaves. The derivative of the sum of corresponding residual rows is then
exactly the per-view derivative, without cross-view Jacobian blocks.
"""
from dataclasses import replace

import torch

from .precise_projection import perspective_residual
from .parameters import InvalidGeometry
from . import precision_geometry as pg
from .losses import _landmark_residuals


def _require_supported(parameters, layout):
    config = parameters.config
    camera_names = ("camera_native", "local_focal") if config.lm.fp32_forward == "compensated" else ("camera_delta", "local_log_focal")
    if (config.lm.arithmetic_dtype != "float32" or config.camera_mode != "estimated"
        or config.camera_model != "perspective" or config.focal_mode != "per-observation"
        or not layout.identity or layout.shared_focal or set(layout.names) != {"expression", "pose", *camera_names}
        or any(getattr(config, f"{group}_weight") != 0 for group in ("identity", "expression", "pose", "camera"))):
        raise ValueError("FP32 square-root fitting currently requires the explicit zero-prior joint per-observation perspective protocol")


def _fp32_inputs(parameters, start, stop, values):
    """Shared sparse geometry and decoded physical camera for FP32 fitting."""
    count = stop - start
    initial = parameters.initial
    if (values.identity is None or values.expression is None or values.pose is None
        or values.root is not None or values.shared_log_focal is not None):
        raise ValueError("FP32 square-root projection requires joint per-view perspective fitting")
    identity = initial.identity.expand(count, -1).index_copy(1, parameters.identity_indices,
                                                            values.identity.expand(count, -1))
    expression = initial.expression[start:stop].index_copy(1, parameters.expression_indices, values.expression)
    rotations = initial.rotations[start:stop].index_copy(1, parameters.head_indices, values.pose[:, None, :])
    camera, matrices = parameters.fp32_camera_batch
    camera = replace(camera, rotation=camera.rotation[start:stop],
                     translation=camera.translation[start:stop],
                     focal_length=camera.focal_length[start:stop],
                     principal_point=camera.principal_point[start:stop])
    if parameters.config.lm.fp32_forward == "compensated":
        if values.camera_native is None or values.local_focal is None:
            raise ValueError("Compensated FP32 requires native camera coordinates")
        translation, focal = values.camera_native, values.local_focal.expand(-1, 2)
    else:
        if values.camera_delta is None or values.local_log_focal is None:
            raise ValueError("Projection-only FP32 requires log camera coordinates")
        offsets = values.camera_delta
        initial_z = camera.translation[:, 2:]
        translation = torch.cat((camera.translation[:, :2] + offsets[:, :2] * initial_z,
                                 initial_z * offsets[:, 2:].exp()), dim=1)
        focal = camera.focal_length * values.local_log_focal.exp()
    camera = replace(camera, translation=translation, focal_length=focal)
    return (identity, expression, rotations, initial.translation[start:stop]), camera, matrices[start:stop]


def fp32_geometry_batch(parameters, start, stop, values):
    args, camera, matrices = _fp32_inputs(parameters, start, stop, values)
    landmarks = parameters.landmark_geometry(*args)
    rotated = (landmarks * landmarks.new_tensor((1, -1, -1))) @ matrices.mT
    return rotated, camera, landmarks


def residual_fp32_batch(data, parameters, start, stop, values):
    """Evaluate one common compensated residual for trials and derivatives."""
    if parameters.config.lm.fp32_forward == "compensated":
        args, camera, matrices = _fp32_inputs(parameters, start, stop, values)
        indices = data.gnm_indices[start:stop, :, None].expand(-1, -1, 2)
        eligible = data.eligible[start:stop, :, None]
        with torch.no_grad():
            try:
                projected = pg.projected_pair(pg.landmark_pair(parameters.landmark_geometry, *args),
                    camera.rotation, camera.translation, camera.focal_length, camera.principal_point)
            except ValueError as error:
                raise InvalidGeometry(str(error)) from error
            observed = data.pixels[start:stop]
            low = torch.zeros_like(observed) if data.pixel_low is None else data.pixel_low[start:stop]
            hi, lo = pg.sub(tuple(t.gather(1, indices) for t in projected), (observed, low))
            pixels = (hi + lo).masked_fill(~eligible, 0)
        if torch.is_grad_enabled():
            # Smooth native GNM derivative, not the derivative of rounding-error transforms.
            landmarks = parameters.landmark_geometry(*args)
            rotated = (landmarks * landmarks.new_tensor((1, -1, -1))) @ matrices.mT
            xyz = rotated + camera.translation[:, None, :]
            smooth = (xyz[..., :2] / xyz[..., 2:] * camera.focal_length[:, None, :]).gather(1, indices)
            smooth = smooth.masked_fill(~eligible, 0)
            pixels = pixels + (smooth - smooth.detach())
        return pixels / data.short_edges[start:stop, None, None], pixels
    rotated, camera, _ = fp32_geometry_batch(parameters, start, stop, values)
    indices = data.gnm_indices[start:stop, :, None].expand(-1, -1, 3)
    mapped = rotated.gather(1, indices)
    eligible = data.eligible[start:stop, :, None]
    # Unsupported rows cannot affect the loss; use a finite centered target.
    observed = torch.where(eligible, data.pixels[start:stop], camera.principal_point[:, None, :])
    try:
        pixels = perspective_residual(mapped, camera.translation, camera.focal_length,
                                      camera.principal_point, observed)
    except ValueError as error:
        raise InvalidGeometry(str(error)) from error
    pixels = pixels.masked_fill(~eligible, 0)
    return pixels / data.short_edges[start:stop, None, None], pixels


@torch.enable_grad()
def batch_jacobian(data, parameters, layout, position, start, stop, *, legacy_absolute=False):
    _require_supported(parameters, layout)
    count = stop - start
    width = position.identity.numel()
    point = torch.cat((position.identity.expand(count, -1), position.local[start:stop]), dim=1).detach().requires_grad_()
    values = replace(parameters.values().batch(start, stop), identity=point[:, :width], **{
        name: point[:, width:][:, columns] for name, columns in layout.columns()
    })
    if legacy_absolute:
        # Explicit diagnostic only: V6.1-style absolute raster residual.
        rotated, camera, _ = fp32_geometry_batch(parameters, start, stop, values)
        xyz = rotated + camera.translation[:, None, :]
        projected = xyz[..., :2] / xyz[..., 2:] * camera.focal_length[:, None, :] + camera.principal_point[:, None, :]
        normalized, _ = _landmark_residuals(projected, data, start, stop)
    else:
        normalized, _ = residual_fp32_batch(data, parameters, start, stop, values)
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
    if not bool(torch.isfinite(residual).all() & torch.isfinite(jacobian).all()):
        raise FloatingPointError("Nonfinite batched residual or Jacobian")
    return residual.detach(), jacobian.detach()
