"""Compensated FP32 perspective residual, without absolute pixel rounding.

Error-free sums and products recover low parts of ordinary FP32 operations.
The analytic derivative is that of the underlying smooth perspective map,
not of the discrete rounding corrections. No float64 arithmetic is used.
"""

import torch


def _two_sum(a, b):
    high = a + b
    tail = high - a
    low = (a - (high - tail)) + (b - tail)
    return high, low


def _split(a):
    # Dekker's splitter for the 24-bit binary32 significand. Camera coordinates
    # and raster sizes are far inside the normal range of this arithmetic.
    c = a * 4097.0
    high = c - (c - a)
    return high, a - high


def _two_product(a, b):
    high = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    low = ((ah * bh - high) + ah * bl + al * bh) + al * bl
    return high, low


def perspective_residual(rotated, translation, focal, principal, observed):
    """Return (..., L, 2) residuals in pixels from already mapped 3D points.

    Evaluate [f*X - (observed-c)*Z]/Z, preserving low parts of camera
    translation, centering and products. Inputs are validated by the fitting
    boundary; nonfinite compensated results are errors, not an alternate path.
    """
    if any(t.dtype != torch.float32 or t.device != rotated.device
           for t in (rotated, translation, focal, principal, observed)):
        raise ValueError("Compensated projection requires FP32 on one device")
    translation, focal, principal = (t.unsqueeze(-2) for t in (translation, focal, principal))
    # Preserve the ordinary analytic Jacobian. Do not backpropagate through
    # error-free transforms, whose rounded intermediates are not smooth.
    camera = rotated + translation
    depth = camera[..., 2:]
    smooth = (camera[..., :2] / depth) * focal - (observed - principal)
    with torch.no_grad():
        xyz, xyz_low = _two_sum(rotated, translation)
        xy, z = xyz[..., :2], xyz[..., 2:]
        xy_low, z_low = xyz_low[..., :2], xyz_low[..., 2:]
        center, center_low = _two_sum(observed, -principal)
        left, left_low = _two_product(focal, xy)
        right, right_low = _two_product(center, z)
        numerator, subtraction_low = _two_sum(left, -right)
        correction = (subtraction_low + (left_low - right_low)
                      + focal * xy_low - center * z_low - center_low * z
                      - center_low * z_low)
        residual = (numerator + correction) / z
        # The low part of Z also changes the quotient; include first-order
        # correction without rounding the reconstructed denominator again.
        residual = residual - residual * (z_low / z)
        valid = (torch.isfinite(residual).all() & torch.isfinite(smooth).all()
                 & (depth > 0).all() & (translation[..., 2:] > 0).all()
                 & (focal > 0).all() & torch.isfinite(focal).all())
        if not bool(valid):
            raise ValueError("Invalid or nonfinite compensated perspective residual")
    return residual + (smooth - smooth.detach()) if torch.is_grad_enabled() else residual
