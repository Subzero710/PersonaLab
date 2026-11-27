"""Experimental high/low FP32 forward with a fixed reduction tree.

No float64 tensors, no TF32, no shape-dependent BLAS accumulation in this
forward. The low part is carried through the final projected residual. This
is compensated arithmetic, not a guarantee of exact real arithmetic.
"""
import math
import struct
import torch

from .precise_projection import _two_sum, _two_product


def pair(x):
    if x.dtype != torch.float32:
        raise ValueError("Compensated geometry requires FP32")
    return x, torch.zeros_like(x)


def add(a, b):
    h, e = _two_sum(a[0], b[0])
    return _two_sum(h, e + (a[1] + b[1]))


def neg(a):
    return -a[0], -a[1]


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    h, e = _two_product(a[0], b[0])
    return _two_sum(h, e + (a[0] * b[1] + a[1] * b[0]) + a[1] * b[1])


def div(a, b):
    q = a[0] / b[0]
    remainder = sub(a, mul(pair(q), b))
    return _two_sum(q, (remainder[0] + remainder[1]) / b[0])


def at(a, index):
    return a[0][index], a[1][index]


def reshape(a, *shape):
    return a[0].reshape(*shape), a[1].reshape(*shape)


def stack(items, dim):
    return tuple(torch.stack([item[i] for item in items], dim) for i in range(2))


def sum_pair(a, dim):
    """A fixed pairwise tree independent of observation batch shape."""
    h, l = (x.movedim(dim, -1) for x in a)
    if h.shape[-1] == 0:
        return pair(h.new_zeros(h.shape[:-1]))
    while h.shape[-1] > 1:
        if h.shape[-1] % 2:
            h, l = (torch.cat((x, torch.zeros_like(x[..., :1])), -1) for x in (h, l))
        h, l = add((h[..., ::2], l[..., ::2]), (h[..., 1::2], l[..., 1::2]))
    return h.squeeze(-1), l.squeeze(-1)


def matmul(a, b):
    """Matrix product using fixed-order FP32 two-product reductions."""
    return sum_pair(mul(tuple(x.unsqueeze(-1) for x in a),
                        tuple(x.unsqueeze(-3) for x in b)), -2)


def matvec(a, b):
    return sum_pair(mul(a, tuple(x.unsqueeze(-2) for x in b)), -1)


def linear(coefficients, basis, chunk=32):
    """Compute a native basis contraction in bounded coefficient chunks."""
    n = coefficients[0].shape[-1]
    if basis.shape[0] != n:
        raise ValueError("Inconsistent compensated basis dimensions")
    flat = basis.reshape(n, -1)
    result = pair(coefficients[0].new_zeros((*coefficients[0].shape[:-1], flat.shape[-1])))
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        c = tuple(x[..., start:end, None] for x in coefficients)
        result = add(result, sum_pair(mul(c, pair(flat[start:end])), -2))
    return reshape(result, *coefficients[0].shape[:-1], *basis.shape[1:])


def _constant(value, like):
    # Split compile-time scalar coefficients; no float64 tensor operations.
    hi = struct.unpack('f', struct.pack('f', value))[0]
    return like.new_tensor(hi), like.new_tensor(value - hi)


def _series(squared, offset):
    # sinc and cosc as polynomials in theta^2, through order 36.
    # On |theta| <= 2*pi, the omitted term is below 4e-15 for sinc.
    value = _constant((-1.0)**18 / math.factorial(36 + offset), squared[0])
    for k in range(17, -1, -1):
        value = add(mul(value, squared), _constant((-1.0)**k / math.factorial(2*k + offset), squared[0]))
    return value


def rotation_pair(axis_angle):
    """Rodrigues with the GNM epsilon convention, no device trig discrepancy."""
    v = pair(axis_angle)
    squared = sum_pair(mul(v, v), -1)
    if not bool(torch.isfinite(squared[0]).all() & (squared[0] <= (2*math.pi)**2).all()):
        raise ValueError("Compensated rotation requires finite axis-angle norm <= 2*pi")
    eps = squared[0].new_tensor(1e-8)
    squared = (torch.maximum(squared[0], eps), torch.where(squared[0] >= eps, squared[1], 0))
    x, y, z = axis_angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = pair(torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*x.shape, 3, 3))
    sinc, cosc = (tuple(t[..., None, None] for t in _series(squared, off)) for off in (1, 2))
    return add(pair(torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)),
               add(mul(sinc, skew), mul(cosc, matmul(skew, skew))))


@torch.no_grad()
def landmark_pair(sparse, identity, expression, rotations, translation):
    """Full sparse GNM basis, pose correctives, FK, skinning, barycentrics."""
    head = sparse.head
    identity = identity.expand(*expression.shape[:-1], head.identity_dim)
    vertices = add(add(pair(sparse.template), linear(pair(identity), sparse.identity_basis)),
                   linear(pair(expression), sparse.expression_basis))
    joints = add(pair(head.template_joint_positions), linear(pair(identity), head.joint_identity_basis))
    matrices = rotation_pair(rotations)
    if sparse.pose_regressor is not None:
        features = reshape(sub(matrices, pair(torch.eye(3, dtype=identity.dtype, device=identity.device))),
                           *rotations.shape[:-2], head.num_joints * 9)
        vertices = add(vertices, reshape(linear(features, sparse.pose_regressor), *vertices[0].shape))
    world_r, world_t = [], []
    for j in range(head.num_joints):
        r = at(matrices, (..., j, slice(None), slice(None)))
        joint = at(joints, (..., j, slice(None)))
        if j == 0:
            t = add(joint, pair(translation))
        else:
            parent = head.joint_parent_indices[j]
            delta = sub(joint, at(joints, (..., parent, slice(None))))
            t = add(world_t[parent], matvec(world_r[parent], delta))
            r = matmul(world_r[parent], r)
        world_r.append(r)
        world_t.append(t)
    # Native LBS, evaluated in an algebraically equivalent centered form.
    posed = []
    for j, (r, t) in enumerate(zip(world_r, world_t, strict=True)):
        joint = at(joints, (..., j, None, slice(None)))
        local = sub(vertices, joint)
        moved = add(matvec(tuple(x.unsqueeze(-3) for x in r), local), tuple(x.unsqueeze(-2) for x in t))
        posed.append(mul(moved, pair(sparse.skinning_weights[j, :, None])))
    posed = sum_pair(stack(posed, -3), -3)
    return sum_pair(mul(at(posed, (..., sparse.triangles, slice(None))), pair(sparse.weights[..., None])), -2)


@torch.no_grad()
def projected_pair(landmarks, rotation, translation, focal, principal):
    """Absolute projected pixel high/low pair, retaining compensated 3D lows."""
    axes = mul(landmarks, pair(landmarks[0].new_tensor((1, -1, -1))))
    rotated = matvec(tuple(x.unsqueeze(-3) for x in rotation_pair(rotation)), axes)
    camera = add(rotated, pair(translation.unsqueeze(-2)))
    z = at(camera, (..., slice(2, 3)))
    if not bool((z[0] > 0).all() & (translation[..., 2] > 0).all() & (focal > 0).all()):
        raise ValueError("Invalid compensated perspective depth or focal length")
    pixels = add(div(mul(at(camera, (..., slice(0, 2))), pair(focal.unsqueeze(-2))), z),
                 pair(principal.unsqueeze(-2)))
    if not bool(torch.isfinite(pixels[0]).all() & torch.isfinite(pixels[1]).all()):
        raise ValueError("Nonfinite compensated projection")
    return pixels
