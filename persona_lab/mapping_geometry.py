"""Deterministic geometry primitives for cross-template mapping proposals."""
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay


def register_template(source, target, queries):
    """Interpolate anchors in 3D. Return positions, extrapolation and LOO error in m."""
    source, target, queries = map(lambda x: np.asarray(x, dtype=np.float64), (source, target, queries))
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 5:
        raise ValueError("Registration requires paired 3D anchors")
    if queries.ndim != 2 or queries.shape[1] != 3 or not all(np.isfinite(x).all() for x in (source, target, queries)):
        raise ValueError("Registration requires finite 3D coordinates")
    center = source.mean(0)
    scale = np.linalg.norm(source - center, axis=1).max()
    if scale <= 0 or np.linalg.matrix_rank(source - center) != 3 or len(np.unique(source, axis=0)) != len(source):
        raise ValueError("Registration anchors must be unique and span 3D")
    x, q = (source - center) / scale, (queries - center) / scale
    def fit(a, b):
        return RBFInterpolator(a, b, kernel="thin_plate_spline", degree=1, smoothing=0)
    warped = fit(x, target)(q)
    errors = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        errors.append(np.linalg.norm(fit(x[keep], target[keep])(x[i:i+1])[0] - target[i]))
    if not np.isfinite(warped).all() or not np.isfinite(errors).all():
        raise ValueError("Nonfinite registration")
    return warped, Delaunay(x).find_simplex(q) < 0, np.asarray(errors)


def closest_surface(point, triangles):
    """Exact Euclidean closest point on a nondegenerate triangle set, with weights.

    Returns the triangle row within the supplied set, barycentric weights,
    projected location and distance. Ties preserve input row order.
    """
    p = np.asarray(point, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.float64)
    if p.shape != (3,) or t.ndim != 3 or t.shape[1:] != (3, 3) or not len(t):
        raise ValueError("Expected a point and nonempty 3D triangle set")
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("Nonfinite triangle projection input")
    a, b, c = t.transpose(1, 0, 2)
    u, v, w = b - a, c - a, p - a
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)
    uu, uv, vv = dot(u, u), dot(u, v), dot(v, v)
    det = uu * vv - uv * uv
    good = det > np.finfo(np.float64).eps * uu * vv
    if not good.any():
        raise ValueError("No nondegenerate native triangles in candidate region")
    s = np.divide(vv * dot(w, u) - uv * dot(w, v), det, out=np.zeros_like(det), where=good)
    r = np.divide(uu * dot(w, v) - uv * dot(w, u), det, out=np.zeros_like(det), where=good)
    bary = np.stack((1 - s - r, s, r), axis=1)
    inside = good & (bary >= 0).all(1)
    best = np.where(inside, np.sum((np.einsum("ni,nij->nj", bary, t) - p)**2, axis=1), np.inf)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = t[:, j] - t[:, i]
        length = dot(edge, edge)
        alpha = np.clip(np.divide(dot(p - t[:, i], edge), length, out=np.zeros_like(length), where=length > 0), 0, 1)
        distance = np.sum((t[:, i] + alpha[:, None] * edge - p)**2, axis=1)
        better = good & (distance < best)
        best[better] = distance[better]
        bary[better] = 0
        bary[better, i] = 1 - alpha[better]
        bary[better, j] = alpha[better]
    index = int(best.argmin())
    weights = bary[index]
    projected = weights @ t[index]
    return index, weights, projected, float(np.linalg.norm(projected - p))


def mesh_edges(faces):
    """Unique unoriented edges in deterministic order."""
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu' or (faces < 0).any():
        raise ValueError('Expected nonnegative integer triangle indices')
    return np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                             faces[:, [2, 0]])), axis=1), axis=0)


def curve_guides(canonical, source_slots, native_knots):
    """Ordered native polyline guide at canonical arc-length fractions.

    This is not a measured anatomical rim or an exact surface geodesic.
    """
    source = np.asarray(canonical)[list(source_slots)]
    target = np.asarray(native_knots, dtype=np.float64)
    if len(source) < 2 or len(target) < 2 or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError('Curve requires at least two source and target points')
    def fractions(points):
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if not np.isfinite(steps).all() or (steps <= 0).any():
            raise ValueError('Curve knots must be finite and distinct along the curve')
        arc = np.r_[0., np.cumsum(steps)]
        return arc / arc[-1]
    x, y = fractions(source), fractions(target)
    return np.stack([np.interp(x, y, target[:, j]) for j in range(3)], axis=1)


def refine_surface_graph(canonical, warped, initial, edges, candidates, fixed, vertices, triangles,
                         *, sweeps=8, graph_weight=1., canonical_fraction=.5):
    """Coordinate descent on a data + relative edge-vector energy.

    Each free update is the exact closest point in its explicit triangle set.
    Unsupported slots are not variables. Fixed points remain fixed. No slots
    are removed as a consequence of the energy or its diagnostics.
    """
    canonical, warped, initial, vertices = [np.asarray(a, dtype=np.float64) for a in
                                            (canonical, warped, initial, vertices)]
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if type(sweeps) is not int or sweeps < 1 or not np.isfinite(graph_weight) or graph_weight <= 0 or not 0 <= canonical_fraction <= 1:
        raise ValueError('Invalid graph refinement settings')
    if canonical.shape != warped.shape or initial.shape != warped.shape or canonical.ndim != 2 or canonical.shape[1] != 3:
        raise ValueError('Graph geometry must have equal N x 3 shapes')
    active = np.isfinite(initial).all(1)
    if not np.isfinite(canonical).all() or not np.isfinite(warped).all() or not np.isfinite(vertices).all():
        raise ValueError('Nonfinite graph reference')
    if set(candidates) | set(fixed) != set(np.flatnonzero(active)) or set(candidates) & set(fixed):
        raise ValueError('Every active point must be fixed or have an explicit candidate set')
    edges = edges[active[edges].all(1)]
    if not len(edges):
        raise ValueError('No active canonical edges')
    i, j = edges.T
    raw = canonical[i] - canonical[j]
    length = np.linalg.norm(raw, axis=1)
    if (length <= 0).any():
        raise ValueError('Degenerate canonical edge')
    # Estimate a single positive size ratio from fixed anchors, not from the
    # potentially collapsed nearest-surface result. The two meshes share axes.
    anchors = np.array(sorted(fixed), dtype=int)
    if len(anchors) < 2:
        raise ValueError('Graph refinement requires fixed anatomical anchor pairs')
    src = canonical[anchors] - canonical[anchors].mean(0)
    dst = initial[anchors] - initial[anchors].mean(0)
    scale = np.linalg.norm(dst) / np.linalg.norm(src)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid fixed-anchor scale')
    desired = (1 - canonical_fraction) * (warped[i] - warped[j]) + canonical_fraction * scale * raw
    edge_weight = graph_weight / (scale * length)**2
    neighbors = [[] for _ in canonical]
    for a, b, vector, weight in zip(i, j, desired, edge_weight, strict=True):
        neighbors[a].append((b, vector, weight))
        neighbors[b].append((a, -vector, weight))
    # A point with no mapped neighbor still has a defined data term.
    data_weight = np.full(len(canonical), 1 / (scale * np.median(length))**2)
    for k in np.flatnonzero(active):
        if neighbors[k]:
            data_weight[k] = np.mean([n[2] for n in neighbors[k]]) / graph_weight
    p = initial.copy()
    def objective():
        residual = p[active] - warped[active]
        data = float(np.sum(data_weight[active] * np.sum(residual**2, axis=1)))
        graph = float(np.sum(edge_weight * np.sum((p[i] - p[j] - desired)**2, axis=1)))
        return {'data': data, 'edges': graph, 'total': data + graph}
    history = [objective()]
    locations = {}
    for sweep in range(sweeps):
        order = sorted(candidates, reverse=bool(sweep % 2))
        for k in order:
            allowed = np.asarray(candidates[k], dtype=np.int64)
            if allowed.ndim != 1 or not len(allowed):
                raise ValueError(f'Empty native candidate set for slot {k}')
            rhs = data_weight[k] * warped[k]
            total = data_weight[k]
            for other, vector, weight in neighbors[k]:
                rhs += weight * (p[other] + vector)
                total += weight
            local, weights, xyz, _ = closest_surface(rhs / total, vertices[triangles[allowed]])
            p[k] = xyz
            locations[k] = (int(allowed[local]), weights)
        value = objective()
        if not np.isfinite(value['total']) or value['total'] > history[-1]['total'] + 1e-9 * max(1., history[-1]['total']):
            raise ValueError('Surface graph objective increased; refusing to publish proposals')
        history.append(value)
    return p, locations, {'fixed_anchor_scale': float(scale), 'history': history,
                          'sweeps': sweeps, 'graph_weight': graph_weight,
                          'canonical_fraction': canonical_fraction}


def distortion_report(reference, mapped, faces):
    """All supported edges/faces; ratios and normal disagreements, no pruning."""
    reference, mapped = map(lambda x: np.asarray(x, dtype=np.float64), (reference, mapped))
    active = np.isfinite(mapped).all(1) & np.isfinite(reference).all(1)
    faces = np.asarray(faces, dtype=np.int64)
    edges = mesh_edges(faces)
    edges = edges[active[edges].all(1)]
    a = np.linalg.norm(reference[edges[:, 1]] - reference[edges[:, 0]], axis=1)
    b = np.linalg.norm(mapped[edges[:, 1]] - mapped[edges[:, 0]], axis=1)
    edge_rows = [{'slots': e.tolist(), 'reference_m': float(x), 'mapped_m': float(y),
                  'ratio': float(y/x) if x > 0 else None} for e, x, y in zip(edges, a, b, strict=True)]
    faces = faces[active[faces].all(1)]
    def normal(v):
        t = v[faces]
        return np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
    n, m = normal(reference), normal(mapped)
    na, ma = np.linalg.norm(n, axis=1), np.linalg.norm(m, axis=1)
    face_rows = []
    for f, u, v, x, y in zip(faces, n, m, na, ma, strict=True):
        cosine = float(np.clip(np.dot(u, v)/(x*y), -1, 1)) if x > 0 and y > 0 else None
        face_rows.append({'slots': f.tolist(), 'area_ratio': float(y/x) if x > 0 else None,
                          'normal_cosine': cosine, 'degenerate_mapped': bool(y <= 1e-15)})
    ratios = np.array([r['ratio'] for r in edge_rows if r['ratio'] is not None])
    return {'edge_count': len(edges), 'face_count': len(faces),
            'edge_ratio_quantiles': np.quantile(ratios, [0,.01,.05,.5,.95,.99,1]).tolist() if len(ratios) else [],
            'edges_below_half': int((ratios < .5).sum()), 'edges_below_tenth': int((ratios < .1).sum()),
            'edges_above_twice': int((ratios > 2).sum()),
            'opposite_normals': sum(r['normal_cosine'] is not None and r['normal_cosine'] < 0 for r in face_rows),
            'degenerate_faces': sum(r['degenerate_mapped'] for r in face_rows),
            'edges': edge_rows, 'faces': face_rows}


def polyline_distances(points, knots):
    """Euclidean distance to a finite ordered polyline, with nondegenerate segments."""
    points, knots = np.asarray(points, dtype=np.float64), np.asarray(knots, dtype=np.float64)
    if points.ndim != 2 or knots.ndim != 2 or points.shape[1:] != (3,) or knots.shape[1:] != (3,) or len(knots) < 2:
        raise ValueError('Expected points and ordered 3D polyline knots')
    edge = np.diff(knots, axis=0)
    length2 = np.sum(edge**2, axis=1)
    if not np.isfinite(points).all() or not np.isfinite(knots).all() or (length2 <= 0).any():
        raise ValueError('Invalid polyline geometry')
    rel = points[:, None, :] - knots[None, :-1, :]
    t = np.clip(np.einsum('nsi,si->ns', rel, edge) / length2, 0, 1)
    return np.sqrt(np.min(np.sum((rel - t[..., None] * edge)**2, axis=2), axis=1))
