import numpy as np
import pytest

from persona_lab.mapping_geometry import (closest_surface, register_template, mesh_edges, curve_guides,
                                          refine_surface_graph, distortion_report, polyline_distances)


def test_registration_affine_and_extrapolation():
    rng = np.random.default_rng(42)
    anchors = rng.normal(size=(18, 3))
    transform = np.array([[2., 0, 1], [0, .5, .2], [.3, 0, 1.]])
    target = anchors @ transform + np.array([2., 1., -3.])
    queries = np.stack((anchors.mean(0), np.full(3, 100.)))
    result, outside, loo = register_template(anchors, target, queries)
    np.testing.assert_allclose(result, queries @ transform + [2., 1., -3.], atol=1e-10)
    assert outside.tolist() == [False, True]
    assert loo.max() < 1e-10


@pytest.mark.parametrize("query, expected", [([.2,.3,1], [.2,.3,0]),
    ([.8,.8,1], [.5,.5,0]), ([-1,-1,1], [0,0,0]), ([.5,-1,0], [.5,0,0])])
def test_triangle_projection(query, expected):
    triangles = np.array([[[0.,0,0], [1,0,0], [0,1,0]], [[8,0,0], [9,0,0], [8,1,0]]])
    index, weights, projected, distance = closest_surface(query, triangles)
    assert index == 0
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.)
    np.testing.assert_allclose(projected, expected, atol=1e-14)
    np.testing.assert_allclose(weights @ triangles[index], expected, atol=1e-14)
    assert distance == pytest.approx(np.linalg.norm(np.array(query)-expected))


def test_triangle_degenerate_rejection_and_registration_rejection():
    with pytest.raises(ValueError, match="nondegenerate"):
        closest_surface([0,0,0], np.zeros((2,3,3)))
    with pytest.raises(ValueError, match="span 3D"):
        register_template(np.zeros((5,3)), np.zeros((5,3)), np.zeros((1,3)))



def test_curve_order_and_distances():
    source = np.array([[0.,0,0], [1,0,0], [3,0,0]])
    target = np.array([[0.,0,0], [0,1,0], [0,2,0], [1,2,0]])
    np.testing.assert_allclose(curve_guides(source, (0,1,2), target),
                               [[0,0,0], [0,1,0], [1,2,0]])
    np.testing.assert_allclose(polyline_distances([[0,1,0], [2,1,0], [0,1,2]], target),
                               [0, np.sqrt(2), 2])
    with pytest.raises(ValueError, match='distinct'):
        curve_guides(source, (0,0,2), target)


def test_graph_refinement_recovers_spacing_without_moving_anchors():
    canonical = np.array([[0.,0,0], [2,0,0], [1,1,0], [1,2,0]])
    faces = np.array([[0,1,2], [0,2,3], [1,3,2]])
    # Artificial collapse of two adjacent samples in the initial projection.
    warped = canonical.copy()
    warped[2:] = [[1,.1,0], [1,.1,0]]
    vertices = np.array([[-5.,-5,0], [10,-5,0], [-5,10,0], [10,10,0]])
    triangles = np.array([[0,1,2], [1,3,2]])
    candidates = {2: np.array([0,1]), 3: np.array([0,1])}
    result, locations, info = refine_surface_graph(canonical, warped, warped, mesh_edges(faces),
                                                   candidates, {0,1}, vertices, triangles)
    np.testing.assert_array_equal(result[:2], canonical[:2])
    assert np.linalg.norm(result[3]-result[2]) > 0.1
    assert info['history'][-1]['total'] < info['history'][0]['total']
    assert all(b['total'] <= a['total'] + 1e-10 for a,b in zip(info['history'], info['history'][1:]))
    for k,(row,w) in locations.items():
        assert row in candidates[k]
        np.testing.assert_allclose(w @ vertices[triangles[row]], result[k], atol=1e-13)
    again, _, _ = refine_surface_graph(canonical, warped, warped, mesh_edges(faces),
                                       candidates, {0,1}, vertices, triangles)
    np.testing.assert_array_equal(again, result)


def test_distortion_reports_all_supported_edges_and_inverted_normal():
    ref = np.array([[0.,0,0], [1,0,0], [0,1,0], [1,1,0]])
    faces = np.array([[0,1,2], [1,3,2]])
    mapped = ref.copy()
    mapped[1] = [-.05,0,0]
    mapped[3] = np.nan
    report = distortion_report(ref, mapped, faces)
    assert report['edge_count'] == 3 and report['face_count'] == 1
    assert report['edges_below_tenth'] == 1
    assert report['opposite_normals'] == 1
    assert len(report['edges']) == 3


def test_graph_rejects_missing_candidate_without_substitution():
    canonical = np.array([[0.,0,0], [1,0,0], [0,1,0]])
    faces = np.array([[0,1,2]])
    with pytest.raises(ValueError, match='Every active point'):
        refine_surface_graph(canonical, canonical, canonical, mesh_edges(faces), {}, {0,1}, canonical, faces)
