import numpy as np
import pytest
from persona_lab.facepoints_inference import validate_predictions
from persona_lab.facepoints_audit import native_rows


def prediction():
    return {"points": np.ones((620, 2)), "std": np.ones(620),
            "300w": np.ones((68, 2)), "wflw": np.ones((98, 2))}


def test_pixels_and_uncertainty():
    p = prediction()
    p["points"][0] = [512, 0]
    p["std"][0] = 23.5
    result = validate_predictions(p, 512, 512)
    assert result["validity"].sum() == 619
    assert result["std"][0] == 23.5
    assert result["points"][0, 0] == 512


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_std(bad):
    p = prediction()
    p["std"][0] = bad
    with pytest.raises(ValueError):
        validate_predictions(p, 512, 512)


def test_exact_triangle_reorder_and_reject():
    rows = [{"index": 0, "class": "skin", "region": "cheek", "vertices": [2, 0, 1], "weights": [.25,.5,.25]}]
    t = np.array([[0, 1, 2]])
    g = {"skin": np.ones(4)}
    result, = native_rows(rows, t, g, 4)
    assert not result["issues"]
    assert result["location"]["weights"] == [.25,.5,.25]
    assert result["location"]["vertex_indices"] == [2,0,1]
    rows[0]["vertices"] = [3,0,1]
    assert native_rows(rows, t, g, 4)[0]["issues"]
    rows[0]["vertices"] = [2,0,1]
    rows[0]["weights"] = [.25,.5,.249999]
    assert not native_rows(rows, t, g, 4)[0]["issues"]
    rows[0]["weights"] = [.25,.5,.249998]
    assert native_rows(rows, t, g, 4)[0]["issues"]


def test_anatomy_mismatch_is_reported():
    row = {"index": 0, "class": "teeth_u", "region": "teeth", "vertices": [0], "weights": [1.0]}
    result, = native_rows([row], np.array([[0,1,2]]), {"skin": np.ones(3)}, 3)
    assert result["issues"] == ["Expected native group upper_teeth_and_gums absent"]


def test_vertex_and_iris_and_explicit_alias():
    t = np.array([[0, 1, 2]])
    g = {name: np.ones(146) for name in ("skin", "upper_teeth_and_gums", "eye_interiors", "left")}
    rows = [dict(index=0, **{"class":"skin"}, region="cheek", vertices=[0,1,2], weights=[.2,.3,.499999]),
            dict(index=1, **{"class":"lmk"}, region="", vertices=[2,0,1], weights=[.499999,.2,.3]),
            dict(index=2, **{"class":"teeth_u"}, region="upper_edge", vertices=[4], weights=[1.]),
            dict(index=3, **{"class":"iris"}, region="L", vertices=list(range(145)), weights=[.006896552]*145)]
    result = native_rows(rows, t, g, 146)
    assert all(not r["issues"] for r in result)
    assert result[1]["duplicate_of"] == "fp:0"
    assert [r["location"]["weights"] for r in result] == [r["weights"] for r in rows]
    rows[-1]["vertices"][-1] = 145 + 1
    assert native_rows(rows, t, g, 146)[-1]["issues"]


def test_iris_side_and_vertex_weight():
    rows = [dict(index=0, **{"class":"iris"}, region="R", vertices=list(range(145)), weights=[.006896552]*145),
            dict(index=1, **{"class":"teeth_l"}, region="lower_edge", vertices=[0], weights=[.999999])]
    g = {name: np.ones(145) for name in ("eye_interiors", "left", "lower_teeth_and_gums")}
    result = native_rows(rows, np.empty((0,3), dtype=int), g, 145)
    assert all(r["issues"] for r in result)
