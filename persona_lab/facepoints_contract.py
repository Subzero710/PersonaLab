"""Numerical and anatomical contract for the pinned Facepoints 1.0.0 point set."""

# Six-decimal triangle weights differ from unity by at most 1e-6 in the
# audited release. The tolerance includes floating-point roundoff, not rescaling.
FACEPOINTS_WEIGHT_ATOL = 1.1e-6
FACEPOINTS_POINTS_SHA256 = "65523bc63b497ab25f8d29655719176473608b717ac6c118cecad7c19729815c"
FACEPOINTS_GROUPS = {"lmk": "skin", "skin": "skin", "lid": "skin",
                     "teeth_u": "upper_teeth_and_gums", "teeth_l": "lower_teeth_and_gums",
                     "iris": "eye_interiors"}
FACEPOINTS_CLASS_COUNTS = {"lmk": 416, "skin": 150, "lid": 36, "teeth_u": 7, "teeth_l": 9, "iris": 2}
FACEPOINTS_CLASS_ORDER = tuple(name for name, count in FACEPOINTS_CLASS_COUNTS.items() for _ in range(count))
FACEPOINTS_LENGTHS = {"lmk": 3, "skin": 3, "lid": 3, "teeth_u": 1, "teeth_l": 1, "iris": 145}


def location_key(vertices, weights):
    """Exact nonzero coefficients, independent of vertex ordering."""
    return tuple(sorted((int(i), float(w)) for i, w in zip(vertices, weights, strict=True) if w > 0))
