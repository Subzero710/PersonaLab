"""Pinned MediaPipe Face Landmarker measurements using the CPU IMAGE API."""

from importlib.metadata import PackageNotFoundError, version
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .coordinates import ImageSize
from .dense_observations import ModelAsset, ObservationPoint, ObservationSchema


PROVIDER = "mediapipe-face-landmarker"
PROVIDER_VERSION = "0.10.32"
ASSET = ModelAsset(
    "float16/1",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
)
# Ordered by published MediaPipe connections and iris refinement. Keep the
# remaining slots in order without assigning finer anatomy from proximity.
_FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)
_FEATURES = (
    ("facial_oval", "unspecified", _FACE_OVAL),
    ("outer_lip", "unspecified", (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                                  185, 40, 39, 37, 0, 267, 269, 270, 409)),
    ("inner_lip", "unspecified", (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                                  191, 80, 81, 82, 13, 312, 311, 310, 415)),
    ("eyelid_corner", "left", (263, 362)),
    ("eyelid_corner", "right", (33, 133)),
    ("upper_eyelid", "left", (466, 388, 387, 386, 385, 384, 398)),
    ("upper_eyelid", "right", (246, 161, 160, 159, 158, 157, 173)),
    ("lower_eyelid", "left", (249, 390, 373, 374, 380, 381, 382)),
    ("lower_eyelid", "right", (7, 163, 144, 145, 153, 154, 155)),
    ("eyebrow", "left", (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)),
    ("eyebrow", "right", (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)),
    ("iris_boundary", "left", (474, 475, 476, 477)),
    ("iris_boundary", "right", (469, 470, 471, 472)),
    ("iris_center", "left", (473,)),
    ("iris_center", "right", (468,)),
)
_LABELS = {index: (region, side) for region, side, indices in _FEATURES for index in indices}
_POINTS = tuple(ObservationPoint(f"mp:{i}", *_LABELS.get(i, ("face_surface_unclassified", "unspecified")))
                for i in range(478))
SCHEMAS = {
    "mediapipe-face-478": ObservationSchema("mediapipe-face-478", 1, "source_pixels", _POINTS),
    "mediapipe-face-478-dfl": ObservationSchema("mediapipe-face-478-dfl", 1, "aligned_pixels", _POINTS),
}
PREPROCESSING = {
    "decoded_raster": "Pillow RGB; stored orientation; no EXIF transform",
    "image_mode": "IMAGE",
    "application_transform": "none",
    "graph_transform": "face detection ROI, rotated crop and 256x256 face mesh tensor; graph projects to input raster",
    "output_transform": "u = x * input width, v = y * input height",
    "depth": "discarded; task Z is relative, not metric",
}
_REGION_COLORS = {
    "facial_oval": (255, 202, 88), "outer_lip": (255, 99, 121), "inner_lip": (247, 132, 235),
    "eyelid_corner": (130, 246, 203), "upper_eyelid": (87, 218, 214),
    "lower_eyelid": (98, 180, 251), "eyebrow": (187, 229, 140),
    "iris_boundary": (205, 171, 255), "iris_center": (255, 255, 255),
    "face_surface_unclassified": (177, 199, 224),
}


def as_pixels(landmarks, size: ImageSize):
    """Retain finite coordinates and explicitly record absent/out-of-raster slots."""
    if len(landmarks) != len(_POINTS):
        raise ValueError(f"MediaPipe returned {len(landmarks)} slots; expected {len(_POINTS)}")
    coordinates = np.array([(point.x, point.y) for point in landmarks], dtype=np.float64)
    pixels = coordinates * (size.width, size.height)
    finite = np.isfinite(pixels).all(axis=1)
    valid = finite & ((pixels >= 0) & (pixels < (size.width, size.height))).all(axis=1)
    pixels[~finite] = 0
    if not valid.any():
        raise ValueError("MediaPipe produced no valid in-raster point")

    def metadata(name):
        values = [getattr(point, name) for point in landmarks]
        if all(value is None for value in values):
            return None
        return torch.tensor([-1 if value is None else value for value in values], dtype=torch.float64)

    return {
        "landmarks": torch.from_numpy(pixels),
        "validity": torch.from_numpy(valid),
        "visibility": metadata("visibility"),
        "presence": metadata("presence"),
    }


class MediaPipeFaceLandmarker:
    """Open the exact declared task asset, with explicit CPU delegation."""

    def __init__(self, asset_path, *, asset_version, device, score_threshold):
        if device != "cpu":
            raise ValueError("The MediaPipe Python IMAGE provider supports explicit CPU execution only")
        if asset_version != ASSET.version:
            raise ValueError(f"Unknown MediaPipe model version {asset_version!r}; expected {ASSET.version}")
        if type(score_threshold) not in (float, int) or not 0 < score_threshold <= 1:
            raise ValueError("Face detection thresholds must be in (0, 1]")
        path = Path(asset_path)
        try:
            with path.open("rb") as file:
                model_bytes = file.read(3_758_597)
        except OSError as error:
            raise ValueError(f"Cannot open the pinned MediaPipe asset at {path}") from error
        if sha256(model_bytes).hexdigest() != ASSET.sha256:
            raise ValueError(f"MediaPipe model SHA-256 mismatch at {path}")
        try:
            installed = version("mediapipe")
        except PackageNotFoundError as error:
            raise ValueError(f"Install the pinned mediapipe=={PROVIDER_VERSION} package") from error
        if installed != PROVIDER_VERSION:
            raise ValueError(f"Expected mediapipe=={PROVIDER_VERSION}, found {installed}")
        try:
            import mediapipe as mp
        except ImportError as error:
            raise ValueError("Cannot import the pinned MediaPipe runtime and its dependencies") from error
        self._mp = mp
        tasks = mp.tasks
        # A second detected face makes the image ambiguous. Never select one
        # implicitly from an image containing multiple detected faces.
        options = tasks.vision.FaceLandmarkerOptions(
            base_options=tasks.BaseOptions(model_asset_buffer=model_bytes, delegate=tasks.BaseOptions.Delegate.CPU),
            running_mode=tasks.vision.RunningMode.IMAGE, num_faces=2,
            min_face_detection_confidence=score_threshold, min_face_presence_confidence=score_threshold,
            output_face_blendshapes=False, output_facial_transformation_matrixes=False,
        )
        self._task = tasks.vision.FaceLandmarker.create_from_options(options)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._task.close()

    def extract(self, rgb, size):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        faces = self._task.detect(image).face_landmarks
        if len(faces) != 1:
            raise ValueError(f"Require exactly one detected face; MediaPipe returned {len(faces)}")
        return as_pixels(faces[0], size)
