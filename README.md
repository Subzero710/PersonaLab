# GNM: Generative aNthropometric Model and Ecosystem

![GNM Teaser Image](assets/readme/gnm_logo.png)

Welcome to the **GNM Ecosystem** repository. GNM - pronounced as genome
(/ˈdʒiː.noʊm/) in reference to the human genome - strives to be the most
accurate and complete 3D parametric human model.

3D Morphable Models (3DMMs) are widely used across computer vision, computer
graphics, and generative AI for representing human geometry and appearance. GNM
introduces a state-of-the-art family of parametric statistical human models and
its associated perception stack.

Our roadmap includes releasing a comprehensive suite of statistical models
complemented by perception and analysis technology. To facilitate early
community research and open development, we are beginning our open-source
release with **GNM Head**, our high-fidelity statistical 3D model of the human
head.

The ecosystem is released under the Apache 2.0 permissive license suitable for
both non-commercial and commercial applications.

## News

  - [2026-07-28]: Official [technical report](https://arxiv.org/abs/2607.23687) for the GNM Head released! 🚀🚀🚀

## PersonaLab workflow

Docker Engine and Docker Compose v2 are the PersonaLab execution environment.
No host Python installation or virtual environment is required.
Run these commands from the repository root on Linux (x86-64):

```sh
mkdir -p datasets models outputs .gnm-cache
export PERSONALAB_UID=$(id -u) PERSONALAB_GID=$(id -g)
docker compose build personalab
docker compose run --rm personalab python -m persona_lab.gnm_mesh --output outputs/neutral.obj
docker compose run --rm personalab python -m pytest -v tests
```

The official GNM model is downloaded by GNM on first use and cached in
`.gnm-cache/`. To use a specific local model, place its NPZ in `models/` and
pass `--model-file models/gnm_head.npz`. For the tests, pass
`-e PERSONALAB_GNM_MODEL_FILE=/app/models/gnm_head.npz` to `docker compose run`.
The datasets and local models are mounted read-only; cache and outputs are
persistent and writable. Pytest stores its cache in `.gnm-cache/.pytest_cache/`
using the same unprivileged container UID/GID. Rebuild the image after source
or dependency changes.

The Issue #1 validation used GNM v3.0 Head from Hugging Face repository
`google/gnm-v3`, revision `c01e90d298d82301f9fd18f54806be751775cb7c`, file
`v3_0/gnm_head.npz`, with SHA-256
`61d78bbfb4ad8e0b38495804a4caef3214d3df00f8c3f68761e63b41ce3747eb`.
Use this exact asset for comparisons with that validation.

For CUDA workloads, Docker Compose 2.30+ and a working NVIDIA Container
Toolkit installation are required. Build and check the explicitly selected
CUDA 12.8 PyTorch environment:

```sh
docker compose --profile cuda build personalab-cuda
docker compose --profile cuda run --rm personalab-cuda python -c 'import torch; print(torch.cuda.get_device_name(0))'
```

The CUDA service refuses to start when PyTorch cannot access CUDA. The existing
Issue #1 mesh CLI intentionally executes on CPU; GPU fitting is not implemented
by this container configuration. The image pins PyTorch and the direct
numerical/test dependencies reported in Issue #1, but does not lock every
transitive dependency or the base image digest.

## Import DeepFaceLab aligned JPEGs

PersonaLab accepts one `.jpg` or `.jpeg`, a directory of JPEGs, or a `.zip`
archive containing JPEGs. Extensions are case-insensitive, and directory and
ZIP scans include nested files. Use the CPU service from the repository root
after building the image.

```sh
docker compose run --rm personalab python -m persona_lab.dfl_import \
  --input datasets/studio_light.zip --output outputs/dfl-validation.json
docker compose run --rm personalab python -m persona_lab.dfl_import \
  --input datasets/aligned --output outputs/dfl-directory-validation.json
```

Omit `--output` to write JSON to stdout. A report path must be new and outside
a directory input. Exit status 0 indicates a successful scan with zero
rejections, 1 indicates one or more sample rejections, and 2 indicates a
dataset or CLI error. Dataset failures produce a `dataset_error` object with
a stable `code` and a `message`. Command and output I/O errors go to stderr.

```python
from persona_lab.dfl import DFLFaceSample, iter_dataset, read_jpeg, validate_dataset

sample = read_jpeg("datasets/aligned/0001.jpg")
original_jpeg = sample.read_image_bytes()

for result in iter_dataset("datasets/studio_light.zip"):
    if isinstance(result, DFLFaceSample):
        print(result.sample_id, result.landmarks.shape)
    else:
        print(result.sample_id, [issue.code for issue in result.issues])

report = validate_dataset("datasets/aligned").to_dict()
```

`read_jpeg` raises `SampleError` with a `rejection` for an invalid JPEG.
`iter_dataset` yields a `DFLFaceSample` or `SampleRejection` for each
discovered JPEG. Both API entry points raise `DatasetError` for input-level
failures. Exhaust the iterator or call `close()` if you stop early.
The iterator releases its files and archive on completion. Samples retain
metadata and an image reference. `read_image_bytes()` reopens that reference,
verifies a SHA-256 digest, and returns caller-owned immutable bytes suitable
for a pixel decoder. It raises `DFLError` with code `dataset_changed` if the
original bytes are no longer available. Scan-time pixel buffers are released
after each image. Keep only the samples needed by the consumer.

| Sample field | Python type and convention |
| --- | --- |
| `sample_id`, `image_ref` | `str`. The reference is a dataset-relative POSIX path, or the JPEG basename for a single input. The ID uses the reference with a `zip:` prefix for accepted archive members. |
| `image_size` | `tuple[int, int]` in `(width, height)` order from the JPEG frame header. |
| `landmarks` | Required `np.ndarray` of shape `(68, 2)`. Original ordered `(x, y)` aligned JPEG pixel coordinates, with origin at the upper left. Finite coordinates outside the frame are accepted. |
| `face_type`, `source_filename` | `str \| None`. DFL face type and original source filename as metadata. |
| `source_rect` | `np.ndarray \| None`, shape `(4,)`, ordered left, top, right, bottom in source pixels. |
| `source_landmarks` | `np.ndarray \| None`, shape `(68, 2)`, in the original source-image pixel coordinate system. |
| `image_to_face_mat` | `np.ndarray \| None`, shape `(2, 3)`. The stored affine matrix maps source coordinates to the aligned JPEG. |
| `xseg_mask` | `np.ndarray \| None`, `float32` of shape `(mask_height, mask_width, 1)`. Rows and columns index the mask's native grid over the aligned image. Values are decoded grayscale divided by 255, with zero for background and one for fully selected pixels. |
| `additional_metadata` | `dict[str, object]` of remaining safely decoded values, including `eyebrows_expand_mod` and `seg_ie_polys` when present. Polygons preserve their stored representation. |

Coordinate and transform arrays are read-only. Stored NumPy arrays keep their
real numeric dtype and byte order. Python sequences use NumPy dtype inference
with exact numeric values required. Missing or null optional fields become
`None` or are absent from `additional_metadata`. Each invalid required or
available optional field rejects the sample with its own issue. DFL
`mark_only` files contain an unaligned image and are rejected. XSeg accepts
compressed grayscale PNG or JPEG in a one-dimensional `uint8` array or a
two-dimensional single row or column. The mask keeps its resolution even
when it differs from the JPEG.

### Validation report

`discovered_samples` equals `accepted_samples` plus `rejected_samples`.
`resolution_distribution` counts readable frame headers by `WIDTHxHEIGHT`.
`unknown_resolution` accounts for the rest. The `landmarks` and `xseg_masks`
counts group samples by `valid`, `invalid`, `missing`, and `unknown`. Unknown
means inspection stopped before field validation. The report counts missing
optional fields after successful metadata decoding, conflicting APP15
segments or invalid alignment data under `inconsistent_metadata`, and
unsupported representations under `unsupported_metadata`.

Each `rejections` entry contains a sample ID, relative image reference where
safe, dimensions where known, field statuses, missing optional fields, and
one or more `issues` with `code`, `message`, and optional `field`.
`rejection_counts` counts samples per distinct issue code. The stable codes
are defined by `persona_lab.dfl.ErrorCode`. Reports omit source filenames and
pixel, coordinate, and mask values.

### Format and resource limits

The decoder reads DFL's raw binary pickle dictionary in JPEG APP15. A
recognized DFL key identifies a decoded dictionary. Multiple DFL segments
are an error. Supported binary protocols are 2 through 5, including NumPy
1.x and 2.x legacy array state, fixed-width numeric and boolean dtypes,
scalars, and protocol 5 in-band buffers. Legacy NumPy dtype reducers using
integer flags `(0, 1)` and modern boolean flags `(False, True)` are supported. Plain strings, numbers, bytes,
booleans, lists, tuples, string-keyed dictionaries, and nulls are supported.
The decoder resolves known global names to inert symbols and validates each
state before reconstructing data. It rejects arbitrary globals, constructors,
object and structured arrays, persistent and extension references, and
external buffers. Other JPEG frame formats and trailing JPEG data are
rejected. Pixel checks use Pillow, and JPEG corruption that still decodes
successfully can remain undetected.

| Scope | Limits |
| --- | --- |
| JPEG | 32 MiB, up to 8,192 pixels per axis and 16 Mi pixels in total, 8,192 markers, 128 scans, and 16 APP15 segments. Supported frames are 8-bit sequential and progressive. |
| APP15 | 65,533 bytes per payload, 16,384 opcodes and expanded values, 4,096 memo slots, nesting up to 32, and up to four NumPy axes. Array buffers are at most 65,533 bytes each and 262,132 bytes cumulatively. Legacy byte reconstructions have their own 262,132-byte budget. |
| XSeg | Up to 4,096 pixels per axis and 4 Mi pixels in total. PNG support covers streams with only IHDR, IDAT, and IEND chunks. |
| Directory | Up to 20,000 entries, 64 relative path components, and 1,024 bytes per relative path. Entries are visited in depth-first name order. Symlink directories are skipped and JPEG-named symlinks are rejected. |
| ZIP | Classic single-volume ZIP, up to 4 GiB minus one byte, an 8 MiB index, 20,000 entries, and 8 GiB of cumulative JPEG expansion. Members use stored or DEFLATE compression, up to 32 MiB compressed and uncompressed and a 200:1 expansion ratio. |

Directory and archive contents are read-only. ZIP members are inspected in
name order without extracting files. Duplicate names, unsafe paths,
nonregular members, ZIP64 headers, and encrypted JPEG entries are rejected.
Unsafe member paths are omitted and identified by central-directory index.
Limits on the entire dataset or an unreadable directory surface as
dataset-level errors.

The format references are the upstream DeepFaceLab
[JPEG writer](https://github.com/iperov/DeepFaceLab/blob/e4b7543ffa1d73b26fce1e31852727f658ba490c/DFLIMG/DFLJPG.py),
[extractor](https://github.com/iperov/DeepFaceLab/blob/e4b7543ffa1d73b26fce1e31852727f658ba490c/mainscripts/Extractor.py),
and [polygon format](https://github.com/iperov/DeepFaceLab/blob/e4b7543ffa1d73b26fce1e31852727f658ba490c/core/imagelib/SegIEPolys.py).
NumPy's [array reducers](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/src/multiarray/methods.c)
and [dtype state](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/src/multiarray/descriptor.c)
describe the supported binary conventions.

To measure streaming memory on 600 synthetic JPEGs with masks per input
type, run the following test. It measures elapsed time, Python allocations
with `tracemalloc`, and Linux process RSS high-water marks after imports and
after iteration. The process keeps only the current sample.

```sh
docker compose run --rm personalab python -m pytest -s -v tests/test_dfl_dataset.py -k streaming_memory
```

## Project GNM landmarks into an aligned crop

`persona_lab.gnm_geometry.load_sparse_landmarks(model)` loads the installed
GNM `HEAD_SPARSE_68` resource and checks its arrays against the v3.0 Head
topology. Each of the 68 definitions holds three vertex indices and three
finite weights in `[0, 1]`. The absolute sum tolerance is 0.00101 for the
three-decimal file and float32 rounding. The official weights are used as stored.
`extract_landmarks()` calls GNM's differentiable `vertices_and_landmarks()`
and returns vertices `(..., V, 3)` and GNM-ordered points `(..., 68, 3)`.
The model and input tensors must share a device and a float32 or float64 dtype.
Expression `(..., E)`, joint rotations `(..., J, 3)` and root translation
`(..., 3)` share leading dimensions. A single identity `(I,)` expands over
those dimensions with one gradient back to the shared coefficients. A batched
identity `(..., I)` must match the other inputs. GNM supplies its native neutral
values for omitted inputs.

### Anatomical correspondence and fitting masks

`persona_lab.correspondences.DFL_TO_GNM` records the subject-side anatomy,
region, GNM index, default eligibility and any limitation for each DFL index.
The official GNM v3.0 template, its anatomical vertex groups, and its vertex
mirror table match DeepFaceLab's indexed convention across the following regions.
In a frontal image, the subject's right is image-left.

| DeepFaceLab indices | GNM indices | Anatomy | Enabled by default |
| --- | --- | --- | --- |
| 0–16 | 0–16 | Right facial outline through chin to left facial outline | 17 |
| 17–26 | 17–26 | Right and left eyebrows | 10 |
| 27–30 | 27–30 | Nasal bridge and tip | 4 |
| 31–35 | 31–35 | Nasal base and nostril region | 5 |
| 36–47 | 36–47 | Right and left eye corners and eyelids | 12 |
| 48–59 | 48–59 | Outer lips | 12 |
| 60–67 | 60–67 | Inner lips | 8 |

The default map has 68 direct anatomical correspondences and no remapped or
unsupported rows. All 68 are enabled by default, including the 17 jawline
and facial outline landmarks (0–16). DeepFaceLab's jaw landmarks are not
assumed to be geometric silhouette samples. The fixed GNM surface locations
can exhibit viewpoint-dependent correspondence errors, but such errors have
not been established for the project's DeepFaceLab observations. The
`view_dependent` annotation describes this potential limitation and has no
masking effect. No region is automatically excluded based on viewpoint.
Use an explicit validity mask, `pose_validity`, subset or zero weights to
exclude individual unreliable observations in any region. The API applies no
automatic visibility threshold.

A custom `CorrespondenceMap` can remap indices. An unsupported row keeps
its original DFL index and carries `gnm_index=None`, an exclusion reason
and disabled fitting eligibility. `LandmarkObservation.landmark_pairs()`
selects supported, enabled, valid rows with positive weights and preserves
their DFL order. Pass a subset of DFL indices, observation validity, pose
validity, or per-point weights to restrict the result. It returns both index
arrays, pixel pairs and weights for the objective. The diagnostic API can
display excluded points without putting them into the residual.

References are DeepFaceLab's [LandmarksProcessor](https://github.com/iperov/DeepFaceLab/blob/e4b7543ffa1d73b26fce1e31852727f658ba490c/facelib/LandmarksProcessor.py),
the [official landmark resource](gnm/shape/data/landmarks/head_sparse_68.txt),
and the [GNM v3.0 Head asset](https://huggingface.co/google/gnm-v3/tree/main/v3_0).

### Coordinate conventions

| Space | Axes, units and origin |
| --- | --- |
| GNM native template and posed coordinates | Right-handed meters. +X toward the subject's left, +Y toward the top of the head, +Z anterior. Posed coordinates keep the template's scene axes and origin. GNM applies root translation in meters and joint rotations about the bind-pose joints in parent-to-child order. |
| OpenCV camera | Right-handed. +X image-right, +Y image-down, +Z forward from the optical center. 3D positions and perspective translation share the model's units. |
| OpenGL camera | Right-handed. +X image-right, +Y image-up, camera looks along −Z. |
| DFL aligned and original-source pixels | +X right, +Y down, top-left pixel center at `(0, 0)`. Pixel centers lie at integer coordinates. The spaces refer to their respective rasters. Coordinates outside the image remain valid. |
| Normalized aligned raster | +X right, +Y down, dimensionless. Edges span 0 to 1. From pixel centers use `(u + 0.5) / width` and `(v + 0.5) / height`. Invert with `u = x * width - 0.5`, likewise for `v`. |

`gnm_to_frontal_axes()` multiplies GNM points by `C = diag(1, -1, -1)`
and keeps the GNM origin. It expresses the geometry in world axes aligned
with a frontal OpenCV camera. The separate world-to-camera transform is
`P_cam = R @ P_world + t`. With row-stored point tensors this is
`points @ R.T + t`. `camera_position(rotation, translation)` returns the
optical center in world coordinates `-R.T @ t`. GNM joint and camera
rotations both use axis-angle vectors in radians. Each vector represents one
right-handed rotation with no Euler order. Use GNM rotations for articulation
and camera extrinsics for viewpoint. Apply each transform once.

`flip_camera_axes()` switches OpenCV and OpenGL camera point conventions.
`convert_camera_matrix()` changes only the camera side of a homogeneous
4×4 transform. For world-to-camera it left-multiplies by
`diag(1, -1, -1, 1)` as in [GNM's reference](gnm/shape/visualization/camera_conversions.py).
For camera-to-world it right-multiplies by that matrix. Each conversion
is its own inverse.

`source_to_aligned()` applies DeepFaceLab's `image_to_face_mat` in its stored
direction, `pixel_aligned = A @ pixel_source + b`. `aligned_to_source()`
solves that 2D affine equation in reverse. Missing, invalid and singular
matrices raise `ValueError`. The affine describes a raster-to-raster crop.
Camera calibration remains independent of it. Reversing 3D projection
requires additional depth information.

### Cameras and observations

`persona_lab.camera.project(points, camera)` consumes `(..., N, 3)`
and returns `(..., N, 2)` pixel centers in input point order. Inputs and
camera parameters share a CPU or CUDA device and float32 or float64 dtype.
Their leading dimensions broadcast using PyTorch rules. Each camera batch
shares one `ImageSize(width, height)`. Changing the raster dimensions
alone leaves pixel projections unchanged. To resize by a factor of 2 with
the same edge-normalized framing, double the focal lengths or weak scale
and apply `(center + 0.5) * 2 - 0.5` to the principal point or weak translation.

- `PerspectiveCamera` takes axis-angle rotation `(..., 3)`, translation
  `(..., 3)`, focal lengths `(..., 2)` and principal point `(..., 2)`.
  For `(X, Y, Z) = R @ P_world + t`, the result is
  `(fx * X/Z + cx, fy * Y/Z + cy)`. The intrinsics have zero skew and no
  distortion. Nonpositive camera depth, nonfinite coordinates and invalid
  parameters raise `ValueError`.
- `WeakPerspectiveCamera` takes axis-angle rotation `(..., 3)`, image
  translation `(..., 2)` and scalar scale per camera `(...)` in pixels
  per scene unit. The result is `scale * (R @ P_world)[:2] + translation`.
  It has one image scale and one image translation. Perspective intrinsics
  are optional for observations. In future fitting, fix or constrain head
  pose versus camera rotation, scene placement versus camera translation,
  and object size versus image scale or camera distance.

`LandmarkObservation.from_dfl(sample)` works with an accepted `DFLFaceSample`.
It copies aligned pixels, dimensions and optional source/alignment metadata
in the requested dtype and device, without reopening the JPEG. The default
observation dtype is float64. XSeg and additional metadata remain in the
ingested sample. `source_pixels()` inverts the supplied affine. Stored
source landmarks are also accessible through `alignment.source_landmarks`.
Landmarks have shape `(68, 2)` in original DFL order, validity is a boolean
`(68,)` mask, and fitting weights are a nonnegative `(68,)` tensor.
`known_camera` and `camera_initialization` are separate optional fields.
Both remain absent for uncalibrated crops unless supplied explicitly.

```python
from persona_lab.camera import PerspectiveCamera, project
from persona_lab.coordinates import ImageSize, gnm_to_frontal_axes
from persona_lab.gnm_geometry import extract_landmarks
from persona_lab.gnm_mesh import load_head_model
from persona_lab.observations import LandmarkObservation

model = load_head_model()
ref = model.template_vertex_positions
identity = ref.new_zeros(model.identity_dim).requires_grad_()
_, landmarks = extract_landmarks(model, identity=identity)
size = ImageSize(512, 384)
camera = PerspectiveCamera(
    image_size=size,
    rotation=ref.new_zeros(3),
    translation=ref.new_tensor([0, 0.265, 0.65]),
    focal_length=ref.new_tensor([820, 760]),
    principal_point=ref.new_tensor([255.5, 191.5]),
)
pixels = project(gnm_to_frontal_axes(landmarks), camera)
# The verified default correspondences use the same ordering.
observation = LandmarkObservation(
    observation_id="synthetic-0", image_size=size,
    landmarks=pixels.detach() + ref.new_tensor([2, -1]), known_camera=camera,
)
pairs = observation.landmark_pairs(pixels)
loss = ((pairs.projected - pairs.observed).square().sum(-1) * pairs.weights).mean()
loss.backward()
```

The CPU suite includes independent NumPy and SciPy projection references,
geometry and affine round trips, and complete reprojection gradients to GNM
identity, expression, joint pose and camera parameters. The float64 camera
tests use 1e-11 pixel absolute tolerance for batched synthetic points and
1e-10 pixels for neutral GNM landmarks. Pose and expression comparisons allow
7e-4 pixels for the rig geometry. Central finite differences use step 1e-5,
absolute tolerance 2e-5 and relative tolerance 2e-4. Tests exercise shared
identity and native landmark caching. The CUDA checks require a compatible
GPU and the existing CUDA Compose service.

### Headless landmark diagnostics

```sh
docker compose run --rm personalab python -m persona_lab.landmark_diagnostics \
  --output outputs/landmark-diagnostics
```

The command writes frontal, two three-quarter, profile and lower-face
expression PNGs. The two panels show the projected GNM mesh and official
indices and the observed DFL indices, residual lines and exclusion indicators.
The examples use explicitly offset synthetic landmarks. To inspect your own
observations, call `save_landmark_overlay()` with your aligned pixel projections
and a `LandmarkObservation`. Pass an aligned Pillow image optionally.

## Synthetic identity fitting benchmark

The Issue #4 benchmark generates DFL-ordered 2D landmark observations from known
GNM identities, expressions, poses, and cameras. It evaluates recovery of identity
coefficients, the full neutral mesh, and projected landmarks. Generation and
evaluation operate on numeric arrays and use the existing GNM, correspondence,
and camera APIs. They require only the public model asset.

### Generate and evaluate with Docker

Run from the repository root with the existing CPU service:

```sh
docker compose build personalab
docker compose run --rm personalab python -m pytest -v tests

docker compose run --rm personalab python -m persona_lab.benchmark.generate \
  --scenario F --subjects 5 --observations-per-subject 50 --seed 42 \
  --output outputs/benchmarks/synthetic

docker compose run --rm personalab python -m persona_lab.benchmark.evaluate \
  --ground-truth outputs/benchmarks/synthetic \
  --predictions outputs/predictions/synthetic \
  --mode known-camera --output outputs/reports/synthetic
```

The evaluation command expects external predictions for every subject. Pass
`--subject subject_0000` to evaluate one subject explicitly. It can be repeated
for several subjects, and the report lists omitted subject IDs. Generation,
prediction writing, and evaluation require a fresh output path. Evaluation
publishes `report.json` and prints a human-readable summary.

Both commands accept `--model-file models/gnm_head.npz`. The file must match
GNM v3.0 Head from `google/gnm-v3`, revision
`c01e90d298d82301f9fd18f54806be751775cb7c`, path `v3_0/gnm_head.npz`
and SHA-256 `61d78bbfb4ad8e0b38495804a4caef3214d3df00f8c3f68761e63b41ce3747eb`.
The benchmark verifies the bytes and uses the existing GNM cache directory
(`$GNM_CACHE_DIR/v3_0/gnm_head.npz`). It downloads the pinned revision on a cache
miss. An incompatible cached or explicitly selected file fails with a checksum
error. The asset contains 253 identity coefficients, 383 expression
coefficients, four joints (`neck`, `head`, `left_eye`, `right_eye`), 17,821
vertices, and 35,324 triangles. The manifest records ordered model parameter
names, topology and landmark-resource digests, and coordinate conventions.

The default numerical dtype is `float64` and the default device is `cpu`.
Generation also accepts `--dtype float32`. For a CUDA host configured as above,
use `docker compose --profile cuda run --rm personalab-cuda` and pass
`--device cuda` to the module. An unavailable requested device fails explicitly.

### Scenarios and generation options

| Scenario | Subjects | Observations per subject by default | Expressions and views | Noise profiles |
| --- | ---: | ---: | --- | --- |
| A | 1 | 1 | Neutral frontal | Clean |
| B | 1 | 9 | Neutral, all nine views | Clean |
| C | 1 | 9 | All seven presets, all nine views | Clean |
| D | 1 | 9 | Same clean inputs as C | Alternating 0.5 and 2 px Gaussian standard deviations |
| E | 1 | 9 | Same clean inputs as C | Four missing landmarks and three outliers, with optional region removal |
| F | 5 by default, at least 2 | 9 | All presets and views for each subject | Cycles clean, small Gaussian, moderate Gaussian, structured, and combined |

Scenario F is the default. Scenario A takes exactly one observation and one
projection type. B–F require at least nine observations to cover the views.
A defaults to perspective. B–F alternate perspective and weak perspective
when `--camera both` is selected. `--camera perspective` and
`--camera weak-perspective` select one projection throughout. Set the raster
with `--image-size WIDTH HEIGHT` (default 512 × 512) and the local random seed
with `--seed` (default 42).

Pass `--config path.json` with a JSON object of generation options. Explicit
command-line options take precedence. The manifest's resolved `config` object
can be copied into a JSON file and reused directly. In addition to CLI options,
the file accepts `expression_strength`, `expressions` (preset order),
`view_degrees` (signed angles by view name), `distance_m` (positive lower and
upper bounds), `weak_scale_per_short_edge` (positive lower and upper bounds),
and `noise`. Supply every required view and expression for the scenario.
A `noise` override supplies a complete map of the scenario's profile names
(`clean`, `small`, `moderate`, `structured`, and/or `combined`). Each profile
object can define `gaussian_px`, `missing`, `outliers`, `outlier_radius_px`,
and `occlude`, a list of correspondence region names such as `eyebrow` or
`inner_mouth`. Scenario constraints and insufficient landmark selections report
explicit errors.

One identity vector is drawn per subject, independently across all native
identity coefficients, from a uniform distribution on `[-h, h]`. The default
`h=0.5` covers a modest part of GNM's documented typical coefficient range
of −3 to +3. Use `--identity-half-range` or the corresponding JSON setting to
choose a positive half-range up to 3. These are controlled benchmark samples.
The upstream semantic sampler models demographic and emotion classes. The
benchmark samples the native identity basis directly and uses explicit
anatomical probes for its expression directions.
Realistic geometry for arbitrary vectors is outside this sampling guarantee.
Neutral vertices use zero expressions, joint rotations, and root translation,
in native meter units and original vertex order.

The expression controls below use signed native basis components. Probes
measure movement on the official sparse landmark asset, in GNM's native axes.
For each subject, generation checks directions at the configured strength
endpoints with a 10 µm displacement margin. Eye closing yields a reduced
positive eyelid aperture. Each observation independently draws a multiplier
uniformly from `expression_strength` (default `[0.8, 1.0]`). GNM documents a
typical expression coefficient range of −3 to +3. The table describes the
benchmark's selected directions and magnitudes.

| Preset | Base GNM expression coefficients | Anatomical probe |
| --- | --- | --- |
| `neutral` | All zero | Neutral reference |
| `smile` | `lower_face_region_005=-1.5`, `lower_face_region_004=-0.7` | Both outer lip corners rise relative to the center and get farther apart |
| `mouth_open` | `lower_face_region_000=-1.5` | Upper-to-lower inner lip gap increases |
| `raise_brows` | For each of `left_eye_region` and `right_eye_region`, `_000=1`, `_001=-1` | Each brow rises relative to its eye corners |
| `eyes_close` | For each eye region, `_000=-0.95`, `_001=-0.95` | Both eyelid apertures decrease |
| `eyes_open` | For each eye region, `_000=0.75`, `_001=0.75` | Both eyelid apertures increase |
| `smile_raise_brows` | Sum of the smile and brow directions | Lip and brow probes both pass |

This Head asset has no jaw joint. Opening uses its lower-face expression
basis. Eye joint rotations orient the eyeballs. The eyelid presets use
the eye-region expression basis.

| View | Native `head` joint axis | Default angle |
| --- | --- | ---: |
| `frontal` | — | 0° |
| `three_quarter_left` / `three_quarter_right` | Y | −35° / +35° |
| `profile_left` / `profile_right` | Y | −70° / +70° |
| `pitch_up` / `pitch_down` | X | −18° / +18° |
| `roll_positive` / `roll_negative` | Z | +15° / −15° |

Left and right view labels denote the direction the nose moves in the image.
The view schedule cycles through this table. Expressions cycle through their
own ordered list with a phase advance after each rotation through the views.
The head joint creates the viewpoint once, with zero camera rotation and zero
root translation. Both cameras view the frontal OpenCV axes defined above.
Perspective focal lengths equal 1.4 times the image short edge in pixels.
The principal point is the raster center, and the camera adds 0.265 m along
its Y axis to center the face. Camera Z translation is sampled within
the configured depth interval (default 0.60–0.80 m). Weak-perspective scale
is the image short edge times a factor sampled in the configured interval
(default 2.0–2.4). Weak image translation centers the raster with
an additional Y offset of scale × 0.265 m. Each camera type alternates
sampling around fractions 0.25 and 0.75 of its interval with uniform
jitter in `[-0.125, 0.125]`, which guarantees near and far tiers.
Three-quarter magnitudes are 20–55°, profile magnitudes are 60–<90°, and
pitch and roll magnitudes are 5–35° when configuring view angles.

### Landmarks, noise, and reproducibility

The benchmark projects GNM's official `HEAD_SPARSE_68` locations and applies
`DFL_TO_GNM` in DFL index order. The default corresponds to all 68 slots
including all 17 contour slots, with unit weights. Eligibility requires a
supported, enabled correspondence and a clean projection within the raster
edges `[-0.5, width - 0.5)` and `[-0.5, height - 0.5)`. Coordinates remain
unclamped. Unsupported slots carry a finite zero placeholder and false masks.
Perspective projection rejects nonpositive depths. All generated observations
have physical visibility status **not evaluated**. Simulated occlusion is
recorded separately from frame status and projection support.

Corruption first removes configured anatomical regions and selects additional
missing slots uniformly without replacement among the remaining eligible
points. The removed slots retain their clean coordinates under a false
fitting mask. Independent zero-mean Gaussian samples in pixel units are added
to valid X and Y coordinates. Then outlier indices are sampled uniformly
without replacement from the remaining valid points. Each receives an
additive displacement sampled with uniform angle and uniform radius
(default 8–18 px), independent of the Gaussian noise. Outliers stay valid,
including when their corrupted coordinates move outside the raster. The
combined profile defaults to 2 px Gaussian noise, eyebrow occlusion, four
randomly missing slots, and three outliers. Missing, occluded, and outlier
labels are disjoint. Gaussian noise can overlap with outliers. The dataset
stores both clean and observed coordinates and their final signed offsets.
These noise levels are controlled perturbations in pixel units.

The generator uses local NumPy PCG64 streams with `SeedSequence` keyed by
the root seed and `(stream, subject index, observation index)`. Stream IDs
are recorded in the manifest. Identity, expression, camera, Gaussian,
missing, outlier selection, and outlier displacement use separate streams.
Views and region removal follow explicit schedules. Noise changes leave
the clean parameters and projections paired at the same seed and configuration.
Stable identifiers use zero-based indices. The dataset ID hashes canonical
metadata and numeric array digests, independent of the archive container bytes
and descriptive runtime metadata. The reproducibility target is equal arrays
with identical settings, model, software, and device. Exact agreement across
platforms or GPU and CPU and byte-identical NPZ archives are outside that target.

### Dataset, predictions, and fitting API

A dataset directory contains `manifest.json`, `topology.npz`, and one
`subject_XXXX.npz` per subject. Schema version 1 records the resolved config,
the asset and parameter basis, the runtime, correspondences, and ordered subject
and observation records. Each camera is stored in its observation's metadata
with the raster size, projection type, rotation, and projection-specific
parameters. IDs have the form `subject_0000/observation_0000`. The completion
manifest is published after validation. Loading checks the archive shapes,
dtypes, finiteness, topology, digests, identifiers, cameras, and mask relationships
through NumPy's non-pickle loading path.

| Archive array | Shape | Type |
| --- | --- | --- |
| `triangles` in `topology.npz` | `(35324, 3)` | int32 |
| `identity` | `(253,)` | Configured float dtype |
| `neutral_vertices` | `(17821, 3)` | Configured float dtype |
| `expression` | `(N, 383)` | Configured float dtype |
| `rotations`, `translation` | `(N, 4, 3)`, `(N, 3)` | Configured float dtype |
| `clean`, `observed`, `offset_px` | `(N, 68, 2)` | Configured float dtype |
| `weights` | `(N, 68)` | Configured float dtype |
| `supported`, `projection_valid`, `in_frame`, `validity` | `(N, 68)` | bool |
| `missing`, `occluded`, `outliers` | `(N, 68)` | bool |

`load_dataset(path).fitting_observations(subject_id)` returns a tuple of
`LandmarkObservation` instances with observed pixels, dimensions, weights,
validity, correspondence map, and `known_camera`. It accepts an explicit
`observation_ids` subset and `dtype` and `device` arguments. To enumerate IDs,
use `subject_ids` and `observation_ids(subject_id)`. Evaluation code can
request the hidden state explicitly through `ground_truth(subject_id)`.

An external producer can write predictions with `write_predictions`.
For example, this function saves a shared identity obtained from fitting:

```python
from persona_lab.benchmark.dataset import load_dataset
from persona_lab.benchmark.predictions import SubjectPrediction, write_predictions

dataset = load_dataset("outputs/benchmarks/synthetic")
subject = dataset.subject_ids[0]
observations = dataset.fitting_observations(subject, dtype=dataset.config.dtype)
cameras = tuple(item.known_camera for item in observations)

def save_estimated_identity(identity):
    # The caller supplies the optimizer's ordered NumPy vector in the dataset dtype.
    write_predictions(
        "outputs/predictions/identity",
        dataset,
        {subject: SubjectPrediction(identity=identity)},
    )
```

Use `LandmarkObservation.landmark_pairs(projected_gnm)` inside a fitting
objective to obtain valid pairs. Save each subject's single estimated identity
after fitting. For observation-specific estimates, populate
`SubjectPrediction.observations` with `ObservationPrediction` objects keyed
by the matching observation ID. Pass the dataset and output path to
`write_predictions`, then evaluate through the CLI:

```sh
docker compose run --rm personalab python -m persona_lab.benchmark.evaluate \
  --ground-truth outputs/benchmarks/synthetic \
  --predictions outputs/predictions/identity \
  --mode identity-only --subject subject_0000 --output outputs/reports/identity
```

Prediction schema version 1 is a directory with a `manifest.json` containing
`schema_version`, `dataset_id`, the full `model` object from the dataset, and
a `subjects` list. Each subject object has `id`, an `arrays` list of archive
keys, and an `observations` list of `{"id": "..."}` objects with optional
`camera` objects. The corresponding `<subject_id>.npz` contains `identity`
plus any supplied `expression`, `rotations` and `translation` (supplied
together for pose), or `projected` (`N × 68 × 2` DFL-ordered pixels) in
the dataset float dtype. Observation arrays follow that prediction manifest's
row order, and the evaluator associates them by ID. Every supplied observation
group covers the full selected subject. Camera objects use the same fields as
the dataset, with matching projection type and dimensions. Foreign, missing,
duplicate, or incompatible IDs and invalid arrays produce explicit errors.

| Mode | Required predictions | Reprojection and parameter recovery |
| --- | --- | --- |
| `identity-only` | Identity | Coefficient and neutral geometry recovery |
| `conditional` | Identity | Reprojects with the saved nuisance parameters, labeled conditional |
| `known-camera` | Identity, expression, pose | Reprojects with fixed benchmark cameras and scores predicted expression and pose |
| `complete` | Identity, expression, pose, camera | Reprojects the supplied parameters and scores each estimated group |
| `projected` | Identity and external DFL pixels | Scores supplied pixels and any independently supplied expression, pose, or camera groups |

For evaluator checks, `oracle_predictions(dataset, mode="complete")` explicitly
builds predictions from the saved ground truth. Pass its result to
`write_predictions` and evaluate that output. The CLI always requires a
prediction directory.

### Numerical interpretation

The evaluator reports raw absolute coefficient mean, RMS, and maximum. It
evaluates the neutral identity with zero expression and pose and compares
corresponding native mesh vertices using their Euclidean distances in meters.
Mesh reductions are mean, RMS, and maximum. Vertex correspondence uses the
whole verified topology and its original positions.

Clean- and observed-reference reprojection both use the benchmark fitting mask.
Eligible distances are Euclidean pixels, with mean, median, RMS, and counts.
Per-observation reports retain all 68 slots and use JSON `null` for excluded
slots. Entirely empty masks produce unavailable pixel summaries. A noisy
oracle can have positive observed-reference error. Subject and scenario
reprojection aggregates pool individual valid landmark distances, including
in the noise-profile slices. Coefficient and vertex aggregates pool
equal-sized vectors and meshes across selected subjects. RMS is calculated
from squared errors before taking the square root.

Expression recovery uses its own coefficient reductions. Each joint or
camera rotation error is the shortest relative rotation angle in degrees.
Root and perspective camera translations use Euclidean meters. Perspective
focal length and principal point errors are separate absolute pixel
components. Weak camera image translation is Euclidean pixels and scale
error is in pixels per meter. Reports label absent groups `not_provided`
and fixed cameras `fixed_known`.

Float64 tests use exact array equality for serialization and paired sampling,
1e-10 pixels for the independent neutral pinhole calculation and clean oracle
reprojection, and 1e-12 m for neutral vertex roundoff. They use the Issue #3
float64 projection checks and a native-coordinate reference. The expression
direction checks use the 10 µm margin above to distinguish movement from
roundoff and small variations in the landmark resource. These tolerances
cover benchmark arithmetic. Optimization thresholds will be measured in
Issue #5.

Sparse landmarks can constrain some identity dimensions weakly, especially
with one frontal view. Changes to expression, pose, and camera can compensate
for identity errors in the raster. Compare coefficient and geometry recovery
alongside reprojection. Synthetic recovery establishes correctness under GNM's
assumptions. Evaluating real photographs requires the separate real-data study.

## Fit one identity from multiple observations

The fitting API accepts `LandmarkObservation` objects from the DeepFaceLab
importer or the Issue #4 benchmark. Each fit uses one verified GNM v3.0 Head
model and one shared vector of 253 identity coefficients. Each observation
has a 383-coefficient expression, four native axis-angle joint rotations,
a root translation in meters, and its own camera. The model's buffers and
the supplied landmark data stay fixed. Joint rotations follow GNM's
`neck`, `head`, `left_eye`, `right_eye` order in radians.

### Parameter selection and camera gauges

The default identity selection is the 170-component `head` region. The
pinned asset has nonzero head basis entries at the 186 positively weighted
vertices of the official sparse landmarks. The `eyes` and `teeth` vertex
bases are zero at those vertices. The teeth joint basis still affects the
neck and head pivots, so pose can introduce indirect effects. Direct vertex
support gives evidence of sensitivity and leaves independent recovery open.

The default expression selection uses the first 12 `lower_face_region`
coefficients and the first six coefficients in each `left_eye_region` and
`right_eye_region`. The benchmark's anatomical probes exercise mouth,
eyelid, and brow movements within these regions. Tongue and pupil
expression bases have zero direct support at the sparse vertices. The
initial subset limits how much each observation can explain with its own
expression. Its size and regularization are configurable. The selected
expression values are projected to the configured operating interval after
each Adam update. The default interval is ±3, the typical range documented
by GNM. It is an optimizer policy.

Select native names such as `head_000` or regions such as `head` in
`train_identity` and `train_expression`. The `fixed_identity` and
`fixed_expression` lists remove explicitly fixed names or regions from
those selections. The parameter arrays saved with a result contain every
native coefficient. Only selected coordinates are stored in trainable
tensors. The same representation preserves externally supplied fixed values
when stages change. The default starting identity and expressions are the
mean identity and neutral expressions, both represented by zero.

| Camera mode | Optimized camera values | Fixed camera values | GNM degrees of freedom |
| --- | --- | --- | --- |
| Known, either projection | None | All supplied camera tensors | Head rotation, plus optional root translation with fixed perspective cameras |
| Estimated weak perspective | Pixel XY offset and log scale ratio | Initial camera rotation | Head rotation with fixed root translation |
| Estimated perspective | Metric XY offset and log distance ratio | Initial rotation, focal length and principal point | Head rotation with fixed root translation |

The neck and eyes keep their initial rotations. Estimated weak perspective
has no depth parameter. Estimated perspective uses the raster center as the
principal point and a focal length of 1.5 times the shorter raster edge
by default. This defines an artificial gauge for aligned crops. An explicit
initial camera can set these fixed quantities. For estimated cameras the
weak-perspective pixel offsets are scaled by the shorter edge and the
perspective XY offsets by the initial Z. A positive initial scale or Z
multiplied by the exponential of its corresponding change remains
positive. Projection checks all 68 official landmark depths. Invalid
depths terminate the fit. Focal length, depth, physical head size and
orientation do not have independent calibration in this problem.

### Initialization, objective, and optimization

Initialization tests a deterministic set of 45 head rotations at yaw
0°, ±40°, and ±75°, pitch 0° and ±25°, and roll 0° and ±20°. GNM
receives the resulting axis-angle vectors from SciPy's extrinsic
`yxz` conversion. Known or explicitly initialized cameras are used directly
to score candidates. With an uncalibrated weak-perspective camera, weighted
centroids and least-squares scale place each candidate in the image.
Perspective initialization uses that scale at the weighted depth centroid
and solves XY translation with the existing pinhole projector. The candidate
with the lowest eligible weighted residual is selected from the candidates
with valid camera geometry. The result records the strategy, selected index
and admissible candidate count. `pose_initializer="neutral"` starts from
the native zero pose. Passing explicit rotations overrides pose search for
that observation. The fitting loop uses no randomness. The seed controls
CLI sampling when `--selection sample` is used.

For image \(i\) let \(s_i=\min(w_i,h_i)\) be its shorter raster edge and
\(a_{ik}\) be the eligible landmark weight. The landmark term pools all
eligible weights across the subject.

```text
L_landmark = Σ a_ik || (projected_ik - observed_ik) / s_i ||² / Σ a_ik
L_total = L_landmark + L_identity + L_expression + L_pose + L_camera
```

All 68 supported correspondences, including the 17 contour landmarks, have
unit weight and are enabled by default. An observation's mask and weights,
disabled or unsupported correspondences, and an explicit subset determine
eligibility. An observation with zero eligible pairs is rejected with its ID.
Logging reports independent unweighted Euclidean mean and RMS in pixels.

Identity regularization is the mean square of the selected native identity
coefficients. Expression regularization is the mean square of the selected
native expression coefficients over observations. Both are zero-centered
penalties in the model's native scaling. Pose penalties, when enabled,
measure squared changes from the initial head rotation and root translation.
The camera penalty is the mean square of optimized camera coordinates in
their normalized parameterization. The default strengths for identity,
expression, pose and camera are `1e-6`, `2e-5`, `0` and `0`.
The expression penalty and restricted subset are intended to limit
identity-expression compensation. Benchmark evaluation can measure that trade-off.

| Stage | Updates | Available groups | Default learning rates |
| --- | ---: | --- | --- |
| Pose and camera | 30 | Pose, camera | 0.018 |
| Identity | 70 | Identity, pose, camera | Identity 0.045, pose 0.008, camera 0.006 |
| Joint | 100 | Identity, expression, pose, camera | Identity and expression 0.022, pose 0.005, camera 0.004 |

The stage budgets and rates are initial implementation settings. A JSON
`FitConfig` can override the order, groups, step counts, and per-group rates.
An Adam optimizer starts fresh for each stage and receives only the enabled
groups. A global iteration accumulates normalized batch gradients and
applies the subject's regularizers once before the Adam step. The default
batch size is four. Only a batch of GNM graphs is live at a time. The
fitting loop records losses, eligible counts, pixel errors, gradient norms,
and elapsed time.

The result distinguishes gradient-tolerance convergence, an improvement
plateau after configured patience, an exhausted iteration budget, and
failure. Tolerance-based stopping is opt-in. The default divergence check
fails a stage if its objective exceeds 100 times the stage-entry loss,
with a floor of `1e-12` on the reference loss. Missing or nonfinite
gradients, invalid projections, and nonfinite parameters or Adam state
fail explicitly. History contains the last logged valid step. Failed
results have no final parameters, mesh, overlays, or predictions.

### CLI and result files

Commands below use the existing Docker service. They require local
validation before using the fitting pipeline on private images.

```sh
docker compose run --rm personalab python -m persona_lab.fit \
  --input datasets/light.zip --camera weak-perspective --max-images 50 \
  --selection first --device cpu --batch-size 4 --overlays 6 \
  --output outputs/fitting/first_identity

docker compose run --rm personalab python -m persona_lab.fit \
  --benchmark outputs/benchmarks/scenario-C --subject subject_0000 \
  --camera-mode known --device cpu --batch-size 4 \
  --output outputs/fitting/scenario-C --predictions outputs/predictions/scenario-C

docker compose run --rm personalab python -m persona_lab.benchmark.evaluate \
  --ground-truth outputs/benchmarks/scenario-C --predictions outputs/predictions/scenario-C \
  --mode known-camera --subject subject_0000 --output outputs/reports/fitted-scenario-C
```

Use `--model-file models/gnm_head.npz` to provide the exact pinned asset or
omit the flag to use the verified model cache. `--model-version v3`
selects GNM v3.0 Head. `--device cpu` and `--dtype float64` are the
defaults, and `--dtype float32` is supported. Request CUDA through
`docker compose --profile cuda run --rm personalab-cuda` with `--device cuda`.
An unavailable requested device raises an error. Other controls include
`--batch-size`, `--iterations A B C`, `--learning-rate`, `--log-every`,
`--seed`, the four penalty weights, and the trainable coefficient lists.
`--learning-rate` overrides all stage and group rates. `--config` accepts
a JSON object with the fields of `FitConfig`. Each stage has `name`,
`iterations`, `groups`, `learning_rate`, and optional `group_rates` fields.

The real input uses the stable importer order and records accepted,
rejected, and selected counts. `--selection sample` selects without
replacement from accepted observations and preserves their importer order.
The result directory must be new. `result.json` records the verified model
and parameter order, selection, camera gauge, observation IDs, cameras,
configuration, initialization, history, and termination status.
`parameters.npz` contains the initial and final arrays and loads with
`allow_pickle=False`. `neutral.obj` uses fitted identity with zero
expressions, rotations, and translation. `--no-neutral-mesh` disables
that diagnostic. `--overlays N` writes initial and final pairs for the
highest-RMS observation and a sample spaced across fitted head yaw. The
overlays use the existing GNM wireframe and landmark diagnostic. Keep real
results and overlays in the local outputs directory.

### Python API and benchmark use

```python
from persona_lab.benchmark.dataset import load_dataset
from persona_lab.benchmark.model import load_benchmark_model
from persona_lab.benchmark.predictions import write_predictions
from persona_lab.fitting import FitConfig, fit_identity, save_result

model = load_benchmark_model(dtype="float64", device="cpu")
dataset = load_dataset("outputs/benchmarks/scenario-C")
subject = dataset.subject_ids[0]
observations = dataset.fitting_observations(subject)
result = fit_identity(model, observations, FitConfig())
save_result("outputs/fitting/from_api", result)
if result.final is not None:
    write_predictions("outputs/predictions/from_api", dataset, {
        subject: result.to_prediction(dtype=dataset.config.dtype),
    })
```

For explicit inputs, pass a `FitInitialization` with a native identity
tensor and `ObservationInitialization` records keyed by observation ID.
The optional observation camera seed is `LandmarkObservation.camera_initialization`.
`load_result(path, model=model)` checks archive shapes, dtype, camera
fields, fixed coordinates, and the model asset and native ordering.

For the neutral multiview experiment generate benchmark Scenario B and
pass an empty `--train-expression` list. Scenario C covers multiple
expressions and unknown poses with known cameras. Generate separate
Scenario C datasets with `--camera weak-perspective` and `--camera perspective`
for estimated-camera experiments. Fit with `--camera-mode estimated --camera`
and the corresponding model. Estimated-camera input is built with
`include_known_cameras=False`. It contains observed landmarks and fitting
metadata. Independent evaluation uses `complete` mode for estimated cameras
and `known-camera` mode for fixed cameras.

For a count study generate one Scenario C dataset with 100 observations and
use ordered prefixes of 1, 5, 10, 25, 50, and 100. In the Python API, pass
the explicit prefix to `fitting_observations`. Evaluate subsets by creating
`dataset.select_observations(subject, ids)` with the same IDs and order.
The evaluator CLI accepts a repeated `--observation-id` with exactly one
`--subject` for the same subset. Evaluate the initial state using
`result.to_prediction(initial=True)`. For several subjects, fit each
separately and collect the resulting predictions by subject ID.

Compare coefficient and neutral mesh errors against the mean-identity
baseline. A reprojection improvement can coincide with a worse neutral
mesh. Report full and selected coefficient errors and native neutral
geometry before and after fitting. The `identity_observability` function
in `persona_lab.fitting.observability` computes local weighted 2D
Jacobian column norms and singular spectra with fixed nuisance parameters,
after removing expression directions, and after removing all active
nuisance directions. It uses one observation graph at a time.
Fixed-nuisance rank is a conditional diagnostic. Joint identifiability
depends on the views, camera gauge, and strength of nuisance compensation.

### Validation status

The official asset checksum, native dimensions and direct basis support
were inspected. CPU and CUDA fitting, the Docker test suite, numerical
Jacobian ranks, synthetic recovery experiments, the observation-count
study and the private DeepFaceLab run remain unverified. The implementation
records model loading, initialization, optimization, update counts, process
peak RSS and, on CUDA, peak allocated memory. Identical ordered inputs
and configuration use deterministic initialization. Numerical equivalence
across processes, batch sizes, dtypes, devices and software versions
requires measured tolerances.

## GNM Ecosystem Packages

Here we list all the available GNM packages:

| Package | Description | Status | Teaser |
| :---: | :--- | :---: | :---: |
| **[GNM Head](gnm/shape/README.md)**  | Parametric 3D statistical human head and face geometry model providing fine-grained, disentangled control over identity, expressions, and head pose. The model contains controllable internal anatomy including eyeballs, teeth and tongue. Includes multi-framework backend support for **NumPy**, **JAX**, **PyTorch**, and **TensorFlow**, along with semantic parameter sampling. <br><br>*Technical report*: [GNM Head: A Generative aNthropometric Model of the human head](https://arxiv.org/abs/2607.23687) | [![CI Linux](https://github.com/google/gnm/actions/workflows/ci-shape-linux.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/ci-shape-linux.yml)<br>[![CI macOS](https://github.com/google/gnm/actions/workflows/ci-shape-macos.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/ci-shape-macos.yml)<br>[![CI Windows](https://github.com/google/gnm/actions/workflows/ci-shape-windows.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/ci-shape-windows.yml)<br>[![Lint](https://github.com/google/gnm/actions/workflows/lint.yml/badge.svg)](https://github.com/google/gnm/actions/workflows/lint.yml) | ![GNM Head Teaser](gnm/shape/assets/readme/teaser_heads_cropped.gif) ![GNM Head demo teaser](gnm/shape/assets/readme/gnm_head_demo.gif)

## Citation
If you use any part of the GNM Ecosystem in your work, please consider citing
the corresponding package. Relevant bibtex entries are listed below as well as
within the individual packages.

**GNM Head**

```bash
@article{ploumpis2026gnmhead,
  title={GNM Head: A Generative aNthropometric Model of the human head},
  author={Ploumpis, S. and Bednarik, J. and Zoss, G. and Guseinov, R. and Prasso, L. and Chandran, P. and Boyne, O. and Choutas, V. and Bolkart, T. and Wang, D. and Chai, M. and Qiu, D. and Winberg, S. and Rainer, G. and Bridgeman, L. and Vicini, D. and Riviere, J. and Boetzel, Y. and Koumis, A. and Busch, J. and Herrera, C. and Still, J. and Ysebert, S. and Lincoln, P. and Escolano, S. O. and Rhemann, C. and Wood, E. and Beeler, T. and Zafeiriou, S.},
  year={2026},
  eprint={2607.23687},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2607.23687},
}
```

## Contributing
We'd love to accept your patches and contributions to this project! See
[CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to get started
and how we handle external contributions.

## License
This project is licensed under the Apache License, Version 2.0. See the
[LICENSE](LICENSE) file for details.
