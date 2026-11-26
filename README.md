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
failure. Adam's tolerance-based stopping is opt-in. The default divergence check
fails a stage if its objective exceeds 100 times the stage-entry loss,
with a floor of `1e-12` on the reference loss. Missing or nonfinite
gradients, invalid projections, and nonfinite parameters or Adam state
fail explicitly. History contains the last logged valid step. Failed
results have no final parameters, mesh, overlays, or predictions.

### Levenberg–Marquardt

Levenberg–Marquardt with a Schur complement is the default in `FitConfig` and
the fitting CLI. Select `--solver adam` or `FitConfig(solver="adam")` for
historical comparisons. The selection applies to the entire fitting schedule.
Both solvers use the same parameter
groups, starting values, observation masks, and objective. Adam uses the
existing learning rates and divergence check. Levenberg–Marquardt (LM) uses a
stage's groups and accepted-update budget. Stage learning rates are Adam
settings.

Each eligible landmark contributes two residual coordinates,
`sqrt(a_ik / Σ a) * (projected_ik - observed_ik) / s_i`. The weighted
denominator spans the subject. Let `p` be the number of selected identity
coefficients, `N` the number of observations, and `d` the width of one selected
local coordinate group. The identity prior contributes
`sqrt(identity_weight / p) * identity` once. For a local group, the prior
contributes `sqrt(group_weight / (N * d)) * delta` per view. The head
rotation and root translation use separate means centered on initialization.
Expression and normalized camera coordinates are centered on zero.
Configured groups frozen during a stage still contribute their constant
penalties. The objective is the sum of squares of these residuals.

At the accepted state, let `A_i` and `B_i` be the derivatives of the view
residual with respect to shared identity and that view's local coordinates.
The Gauss–Newton system has one shared block `U`, independent local blocks
`V_i`, cross blocks `E_i`, and gradient `g = Jᵀr`. The shared block sums
`A_iᵀA_i` and the single identity prior. Each cross block is `A_iᵀB_i`,
and each local block is `B_iᵀB_i` plus local prior curvature. LM equilibrates
columns with inverse square roots of the normal diagonals. Each block uses
a relative diagonal floor and a positive machine-precision floor. It adds
`λI` to the shared and local blocks in scaled coordinates.

With `V_i = L_i L_iᵀ` after scaling and damping, the local triangular solves
form `W_i = L_i⁻¹ E_iᵀ` and `v_i = L_i⁻¹ g_i`. Cholesky solves the
reduced system

```text
(U - Σ W_iᵀ W_i) Δa = -g_a + Σ W_iᵀ v_i
Δb_i = L_i⁻ᵀ (-v_i - W_i Δa)
```

The algorithm back-substitutes the local updates and maps them to native
coordinates. The Jacobian evaluates the vertices used by the official
landmarks with the native GNM basis, joint regressors, pose correctives and
skinning operations. Trial objectives still evaluate full mesh geometry.
Differentiation uses one observation graph and bounded groups of reverse-mode
rows. A value of `jacobian_rows=1` selects scalar
reverse-mode derivatives. The retained matrices take
`O(p² + N(pq + q²))` storage for local width `q`. Trial scoring uses
`batch_size`. Jacobian blocks are accumulated, scaled, and factored in
float64 on the requested device. Forward projection and differentiation
use the model's selected precision. Float32 can limit progress when a
candidate rounds to the current coordinates. The diagonal ratios of the
Cholesky factors are conditioning indicators.

Coordinates at an expression bound with an outward descent direction are
held for the linear solve. The trial then clips expressions and casts to
the model dtype. For the resulting actual step `δ`, the predicted
reduction in the undamped objective is `-2 gᵀδ - δᵀ (JᵀJ) δ`.
The actual reduction is the old objective minus the trial objective. An
accepted trial has positive actual reduction and meets the configured
acceptance ratio. Damping is divided by three after an acceptance ratio
above 0.75 and multiplied by four after a ratio below 0.25 or after a
rejection, within its bounds. Each stage starts with its configured initial
damping. Rejected trial coordinates are discarded. Existing camera gauges,
positive scales and positive perspective landmark depths are checked, as is
finite trial geometry.

`LevenbergMarquardtConfig` is exported from `persona_lab.fitting` and
serialized under `FitConfig.lm`. The defaults are listed below.

| Key | Default | Role |
| --- | ---: | --- |
| `initial_damping` | `1e-3` | Damping at each stage entry |
| `minimum_damping` | `1e-8` | Lower damping bound |
| `maximum_damping` | `1e8` | Upper damping bound |
| `max_trials` | `12` | Trials per linearization |
| `acceptance_threshold` | `1e-4` | Minimum actual-to-predicted reduction ratio |
| `diagonal_floor` | `1e-12` | Relative floor for each block's squared column norms |
| `scaled_gradient_tolerance` | `1e-10` | Tolerance on the scaled projected objective gradient |
| `jacobian_rows` | `8` | Output rows per reverse-mode call |

LM also supports the common `gradient_tolerance` in native coordinates
and `improvement_tolerance` with `patience`. `converged` records a
gradient tolerance, `stopped` an objective plateau, and `budget_exhausted`
the accepted-update budget. A failure to obtain an acceptable trial returns
`no_acceptable_step` with the trial or damping limit in the reason.
`numerical_failure` reports a bad linearization, failed factorization or
other numerical error. Both failure statuses return a nonzero CLI exit
and omit final parameters. Adam retains its `failed` status.

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
  --solver levenberg-marquardt --camera-mode known --device cpu --batch-size 4 \
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
`--learning-rate` overrides all Adam stage and group rates. `--config` accepts
a JSON object with the fields of `FitConfig`. Each stage has `name`,
`iterations`, `groups`, `learning_rate`, and optional `group_rates` fields.

An LM configuration can set its bounds and trial budget in the same JSON.

```json
{
  "solver": "levenberg-marquardt",
  "lm": {
    "initial_damping": 0.001,
    "maximum_damping": 1000000,
    "max_trials": 12
  }
}
```

The real input uses the stable importer order and records accepted,
rejected, and selected counts. `--selection sample` selects without
replacement from accepted observations and preserves their importer order.
The result directory must be new. `result.json` records the verified model
and parameter order, selection, camera gauge, observation IDs, cameras,
configuration, initialization, history, and termination status.
`result.json` includes an `optimization` record with objective terms
at initialization and the last valid state, objective and Jacobian evaluation
counts, updates and their norms, and LM trials with damping, reductions,
acceptance ratio, and factor diagnostics. Runtime metadata includes
initialization, objective, Jacobian and block-solve times, fitting time,
process RSS and CUDA peak allocated bytes where available. Objective time for
Adam includes the backwards pass. A Jacobian evaluation is an attempted
full linearization, including block accumulation. An objective evaluation
is an attempted full scoring pass. On CUDA these timings synchronize the
requested device. RSS and CUDA allocated-byte peaks are process high-water
marks. Model loading is timed by the CLI. It is unavailable to
`fit_identity` when the caller supplies a loaded model. `load_result` also
reads version 1 Adam and version 2 solver-explicit results.
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
Fitting and evaluation accept a repeated `--observation-id` with exactly one
`--subject` for the same ordered subset. Fitting uses explicit IDs or
`--max-images` and optional sampling. A complete selection in the subject's
original order retains the source identifier and available subjects, just as
an omitted selection does. A subset or reorder has an identifier derived
from the source, subject, and ordered IDs. Predictions version 2 records
both `source_dataset_id` and `dataset_id`, the evaluation binding.
Version 1 predictions are readable when their dataset binding matches the
current selection. Recreate a legacy full-selection prediction if it was
written with a separate full-subset identifier. Evaluate the initial state using
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

### Reproduce the solver comparisons

Use a new output directory and the existing pinned GNM asset. The study
regenerates its synthetic datasets and writes case results and a consolidated
`report.json`. Each fit runs in its own process for process RSS measurements.

```sh
docker compose build personalab
docker compose run --rm -e PERSONALAB_GNM_MODEL_FILE=/app/models/gnm_head.npz \
  personalab python -m pytest -v tests
docker compose run --rm personalab python -m persona_lab.benchmark.fitting_study \
  --model-file /app/models/gnm_head.npz --device cpu --dtype float64 \
  --output outputs/issue5 --experiments A B C D E F G H --seeds 42 43 44 --spectra
```

Experiments A–C cover controlled sparse identity, neutral views, and variable
expressions. A runs both a known-pose diagnostic and ordinary unknown-pose
fitting. B and C include the two Adam stage-transition configurations with
the original Adam rates. D changes one regularizer at a time in the paired
comparisons. E uses nested prefixes of each 100-observation dataset and
reports per-seed scores, distributions, and adjacent geometry changes.
F adds the same held-out views for every prefix of an identity. It reports
conditional projection with true nuisance parameters and projection after
estimating expressions and poses with identity frozen. Cameras are known
for both held-out diagnostics. G estimates weak-perspective and constrained
perspective cameras from the observed landmarks. The perspective fit uses
its configured focal length (1.5 times the short edge by default). The
synthetic cameras use 1.4 times the short edge. G scores neutral geometry
and reprojects with the fitted cameras. Camera rotation and perspective
intrinsic differences are reported as fixed-parameter diagnostics. Direct
camera translation error requires matching rotation and root translation,
plus matching focal length and principal point for perspective. Other
observations have a null translation error and list the mismatched
fields in `translation_comparability`.

H measures 50, 100, 300, and 600 observations. It uses one update per stage
for throughput by default.
Use `--scaling-full` to apply the fitting budgets to H.

The protocol sets Adam budgets to 30/70/900 and divides the original Adam
joint learning rates by ten for the main comparisons. The default LM
budgets in the study are 10/12/30. `--adam-iterations` and
`--lm-iterations` explicitly set other computational budgets. Compare
elapsed time, evaluations, and reconstructed geometry across budgets.
`--historical-report` accepts the original `count-study-v2.json` at its
container path. The report verifies the source identifier and the exact
observation order before calculating historical score deltas. For a CUDA
study, use the existing CUDA Compose profile and pass `--device cuda`.

The historical CPU Adam study is the available measured baseline. The
second-pass implementation requires Docker verification of CPU and CUDA
fitting, comparative reconstruction, and scalability. Run the private
DeepFaceLab experiment only after the relevant synthetic validations pass.
The local fit command can then use an explicitly chosen camera model,
approximately 50 observations, overlays, and a neutral mesh.

### Corrected identity recovery

The original variable-expression study sampled all 253 native identity
coefficients while fitting the 170 head coefficients. That left 83 fixed eye
and teeth coefficients with a nonzero global coefficient error floor. Uniform
L2 penalties also changed the optimum. In the nine-view seed-42 variable
expression study, the trainable identity RMSE was 0.240809 with the production
weights, 2.764496 with only the identity penalty disabled, and 0.010869 with
both penalties disabled. The local rank was 170 in the nine-view diagnostic.
Small singular values still permit substantial displacement at a small
projected gradient.

Set `--identity-selection head` in the generator to sample the native head
group and leave the eye and teeth coefficients exactly zero. The vector still
has 253 entries in native order. `--identity-selection full` explicitly samples
the historical full-native stream and records the selection. Omitting the
option reproduces the original manifest and numeric stream. The corrected
study uses the default 24 expression coefficients, known cameras, float64 and
explicitly zero identity and expression weights for clean diagnostics.
Production fitting retains its independent regularization weights. The noisy
study crosses identity and expression weights at each Gaussian deviation.
`--gaussian-noise-px` creates a uniform Gaussian profile in Scenario C with
the same observation schedule and independently seeded noise.

```sh
docker compose run --rm personalab python -m persona_lab.benchmark.fitting_study \
  --protocol corrected-identity --identity-studies exact convergence held-out \
  --model-file /app/models/gnm_head.npz --seeds 42 43 44 \
  --output outputs/identity-recovery
docker compose run --rm personalab python -m persona_lab.benchmark.fitting_study \
  --protocol corrected-identity --identity-studies count \
  --model-file /app/models/gnm_head.npz --seeds 42 43 44 \
  --output outputs/identity-count
docker compose run --rm personalab python -m persona_lab.benchmark.fitting_study \
  --protocol corrected-identity --identity-studies robustness \
  --model-file /app/models/gnm_head.npz --seeds 42 43 44 \
  --noise-levels 0.5 2.0 --identity-weights 0 1e-6 --expression-weights 0 2e-6 2e-5 \
  --output outputs/identity-noise
```

The nine-view exact cases cover neutral and varied expressions. The
corrected protocol uses a single joint stage with a default budget of 30
updates. The convergence comparison uses the same data and budget for each
setting. The stricter experiment uses a scaled gradient tolerance and
damping floor of `1e-12`,
plus native and scaled displacement tolerances of `1e-6` and `1e-10`.
When displacement tolerances are set, LM tests a projected step at the damping
floor before reporting convergence. Each stage records its termination reason.
Accepted updates and rejected trials have separate records. Count cases use
nested prefixes, and the single-view case reports its rank and nullity.
Held-out cases reserve separate observations and report oracle reprojection
with true local parameters and reprojection with estimated expression and
pose. Both held-out scores use known cameras and the frozen fitted identity.

Fit result schema 3 adds stage reasons, native and scaled step norms, and
linearization diagnostics. Versions 1 and 2 remain readable. Study report
schema 2 adds the protocol, coefficient splits, model regions, numeric
representability, and proposed target checks. All geometry is in meters
without alignment. A fixed-coefficient RMSE floor applies to the full native
coefficient metric. Geometry and reprojection have independent errors.
External predictions report full-native and anatomical group scores.
Fitting studies also score trainable and fixed groups using the recorded
configuration. Existing external evaluation metric keys remain readable in
the expanded report.

Observability reports data-only spectra at initialization and at actual
fitted states. The report lists the rank thresholds, observable condition
number, and implicit zeros for wide Jacobians. Full-space condition numbers
are null for rank-deficient Jacobians. The separately reported prior
curvature is excluded from those spectra. Use `--spectra` to measure both
states for other corrected cases. Proposed clean targets appear alongside
actual measured errors. The clean targets describe known-camera synthetic
recovery. A single 68-point image has at most 136 data rows, and the joint
rank can be lower after removing nuisance directions. The noise sweep
measures sensitivity to its supplied levels and weights. Production weight
selection and estimated-camera recovery require their own validation.

#### Full-native identity and estimated cameras

Select `full-native-camera` explicitly to run the V5 diagnostic. Each primary
case uses nine clean Scenario C observations, a single joint stage with strict
LM, and zero identity and expression regularization. The fitter optimizes
170 identity and 24 expression coefficients. The other 83 identity
coefficients stay at their normal initialized values.

| Condition | Ground-truth identity | Cameras |
| --- | --- | --- |
| A | Full native | Known |
| B | Full native | Estimated |
| C | Head only | Estimated |
| D | Head only | Known |

Each seed and projection model shares one source dataset between the known
and estimated runs for each identity mode. The two identity modes keep the
head draws, expressions, poses, cameras and ordering identical. The historical
reference uses the original nine-view mixed-camera generator configuration
for seed 42. The report checks its source dataset ID and compares its metrics
with V2 when the ID matches. The schedule and convergence settings both
changed since V2.

```sh
docker compose build personalab
docker compose run --rm personalab python -m persona_lab.benchmark.fitting_study \
  --protocol corrected-identity --identity-studies full-native-camera \
  --model-file /cache/gnm/v3_0/gnm_head.npz --device cpu --dtype float64 \
  --seeds 42 43 44 --lm-iterations 0 0 30 --spectra \
  --output outputs/identity-study-v5
```

The command runs 24 primary cases and one historical case.
`outputs/identity-study-v5/report.json` retains the full per-case diagnostics
and `outputs/identity-study-v5/summary.json` links to them and summarizes the
matched comparisons. Paired deltas subtract the first condition from the
second. Geometry errors use unaligned native vertices in meters. The fixed
coefficient floor describes full-native coefficient RMSE. Geometry error
depends on how the deformation bases interact.

Estimated cameras use the observed landmarks and the production
initialization. Perspective focal length is fixed at the prescribed
initialization value. A focal mismatch makes raw metric camera translation
incomparable. Weak-perspective scale comparisons use native model units, and
estimated weak-perspective observations leave absolute metric scale
ambiguous. The report records nulls with reasons for incompatible camera
comparisons. The head-only known-camera control uses the established clean
targets. The other cases compare against those values as numerical references.

Execution, convergence, representability, and observability appear as
separate fields. The expected fixed identity discrepancy leaves the case
incompatible without marking its execution as failed. Nonconverged runs
retain their final metrics when the solver returns a final state.
The report measures the projected gradient at that state independently of
the solver's iteration trace. Estimated-camera gradients use offsets from
the final camera, as in final-state observability.

The detailed report can be compressed losslessly while keeping the
original available for the summary's references.

```sh
gzip -k outputs/identity-study-v5/report.json
```

#### Camera calibration and convergence

The V5.1 `camera-calibration` study uses nine clean, head-only Scenario C
observations for each of seeds 42, 43 and 44. Each seed shares one dataset
across the perspective cases and one across the weak-perspective cases.
Each fit starts independently with the standard estimated-camera initializer.

| Cases | Camera model | Fixed focal factor | Joint LM budgets |
| --- | --- | ---: | --- |
| A1, A2 | Estimated weak perspective | — | 60, 100 |
| B1, B2 | Estimated perspective with calibrated intrinsics | 1.4 | 60, 100 |
| C1, C2 | Estimated perspective with incorrect fixed intrinsics | 1.5 | 60, 100 |

Perspective fits estimate camera translation with a prescribed focal length.
For B1 and B2, the known focal factor is the only supplied calibration.
The strict V5 joint LM criteria apply, with zero regularization for this
diagnostic. Run from the repository root with the verified official asset
at `.gnm-cache/v3_0/gnm_head.npz`.

```sh
docker compose build personalab
docker compose run --rm personalab python -m persona_lab.benchmark.fitting_study \
  --protocol corrected-identity --identity-studies camera-calibration \
  --model-file /cache/gnm/v3_0/gnm_head.npz --device cpu --dtype float64 \
  --seeds 42 43 44 --camera-study-budgets 60 100 --spectra \
  --output outputs/identity-study-v5.1
```

Select `camera-calibration` by itself and supply both budgets with
`--camera-study-budgets`. The command writes 18 cases to
`outputs/identity-study-v5.1/report.json` and one linked row per case to
`outputs/identity-study-v5.1/summary.json`. The comparisons require verified
source and evaluation dataset IDs, identical ordered observations and the
same settings outside the varied factor or budget. Metric deltas subtract
the first condition from the second. For a comparison with V5 condition C,
check both dataset IDs, observation order and the recorded fit configuration
with the joint budget omitted. B1 and B2 use a different focal factor from V5.

The report preserves the complete available solver trace and snapshots at
initialization and termination. Checkpoints at accepted updates 10, 20, 30,
60 and 100 indicate which diagnostics and snapshot metrics are available.
The budget comparisons check initial parameters and objectives, then the
common prefixes of the accepted updates, trial records, linearizations and
linearization-time iteration logs. Structural fields, integers, booleans and
labels match exactly. Floats use relative tolerance `1e-10` and absolute
tolerance `1e-15`. Elapsed times and budget-dependent stage-end rows are
excluded from prefix checks.

Convergence, execution status and the known-camera recovery references are
reported independently. Budget exhaustion can retain final metrics. A solver
failure keeps the original termination reason and available diagnostics.
Geometry is scored in unaligned native meters. With `--spectra`, use the
absolute singular values alongside local numerical ranks and condition numbers
when comparing identity sensitivity.

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

#### Joint identity and focal length estimation

`focal-estimation` is an opt-in CPU float64 corrected-identity study. It uses
Scenario C, the official GNM v3.0 Head asset and seeds 42, 43 and 44. Each
dataset has nine clean observations and head-only ground truth. The model and
scoring retain all 253 identity and 383 expression coefficients, all 68 sparse
landmarks, the existing correspondence mapping and the 17,821-vertex neutral
mesh. The fits train 170 head identity and 24 expression coefficients. The 83
remaining identity coefficients are fixed. The recovery claims apply to this
head-only ground truth.

| Condition | Reference intrinsics | Fitting camera | Initial normalized focal factor |
| --- | --- | --- | ---: |
| A1 | Uniform 1.4 | Fixed focal calibrated control | 1.4 |
| A2 | Uniform 1.4 | Fixed focal incorrect control | 1.5 |
| A3 | Uniform 1.4 | Shared focal | 1.5 |
| A4 | Uniform 1.4 | Per-observation focal | 1.5 |
| A5 | Uniform 1.4 | Shared focal | 1.0 |
| A6 | Uniform 1.4 | Shared focal | 2.0 |
| A7 | Uniform 1.4 | Per-observation focal | 1.0 |
| A8 | Uniform 1.4 | Per-observation focal | 2.0 |
| B1 | Heterogeneous | Fixed focal | 1.5 |
| B2 | Heterogeneous | Shared focal | 1.5 |
| B3 | Heterogeneous | Per-observation focal | 1.5 |
| B4 | Heterogeneous | Known-camera control | Supplied cameras |

The B schedule is `[1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 1.1, 1.5, 1.9]`
in observation order. B2 tests the compatibility of a shared parameter with
heterogeneous data. For each seed and experiment, the conditions share the
same dataset and ordered observations. Each case uses ordinary deterministic
initialization and one joint LM stage. A1 receives reference focal length in
its configuration, B4 receives the known cameras, and all other fits receive
landmarks and their explicit fitting settings.

**Fitting configuration.** The default `focal_mode` is `fixed`, which uses
`perspective_focal_per_short_edge` for the existing uncalibrated perspective
initializer and preserves supplied known cameras. Select `shared` or
`per-observation` with `camera_mode="estimated"`,
`camera_model="perspective"`, the LM solver and a positive
`initial_focal_factor`. The same controls are available in the fit CLI as
`--focal-mode` and `--initial-focal-factor`. An explicit perspective camera
seed supplies rotation, principal point and translation. Its focal length
is initialized from fitting configuration in trainable focal modes.

For a raster short edge $s_i$, initial focal factor $k_0$ and log
coordinate $\delta_i$, both focal axes are $f_i=s_i k_0\exp(\delta_i)$.
The coordinate starts at zero. Shared mode uses one $\delta$ across the
dataset, and per-observation mode gives each view its own $\delta_i$.
Cropping and resizing each image independently
can change its effective normalized focal factor even when the physical
camera is the same. Existing log-depth and lateral translation coordinates,
camera rotation, principal point, root gauge and expression bounds still
apply. The focal coordinate has zero prior curvature. The original
`camera_weight` applies to the existing camera offset coordinates.
Invalid trial intrinsics follow the LM invalid-geometry rejection path.

**Schur system.** Let $A_i$ and $B_i$ be the Jacobians of the normalized,
weighted residual for view $i$ with respect to shared and local coordinates.
The shared block contains identity and the shared log focal when active.
The independent blocks contain expression, pose and translation, plus a log
focal in per-observation mode. The unscaled data curvature blocks are
$H_{gg}=\sum_i A_i^\mathsf{T} A_i$,
$H_{gi}=A_i^\mathsf{T}B_i$, and $H_{ii}=B_i^\mathsf{T}B_i$.
For a projected pixel $u$ and principal point $c$, the isotropic log-focal
derivative is $u-c$. The log-depth derivative is $-(u-c)t_z/Z$ for
translation depth $t_z$ and projected landmark depth $Z$. Their near
collinearity at similar landmark depths motivates the joint diagnostics.
The solver accumulates the corresponding $J^\mathsf{T}r$ blocks and applies
the existing priors, equilibration, active bounds and damping. In the resulting
scaled system the shared solve eliminates the local blocks through
$H_{gg}-\sum_i H_{gi}H_{ii}^{-1}H_{ig}$ with right-hand side
$-g_g+\sum_i H_{gi}H_{ii}^{-1}g_i$. Back-substitution gives the view
updates. The objective reduction and strict gradient and minimum-damping
displacement checks include the active focal coordinates. Fit result schema 4
records their mode and accepted focal trajectory. Fixed-focal results retain
their established on-disk format.

**Observability.** The spectra use the data Jacobian with the fitting residual
weights. Local singular vectors above the recorded relative threshold define
the nuisance projections. The rank threshold uses the largest singular value
of the fixed-nuisance identity Jacobian. The joint shared-focal diagnostic
streams reduced QR across views. The independent-focal diagnostic compresses
each view to identity-only rows plus one row carrying its focal coordinate.
The identity-only reductions keep a fixed-width factor. A complete
independent-focal spectrum uses a compressed matrix of at most
$(p+n)\times(p+n)$ for $p$ identity coefficients and $n$ views.
The weak directions include the corresponding least-squares local log-depth
compensation. Local rank measures sensitivity near the evaluated snapshot.
Absolute calibration and uniqueness remain separate scientific questions.

**Running the study.** Put the pinned model asset at
`.gnm-cache/v3_0/gnm_head.npz` and use a fresh output directory.

```sh
docker compose build personalab
docker compose run --rm personalab python -m persona_lab.benchmark.fitting_study \
  --protocol corrected-identity --identity-studies focal-estimation \
  --model-file /cache/gnm/v3_0/gnm_head.npz --device cpu --dtype float64 \
  --seeds 42 43 44 --focal-study-budget 100 --spectra \
  --output outputs/identity-study-v5.2
```

The command writes all 36 cases to `outputs/identity-study-v5.2/report.json`
and linked case rows, matched comparisons and counts to `summary.json`.
The budget is explicit and accepts 1 through 100 joint updates. Use
`--v51-report outputs/identity-study-v5.1/report.json` to compare the A1
and A2 controls with V5.1 B2 and C2 when both dataset identifiers, observation
order, model asset, generation settings and fit settings match. The study
keeps the strict LM stopping criteria and original exact-recovery targets.
The focal experiment additionally uses a maximum relative error of
`1e-5` as its recovery target. Reported comparable camera translation errors
require matching raster, rotation, principal point and root frame, with
matching fixed focal lengths or trainable focal lengths within the experimental
relative tolerance. Fitted and reference depths are recorded with the gauge
status. Solver failures give a nonzero command exit status, including in
intentionally incompatible conditions. Reports preserve each case and its
termination reason.

#### Focal convergence and initialization stability

The opt-in `focal-convergence` study reuses V5.2's six Scenario C datasets
with nine clean perspective observations each. It runs the 21 compatible
trainable-focal configurations at three explicit accepted-update schedules.

| Strategy | Pose and camera | Identity | Joint |
| --- | ---: | ---: | ---: |
| J100 | 0 | 0 | 100 |
| J200 | 0 | 0 | 200 |
| P100 | 20 | 0 | 80 |

Each fit starts from the ordinary deterministic initializer. P100 first
fits pose, camera translation, depth and the selected focal coordinates.
Identity and expression stay frozen until the joint stage. Both stages use
the same parameter object and the original camera reference coordinates.
The solver initializes LM damping from the configuration at each executed
stage. Stage budgets are independent. A calibration stage that converges or
exhausts its budget can proceed to the joint stage. A failed calibration ends
the fit. The production focal mode is fixed and the production stage budgets
are 30, 70 and 100.

Put the pinned model asset at `.gnm-cache/v3_0/gnm_head.npz` and use a fresh
output directory.

```sh
docker compose build personalab
docker compose run --rm personalab python -m persona_lab.benchmark.fitting_study \
  --protocol corrected-identity --identity-studies focal-convergence \
  --model-file /cache/gnm/v3_0/gnm_head.npz --device cpu --dtype float64 \
  --seeds 42 43 44 --spectra --output outputs/identity-study-v5.3
```

The command runs all 63 cases. Select schedules explicitly with
`--focal-strategies J100 J200`, for example. The reports include the ordered
dataset identifiers, full optimizer histories, per-stage counters and the
physical state and full objective on both sides of the P100 boundary.
Boundary objective evaluations are counted separately from solver evaluations.
P100's calibration metrics describe its frozen identity and expression state.
With `--spectra`, the report includes both the active calibration spectrum
and the prospective joint identity and focal spectra at that state.

The dataset comparison uses independently regenerated V5.2 identifiers from
the pinned source commit and checks observation order. Pass an archived
detailed V5.2 report with `--v52-report` to compare matching J100
configurations, snapshots and recorded trajectories. `summary.json` links
each case to `report.json` and records matched comparisons, statuses and
recovery targets. Solver failures retain their outcomes and give the study
a nonzero exit status.
