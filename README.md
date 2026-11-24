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
docker compose run --rm personalab python -m personalab.gnm_mesh --output outputs/neutral.obj
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
docker compose run --rm personalab python -m personalab.dfl_import \
  --input datasets/studio_light.zip --output outputs/dfl-validation.json
docker compose run --rm personalab python -m personalab.dfl_import \
  --input datasets/aligned --output outputs/dfl-directory-validation.json
```

Omit `--output` to write JSON to stdout. A report path must be new and outside
a directory input. Exit status 0 indicates a successful scan with zero
rejections, 1 indicates one or more sample rejections, and 2 indicates a
dataset or CLI error. Dataset failures produce a `dataset_error` object with
a stable `code` and a `message`. Command and output I/O errors go to stderr.

```python
from personalab.dfl import DFLFaceSample, iter_dataset, read_jpeg, validate_dataset

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
are defined by `personalab.dfl.ErrorCode`. Reports omit source filenames and
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
