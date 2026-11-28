"""Extract reproducible MediaPipe face observations from image files or DFL datasets."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
from itertools import batched, islice
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
import torch

from .benchmark.storage import publish_directory, write_json
from .coordinates import ImageSize
from .dense_dataset import ExtractionAttempt, _write_rows
from .dense_observations import DenseObservation, ImageProvenance
from .dfl import DFLFaceSample, SampleRejection, iter_dataset
from .dfl.dataset import _ImageSource, _MAX_DIRECTORY_DEPTH, _MAX_DIRECTORY_ENTRIES, _MAX_REFERENCE_BYTES
from .dfl.errors import DFLError
from .fitting.monitoring import peak_rss_bytes
from .mediapipe_face import (
    ASSET, PREPROCESSING, PROVIDER, PROVIDER_VERSION, SCHEMAS, _REGION_COLORS, MediaPipeFaceLandmarker,
)
from .observations import LandmarkObservation


_IMAGE_FORMATS = {".jpeg": "JPEG", ".jpg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
_MAX_PIXELS = 64 * 1024 * 1024
_UNSUPPORTED = ("ears", "pupil_boundary", "visible_sclera", "upper_teeth", "lower_teeth",
                "dental_contours", "gingival_boundaries", "tongue", "other_mouth_interior")


def _ordinary_sources(path):
    if not path.is_symlink() and path.is_dir():
        count = 0

        def on_error(error):
            raise error

        for folder, dirs, files in path.walk(on_error=on_error, follow_symlinks=False):
            dirs.sort()
            count += len(dirs) + len(files)
            if count > _MAX_DIRECTORY_ENTRIES or len(folder.relative_to(path).parts) > _MAX_DIRECTORY_DEPTH:
                raise ValueError("Image directory exceeds the supported scan limits")
            for name in sorted(files):
                if Path(name).suffix.lower() in _IMAGE_FORMATS:
                    reference = (folder / name).relative_to(path).as_posix()
                    yield _ImageSource(path, "directory", reference)
    elif path.suffix.lower() in _IMAGE_FORMATS:
        yield _ImageSource(path, "file", path.name)
    else:
        raise ValueError("Image input must be JPEG, PNG, WebP or an image directory; use dfl for a DFL ZIP")


def _selected(items, limit, selection, seed):
    if selection == "first":
        yield from items if limit is None else islice(items, limit)
        return
    if limit is None:
        raise ValueError("Sample selection requires an explicit maximum image count")
    rng = np.random.default_rng(seed)
    selected = []
    for index, item in enumerate(items):
        if index < limit:
            selected.append((index, item))
        else:
            slot = int(rng.integers(index + 1))
            if slot < limit:
                selected[slot] = index, item
    yield from (item for _, item in sorted(selected))


def _decode(data, reference):
    expected = _IMAGE_FORMATS[Path(reference).suffix.lower()]
    with Image.open(BytesIO(data), formats=[expected]) as image:
        if image.width * image.height > _MAX_PIXELS:
            raise ValueError("Decoded image exceeds pixel limit")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Animated image inputs are unsupported")
        return image.convert("RGB")


def save_dense_overlay(path, image, observation, *, projected=None, name="measurements", dfl=None):
    """Draw source coordinates, confidence where observed, and optional model predictions."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    if image.size != (observation.image_size.width, observation.image_size.height):
        raise ValueError("Diagnostic raster dimensions do not match its observation")
    pixels = observation.landmarks.detach().cpu().tolist()
    validity = observation.validity.detach().cpu().tolist()
    visibility = None if observation.visibility is None else observation.visibility.detach().cpu().tolist()
    confidence = None if observation.confidence is None else observation.confidence.detach().cpu().tolist()
    for index, (x, y) in enumerate(pixels):
        if not validity[index]:
            continue
        point = observation.schema.points[index]
        color = _REGION_COLORS.get(point.region, (177, 199, 224))
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        if visibility is not None and float(visibility[index]) >= 0:
            visible = round(255 * float(visibility[index]))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(255 - visible, visible, 0))
        if confidence is not None and float(confidence[index]) >= 0:
            radius = 3 + round(4 * float(confidence[index]))
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), outline=(238, 238, 238))
    if dfl is not None:
        for x, y in dfl.detach().cpu().tolist():
            draw.rectangle((x - 2, y - 2, x + 2, y + 2), outline=(255, 255, 255))
    if projected is not None:
        for index, (x, y) in projected.items():
            if validity[index]:
                draw.line((*pixels[index], x, y), fill=(252, 111, 92))
            draw.line((x - 3, y, x + 3, y), fill=(255, 255, 255))
            draw.line((x, y - 3, x, y + 3), fill=(255, 255, 255))
    title = f"{name} · {sum(validity)}/{len(validity)} valid"
    draw.text((5, 5), title, fill="white", stroke_width=2, stroke_fill="black")
    image.save(path)


def observation_statistics(obs, mapped_slots=()):
    mapped_slots = set(mapped_slots)
    regions = {}
    missing, unsupported = [], []
    for index, point in enumerate(obs.schema.points):
        valid = bool(obs.validity[index])
        mapped = index in mapped_slots
        entry = regions.setdefault(point.region, Counter())
        entry.update(slots=1, valid=int(valid), mapped=int(mapped), valid_mapped=int(valid and mapped))
        if not valid:
            missing.append(point.identifier)
        if not mapped:
            unsupported.append(point.identifier)
    return {
        "valid": int(obs.validity.sum()), "mapped": len(mapped_slots),
        "valid_mapped": sum(region["valid_mapped"] for region in regions.values()),
        "missing_point_ids": missing, "unmapped_point_ids": unsupported,
        "score_slots": {
            name: 0 if getattr(obs, name) is None else int((getattr(obs, name) >= 0).sum())
            for name in ("confidence", "visibility", "presence")
        },
        "regions": regions,
    }


def extract_dataset(path, output, *, input_kind, schema_name, provider, asset_version, asset_path,
                    device, batch_size, max_images=None, selection="first", seed=42, overlays=0,
                    score_threshold=0.5):
    """Write observations and failure/coverage diagnostics even for zero detections."""
    if provider != PROVIDER or schema_name not in SCHEMAS:
        raise ValueError("Select an implemented provider and schema explicitly")
    if (input_kind, schema_name) not in (("images", "mediapipe-face-478"), ("dfl", "mediapipe-face-478-dfl")):
        raise ValueError("Input kind and the schema's source or aligned raster must agree")
    if type(batch_size) is not int or batch_size < 1 or (
        max_images is not None and (type(max_images) is not int or max_images < 1)
    ) or type(seed) is not int or seed < 0 or type(overlays) is not int or overlays < 0:
        raise ValueError("Require a positive batch size and limit, and nonnegative seed and overlay count")
    if selection not in ("first", "sample") or (selection == "sample" and max_images is None):
        raise ValueError("Select first or sample with an explicit maximum image count")
    schema = SCHEMAS[schema_name]
    path = Path(path).absolute()
    configuration = {
        "input": str(path), "input_kind": input_kind, "device": device, "application_batch_size": batch_size,
        "native_inference_batch_size": 1, "max_images": max_images, "selection": selection,
        "seed": seed, "overlays": overlays, "score_threshold": score_threshold, "num_faces": 2,
        "model_path": str(Path(asset_path).absolute()), "preprocessing": PREPROCESSING,
    }
    start = perf_counter()
    report_rows = []
    aggregates = {}
    with publish_directory(output, completion="manifest.json") as stage:
        with MediaPipeFaceLandmarker(asset_path, asset_version=asset_version, device=device,
                                      score_threshold=score_threshold) as detector:
            initialization_seconds = perf_counter() - start
            items = iter_dataset(path) if input_kind == "dfl" else _ordinary_sources(path)
            candidates = _selected(items, max_images, selection, seed)
            overlay_count = 0
            input_error = None

            def scan():
                nonlocal input_error
                try:
                    yield from candidates
                except (DFLError, OSError, ValueError) as exc:
                    input_error = f"{type(exc).__name__}: {exc}"
                    if not report_rows:
                        raise

            def attempts():
                nonlocal overlay_count
                for group in batched(scan(), batch_size):
                    for item in group:
                        before = perf_counter()
                        obs, image, dfl = None, None, None
                        size, digest, error = None, None, None
                        if isinstance(item, SampleRejection):
                            identifier, reference = item.sample_id, item.image_ref
                            size = None if item.image_size is None else ImageSize(*item.image_size)
                            error = "; ".join(f"{issue.code}: {issue.message}" for issue in item.issues)
                        else:
                            if isinstance(item, DFLFaceSample):
                                identifier, reference = item.sample_id, item.image_ref
                                size = ImageSize(*item.image_size)
                                dfl = LandmarkObservation.from_dfl(item, dtype=torch.float64)
                            else:
                                identifier = reference = item.reference
                            try:
                                if len(os.fsencode(reference)) > _MAX_REFERENCE_BYTES:
                                    raise ValueError("Image relative path exceeds limit")
                                data = item.read_image_bytes() if dfl is not None else item.read_bytes()
                                digest = sha256(data).hexdigest()
                                image = _decode(data, reference)
                                raster = ImageSize(*image.size)
                                if size is not None and size != raster:
                                    raise ValueError("DFL raster dimensions disagree with the decoded image")
                                size = raster
                                values = detector.extract(np.asarray(image), size)
                                obs = DenseObservation(
                                    observation_id=identifier, schema=schema, provider=provider,
                                    provider_version=PROVIDER_VERSION, asset=ASSET,
                                    source=ImageProvenance(reference, digest, size if dfl is None else None),
                                    extraction=configuration, image_size=size,
                                    alignment=None if dfl is None else dfl.alignment, **values,
                                )
                            except (DFLError, OSError, Image.DecompressionBombError, ValueError, RuntimeError) as exc:
                                error = f"{type(exc).__name__}: {exc}"
                        row = {
                            "observation_id": identifier, "image_ref": reference,
                            "extraction_seconds": perf_counter() - before, "error": error,
                            "dfl_valid": None if dfl is None else int(dfl.validity.sum()),
                            "dfl_mapped": None if dfl is None else int(dfl.supported_mask.sum()),
                        }
                        if obs is not None:
                            stats = observation_statistics(obs)
                            row |= stats
                            for region, counts in stats["regions"].items():
                                aggregates.setdefault(region, Counter()).update(counts)
                            if overlay_count < overlays:
                                overlay_name = f"observation_{len(report_rows):05d}.png"
                                save_dense_overlay(stage / overlay_name, image, obs,
                                                   dfl=None if dfl is None else dfl.landmarks)
                                row["overlay"] = overlay_name
                                overlay_count += 1
                        report_rows.append(row)
                        yield ExtractionAttempt(identifier, reference, obs, error, size, digest)

            _write_rows(stage, schema, provider, PROVIDER_VERSION, ASSET, configuration, attempts())
        report = {
            "attempted": len(report_rows), "succeeded": sum(row["error"] is None for row in report_rows),
            "failed": sum(row["error"] is not None for row in report_rows),
            "input_error": input_error,
            "initialization_seconds": initialization_seconds, "elapsed_seconds": perf_counter() - start,
            "peak_process_rss_bytes": peak_rss_bytes(), "peak_gpu_bytes": None,
            "mapping": "No reviewed GNM mapping supplied to extraction",
            "unsupported_anatomy": list(_UNSUPPORTED), "regions": aggregates,
            "region_colors_rgb": _REGION_COLORS, "attempts": report_rows,
        }
        write_json(stage / "extraction.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-kind", required=True, choices=("images", "dfl"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", required=True, choices=(PROVIDER,))
    parser.add_argument("--schema", required=True, choices=SCHEMAS)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--model-file", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", required=True, type=int, help="application batches; the task detects one image at a time")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--selection", choices=("first", "sample"), default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlays", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        report = extract_dataset(
            args.input, args.output, input_kind=args.input_kind, schema_name=args.schema,
            provider=args.provider, asset_version=args.model_version, asset_path=args.model_file,
            device=args.device, batch_size=args.batch_size, max_images=args.max_images,
            selection=args.selection, seed=args.seed, overlays=args.overlays, score_threshold=args.score_threshold,
        )
    except (OSError, DFLError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"{report['succeeded']} of {report['attempted']} extractions succeeded -> {args.output}")
    if report["failed"] or report["input_error"]:
        parser.exit(1, "Inspect extraction.json for each failed attempt.\n")


if __name__ == "__main__":
    main()
