"""Isolated Facepoints 1.0.0 inference; no GNM or MediaPipe dependency."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
from io import BytesIO
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

REVISION = "88607d8400e2fce5d429dd0bf8e5db9404f852b7"
WEIGHTS_SHA256 = "b37075caa3eaf0d00f23f32925b16ec3a8e2e45c42aa9c00956f94f2574e6526"
CLASSES = {"lmk": 416, "skin": 150, "lid": 36, "teeth_u": 7, "teeth_l": 9, "iris": 2}
COLORS = {"lmk": "#5fbbf5", "skin": "#d6d8de", "lid": "#63e3ae",
          "teeth_u": "#ffd76b", "teeth_l": "#f59e69", "iris": "#e5a0ff"}


def digest(path):
    with Path(path).open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def load_selection(path, dataset):
    if digest(path) != "96524f6a9b93aa9acc94cb1d48816c171474882c35c8259367e7bab73a681a68":
        raise ValueError("Unexpected V7 selection file SHA-256")
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = record["observation_ids"]
    if (record["format"] != "personalab-issue14-v7-selection-v1"
            or not isinstance(ids, list) or len(ids) != 50
            or any(not isinstance(i, str) or not i.startswith("zip:") or len(i) <= 4 for i in ids)
            or len(set(ids)) != 50 or digest(dataset) != record["input_zip_sha256"]):
        raise ValueError("Require the exact verified 50-image V7 ZIP selection")
    return ids


def point_rows(root):
    rows = json.loads((root / "points_gnm.json").read_text(encoding="utf-8"))["points"]
    config = json.loads((root / "model/config.json").read_text(encoding="utf-8"))
    if (len(rows) != 620 or [r["index"] for r in rows] != list(range(620))
            or Counter(r["class"] for r in rows) != CLASSES
            or config["classes"] != [r["class"] for r in rows]):
        raise ValueError("Facepoints output ordering/classes disagree with the release")
    return rows


def validate_predictions(result, width, height):
    arrays = {key: np.asarray(result[key], dtype=np.float64)
              for key in ("points", "std", "300w", "wflw")}
    for key, shape in (("points", (620, 2)), ("std", (620,)), ("300w", (68, 2)), ("wflw", (98, 2))):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid Facepoints {key} output")
    if not (arrays["std"] > 0).all():
        raise ValueError("Facepoints std must be strictly positive")
    p = arrays["points"]
    arrays["validity"] = ((p >= 0) & (p < (width, height))).all(axis=1)
    return arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "selection", "release", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = p.parse_args()
    import torch
    root = args.release.resolve()
    # Verify the actual checkout, not merely a claimed revision string.
    cmd = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    if subprocess.check_output(cmd + ["rev-parse", "HEAD"], text=True).strip() != REVISION:
        raise ValueError("Unexpected Facepoints source revision")
    if subprocess.check_output(cmd + ["status", "--porcelain", "--untracked-files=all"], text=True).strip():
        raise ValueError("Facepoints checkout is not clean")
    if digest(root / "model/model.safetensors") != WEIGHTS_SHA256:
        raise ValueError("Facepoints weight SHA-256 mismatch")
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    import facepoints
    if Path(facepoints.__file__).resolve().parent != root:
        raise ValueError("Imported Facepoints package is not the verified checkout")
    ids = load_selection(args.selection, args.input)
    rows = point_rows(root)
    args.output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    detector = facepoints.Detector(str(root / "model"), device=args.device)
    if str(detector.device) != args.device:
        raise RuntimeError("Facepoints device differs from the requested device")
    if any(str(t.device) != args.device for t in detector.net.parameters()):
        raise RuntimeError("Facepoints network parameters are not on the requested device")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    initialization = perf_counter() - start
    attempts = []
    with ZipFile(args.input) as archive:
        names = Counter(archive.namelist())
        for index, identifier in enumerate(ids):
            name = identifier[4:]
            attempt = {"observation_id": identifier, "image_ref": name}
            before = perf_counter()
            try:
                if names[name] != 1 or archive.getinfo(name).file_size > 32 * 1024 * 1024:
                    raise ValueError("Missing, duplicate or oversized selected ZIP member")
                data = archive.read(name)
                attempt["image_sha256"] = sha256(data).hexdigest()
                with Image.open(BytesIO(data), formats=["JPEG"]) as source:
                    if source.width * source.height > 16 * 1024 * 1024:
                        raise ValueError("Oversized JPEG")
                    image = source.convert("RGB")
                width, height = image.size
                attempt["image_size"] = [width, height]
                # Explicit DFL protocol: the already aligned raster supplies the
                # initial box. No extra detector and no alternative box on failure.
                box = facepoints.square_box(0, 0, width, height, factor=1.0)
                attempt["initial_box"] = list(box)
                out = detector.detect(np.asarray(image)[:, :, ::-1].copy(), box, passes=3)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                arrays = validate_predictions(out, width, height)
                attempt["inference_seconds"] = perf_counter() - before
                filename = f"observation_{index:05d}.npz"
                np.savez_compressed(args.output / filename, **arrays)
                attempt["arrays"] = filename
                attempt["arrays_sha256"] = digest(args.output / filename)
                attempt["valid"] = int(arrays["validity"].sum())
                attempt["error"] = None
                draw = ImageDraw.Draw(image)
                for point, row, valid in zip(arrays["points"], rows, arrays["validity"], strict=True):
                    if valid:
                        x, y = point
                        draw.ellipse((x-2, y-2, x+2, y+2), fill=COLORS[row["class"]])
                image.save(args.output / f"observation_{index:05d}.png")
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                attempt["error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(attempt)
            print(f"{index+1}/50 {identifier}: {attempt['error'] or 'OK'}", flush=True)
    report = {"format": "personalab-facepoints-inference-v1", "revision": REVISION,
              "weights_sha256": WEIGHTS_SHA256, "points_sha256": digest(root / "points_gnm.json"),
              "config_sha256": digest(root / "model/config.json"), "selection_sha256": digest(args.selection),
              "dataset_sha256": digest(args.input), "device": args.device, "torch": str(torch.__version__),
              "numpy": np.__version__, "initial_box_protocol": "entire aligned DFL raster; factor 1.0",
              "passes": 3, "cuda_autocast": args.device.startswith("cuda"),
              "initialization_seconds": initialization, "elapsed_seconds": perf_counter()-start,
              "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
              "attempts": attempts}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if any(a["error"] is not None for a in attempts):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
