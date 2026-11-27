"""Create an independent compensated FP32 high/low reference from verified V6 truth.

Only this offline synthetic generator accesses reference parameters. The fitter
receives high/low pixels, masks and weights, never the generating parameters.
The original reference is read-only and the destination must be new.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch

from ..fitting.precision_geometry import landmark_pair, projected_pair
from ..fitting.parameters import stack_cameras
from ..gnm_geometry import SparseHeadLandmarks
from .dataset import camera_from_record, write_dataset
from .model import MODEL_SHA256, load_benchmark_model, in_frame
from .scalability_data import (REFERENCE_SCHEMA, PRECISION_REFERENCE_SCHEMA, SCHEDULE,
                               file_digests, load_parent)
from .storage import read_json, write_json


@torch.no_grad()
def generate_precision_parent(parent, source_manifest, model, destination, *, batch_size=8):
    if parent.has_precision_pixels or model.reference.dtype != torch.float32 or model.reference.device.type != "cpu":
        raise ValueError("Precision archive generation requires the original raster reference and a CPU FP32 model")
    if parent.model_metadata != model.metadata or not np.array_equal(parent.triangles, model.triangles):
        raise ValueError("Reference model provenance mismatch")
    subject, = parent.subject_ids
    truth = parent.ground_truth(subject)
    arrays = truth.arrays
    if not np.array_equal(arrays["clean"], arrays["observed"]) or np.any(arrays["offset_px"]):
        raise ValueError("Precision control requires an uncorrupted reference")
    # No correspondence removal or visibility change is permitted.
    if not arrays["supported"].all():
        raise ValueError("Precision control requires all original supported landmarks")
    sparse = SparseHeadLandmarks(model.head)
    count = len(truth.observations)
    high, low = [], []
    indices = torch.tensor([item.gnm_index for item in parent.correspondences.entries], dtype=torch.long)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        cameras = stack_cameras([camera_from_record(item["camera"], dtype=torch.float32, device=torch.device("cpu"))
                                 for item in truth.observations[start:stop]])
        landmarks = landmark_pair(sparse, model.tensor(arrays["identity"]),
                                  *(model.tensor(arrays[name][start:stop]) for name in ("expression", "rotations", "translation")))
        h, l = projected_pair(landmarks, cameras.rotation, cameras.translation, cameras.focal_length, cameras.principal_point)
        h, l = h.index_select(1, indices), l.index_select(1, indices)
        if not np.array_equal(in_frame(h, cameras.image_size, torch.from_numpy(arrays["supported"][start:stop])).numpy(),
                              arrays["in_frame"][start:stop]):
            raise ValueError("Precision projection changed the archived visibility mask")
        high.append(h.numpy()); low.append(l.numpy())
    high, low = np.concatenate(high), np.concatenate(low)
    new_arrays = arrays | {"clean": high, "observed": high.copy(), "pixel_low": low}
    return write_dataset(destination, config=parent.config, model=model,
                         subjects=(replace(truth, arrays=new_arrays),), mapping=parent.correspondences,
                         runtime={"python": platform.python_version(), "torch": str(torch.__version__),
                                  "device": "cpu", "dtype": "float32", "precision_batch_size": batch_size},
                         generator=source_manifest["generator"] | {
                             "pixel_representation": "compensated-fp32-high-low-v1",
                             "precision_source_dataset_id": parent.dataset_id,
                             "precision_source_files_sha256": file_digests(Path(source_manifest["verified_directory"]))},
                         precise=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive batch size and unique seeds required")
    if Path("outputs").resolve() in (args.output.resolve(), *args.output.resolve().parents):
        parser.error("Synthetic reference archives must not be stored in outputs")
    if args.output.resolve() == args.source.resolve() or args.source.resolve() in args.output.resolve().parents:
        parser.error("Precision archives must be separate from the immutable source")
    reference = read_json(args.source / "reference.json")
    if (reference.get("schema") != REFERENCE_SCHEMA or reference.get("model_asset_sha256") != MODEL_SHA256
            or reference.get("focal_schedule") != list(SCHEDULE)):
        raise ValueError("Expected an original verified V6 reference index")
    for seed in args.seeds:
        if str(seed) not in reference["parents"]:
            raise ValueError(f"Original reference for seed {seed} is missing")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_benchmark_model(args.model_file, dtype="float32", device="cpu")
    parents = {}
    for seed in args.seeds:
        started = perf_counter()
        print(f"Generating compensated FP32 high/low reference seed {seed}", flush=True)
        source = args.source / f"seed-{seed}"
        parent = load_parent(source, seed, reference["parents"][str(seed)])
        manifest = read_json(source / "manifest.json") | {"verified_directory": str(source)}
        destination = args.output / f"seed-{seed}"
        dataset = generate_precision_parent(parent, manifest, model, destination, batch_size=args.batch_size)
        parents[str(seed)] = {"dataset_id": dataset.dataset_id, "files_sha256": file_digests(destination),
                              "pixel_representation": "compensated-fp32-high-low-v1",
                              "source_dataset_id": parent.dataset_id,
                              "generation_seconds": perf_counter() - started}
    write_json(args.output / "reference.json", {"schema": PRECISION_REFERENCE_SCHEMA,
               "focal_schedule": SCHEDULE, "model_asset_sha256": MODEL_SHA256, "parents": parents})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
