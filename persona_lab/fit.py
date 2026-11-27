"""Fit one shared GNM v3 identity from a DeepFaceLab dataset or benchmark subject."""

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from .benchmark.configuration import torch_options
from .benchmark.dataset import load_dataset
from .benchmark.model import load_benchmark_model
from .benchmark.predictions import write_predictions
from .benchmark.storage import publish_directory, read_json
from .dfl import DFLFaceSample, iter_dataset
from .fitting import FitConfig, fit_identity
from .fitting.configuration import FOCAL_MODES, SOLVERS
from .fitting.diagnostics import write_neutral_mesh, write_overlays
from .fitting.results import FAILED_STATUSES, _write_result
from .observations import LandmarkObservation


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="DeepFaceLab ZIP, JPEG, or directory")
    inputs.add_argument("--benchmark", type=Path, help="Issue #4 benchmark dataset directory")
    parser.add_argument("--subject", help="explicit benchmark subject ID")
    parser.add_argument("--max-images", type=_positive_int)
    parser.add_argument("--require-images", type=_positive_int, help="reject unless exactly N observations are selected")
    parser.add_argument("--require-convergence", action="store_true", help="return failure unless final stage converges; retain the result")
    parser.add_argument("--selection", choices=("first", "sample"), default="first")
    parser.add_argument("--observation-id", action="append", help="ordered benchmark observation ID; repeatable")
    parser.add_argument("--model-version", choices=("v3",), default="v3")
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--config", type=Path, help="JSON FitConfig")
    parser.add_argument("--solver", choices=SOLVERS, help="fitting solver (default: Levenberg–Marquardt)")
    parser.add_argument("--fp64-qr", action="store_true", help="explicit native FP64 square-root LM with staged priors")
    parser.add_argument("--jacobian-rows", type=_positive_int, help="LM residual rows per autograd batch")
    parser.add_argument("--camera-mode", choices=("known", "estimated"))
    parser.add_argument("--camera", choices=("weak-perspective", "perspective"))
    parser.add_argument("--focal-mode", choices=FOCAL_MODES)
    parser.add_argument("--initial-focal-factor", type=float, help="initial perspective focal length per short edge")
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--iterations", type=int, nargs=3, metavar=("A", "B", "C"))
    parser.add_argument("--learning-rate", type=float, help="override all Adam stage learning rates")
    parser.add_argument("--identity-weight", type=float)
    parser.add_argument("--expression-weight", type=float)
    parser.add_argument("--pose-weight", type=float)
    parser.add_argument("--camera-weight", type=float)
    parser.add_argument("--train-identity", nargs="*", metavar="NAME")
    parser.add_argument("--train-expression", nargs="*", metavar="NAME")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-every", type=_positive_int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="separate Issue #4 prediction directory")
    parser.add_argument("--neutral-mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlays", type=_positive_int, help="number of before/after overlay pairs")
    return parser


def _config(args):
    values = read_json(args.config) if args.config else {}
    if args.input and "camera_mode" not in values:
        values["camera_mode"] = "estimated"
    for name in (
        "camera_mode", "batch_size", "identity_weight", "expression_weight", "pose_weight",
        "camera_weight", "train_identity", "train_expression", "seed", "log_every", "solver",
        "focal_mode", "initial_focal_factor",
    ):
        if (value := getattr(args, name)) is not None:
            values[name] = value
    if args.camera is not None:
        values["camera_model"] = args.camera
    if args.fp64_qr or args.jacobian_rows is not None:
        lm = values.get("lm", {})
        if not isinstance(lm, dict):
            raise ValueError("LM configuration must be a JSON object")
        values["lm"] = dict(lm)
        if args.fp64_qr:
            values["lm"]["fp64_square_root"] = True
        if args.jacobian_rows is not None:
            values["lm"]["jacobian_rows"] = args.jacobian_rows
    config = FitConfig.from_dict(values)
    if config.lm.fp64_square_root and args.dtype != "float64":
        raise ValueError("FP64 square-root requires --dtype float64")
    if args.input and config.camera_mode != "estimated":
        raise ValueError("DeepFaceLab fitting requires estimated-camera mode")
    if config.camera_mode == "estimated" and config.camera_model is None:
        raise ValueError("Select an estimated camera model with --camera or camera_model")
    stages = config.stages
    if args.iterations is not None:
        if len(stages) != len(args.iterations):
            raise ValueError("--iterations requires a three-stage configuration")
        stages = tuple(replace(stage, iterations=steps)
                       for stage, steps in zip(stages, args.iterations, strict=True))
    if args.learning_rate is not None:
        if config.solver != "adam":
            raise ValueError("--learning-rate requires --solver adam")
        stages = tuple(replace(stage, learning_rate=args.learning_rate, group_rates=())
                       for stage in stages)
    return replace(config, stages=stages)


def _selection(ids, args, seed):
    limit = len(ids) if args.max_images is None else min(args.max_images, len(ids))
    if args.selection == "sample" and limit < len(ids):
        order = np.random.default_rng(seed).choice(len(ids), size=limit, replace=False)
        return tuple(ids[int(index)] for index in sorted(order))
    return ids[:limit]


def _real_input(args, config, dtype, device):
    samples = []
    accepted, rejected = 0, 0
    rng = np.random.default_rng(config.seed)
    for item in iter_dataset(args.input):
        if not isinstance(item, DFLFaceSample):
            rejected += 1
            continue
        accepted += 1
        if args.max_images is None or len(samples) < args.max_images:
            samples.append((accepted, item))
        elif args.selection == "sample":
            slot = int(rng.integers(accepted))
            if slot < args.max_images:
                samples[slot] = accepted, item
    selected = tuple(sample for _, sample in sorted(samples))
    observations = tuple(
        LandmarkObservation.from_dfl(sample, dtype=dtype, device=device) for sample in selected
    )
    return observations, selected, {
        "source": str(args.input), "accepted": accepted, "rejected": rejected,
        "selected": len(observations), "selection": args.selection,
    }


def _run(args):
    config = _config(args)
    dtype, device = torch_options(args.dtype, args.device)
    if args.benchmark and args.subject is None:
        raise ValueError("--benchmark requires --subject")
    if args.input and (args.subject is not None or args.predictions is not None):
        raise ValueError("--subject and --predictions require --benchmark")
    if args.observation_id and (not args.benchmark or args.max_images is not None or args.selection != "first"):
        raise ValueError("--observation-id requires --benchmark without sampling or --max-images")
    for path in (args.output, args.predictions):
        if path is not None and (path.exists() or path.is_symlink()):
            raise FileExistsError(f"Output already exists: {path}")
    if args.predictions:
        output, predictions = args.output.resolve(), args.predictions.resolve()
        if output == predictions or output in predictions.parents or predictions in output.parents:
            raise ValueError("Fitting and prediction outputs require independent directories")

    started = perf_counter()
    model = load_benchmark_model(args.model_file, dtype=dtype, device=device)
    load_seconds = perf_counter() - started
    samples = ()
    if args.benchmark:
        dataset = load_dataset(args.benchmark)
        if model.metadata != dataset.model_metadata:
            raise ValueError("Benchmark and fitting model provenance differ")
        all_ids = dataset.observation_ids(args.subject)
        ids = tuple(args.observation_id) if args.observation_id else _selection(all_ids, args, config.seed)
        evaluation_data = dataset.select_observations(args.subject, ids)
        observations = evaluation_data.fitting_observations(
            args.subject, dtype=dtype, device=model.reference.device,
            include_known_cameras=config.camera_mode == "known",
        )
        source_info = {
            "dataset_id": dataset.dataset_id, "source_dataset_id": dataset.source_dataset_id,
            "evaluation_dataset_id": evaluation_data.dataset_id, "subject": args.subject,
            "selected": len(ids), "selection": "explicit" if args.observation_id else args.selection,
        }
    else:
        observations, samples, source_info = _real_input(args, config, dtype, model.reference.device)
    if not observations:
        raise ValueError(f"No fitting observations were selected: {source_info}")

    if args.require_images is not None and len(observations) != args.require_images:
        raise ValueError(f"Expected exactly {args.require_images} observations, selected {len(observations)}")

    def log(item):
        print(f"{item.stage} step {item.stage_iteration} (total {item.iteration}) "
              f"loss={item.losses.total:.6g} RMS={item.reprojection.rms_px:.4f} px", flush=True)

    result = fit_identity(model, observations, config, progress=log)
    runtime = result.runtime | {
        "input": source_info, "model_loading_seconds": load_seconds,
    }
    result = replace(result, runtime=runtime)
    with publish_directory(args.output, completion="result.json") as stage:
        if result.final is not None:
            if args.neutral_mesh:
                write_neutral_mesh(stage / "neutral.obj", model, result)
            if args.overlays:
                write_overlays(stage, model, result, observations, args.overlays, samples)
        _write_result(stage, result)
    if result.status in FAILED_STATUSES:
        print(f"Failed fit: {result.reason}", file=sys.stderr)
        return 1

    if args.require_convergence and result.status != "converged":
        print(f"Convergence required: {result.status}: {result.reason}", file=sys.stderr)
        return 1

    if args.predictions:
        write_predictions(args.predictions, evaluation_data, {
            args.subject: result.to_prediction(dtype=evaluation_data.config.dtype),
        })
    initial_rms = result.initial_reprojection.overall.rms_px
    final_rms = result.final_reprojection.overall.rms_px
    print(f"{result.status} with {len(observations)} observations: "
          f"RMS {initial_rms:.4f} -> {final_rms:.4f} px")
    return 0


def main(argv: Sequence[str] | None = None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
