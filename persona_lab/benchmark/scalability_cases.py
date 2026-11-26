"""The V6 experiment matrix, shared by the runner and the report reader."""

from itertools import product


STUDY = "focal-scalability"
SCHEMA_VERSION = 5
BASE_COMMIT = "0f9940827d2444c10e6f60d0991ef485065a48db"
SEEDS = (42, 43, 44)
COUNTS = (50, 100, 300, 600)
DEVICES = ("cpu", "cuda")
BATCH_SIZES = (1, 4, 16, 32)
PARENT_SIZE = 600
PRIMARY_BATCH_SIZE = 4
BATCHING_SEED = 42
BATCHING_COUNT = 100


def _case(experiment, seed, count, device, batch_size):
    return {
        "case": f"{experiment}-{device.replace(':', '-')}-{seed}-{count}-batch{batch_size}",
        "experiment": experiment, "scenario": "C", "seed": seed,
        "requested_observations": count, "requested_device": device, "dtype": "float32",
        "batch_size": batch_size, "focal_mode": "per-observation",
        "configured_initial_focal_factor": 1.5, "identity_ground_truth": "head-only",
        "strategy": "J200",
    }


def case_matrix(seeds=SEEDS, counts=COUNTS, devices=DEVICES, batch_size=PRIMARY_BATCH_SIZE, *, primary_only=False):
    """List every configuration, including the shared primary/batching case."""
    primary = [_case("primary", seed, count, device, batch_size)
               for device, count, seed in product(devices, counts, seeds)]
    if primary_only:
        return primary
    batching = []
    for device, size in product(devices, BATCH_SIZES):
        case = _case("batching", BATCHING_SEED, BATCHING_COUNT, device, size)
        if size == batch_size == PRIMARY_BATCH_SIZE and BATCHING_SEED in seeds and BATCHING_COUNT in counts:
            case["reuse_case"] = _case("primary", BATCHING_SEED, BATCHING_COUNT, device, size)["case"]
        batching.append(case)
    return primary + batching
