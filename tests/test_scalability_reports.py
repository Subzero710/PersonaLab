"""Exercise report accounting and comparison with small constructed records."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from persona_lab.benchmark import scalability_report
from persona_lab.benchmark.scalability_cases import BASE_COMMIT, SCHEMA_VERSION, STUDY, case_matrix
from persona_lab.benchmark.storage import read_json


DIGEST = "1" * 64
FILES = dict.fromkeys(("manifest.json", "topology.npz", "subject_0000.npz"), DIGEST)
NOT_REQUESTED = {"initial": {"status": "not_requested"}, "final": {"status": "not_requested"}}


def _blocked(case, index):
    return case | {
        "config": {"batch_size": case["batch_size"]},
        "requested_stage_budgets": {"pose_camera": 0, "identity": 0, "joint": 200},
        "report_case": f"report.json#/cases/{index}",
        "dataset_verified": False, "actual_observations": None,
        "actual_execution_device": None, "execution_status": "not_run", "termination_status": None,
        "failure": {"kind": "cuda_unavailable", "reason": "Test device unavailable"},
        "spectra_requested": False, "observability_status": NOT_REQUESTED,
    }


def _completed(device, index=0):
    case = case_matrix(seeds=(42,), counts=(50,), devices=(device,))[0]
    row = _blocked(case, index)
    state = {
        "identity": [0.0, 0.5], "expression": [[0.1]], "rotations": [[[0.0, 0.0, 0.0]]],
        "translation": [[0.0, 0.0, 0.0]],
        "cameras": [{"type": "perspective", "image_size": [100, 100], "focal_length": [150.0, 150.0],
                     "translation": [0.0, 0.0, 1.0], "rotation": [1.0, 0.0, 0.0],
                     "principal_point": [50.0, 50.0]}],
    }
    metrics = {"valid_count": 68, "excluded_count": 0, "mean_px": 0.0, "median_px": 0.0, "rms_px": 0.0}
    return row | {
        "dataset_verified": True, "actual_observations": 50, "reference_files_sha256": FILES,
        "source_dataset_id": DIGEST, "dataset_id": DIGEST, "content_sha256": DIGEST,
        "identity_sha256": DIGEST, "fitter_input_sha256": DIGEST,
        "observation_ids": [f"observation-{index}" for index in range(50)],
        "actual_execution_device": "cpu" if device == "cpu" else "cuda:0",
        "actual_dtype": "float32", "model": {"asset_sha256": DIGEST},
        "failure": None, "execution_status": "completed", "termination_status": "converged",
        "parameter_snapshots": {"initial": deepcopy(state), "final": deepcopy(state)},
        "optimization": {"initial_objective": {"total": 1.0}, "final_objective": {"total": 0.0},
                         "accepted_updates": 3, "rejected_trials": 2, "jacobian_evaluations": 4},
        "recovery": {"exact_compatible_recovery": True},
        "timing": {"optimization": 16.0 if device == "cpu" else 4.0},
        "final": {
            "neutral_vertices_m": [[0.1, 0.2, 0.3]],
            "training_reprojection_and_nuisance_recovery": {
                "aggregate": {"observed": metrics, "clean": metrics}, "observations": [],
            },
        },
    }


def _header(device):
    return {
        "requested_device": device, "schema_version": SCHEMA_VERSION,
        "protocol": "corrected-identity", "requested_identity_studies": [STUDY], "dtype": "float32",
        "reference_base_commit": BASE_COMMIT, "spectra_requested": False,
        "model_asset_sha256": DIGEST, "focal_schedule": [1.0] * 600,
    }


class ScalabilityReportTests(unittest.TestCase):
    def test_report_and_summary_resolve_the_shared_fit(self):
        matrix = case_matrix(devices=("cuda",))
        detailed, pointers = [], {}
        for index, case in enumerate(matrix):
            if "reuse_case" in case:
                detailed.append(case | {"report_case": pointers[case["reuse_case"]]})
            else:
                row = _blocked(case, index)
                pointers[case["case"]] = row["report_case"]
                detailed.append(row)
        with TemporaryDirectory() as temp:
            output = Path(temp)
            summary = scalability_report.write_reports(output, _header("cuda"), iter(detailed))
            self.assertEqual(summary, read_json(output / "summary.json"))
            report = read_json(output / "report.json")
        self.assertEqual(len(summary["cases"]), 16)
        self.assertEqual(report["cases"], detailed)
        self.assertEqual(len(summary["batching_consistency"]), 3)
        for row in summary["cases"]:
            self.assertIsNone(row["accepted_updates"])
            self.assertIsNone(row["memory"])
            self.assertEqual(row["failure"]["kind"], "cuda_unavailable")
            pointer = int(row["report_case"].rsplit("/", 1)[1])
            self.assertEqual(report["cases"][pointer]["case"], row["reuse_case"] or row["case"])
        self.assertTrue(all(group["unexecuted_count"] == 3 for group in summary["primary_by_observation_count"]))

    def test_compatible_results_report_a_matched_measured_speedup(self):
        cpu = _header("cpu") | {"cases": [_completed("cpu")]}
        cuda = _header("cuda") | {"cases": [_completed("cuda")]}
        comparison = scalability_report.compare_reports(cpu, cuda)
        pair, = comparison["pairs"]
        self.assertEqual(pair["status"], "consistent")
        self.assertTrue(pair["both_meet_exact_recovery_targets"])
        self.assertEqual(pair["optimization_speedup_first_over_second"], 4)
        self.assertEqual(pair["observability"]["final"]["status"], "not_requested")

    def test_different_configuration_or_input_disallows_comparison(self):
        left, right = _completed("cpu"), _completed("cuda")
        right["fitter_input_sha256"] = "2" * 64
        pair = scalability_report.compare_cases(left, right)
        self.assertEqual(pair["status"], "incompatible_dataset")
        self.assertNotIn("optimization_speedup_first_over_second", pair)
        right["config"]["batch_size"] = 16
        self.assertEqual(scalability_report.compare_cases(left, right)["status"], "incompatible_configuration")
        right["dataset_verified"] = False
        self.assertEqual(scalability_report.compare_cases(left, right, different_batch=True)["status"], "unavailable")

    def test_numerical_disagreement_is_visible_when_recovery_targets_are_met(self):
        left, right = _completed("cpu"), _completed("cuda")
        right["parameter_snapshots"]["final"]["expression"][0][0] += 2e-6
        pair = scalability_report.compare_cases(left, right)
        self.assertEqual(pair["status"], "disagreement")
        self.assertTrue(pair["both_meet_exact_recovery_targets"])
        self.assertFalse(pair["numerical_comparisons"]["final"]["expression"]["matches"])

    def test_a_cpu_execution_cannot_be_compared_as_cuda(self):
        row = _completed("cuda")
        row["actual_execution_device"] = "cpu"
        with self.assertRaisesRegex(ValueError, "device disagrees"):
            scalability_report.compare_reports(
                _header("cpu") | {"cases": [_completed("cpu")]}, _header("cuda") | {"cases": [row]},
            )

    def test_observability_reports_rank_and_spectrum_disagreements(self):
        left, right = _completed("cpu"), _completed("cuda")
        for row in (left, right):
            row["spectra_requested"] = True
            row["observability_status"] = {at: {"status": "evaluated"} for at in ("initial", "final")}
            for at in ("initial", "final"):
                row[f"{at}_identity_observability"] = {
                    key: {"rank": 1, "nullity": 1, "matrix_dimensions": [6, 2], "compressed_dimensions": [2, 2],
                          "singular_values": [1.0, 0.0], "absolute_threshold": 1e-6}
                    for key in scalability_report.SPECTRAL_BLOCKS
                }
        changed = right["final_identity_observability"]["joint_identity_and_focal"]
        changed.update(rank=2, nullity=0, singular_values=[1.0, 1e-3])
        comparison = scalability_report.compare_cases(left, right)
        self.assertEqual(comparison["status"], "disagreement")
        self.assertEqual(comparison["observability"]["initial"]["status"], "consistent")
        joint = comparison["observability"]["final"]["blocks"]["joint_identity_and_focal"]
        self.assertEqual(joint["ranks"], [1, 2])
        self.assertFalse(joint["singular_values"]["matches"])
