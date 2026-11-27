"""Verify the exact V6 matrix and its batching reuse rules."""

from itertools import product
import unittest

from persona_lab.benchmark.scalability_cases import case_matrix


class ScalabilityCaseTests(unittest.TestCase):
    def test_primary_and_batching_cartesian_products(self):
        rows = case_matrix()
        primary = [case for case in rows if case["experiment"] == "primary"]
        batching = [case for case in rows if case["experiment"] == "batching"]
        self.assertEqual(len(primary), 24)
        self.assertEqual(len(batching), 8)
        self.assertEqual(len({case["case"] for case in rows}), 32)
        self.assertEqual({(case["seed"], case["requested_observations"], case["requested_device"])
                          for case in primary}, set(product((42, 43, 44), (50, 100, 300, 600), ("cpu", "cuda"))))
        self.assertEqual({(case["seed"], case["requested_observations"], case["requested_device"], case["batch_size"])
                          for case in batching}, set(product((42,), (100,), ("cpu", "cuda"), (1, 4, 16, 32))))
        reused = [case for case in batching if "reuse_case" in case]
        self.assertEqual({case["reuse_case"] for case in reused},
                         {"primary-cpu-42-100-batch4", "primary-cuda-42-100-batch4"})
        self.assertEqual(sum("reuse_case" not in case for case in rows), 30)

    def test_reuse_requires_the_primary_batch_four_case(self):
        for options in ({"seeds": (43,)}, {"counts": (50,)}, {"batch_size": 16}):
            self.assertTrue(all("reuse_case" not in case for case in case_matrix(**options)))

    def test_explicit_primary_only_pilot(self):
        rows = case_matrix(seeds=(42,), counts=(50,), devices=("cuda",), batch_size=32, primary_only=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case"], "primary-cuda-42-50-batch32")
