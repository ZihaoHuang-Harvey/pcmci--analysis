import unittest

import numpy as np

from app.core.constraint_engine import ConstraintEngine
from app.core.models import ConstraintType, ManualConstraintRule, VariableRole


class ConstraintEngineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.var_names = ["date_sin", "NDVI_mean", "elevation_mean"]
        self.graph = np.full((3, 3, 3), "", dtype=object)
        self.p_matrix = np.ones((3, 3, 3), dtype=float)
        self.val_matrix = np.zeros((3, 3, 3), dtype=float)

        self.graph[1, 0, 1] = "-->"
        self.graph[0, 1, 1] = "-->"
        self.graph[0, 0, 1] = "-->"
        self.graph[1, 2, 1] = "-->"
        self.graph[2, 1, 1] = "-->"
        self.graph[2, 2, 1] = "-->"

        self.graph[1, 0, 0] = "-->"
        self.graph[0, 1, 0] = "<--"
        self.graph[1, 2, 0] = "-->"
        self.graph[2, 1, 0] = "<--"

        self.p_matrix[:, :, :] = 0.001
        self.val_matrix[:, :, :] = 0.5

    def test_time_driver_blocks_all_incoming_and_keeps_outgoing(self) -> None:
        compiled = ConstraintEngine.compile(
            var_names=self.var_names,
            tau_min=0,
            tau_max=2,
            role_mapping={"date_sin": VariableRole.TIME_DRIVER},
            manual_rules=[],
        )

        graph, p_matrix, val_matrix = ConstraintEngine.apply_to_results(
            graph=self.graph,
            p_matrix=self.p_matrix,
            val_matrix=self.val_matrix,
            var_names=self.var_names,
            tau_min=0,
            tau_max=2,
            compiled=compiled,
        )

        self.assertEqual(graph[1, 0, 1], "")
        self.assertEqual(graph[0, 0, 1], "")
        self.assertEqual(p_matrix[1, 0, 1], 1.0)
        self.assertEqual(val_matrix[0, 0, 1], 0.0)
        self.assertEqual(graph[0, 1, 1], "-->")
        self.assertEqual(graph[0, 1, 0], "-->")
        self.assertEqual(graph[1, 0, 0], "<--")

    def test_terrain_driver_blocks_incoming_but_keeps_outgoing(self) -> None:
        compiled = ConstraintEngine.compile(
            var_names=self.var_names,
            tau_min=0,
            tau_max=2,
            role_mapping={"elevation_mean": VariableRole.TERRAIN_DRIVER},
            manual_rules=[],
        )

        graph, _, _ = ConstraintEngine.apply_to_results(
            graph=self.graph,
            p_matrix=self.p_matrix,
            val_matrix=self.val_matrix,
            var_names=self.var_names,
            tau_min=0,
            tau_max=2,
            compiled=compiled,
        )

        self.assertEqual(graph[1, 2, 1], "")
        self.assertEqual(graph[2, 2, 1], "")
        self.assertEqual(graph[2, 1, 1], "-->")
        self.assertEqual(graph[2, 1, 0], "-->")
        self.assertEqual(graph[1, 2, 0], "<--")

    def test_no_link_reorients_contemporaneous_pair(self) -> None:
        self.graph[0, 1, 0] = "-->"
        self.graph[1, 0, 0] = "<--"

        compiled = ConstraintEngine.compile(
            var_names=self.var_names,
            tau_min=0,
            tau_max=2,
            role_mapping={},
            manual_rules=[
                ManualConstraintRule(
                    source_name="date_sin",
                    target_name="NDVI_mean",
                    constraint_type=ConstraintType.NO_LINK,
                    description="date_sin 不能影响 NDVI_mean",
                )
            ],
        )

        graph, _, _ = ConstraintEngine.apply_to_results(
            graph=self.graph,
            p_matrix=self.p_matrix,
            val_matrix=self.val_matrix,
            var_names=self.var_names,
            tau_min=0,
            tau_max=2,
            compiled=compiled,
        )

        self.assertEqual(graph[0, 1, 1], "")
        self.assertEqual(graph[1, 0, 0], "-->")
        self.assertEqual(graph[0, 1, 0], "<--")


if __name__ == "__main__":
    unittest.main()
