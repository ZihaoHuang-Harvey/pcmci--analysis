import unittest

import numpy as np

from app.gui.result_plotting import (
    build_target_centric_time_series_data,
    build_upstream_graph_data,
    create_diverging_colormap,
    middle_color_hex,
)


class ResultPlottingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.var_names = ["rain", "temp", "10cm_mean", "noise"]
        self.graph = np.full((4, 4, 3), "", dtype=object)
        self.p_matrix = np.ones((4, 4, 3), dtype=float)
        self.val_matrix = np.zeros((4, 4, 3), dtype=float)

        self.graph[0, 1, 2] = "-->"
        self.p_matrix[0, 1, 2] = 0.001
        self.val_matrix[0, 1, 2] = 0.42

        self.graph[0, 1, 1] = "-->"
        self.p_matrix[0, 1, 1] = 0.001
        self.val_matrix[0, 1, 1] = 0.75

        self.graph[1, 2, 0] = "-->"
        self.graph[2, 1, 0] = "<--"
        self.p_matrix[1, 2, 0] = 0.001
        self.p_matrix[2, 1, 0] = 0.001
        self.val_matrix[1, 2, 0] = 0.33
        self.val_matrix[2, 1, 0] = 0.33

        self.graph[3, 0, 1] = "-->"
        self.p_matrix[3, 0, 1] = 0.2
        self.val_matrix[3, 0, 1] = 0.9

    def test_build_upstream_graph_data_collects_full_chain(self) -> None:
        graph_data = build_upstream_graph_data(
            graph=self.graph,
            p_matrix=self.p_matrix,
            val_matrix=self.val_matrix,
            var_names=self.var_names,
            pc_alpha=0.01,
            target_name="10cm_mean",
        )

        self.assertEqual(graph_data.target_index, 2)
        self.assertEqual(graph_data.levels[2], 0)
        self.assertEqual(graph_data.levels[1], 1)
        self.assertEqual(graph_data.levels[0], 2)
        self.assertNotIn(3, graph_data.node_indices)
        self.assertEqual(len(graph_data.edges), 2)

        folded_edge = next(edge for edge in graph_data.edges if edge.source == 0 and edge.target == 1)
        self.assertEqual(folded_edge.representative_lag, 1)
        self.assertIn("t-1", folded_edge.label)

    def test_build_upstream_graph_data_returns_empty_state(self) -> None:
        graph_data = build_upstream_graph_data(
            graph=self.graph,
            p_matrix=self.p_matrix,
            val_matrix=self.val_matrix,
            var_names=self.var_names,
            pc_alpha=0.00001,
            target_name="10cm_mean",
        )

        self.assertEqual(graph_data.node_indices, [2])
        self.assertFalse(graph_data.has_relationships)
        self.assertIn("没有显著上游影响路径", graph_data.empty_message)

    def test_diverging_colormap_middle_is_not_white(self) -> None:
        middle_hex = middle_color_hex(create_diverging_colormap())
        self.assertNotEqual(middle_hex.lower(), "#ffffff")

    def test_target_centric_time_series_only_keeps_edges_to_target_t(self) -> None:
        graph_data = build_target_centric_time_series_data(
            graph=self.graph,
            p_matrix=self.p_matrix,
            val_matrix=self.val_matrix,
            var_names=self.var_names,
            pc_alpha=0.01,
            target_name="10cm_mean",
            only_target_edges=True,
            hide_historical_contemporaneous=True,
            hide_ambiguous_edges=True,
        )

        self.assertTrue(all(edge.target_var == 2 for edge in graph_data.edges))
        self.assertTrue(all(edge.target_tau == 0 for edge in graph_data.edges))

    def test_full_network_mode_replicates_lagged_edges_across_columns(self) -> None:
        graph_data = build_target_centric_time_series_data(
            graph=self.graph,
            p_matrix=self.p_matrix,
            val_matrix=self.val_matrix,
            var_names=self.var_names,
            pc_alpha=0.01,
            target_name="10cm_mean",
            only_target_edges=False,
            hide_historical_contemporaneous=True,
            hide_ambiguous_edges=True,
        )

        replicated = [
            edge
            for edge in graph_data.edges
            if edge.source_var == 0 and edge.target_var == 1 and edge.source_tau - edge.target_tau == 1
        ]
        self.assertEqual(len(replicated), 2)
        self.assertSetEqual({edge.target_tau for edge in replicated}, {0, 1})

    def test_historical_contemporaneous_edges_are_hidden_but_current_target_edge_can_remain(self) -> None:
        self.graph[0, 2, 0] = "-->"
        self.graph[2, 0, 0] = "<--"
        self.p_matrix[0, 2, 0] = 0.001
        self.p_matrix[2, 0, 0] = 0.001
        self.val_matrix[0, 2, 0] = 0.44
        self.val_matrix[2, 0, 0] = 0.44

        graph_data = build_target_centric_time_series_data(
            graph=self.graph,
            p_matrix=self.p_matrix,
            val_matrix=self.val_matrix,
            var_names=self.var_names,
            pc_alpha=0.01,
            target_name="10cm_mean",
            only_target_edges=True,
            hide_historical_contemporaneous=True,
            hide_ambiguous_edges=True,
        )

        contemporaneous_edges = [edge for edge in graph_data.edges if edge.source_tau == edge.target_tau == 0]
        self.assertGreaterEqual(len(contemporaneous_edges), 1)
        self.assertTrue(all(edge.target_var == 2 for edge in contemporaneous_edges))

    def test_ambiguous_edges_can_be_hidden_or_shown(self) -> None:
        self.graph[0, 2, 0] = "o->"
        self.graph[2, 0, 0] = "<-o"
        self.p_matrix[0, 2, 0] = 0.001
        self.p_matrix[2, 0, 0] = 0.001
        self.val_matrix[0, 2, 0] = 0.44
        self.val_matrix[2, 0, 0] = 0.44

        hidden = build_target_centric_time_series_data(
            graph=self.graph,
            p_matrix=self.p_matrix,
            val_matrix=self.val_matrix,
            var_names=self.var_names,
            pc_alpha=0.01,
            target_name="10cm_mean",
            only_target_edges=True,
            hide_historical_contemporaneous=True,
            hide_ambiguous_edges=True,
        )
        shown = build_target_centric_time_series_data(
            graph=self.graph,
            p_matrix=self.p_matrix,
            val_matrix=self.val_matrix,
            var_names=self.var_names,
            pc_alpha=0.01,
            target_name="10cm_mean",
            only_target_edges=True,
            hide_historical_contemporaneous=True,
            hide_ambiguous_edges=False,
        )

        self.assertFalse(any(edge.style.startswith("ambiguous") for edge in hidden.edges))
        self.assertTrue(any(edge.style.startswith("ambiguous") for edge in shown.edges))


if __name__ == "__main__":
    unittest.main()
