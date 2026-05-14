import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QGroupBox

from app.core.models import TEConfig, TEResult
from app.gui.main_window import MainWindow


class GuiSmokeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_window_can_be_created(self) -> None:
        window = MainWindow()
        self.assertEqual(window.windowTitle(), "多方法交互因果分析平台")
        self.assertEqual(window.step_tabs.count(), 4)
        self.assertEqual(window.step_tabs.tabText(0), "1. 数据导入")
        self.assertEqual(window.step_tabs.tabText(1), "2. 变量角色")
        self.assertEqual(window.pcmci_result_tabs.count(), 5)
        self.assertEqual(window.pcmci_result_tabs.tabText(2), "目标变量上游影响图")
        self.assertEqual(window.tau_min_spin.value(), 0)
        self.assertEqual(window._get_default_target_name(["rain", "10cm_mean"]), "10cm_mean")
        self.assertFalse(window.only_target_edges_checkbox.isChecked())
        self.assertFalse(window.hide_historical_contemporaneous_checkbox.isChecked())
        self.assertFalse(window.hide_ambiguous_edges_checkbox.isChecked())
        self.assertFalse(window.step_tabs.isTabEnabled(3))
        window.close()

    def test_pcmci_advanced_constraints_are_not_shown(self) -> None:
        window = MainWindow()
        self.assertFalse(hasattr(window, "advanced_toggle"))
        self.assertFalse(hasattr(window, "advanced_container"))
        self.assertFalse(hasattr(window, "constraint_type_combo"))
        self.assertNotIn(
            "高级边约束（可选）",
            [group.title() for group in window.findChildren(QGroupBox)],
        )
        window.close()

    def test_te_results_render_graph_tab_after_analysis(self) -> None:
        window = MainWindow()
        self.assertEqual(window.te_result_tabs.tabText(0), "TE 结果图")

        te_matrix = np.zeros((2, 2, 3), dtype=float)
        ndte_matrix = np.zeros((2, 2, 3), dtype=float)
        te_matrix[0, 1, 1] = 0.42
        ndte_matrix[0, 1, 1] = 0.73

        window.current_te_result = TEResult(
            var_names=["rain", "yield"],
            config=TEConfig(enabled=True, tau_max=2),
            te_matrix=te_matrix,
            ndte_matrix=ndte_matrix,
            significant_pairs=[("rain", "yield", 1, 0.42, 0.73)],
            summary={"total_pairs": 1},
        )
        window.display_te_results()

        heatmap_axes = [axis for axis in window.te_graph_canvas.fig.axes if axis.images]
        self.assertEqual(len(heatmap_axes), 2)
        self.assertEqual(window.te_matrix_table.item(0, 1).text(), "0.4200")
        self.assertEqual(window.te_table.rowCount(), 1)
        window.close()


if __name__ == "__main__":
    unittest.main()
