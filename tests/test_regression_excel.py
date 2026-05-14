import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.config.settings import load_app_config
from app.core.analysis_service import AnalysisService
from app.core.constraint_engine import ConstraintEngine
from app.core.models import AnalysisConfig, VariableRole


class RegressionExcelTestCase(unittest.TestCase):
    @staticmethod
    def _build_default_role_mapping(var_names: list[str]) -> dict[str, VariableRole]:
        app_config = load_app_config()
        role_mapping = {name: VariableRole.REGULAR for name in var_names}
        for role, template in app_config.role_templates.items():
            if role == VariableRole.REGULAR:
                continue
            for name in template.variable_names:
                if name in role_mapping:
                    role_mapping[name] = role
        return role_mapping

    @staticmethod
    def _has_contemporaneous_incoming(result, source: int, target: int, alpha: float) -> bool:
        if source == target:
            return False
        return (
            result.graph[source, target, 0] in {"-->", "-?>", "o-o", "o?o", "x-x"}
            and min(result.p_matrix[source, target, 0], result.p_matrix[target, source, 0]) <= alpha
        )

    def test_default_roles_on_real_excel(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        excel_path = project_root / "combined_all_with_dem(内插结果)_sin_cos_filtered.xlsx"
        if not excel_path.exists():
            self.skipTest("回归测试缺少真实 Excel 数据文件。")

        raw_df = pd.read_excel(excel_path, engine="openpyxl")
        numeric_df = raw_df.select_dtypes(include=[np.number]).copy()
        var_names = numeric_df.columns.tolist()

        role_mapping = self._build_default_role_mapping(var_names)
        compiled = ConstraintEngine.compile(
            var_names=var_names,
            tau_min=0,
            tau_max=3,
            role_mapping=role_mapping,
            manual_rules=[],
        )

        result = AnalysisService().run_analysis(
            df=numeric_df,
            var_names=var_names,
            config=AnalysisConfig(tau_min=0, tau_max=3, pc_alpha=0.01),
            compiled_constraints=compiled,
        )

        index_by_name = {name: idx for idx, name in enumerate(var_names)}
        driver_vars = ["date_sin", "date_cos", "elevation_mean", "slope_mean", "aspect_mean"]

        for name in driver_vars:
            target = index_by_name[name]
            for source in range(len(var_names)):
                for tau in range(1, 4):
                    self.assertEqual(
                        result.graph[source, target, tau],
                        "",
                        f"{name} 不应保留任何滞后入边，包括自滞后。",
                    )

                self.assertFalse(
                    self._has_contemporaneous_incoming(result, source, target, 0.01),
                    f"{name} 不应保留来自 {var_names[source]} 的同时刻入边。",
                )

        terrain_vars = ["elevation_mean", "slope_mean", "aspect_mean"]
        regular_vars = [
            name
            for name in var_names
            if role_mapping.get(name, VariableRole.REGULAR) == VariableRole.REGULAR
        ]

        terrain_has_outgoing = False
        for terrain_name in terrain_vars:
            source = index_by_name[terrain_name]
            for regular_name in regular_vars:
                target = index_by_name[regular_name]
                if any(
                    result.graph[source, target, tau] and result.p_matrix[source, target, tau] <= 0.01
                    for tau in range(1, 4)
                ):
                    terrain_has_outgoing = True
                    break
                if (
                    result.graph[source, target, 0] == "-->"
                    and min(result.p_matrix[source, target, 0], result.p_matrix[target, source, 0]) <= 0.01
                ):
                    terrain_has_outgoing = True
                    break
            if terrain_has_outgoing:
                break

        self.assertTrue(
            terrain_has_outgoing,
            "地形驱动变量应至少保留一条显著向外关系，避免被错误渲染成完全断开。",
        )


if __name__ == "__main__":
    unittest.main()
