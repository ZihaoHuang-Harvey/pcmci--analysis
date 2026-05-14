import tempfile
import unittest
from pathlib import Path

from app.config.settings import load_app_config, save_app_config
from app.core.models import AnalysisConfig, AppConfig, RoleTemplate, VariableRole


class ConfigLoaderTestCase(unittest.TestCase):
    def test_save_and_load_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "test_config.yaml"
            config = AppConfig(
                defaults=AnalysisConfig(
                    tau_min=1,
                    tau_max=4,
                    pc_alpha=0.01,
                    default_data_path="sample.xlsx",
                ),
                role_templates={
                    VariableRole.TIME_DRIVER: RoleTemplate(
                        role=VariableRole.TIME_DRIVER,
                        variable_names=["date_sin"],
                        description="时间驱动示例。",
                    ),
                    VariableRole.TERRAIN_DRIVER: RoleTemplate(
                        role=VariableRole.TERRAIN_DRIVER,
                        variable_names=["elevation_mean"],
                        description="地形驱动示例。",
                    ),
                },
                help_text={"step_data": "测试说明"},
            )
            save_app_config(config, config_path=config_path)
            loaded = load_app_config(config_path=config_path)

            self.assertEqual(loaded.defaults.tau_min, 1)
            self.assertEqual(loaded.defaults.tau_max, 4)
            self.assertAlmostEqual(loaded.defaults.pc_alpha, 0.01)
            self.assertEqual(loaded.defaults.default_data_path, "sample.xlsx")
            self.assertIn(VariableRole.TIME_DRIVER, loaded.role_templates)
            self.assertEqual(
                loaded.role_templates[VariableRole.TIME_DRIVER].variable_names,
                ["date_sin"],
            )
            self.assertEqual(loaded.help_text["step_data"], "测试说明")


if __name__ == "__main__":
    unittest.main()
