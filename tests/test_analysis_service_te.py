import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import app.core.analysis_service_te as te_module
from app.core.analysis_service_te import TEAnalysisService
from app.core.models import TEConfig


class TEAnalysisServiceTestCase(unittest.TestCase):
    @staticmethod
    def _build_lagged_dataframe() -> pd.DataFrame:
        source = np.tile([0, 0, 1, 1], 20)
        target = np.roll(source, 1)
        target[0] = 0
        return pd.DataFrame({"source": source, "target": target})

    def test_run_analysis_returns_result_with_configured_backend(self) -> None:
        df = self._build_lagged_dataframe()
        result = TEAnalysisService().run_analysis(
            df=df,
            var_names=["source", "target"],
            config=TEConfig(enabled=True, bins=2, k_history=1, tau_max=1),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.te_matrix.shape, (2, 2, 2))
        self.assertTrue(np.any(result.te_matrix[:, :, 1] > 0.0))

    def test_run_analysis_falls_back_when_pyinform_is_unavailable(self) -> None:
        df = self._build_lagged_dataframe()
        log_messages: list[str] = []

        with (
            patch.object(te_module, "TE_AVAILABLE", False),
            patch.object(te_module, "transfer_entropy", None),
        ):
            result = TEAnalysisService().run_analysis(
                df=df,
                var_names=["source", "target"],
                config=TEConfig(enabled=True, bins=2, k_history=1, tau_max=1),
                logger=log_messages.append,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.te_matrix.shape, (2, 2, 2))
        self.assertGreater(result.te_matrix[0, 1, 1], 0.0)
        self.assertTrue(any("内置" in message for message in log_messages))


if __name__ == "__main__":
    unittest.main()
