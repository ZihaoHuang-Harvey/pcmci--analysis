from __future__ import annotations

import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

from app.core.analysis_service import AnalysisService
from app.core.models import AnalysisConfig, CompiledConstraints


class AnalysisWorker(QThread):
    """后台分析线程。

    GUI 线程只负责显示状态，实际计算放在 QThread 中，避免界面卡死。
    """

    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(
        self,
        df: pd.DataFrame,
        var_names: list[str],
        config: AnalysisConfig,
        compiled_constraints: CompiledConstraints,
    ) -> None:
        super().__init__()
        self.df = df
        self.var_names = var_names
        self.config = config
        self.compiled_constraints = compiled_constraints
        self.analysis_service = AnalysisService()

    def run(self) -> None:
        try:
            result = self.analysis_service.run_analysis(
                df=self.df,
                var_names=self.var_names,
                config=self.config,
                compiled_constraints=self.compiled_constraints,
                progress=self.progress_signal.emit,
                logger=self.log_signal.emit,
            )
            self.finished_signal.emit(result)
        except Exception as exc:
            self.error_signal.emit(f"分析出错: {exc}")

