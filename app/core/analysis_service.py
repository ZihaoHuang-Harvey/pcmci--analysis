from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from app.core.constraint_engine import ConstraintEngine
from app.core.models import AnalysisConfig, AnalysisResult, CompiledConstraints, ROLE_LABELS


def robust_scale_dataframe(
    df: pd.DataFrame,
    quantile_range: tuple[int, int],
) -> pd.DataFrame:
    """对输入数据做鲁棒缩放。"""

    scaler = RobustScaler(quantile_range=quantile_range)
    scaled = scaler.fit_transform(df.values).astype(np.float32)
    return pd.DataFrame(scaled, columns=df.columns)


class AnalysisService:
    """PCMCI 分析服务。"""

    def run_analysis(
        self,
        df: pd.DataFrame,
        var_names: list[str],
        config: AnalysisConfig,
        compiled_constraints: CompiledConstraints,
        progress: Callable[[str], None] | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> AnalysisResult:
        self._emit(progress, "开始数据预处理...")
        scaled_df = robust_scale_dataframe(df, config.quantile_range)
        dataframe = pp.DataFrame(
            scaled_df.values,
            datatime=np.arange(len(scaled_df)),
            var_names=var_names,
        )

        self._emit(progress, "初始化独立性检验...")
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=ParCorr(significance="analytic"),
            verbosity=0,
        )

        if compiled_constraints.has_constraints:
            self._log(logger, "应用统一约束编译结果...")
            self._log(
                logger,
                "无入边变量: " + ", ".join(
                    var_names[index] for index in sorted(compiled_constraints.incoming_blocked)
                ),
            )

        self._emit(progress, "运行 PCMCI+ 分析...")
        results = pcmci.run_pcmciplus(
            tau_min=config.tau_min,
            tau_max=config.tau_max,
            pc_alpha=config.pc_alpha,
            link_assumptions=compiled_constraints.link_assumptions,
        )

        raw_graph = results["graph"].copy()
        raw_p = results["p_matrix"].copy()
        raw_val = results["val_matrix"].copy()

        self._log(logger, f"Graph 原始 shape: {raw_graph.shape}")
        self._log(logger, f"Graph 原始唯一值: {self._collect_unique_values(raw_graph)}")

        self._emit(progress, "将约束应用到结果矩阵...")
        filtered_graph, filtered_p, filtered_val = ConstraintEngine.apply_to_results(
            graph=raw_graph,
            p_matrix=raw_p,
            val_matrix=raw_val,
            var_names=var_names,
            tau_min=config.tau_min,
            tau_max=config.tau_max,
            compiled=compiled_constraints,
            logger=logger,
        )

        for message in ConstraintEngine.summarize_remaining_constraints(
            graph=filtered_graph,
            p_matrix=filtered_p,
            var_names=var_names,
            tau_min=config.tau_min,
            tau_max=config.tau_max,
            pc_alpha=config.pc_alpha,
            compiled=compiled_constraints,
        ):
            self._log(logger, message)

        self._emit(progress, "生成矩阵视图...")
        adj_matrix = self._build_adjacency_matrix(
            graph=filtered_graph,
            p_matrix=filtered_p,
            tau_min=config.tau_min,
            tau_max=config.tau_max,
            pc_alpha=config.pc_alpha,
        )

        summary = self._build_summary(
            var_names=var_names,
            config=config,
            compiled=compiled_constraints,
        )

        return AnalysisResult(
            var_names=var_names,
            tau_min=config.tau_min,
            tau_max=config.tau_max,
            pc_alpha=config.pc_alpha,
            graph=filtered_graph,
            p_matrix=filtered_p,
            val_matrix=filtered_val,
            adj_matrix=adj_matrix,
            raw_results=results,
            compiled_constraints=compiled_constraints,
            summary=summary,
        )

    @staticmethod
    def _build_adjacency_matrix(
        graph: np.ndarray,
        p_matrix: np.ndarray,
        tau_min: int,
        tau_max: int,
        pc_alpha: float,
    ) -> np.ndarray:
        """构造 lag>0 的邻接矩阵。行表示源，列表示目标。"""

        if tau_max < 1:
            return np.zeros(graph.shape[:2], dtype=float)

        significance = np.zeros(graph.shape[:2], dtype=bool)
        for tau in range(max(1, tau_min), tau_max + 1):
            significance |= (graph[:, :, tau] != "") & (p_matrix[:, :, tau] <= pc_alpha)
        return significance.astype(float)

    @staticmethod
    def _build_summary(
        var_names: list[str],
        config: AnalysisConfig,
        compiled: CompiledConstraints,
    ) -> dict[str, object]:
        """构造结果页摘要。

        摘要是给 GUI 直接展示的轻量数据，不重复塞入整个矩阵。
        这里按“角色中文名 -> 变量列表”的方式输出，便于用户快速确认
        本次分析里哪些变量被当作时间驱动、地形驱动或普通变量处理。
        """

        role_groups: dict[str, list[str]] = {}
        for index, role in compiled.role_assignments.items():
            label = ROLE_LABELS[role]
            role_groups.setdefault(label, []).append(var_names[index])

        return {
            "var_count": len(var_names),
            "roles": {label: sorted(names) for label, names in role_groups.items()},
            "incoming_blocked": [var_names[index] for index in sorted(compiled.incoming_blocked)],
            "advanced_rule_count": len(compiled.resolved_rules),
            "parameters": {
                "tau_min": config.tau_min,
                "tau_max": config.tau_max,
                "pc_alpha": config.pc_alpha,
                "quantile_range": list(config.quantile_range),
            },
        }

    @staticmethod
    def _collect_unique_values(graph: np.ndarray) -> list[str]:
        return sorted({str(value) for value in np.unique(graph)})

    @staticmethod
    def _emit(callback: Callable[[str], None] | None, message: str) -> None:
        if callback is not None:
            callback(message)

    @staticmethod
    def _log(callback: Callable[[str], None] | None, message: str) -> None:
        if callback is not None:
            callback(message)
