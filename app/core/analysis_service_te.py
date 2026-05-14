from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

import numpy as np
import pandas as pd

try:
    from pyinform.transferentropy import transfer_entropy
    TE_AVAILABLE = True
except ImportError:
    transfer_entropy = None
    TE_AVAILABLE = False

from app.core.models import TEConfig, TEResult


def discretize_series(series: pd.Series, bins: int) -> np.ndarray:
    return pd.cut(series, bins=bins, labels=False).astype(int).values


def estimate_discrete_transfer_entropy(source: np.ndarray, target: np.ndarray, k_history: int) -> float:
    """基于离散计数估算 TE，用作 pyinform 不可用时的后备实现。"""

    k = max(1, int(k_history))
    if len(source) != len(target) or len(target) <= k:
        return 0.0

    joint_counts: Counter[tuple[int, tuple[int, ...], int]] = Counter()
    history_source_counts: Counter[tuple[tuple[int, ...], int]] = Counter()
    future_history_counts: Counter[tuple[int, tuple[int, ...]]] = Counter()
    history_counts: Counter[tuple[int, ...]] = Counter()

    for index in range(k, len(target)):
        future = int(target[index])
        target_history = tuple(int(value) for value in target[index - k:index])
        source_state = int(source[index])
        joint_counts[(future, target_history, source_state)] += 1
        history_source_counts[(target_history, source_state)] += 1
        future_history_counts[(future, target_history)] += 1
        history_counts[target_history] += 1

    sample_count = sum(joint_counts.values())
    if sample_count == 0:
        return 0.0

    te_value = 0.0
    for (future, target_history, source_state), count in joint_counts.items():
        p_joint = count / sample_count
        p_future_given_history_source = count / history_source_counts[(target_history, source_state)]
        p_future_given_history = future_history_counts[(future, target_history)] / history_counts[target_history]
        if p_future_given_history > 0:
            te_value += p_joint * math.log2(p_future_given_history_source / p_future_given_history)

    return max(0.0, float(te_value))


def calculate_transfer_entropy(source: np.ndarray, target: np.ndarray, k_history: int) -> float:
    if transfer_entropy is not None:
        return float(transfer_entropy(source, target, k=k_history))
    return estimate_discrete_transfer_entropy(source, target, k_history)


class TEAnalysisService:
    def run_analysis(
        self,
        df: pd.DataFrame,
        var_names: list[str],
        config: TEConfig,
        progress: Callable[[str], None] | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> TEResult | None:
        if not TE_AVAILABLE:
            self._log(logger, "警告：pyinform 库未安装，改用内置离散 TE 估算。")

        self._emit(progress, "开始 TE 分析...")

        n_vars = len(var_names)
        tau_max = config.tau_max

        self._emit(progress, f"配置参数: bins={config.bins}, k={config.k_history}, tau_max={tau_max}")

        data_te = df.copy()

        lagged_vars: dict[str, tuple[str, int]] = {}

        self._emit(progress, "构建滞后变量...")

        if config.analyze_all:
            target_indices = range(n_vars)
        elif config.target_var and config.target_var in var_names:
            target_indices = [var_names.index(config.target_var)]
        else:
            self._log(logger, "未指定有效目标变量，将分析所有变量")
            target_indices = range(n_vars)

        for j in target_indices:
            target_name = var_names[j]
            for tau in range(1, tau_max + 1):
                for i in range(n_vars):
                    if i == j:
                        continue
                    src_name = var_names[i]
                    colname = f"{src_name}_lag{tau}"
                    data_te[colname] = df[src_name].shift(tau)
                    lagged_vars[colname] = (src_name, tau)

        data_te = data_te.dropna()

        self._emit(progress, f"数据预处理完成，有效样本数: {len(data_te)}")

        self._emit(progress, "离散化数据...")
        discrete_data = pd.DataFrame()
        for var in var_names:
            discrete_data[var] = discretize_series(data_te[var], config.bins)
        for col in lagged_vars.keys():
            discrete_data[col] = discretize_series(data_te[col], config.bins)

        self._emit(progress, "计算转移熵...")

        te_matrix = np.zeros((n_vars, n_vars, tau_max + 1), dtype=float)
        ndte_matrix = np.zeros((n_vars, n_vars, tau_max + 1), dtype=float)

        total_pairs = sum(1 for j in target_indices for tau in range(1, tau_max + 1) for i in range(n_vars) if i != j)
        processed = 0

        for j in target_indices:
            target_name = var_names[j]
            target_array = discrete_data[target_name].values

            for tau in range(1, tau_max + 1):
                for i in range(n_vars):
                    if i == j:
                        continue

                    src_name = var_names[i]
                    colname = f"{src_name}_lag{tau}"

                    if colname not in discrete_data.columns:
                        continue

                    src_array = discrete_data[colname].values

                    try:
                        te_xy = calculate_transfer_entropy(src_array, target_array, config.k_history)
                        te_yx = calculate_transfer_entropy(target_array, src_array, config.k_history)
                        denom = te_xy + te_yx
                        ndte = te_xy / denom if denom > 0 else 0.0

                        te_matrix[i, j, tau] = te_xy
                        ndte_matrix[i, j, tau] = ndte
                    except Exception as e:
                        self._log(logger, f"计算 {colname} -> {target_name} 失败: {e}")

                    processed += 1
                    if processed % 100 == 0:
                        self._emit(progress, f"已处理 {processed}/{total_pairs} 个变量对...")

        self._emit(progress, "构建结果矩阵...")

        significant_pairs: list[tuple[str, str, int, float, float]] = []
        for j in target_indices:
            for tau in range(1, tau_max + 1):
                for i in range(n_vars):
                    if i == j:
                        continue
                    te_val = te_matrix[i, j, tau]
                    ndte_val = ndte_matrix[i, j, tau]
                    if te_val > 0:
                        significant_pairs.append((var_names[i], var_names[j], tau, te_val, ndte_val))

        significant_pairs.sort(key=lambda x: (x[1], x[2], -x[3]))

        summary = {
            "var_count": len(var_names),
            "total_pairs": len(sig_pairs := [
                p for p in significant_pairs if p[3] > 0
            ]),
            "top_contributors": [
                {"source": s, "target": t, "lag": lag, "te": te, "ndte": ndte}
                for s, t, lag, te, ndte in sorted(sig_pairs, key=lambda x: -x[3])[:10]
            ] if sig_pairs else [],
            "parameters": {
                "bins": config.bins,
                "k_history": config.k_history,
                "tau_max": config.tau_max,
                "target_var": config.target_var if not config.analyze_all else "全部",
            },
        }

        self._emit(progress, "TE 分析完成")

        return TEResult(
            var_names=var_names,
            config=config,
            te_matrix=te_matrix,
            ndte_matrix=ndte_matrix,
            significant_pairs=significant_pairs,
            summary=summary,
        )

    @staticmethod
    def _emit(callback: Callable[[str], None] | None, message: str) -> None:
        if callback is not None:
            callback(message)

    @staticmethod
    def _log(callback: Callable[[str], None] | None, message: str) -> None:
        if callback is not None:
            callback(message)
