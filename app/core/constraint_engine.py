from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np

from app.core.models import (
    CompiledConstraints,
    ConstraintType,
    ManualConstraintRule,
    ResolvedConstraintRule,
    VariableRole,
)


class ConstraintEngine:
    """统一约束引擎。

    本版以“方向许可矩阵”为核心：
    - allowed_lagged[source, target] 表示 tau>0 时 source -> target 是否允许
    - allowed_contemporaneous[source, target] 表示 tau=0 时 source -> target 是否允许

    这样可以自然表达“时间/地形无入边，但允许向外同时刻影响普通变量”的需求，
    不再需要把 tau=0 全部清空。
    """

    @classmethod
    def compile(
        cls,
        var_names: list[str],
        tau_min: int,
        tau_max: int,
        role_mapping: dict[str, VariableRole],
        manual_rules: list[ManualConstraintRule],
        logger: Callable[[str], None] | None = None,
    ) -> CompiledConstraints:
        var_count = len(var_names)
        index_by_name = {name: idx for idx, name in enumerate(var_names)}

        # 默认：所有 lagged 方向允许，所有 contemporaneous 非对角方向允许。
        allowed_lagged = np.ones((var_count, var_count), dtype=bool)
        allowed_contemporaneous = np.ones((var_count, var_count), dtype=bool)
        np.fill_diagonal(allowed_contemporaneous, False)

        role_assignments = {
            index_by_name[name]: role_mapping.get(name, VariableRole.REGULAR)
            for name in var_names
        }

        driver_indices = [
            index
            for index, role in role_assignments.items()
            if role in {VariableRole.TIME_DRIVER, VariableRole.TERRAIN_DRIVER}
        ]
        regular_indices = [
            index
            for index, role in role_assignments.items()
            if role == VariableRole.REGULAR
        ]

        incoming_blocked: set[int] = set()
        lagged_outgoing: dict[int, set[int]] = defaultdict(set)
        contemporaneous_outgoing: dict[int, set[int]] = defaultdict(set)
        resolved_rules: list[ResolvedConstraintRule] = []

        for driver in driver_indices:
            incoming_blocked.add(driver)

            # 驱动变量不接收任何 lagged 入边，也不保留自滞后。
            allowed_lagged[:, driver] = False

            # 同时刻关系需要按“方向”单独建模。
            # 这里先整体清空与驱动变量有关的 tau=0 方向，再只放行
            # “驱动变量 -> 普通变量”这一个方向。
            allowed_contemporaneous[:, driver] = False
            allowed_contemporaneous[driver, :] = False

        for driver in driver_indices:
            for target in regular_indices:
                if driver == target:
                    continue
                allowed_lagged[driver, target] = True
                allowed_contemporaneous[driver, target] = True
                lagged_outgoing[driver].add(target)
                contemporaneous_outgoing[driver].add(target)

        # 高级规则在角色约束之后叠加，优先级更高。
        for rule in manual_rules:
            source = index_by_name.get(rule.source_name)
            target = index_by_name.get(rule.target_name)
            if source is None or target is None:
                continue

            resolved_rules.append(
                ResolvedConstraintRule(
                    source_index=source,
                    target_index=target,
                    source_name=rule.source_name,
                    target_name=rule.target_name,
                    constraint_type=rule.constraint_type,
                    description=rule.description,
                )
            )

            if rule.constraint_type == ConstraintType.NO_IN:
                # 禁止其他变量影响目标变量，但仍允许目标变量自己的滞后。
                allowed_lagged[:, target] = False
                allowed_lagged[target, target] = True
                # tau=0 不存在“自滞后”概念，因此这里直接关闭所有指向 target 的同时刻方向。
                allowed_contemporaneous[:, target] = False
            elif rule.constraint_type == ConstraintType.NO_OUT:
                # 禁止该变量影响其他变量，但仍允许其自身滞后。
                allowed_lagged[source, :] = False
                allowed_lagged[source, source] = True
                allowed_contemporaneous[source, :] = False
            elif rule.constraint_type == ConstraintType.NO_LINK:
                allowed_lagged[source, target] = False
                allowed_contemporaneous[source, target] = False

        structural_constraints_exist = bool(driver_indices or resolved_rules)
        link_assumptions = None
        if structural_constraints_exist:
            link_assumptions = cls._build_link_assumptions(
                allowed_lagged=allowed_lagged,
                allowed_contemporaneous=allowed_contemporaneous,
                tau_min=tau_min,
                tau_max=tau_max,
            )

        cls._log(
            logger,
            "角色摘要: "
            + ", ".join(
                f"{var_names[index]}={role_assignments[index].value}"
                for index in range(var_count)
            ),
        )
        cls._log(
            logger,
            "高级边约束数: " + str(len(resolved_rules)),
        )

        return CompiledConstraints(
            link_assumptions=link_assumptions,
            allowed_lagged=allowed_lagged,
            allowed_contemporaneous=allowed_contemporaneous,
            incoming_blocked=incoming_blocked,
            lagged_outgoing={key: set(value) for key, value in lagged_outgoing.items()},
            contemporaneous_outgoing={key: set(value) for key, value in contemporaneous_outgoing.items()},
            role_assignments=role_assignments,
            resolved_rules=resolved_rules,
        )

    @classmethod
    def apply_to_results(
        cls,
        graph: np.ndarray,
        p_matrix: np.ndarray,
        val_matrix: np.ndarray,
        var_names: list[str],
        tau_min: int,
        tau_max: int,
        compiled: CompiledConstraints,
        logger: Callable[[str], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按统一许可矩阵修正 PCMCI 输出。"""

        filtered_graph = graph.copy()
        filtered_p = p_matrix.copy()
        filtered_val = val_matrix.copy()

        for source in range(len(var_names)):
            for target in range(len(var_names)):
                for tau in range(max(1, tau_min), tau_max + 1):
                    if not compiled.allowed_lagged[source, target]:
                        cls._clear_lagged_edge(
                            graph=filtered_graph,
                            p_matrix=filtered_p,
                            val_matrix=filtered_val,
                            source=source,
                            target=target,
                            tau=tau,
                            var_names=var_names,
                            logger=logger,
                            reason="lagged 方向不被当前角色/高级约束允许",
                        )

        for left in range(len(var_names)):
            for right in range(left + 1, len(var_names)):
                allow_left_to_right = compiled.allowed_contemporaneous[left, right]
                allow_right_to_left = compiled.allowed_contemporaneous[right, left]

                if not allow_left_to_right and not allow_right_to_left:
                    cls._clear_contemporaneous_pair(
                        graph=filtered_graph,
                        p_matrix=filtered_p,
                        val_matrix=filtered_val,
                        left=left,
                        right=right,
                        var_names=var_names,
                        logger=logger,
                        reason="同时刻关系整体不被当前角色/高级约束允许",
                    )
                elif allow_left_to_right and not allow_right_to_left:
                    cls._orient_contemporaneous_pair(
                        graph=filtered_graph,
                        p_matrix=filtered_p,
                        val_matrix=filtered_val,
                        source=left,
                        target=right,
                        var_names=var_names,
                        logger=logger,
                        reason="同时刻关系只允许左变量指向右变量",
                    )
                elif not allow_left_to_right and allow_right_to_left:
                    cls._orient_contemporaneous_pair(
                        graph=filtered_graph,
                        p_matrix=filtered_p,
                        val_matrix=filtered_val,
                        source=right,
                        target=left,
                        var_names=var_names,
                        logger=logger,
                        reason="同时刻关系只允许右变量指向左变量",
                    )

        return filtered_graph, filtered_p, filtered_val

    @classmethod
    def summarize_remaining_constraints(
        cls,
        graph: np.ndarray,
        p_matrix: np.ndarray,
        var_names: list[str],
        tau_min: int,
        tau_max: int,
        pc_alpha: float,
        compiled: CompiledConstraints,
    ) -> list[str]:
        """检查驱动变量是否仍存在非法入边。"""

        messages: list[str] = []
        for target in sorted(compiled.incoming_blocked):
            lagged_incoming: list[str] = []
            contemporaneous_incoming: list[str] = []

            for source in range(len(var_names)):
                for tau in range(max(1, tau_min), tau_max + 1):
                    if graph[source, target, tau] and p_matrix[source, target, tau] <= pc_alpha:
                        lagged_incoming.append(f"{var_names[source]}(t-{tau})[{graph[source, target, tau]}]")

                if source == target:
                    continue

                current = graph[source, target, 0]
                if current in {"-->", "-?>", "o-o", "o?o", "x-x"} and (
                    p_matrix[source, target, 0] <= pc_alpha
                    or p_matrix[target, source, 0] <= pc_alpha
                ):
                    contemporaneous_incoming.append(
                        f"{var_names[source]}({graph[source, target, 0]}/{graph[target, source, 0]})"
                    )

            if lagged_incoming or contemporaneous_incoming:
                detail = []
                if lagged_incoming:
                    detail.append("滞后入边: " + ", ".join(lagged_incoming))
                if contemporaneous_incoming:
                    detail.append("同时刻入边: " + ", ".join(contemporaneous_incoming))
                messages.append(
                    f"[警告] 驱动变量 {var_names[target]} 仍存在非法入边；" + "；".join(detail)
                )
            else:
                messages.append(f"驱动变量 {var_names[target]} 已满足“无入边”约束。")

        return messages

    @classmethod
    def _build_link_assumptions(
        cls,
        allowed_lagged: np.ndarray,
        allowed_contemporaneous: np.ndarray,
        tau_min: int,
        tau_max: int,
    ) -> dict[int, dict[tuple[int, int], str]]:
        """把许可矩阵转换成 Tigramite 的 link_assumptions。

        Tigramite 对 tau=0 的方向是成对编码的：
        - 左右都允许：o?o
        - 只允许 source -> target：-?>
        - 只允许 target -> source：<?-

        因此这里不能只看单侧矩阵，必须同时检查两个方向。
        """

        var_count = allowed_lagged.shape[0]
        link_assumptions = {target: {} for target in range(var_count)}

        for target in range(var_count):
            for source in range(var_count):
                if source != target and tau_min <= 0:
                    forward = allowed_contemporaneous[source, target]
                    backward = allowed_contemporaneous[target, source]
                    if forward and backward:
                        link_assumptions[target][(source, 0)] = "o?o"
                    elif forward and not backward:
                        link_assumptions[target][(source, 0)] = "-?>"
                    elif not forward and backward:
                        link_assumptions[target][(source, 0)] = "<?-"

                for tau in range(max(1, tau_min), tau_max + 1):
                    if allowed_lagged[source, target]:
                        link_assumptions[target][(source, -tau)] = "-?>"

        return link_assumptions

    @classmethod
    def _clear_lagged_edge(
        cls,
        graph: np.ndarray,
        p_matrix: np.ndarray,
        val_matrix: np.ndarray,
        source: int,
        target: int,
        tau: int,
        var_names: list[str],
        logger: Callable[[str], None] | None,
        reason: str,
    ) -> None:
        if not graph[source, target, tau]:
            p_matrix[source, target, tau] = 1.0
            val_matrix[source, target, tau] = 0.0
            return

        cls._log(
            logger,
            f"移除滞后边: {var_names[source]}(t-{tau}) -> {var_names[target]}(t)，原因：{reason}",
        )
        graph[source, target, tau] = ""
        p_matrix[source, target, tau] = 1.0
        val_matrix[source, target, tau] = 0.0

    @classmethod
    def _clear_contemporaneous_pair(
        cls,
        graph: np.ndarray,
        p_matrix: np.ndarray,
        val_matrix: np.ndarray,
        left: int,
        right: int,
        var_names: list[str],
        logger: Callable[[str], None] | None,
        reason: str,
    ) -> None:
        if not graph[left, right, 0] and not graph[right, left, 0]:
            p_matrix[left, right, 0] = 1.0
            p_matrix[right, left, 0] = 1.0
            val_matrix[left, right, 0] = 0.0
            val_matrix[right, left, 0] = 0.0
            return

        cls._log(
            logger,
            f"移除同时刻关系: {var_names[left]} <-> {var_names[right]}，原因：{reason}",
        )
        graph[left, right, 0] = ""
        graph[right, left, 0] = ""
        p_matrix[left, right, 0] = 1.0
        p_matrix[right, left, 0] = 1.0
        val_matrix[left, right, 0] = 0.0
        val_matrix[right, left, 0] = 0.0

    @classmethod
    def _orient_contemporaneous_pair(
        cls,
        graph: np.ndarray,
        p_matrix: np.ndarray,
        val_matrix: np.ndarray,
        source: int,
        target: int,
        var_names: list[str],
        logger: Callable[[str], None] | None,
        reason: str,
    ) -> None:
        """把同时刻边规范为 source -> target。

        如果该变量对之间本来没有边，则不凭空造边；若已有显著 adjacency，
        则强制写成满足先验的方向。

        这一步是修复旧问题的关键：
        tau=0 在结果矩阵里是一对单元格 [source, target, 0] / [target, source, 0]。
        如果只清其中一侧，另一侧仍可能残留并在后续绘图中表现成错误方向。
        """

        forward = graph[source, target, 0]
        backward = graph[target, source, 0]
        if not forward and not backward:
            return

        if forward == "-->" and backward == "<--":
            return

        cls._log(
            logger,
            f"重定向同时刻关系: {var_names[source]} -> {var_names[target]}，原因：{reason}",
        )
        best_p = min(p_matrix[source, target, 0], p_matrix[target, source, 0])
        best_val = val_matrix[source, target, 0] if val_matrix[source, target, 0] != 0 else val_matrix[target, source, 0]

        graph[source, target, 0] = "-->"
        graph[target, source, 0] = "<--"
        p_matrix[source, target, 0] = best_p
        p_matrix[target, source, 0] = best_p
        val_matrix[source, target, 0] = best_val
        val_matrix[target, source, 0] = best_val

    @staticmethod
    def _log(logger: Callable[[str], None] | None, message: str) -> None:
        if logger is not None:
            logger(message)
