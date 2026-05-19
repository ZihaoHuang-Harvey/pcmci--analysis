from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from matplotlib import colors, font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyArrowPatch, Patch


# 小米橙主题色板。
THEME_COLORS = {
    "background": "#f5f5f5",
    "surface": "#ffffff",
    "surface_alt": "#fafafa",
    "border": "#dedede",
    "text": "#303133",
    "text_muted": "#7f8c8d",
    "accent": "#ff6900",
    "accent_light": "#fff0e6",
    "slate": "#455a64",
    "warm_gray": "#d8cfc6",
    "negative": "#4d6373",
    "target_fill": "#ff8a3d",
    "node_fill": "#ffffff",
    "node_stroke": "#8d8d8d",
    "node_blue": "#1a4d7c",
}


@lru_cache(maxsize=1)
def get_chinese_font_properties() -> font_manager.FontProperties | None:
    """返回当前系统可用的中文字体，供 Matplotlib 图形标题和标签使用。"""

    preferred_fonts = (
        "Microsoft YaHei",
        "Noto Sans SC",
        "SimHei",
        "Microsoft JhengHei",
        "STSong",
        "FangSong",
    )
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            return font_manager.FontProperties(family=font_name)
    return None


APP_STYLESHEET = f"""
QMainWindow {{
    background-color: {THEME_COLORS["background"]};
    color: {THEME_COLORS["text"]};
}}
QWidget {{
    color: {THEME_COLORS["text"]};
    font-size: 12px;
}}
QGroupBox {{
    background: {THEME_COLORS["surface"]};
    border: 1px solid {THEME_COLORS["border"]};
    border-radius: 10px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: {THEME_COLORS["text"]};
}}
QTabWidget::pane {{
    border: 1px solid {THEME_COLORS["border"]};
    background: {THEME_COLORS["surface"]};
    border-radius: 10px;
}}
QTabBar::tab {{
    background: {THEME_COLORS["surface_alt"]};
    border: 1px solid {THEME_COLORS["border"]};
    padding: 8px 16px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{
    background: {THEME_COLORS["accent_light"]};
    color: {THEME_COLORS["accent"]};
    border-bottom-color: {THEME_COLORS["accent_light"]};
}}
QPushButton {{
    background: {THEME_COLORS["surface"]};
    border: 1px solid {THEME_COLORS["border"]};
    border-radius: 8px;
    padding: 8px 14px;
}}
QPushButton[primary="true"] {{
    background: {THEME_COLORS["accent"]};
    color: {THEME_COLORS["surface"]};
    border-color: {THEME_COLORS["accent"]};
    font-weight: 600;
}}
QPushButton:hover {{
    background: {THEME_COLORS["accent_light"]};
    border-color: {THEME_COLORS["accent"]};
}}
QPushButton:pressed {{
    background: #ffe3cf;
}}
QPushButton[primary="true"]:hover {{
    background: #ff7d1f;
    border-color: #ff7d1f;
}}
QPushButton[primary="true"]:pressed {{
    background: #ef6200;
    border-color: #ef6200;
}}
QPushButton:disabled {{
    color: #a8abb2;
    background: #f3f3f3;
}}
QPlainTextEdit, QListWidget, QTableWidget, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {THEME_COLORS["surface"]};
    border: 1px solid {THEME_COLORS["border"]};
    border-radius: 8px;
    padding: 6px;
    selection-background-color: {THEME_COLORS["accent_light"]};
    selection-color: {THEME_COLORS["text"]};
}}
QHeaderView::section {{
    background: #f7f7f7;
    border: none;
    border-bottom: 1px solid {THEME_COLORS["border"]};
    padding: 6px;
    font-weight: 600;
}}
QProgressBar {{
    border: 1px solid {THEME_COLORS["border"]};
    border-radius: 8px;
    background: #eeeeee;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {THEME_COLORS["accent"]};
    border-radius: 8px;
}}
"""


@dataclass(slots=True)
class UpstreamEdge:
    """折叠后的单条有向边。"""

    source: int
    target: int
    representative_lag: int
    value: float
    label: str


@dataclass(slots=True)
class UpstreamGraphData:
    """目标变量上游子图。"""

    target_index: int
    node_indices: list[int]
    edges: list[UpstreamEdge]
    levels: dict[int, int]
    empty_message: str = ""

    @property
    def has_relationships(self) -> bool:
        return bool(self.edges)


@dataclass(slots=True)
class TimeSeriesNodeSpec:
    """论文版时序图节点。"""

    var_index: int
    tau_back: int
    x: float
    y: float
    is_current: bool
    is_target_row: bool


@dataclass(slots=True)
class TimeSeriesEdgeSpec:
    """论文版时序图单条边。"""

    source_var: int
    target_var: int
    source_tau: int
    target_tau: int
    value: float
    style: str
    directed: bool


@dataclass(slots=True)
class TimeSeriesGraphData:
    """论文版目标中心时序图数据。"""

    target_index: int
    tau_max: int
    nodes: list[TimeSeriesNodeSpec]
    edges: list[TimeSeriesEdgeSpec]
    only_target_edges: bool
    only_related_vars: bool
    hide_historical_contemporaneous: bool
    hide_ambiguous_edges: bool
    related_var_indices: set[int]
    empty_message: str = ""

    @property
    def strict_edge_values(self) -> list[float]:
        return [edge.value for edge in self.edges if edge.style == "strict"]


def create_diverging_colormap() -> LinearSegmentedColormap:
    """创建不含纯白中心的发散配色。"""

    return LinearSegmentedColormap.from_list(
        "xiaomi_diverging",
        [
            THEME_COLORS["negative"],
            THEME_COLORS["warm_gray"],
            THEME_COLORS["accent"],
        ],
    )


def get_tigramite_plot_kwargs() -> dict[str, object]:
    """统一 Tigramite 绘图参数。"""

    colormap = create_diverging_colormap()
    return {
        "cmap_edges": colormap,
        "cmap_nodes": colormap,
        "node_aspect": 1.0,
        "label_fontsize": 9,
        "node_label_size": 9,
        "tick_label_size": 7,
        "arrow_linewidth": 3.5,
        "arrowhead_size": 14,
        "node_size": 0.22,
        "vmin_edges": -1.0,
        "vmax_edges": 1.0,
        "vmin_nodes": -1.0,
        "vmax_nodes": 1.0,
        "link_colorbar_label": "MCI",
        "node_colorbar_label": "auto-MCI",
        "alpha": 0.98,
    }


def style_result_axes(ax, square: bool = False) -> None:
    """统一结果图区样式。"""

    ax.set_facecolor(THEME_COLORS["surface_alt"])
    ax.figure.set_facecolor(THEME_COLORS["surface"])
    if square:
        ax.set_aspect("equal", adjustable="box")
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)


def apply_colorbar_theme(fig) -> None:
    """对图中的 colorbar 和坐标轴做浅色主题修饰。"""

    for axis in fig.axes:
        axis.set_facecolor(THEME_COLORS["surface_alt"])
        for spine in axis.spines.values():
            spine.set_color(THEME_COLORS["border"])
        axis.tick_params(colors=THEME_COLORS["text_muted"])
        axis.title.set_color(THEME_COLORS["text"])
        axis.xaxis.label.set_color(THEME_COLORS["text"])
        axis.yaxis.label.set_color(THEME_COLORS["text"])


def _classify_edge_symbol(symbol: str) -> str:
    """把 Tigramite 的边字符串粗分成严格定向 / 模糊定向 / 模糊无向 / 空。"""

    if not symbol:
        return "none"
    text = str(symbol)
    if text == "-->":
        return "strict_forward"
    if text == "<--":
        return "strict_backward"
    if text.endswith(">") and not text.startswith("<"):
        return "ambiguous_forward"
    if text.startswith("<") and not text.endswith(">"):
        return "ambiguous_backward"
    return "ambiguous_undirected"


def _build_time_series_nodes(var_count: int, tau_max: int, target_index: int) -> list[TimeSeriesNodeSpec]:
    """构造固定网格坐标节点。"""

    nodes: list[TimeSeriesNodeSpec] = []
    x_gap = 1.4
    y_gap = 0.85

    for var_index in range(var_count):
        y_pos = (var_count - 1 - var_index) * y_gap
        for tau_back in range(tau_max, -1, -1):
            x_pos = (tau_max - tau_back) * x_gap
            nodes.append(
                TimeSeriesNodeSpec(
                    var_index=var_index,
                    tau_back=tau_back,
                    x=x_pos,
                    y=y_pos,
                    is_current=tau_back == 0,
                    is_target_row=var_index == target_index,
                )
            )

    return nodes


def build_target_centric_time_series_data(
    graph: np.ndarray,
    p_matrix: np.ndarray,
    val_matrix: np.ndarray,
    var_names: list[str],
    pc_alpha: float,
    target_name: str,
    only_target_edges: bool,
    only_related_vars: bool = False,
    hide_historical_contemporaneous: bool = True,
    hide_ambiguous_edges: bool = True,
) -> TimeSeriesGraphData:
    """按目标变量和过滤开关构造论文版时序图数据。

    该函数不直接绘图，只负责：
    - 解析 PCMCI 结果矩阵；
    - 过滤非显著、非目标、历史同期或未决断边；
    - 在"全网模式"下把滞后边复制到所有合法时间列；
    - 产出固定网格节点和可渲染的边列表。
    """

    if target_name not in var_names:
        raise ValueError(f"目标变量 {target_name} 不在分析结果中。")

    target_index = var_names.index(target_name)
    tau_max = graph.shape[2] - 1

    related_var_indices: set[int] = set()
    if only_related_vars:
        related_var_indices.add(target_index)
        for source in range(len(var_names)):
            for target in range(len(var_names)):
                for lag in range(tau_max + 1):
                    if p_matrix[source, target, lag] <= pc_alpha:
                        if source == target_index or target == target_index:
                            related_var_indices.add(source)
                            related_var_indices.add(target)

    nodes = _build_time_series_nodes(len(var_names), tau_max, target_index)
    edges: list[TimeSeriesEdgeSpec] = []

    for source in range(len(var_names)):
        for target in range(len(var_names)):
            if only_related_vars:
                if source == target:
                    if source != target_index:
                        continue
                else:
                    if source != target_index and target != target_index:
                        continue
                    if source not in related_var_indices or target not in related_var_indices:
                        continue

            for lag in range(1, tau_max + 1):
                symbol = str(graph[source, target, lag])
                if not symbol or p_matrix[source, target, lag] > pc_alpha:
                    continue

                classification = _classify_edge_symbol(symbol)
                if classification == "strict_backward":
                    continue
                if hide_ambiguous_edges and classification != "strict_forward":
                    continue

                style = "strict" if classification == "strict_forward" else "ambiguous_directed"

                if only_target_edges:
                    if target != target_index:
                        continue
                    edges.append(
                        TimeSeriesEdgeSpec(
                            source_var=source,
                            target_var=target,
                            source_tau=lag,
                            target_tau=0,
                            value=float(val_matrix[source, target, lag]),
                            style=style,
                            directed=True,
                        )
                    )
                    continue

                for target_tau in range(0, tau_max - lag + 1):
                    source_tau = target_tau + lag
                    edges.append(
                        TimeSeriesEdgeSpec(
                            source_var=source,
                            target_var=target,
                            source_tau=source_tau,
                            target_tau=target_tau,
                            value=float(val_matrix[source, target, lag]),
                            style=style,
                            directed=True,
                        )
                    )

    for left in range(len(var_names)):
        for right in range(left + 1, len(var_names)):
            if only_related_vars:
                if left != target_index and right != target_index:
                    continue
                if left not in related_var_indices or right not in related_var_indices:
                    continue

            left_to_right = str(graph[left, right, 0])
            right_to_left = str(graph[right, left, 0])
            pair_significant = (
                min(p_matrix[left, right, 0], p_matrix[right, left, 0]) <= pc_alpha
            )
            if not pair_significant:
                continue

            left_class = _classify_edge_symbol(left_to_right)
            right_class = _classify_edge_symbol(right_to_left)

            if left_class == "strict_forward" and right_class == "strict_backward":
                relation_style = "strict"
                relation_directed = True
                source_var = left
                target_var = right
                value = float(val_matrix[left, right, 0])
            elif left_class == "strict_backward" and right_class == "strict_forward":
                relation_style = "strict"
                relation_directed = True
                source_var = right
                target_var = left
                value = float(val_matrix[right, left, 0])
            else:
                if hide_ambiguous_edges:
                    continue

                relation_style = "ambiguous_undirected"
                relation_directed = False
                source_var = left
                target_var = right
                value = float(
                    max(
                        abs(val_matrix[left, right, 0]),
                        abs(val_matrix[right, left, 0]),
                    )
                )

                if left_class in {"ambiguous_forward", "ambiguous_backward"}:
                    relation_style = "ambiguous_directed"
                    relation_directed = True
                    if left_class == "ambiguous_forward":
                        source_var = left
                        target_var = right
                        value = float(val_matrix[left, right, 0])
                    else:
                        source_var = right
                        target_var = left
                        value = float(val_matrix[right, left, 0])
                elif right_class in {"ambiguous_forward", "ambiguous_backward"}:
                    relation_style = "ambiguous_directed"
                    relation_directed = True
                    if right_class == "ambiguous_forward":
                        source_var = right
                        target_var = left
                        value = float(val_matrix[right, left, 0])
                    else:
                        source_var = left
                        target_var = right
                        value = float(val_matrix[left, right, 0])

            if only_target_edges:
                if target_var != target_index:
                    continue
                target_columns = [0]
            else:
                if hide_historical_contemporaneous:
                    target_columns = [0]
                else:
                    target_columns = list(range(0, tau_max + 1))

            for target_tau in target_columns:
                edges.append(
                    TimeSeriesEdgeSpec(
                        source_var=source_var,
                        target_var=target_var,
                        source_tau=target_tau,
                        target_tau=target_tau,
                        value=value,
                        style=relation_style,
                        directed=relation_directed,
                    )
                )

    empty_message = ""
    if not edges:
        empty_message = "当前过滤条件下没有可显示的关系边。"

    return TimeSeriesGraphData(
        target_index=target_index,
        tau_max=tau_max,
        nodes=nodes,
        edges=edges,
        only_target_edges=only_target_edges,
        only_related_vars=only_related_vars,
        hide_historical_contemporaneous=hide_historical_contemporaneous,
        hide_ambiguous_edges=hide_ambiguous_edges,
        related_var_indices=related_var_indices,
        empty_message=empty_message,
    )


def build_upstream_graph_data(
    graph: np.ndarray,
    p_matrix: np.ndarray,
    val_matrix: np.ndarray,
    var_names: list[str],
    pc_alpha: float,
    target_name: str,
) -> UpstreamGraphData:
    """构造“所有最终能到达目标变量”的上游子图。

    规则：
    - 只保留显著关系；
    - 同一 source -> target 多个 lag 折叠为一条边；
    - 代表边取绝对 MCI 最大的 lag；
    - 仅保留所有最终能通向目标变量的节点与边。
    """

    if target_name not in var_names:
        raise ValueError(f"目标变量 {target_name} 不在分析结果中。")

    target_index = var_names.index(target_name)
    collapsed_edges: dict[tuple[int, int], UpstreamEdge] = {}
    tau_count = graph.shape[2]

    for source in range(len(var_names)):
        for target in range(len(var_names)):
            if source == target:
                continue

            candidates: list[UpstreamEdge] = []

            if (
                graph[source, target, 0] == "-->"
                and graph[target, source, 0] == "<--"
                and min(p_matrix[source, target, 0], p_matrix[target, source, 0]) <= pc_alpha
            ):
                candidates.append(
                    UpstreamEdge(
                        source=source,
                        target=target,
                        representative_lag=0,
                        value=float(val_matrix[source, target, 0]),
                        label=f"t0 | {val_matrix[source, target, 0]:.2f}",
                    )
                )

            for tau in range(1, tau_count):
                if graph[source, target, tau] and p_matrix[source, target, tau] <= pc_alpha:
                    candidates.append(
                        UpstreamEdge(
                            source=source,
                            target=target,
                            representative_lag=tau,
                            value=float(val_matrix[source, target, tau]),
                            label=f"t-{tau} | {val_matrix[source, target, tau]:.2f}",
                        )
                    )

            if not candidates:
                continue

            best_edge = max(
                candidates,
                key=lambda edge: (abs(edge.value), -edge.representative_lag),
            )
            collapsed_edges[(source, target)] = best_edge

    predecessors: dict[int, set[int]] = defaultdict(set)
    successors: dict[int, set[int]] = defaultdict(set)
    for edge in collapsed_edges.values():
        predecessors[edge.target].add(edge.source)
        successors[edge.source].add(edge.target)

    node_set = {target_index}
    queue: deque[int] = deque([target_index])
    while queue:
        current = queue.popleft()
        for predecessor in predecessors.get(current, set()):
            if predecessor not in node_set:
                node_set.add(predecessor)
                queue.append(predecessor)

    if node_set == {target_index}:
        return UpstreamGraphData(
            target_index=target_index,
            node_indices=[target_index],
            edges=[],
            levels={target_index: 0},
            empty_message=f"{target_name} 当前没有显著上游影响路径。",
        )

    levels = {target_index: 0}
    queue = deque([target_index])
    while queue:
        current = queue.popleft()
        for predecessor in predecessors.get(current, set()):
            next_level = levels[current] + 1
            if predecessor not in levels or next_level < levels[predecessor]:
                levels[predecessor] = next_level
                queue.append(predecessor)

    filtered_edges = [
        edge
        for edge in collapsed_edges.values()
        if edge.source in node_set and edge.target in node_set
    ]

    return UpstreamGraphData(
        target_index=target_index,
        node_indices=sorted(node_set, key=lambda index: (levels.get(index, 99), var_names[index])),
        edges=filtered_edges,
        levels=levels,
    )


def draw_upstream_graph(ax, data: UpstreamGraphData, var_names: list[str]) -> None:
    """绘制目标变量上游影响图。"""

    ax.clear()
    style_result_axes(ax, square=True)
    ax.set_axis_off()

    max_level = max(data.levels.values(), default=0)
    level_groups: dict[int, list[int]] = defaultdict(list)
    for node in data.node_indices:
        level_groups[data.levels[node]].append(node)

    positions: dict[int, tuple[float, float]] = {}
    for level, indices in level_groups.items():
        x = max_level - level
        ordered_indices = sorted(indices, key=lambda index: var_names[index])
        count = len(ordered_indices)
        if count == 1:
            y_values = [0.0]
        else:
            y_values = np.linspace(-(count - 1), count - 1, count)
        for index, y in zip(ordered_indices, y_values):
            positions[index] = (x * 2.4, y * 1.0)

    radius = 0.32
    color_map = create_diverging_colormap()
    norm = Normalize(vmin=-1.0, vmax=1.0)

    if data.has_relationships:
        for edge in data.edges:
            start_x, start_y = positions[edge.source]
            end_x, end_y = positions[edge.target]
            direction = np.array([end_x - start_x, end_y - start_y], dtype=float)
            length = np.linalg.norm(direction)
            if length == 0:
                continue
            direction /= length
            start = (start_x + direction[0] * radius, start_y + direction[1] * radius)
            end = (end_x - direction[0] * radius, end_y - direction[1] * radius)
            patch = FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4 + abs(edge.value) * 1.8,
                color=color_map(norm(edge.value)),
                alpha=0.95,
                connectionstyle="arc3,rad=0.04",
            )
            ax.add_patch(patch)
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(
                mid_x,
                mid_y + 0.14,
                edge.label,
                ha="center",
                va="center",
                fontsize=7,
                color=THEME_COLORS["text"],
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": THEME_COLORS["surface"],
                    "edgecolor": THEME_COLORS["border"],
                    "linewidth": 0.7,
                },
            )

        sm = ScalarMappable(norm=norm, cmap=color_map)
        colorbar = ax.figure.colorbar(sm, ax=ax, fraction=0.04, pad=0.03)
        colorbar.set_label("MCI", color=THEME_COLORS["text"])
        colorbar.ax.tick_params(colors=THEME_COLORS["text_muted"])
        colorbar.outline.set_edgecolor(THEME_COLORS["border"])

    for index, (x_pos, y_pos) in positions.items():
        is_target = index == data.target_index
        circle = Circle(
            (x_pos, y_pos),
            radius=radius,
            facecolor=THEME_COLORS["target_fill"] if is_target else THEME_COLORS["node_fill"],
            edgecolor=THEME_COLORS["accent"] if is_target else THEME_COLORS["node_stroke"],
            linewidth=1.8 if is_target else 1.0,
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            x_pos,
            y_pos,
            var_names[index],
            ha="center",
            va="center",
            fontsize=8.5,
            color=THEME_COLORS["surface"] if is_target else THEME_COLORS["text"],
            zorder=4,
            wrap=True,
        )

    xs = [position[0] for position in positions.values()]
    ys = [position[1] for position in positions.values()]
    ax.set_xlim(min(xs) - 0.9, max(xs) + 1.4)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)

    title = f"所有最终通向 {var_names[data.target_index]} 的上游影响路径"
    if data.empty_message:
        title += f"\n{data.empty_message}"
    ax.set_title(title, fontsize=11, color=THEME_COLORS["text"], pad=12)


def _compute_trimmed_points(
    start: tuple[float, float],
    end: tuple[float, float],
    radius: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """把连线端点收缩到圆节点边缘，避免穿透节点。"""

    direction = np.array([end[0] - start[0], end[1] - start[1]], dtype=float)
    length = np.linalg.norm(direction)
    if length == 0:
        return start, end
    direction /= length
    start_point = (start[0] + direction[0] * radius, start[1] + direction[1] * radius)
    end_point = (end[0] - direction[0] * radius, end[1] - direction[1] * radius)
    return start_point, end_point


def _edge_connection_radius(source: tuple[float, float], target: tuple[float, float]) -> float:
    """根据时间跨度和垂直跨度给弧线一个稳定曲率。"""

    span_x = abs(target[0] - source[0])
    if span_x == 0:
        return 0.28 if source[1] >= target[1] else -0.28

    base = 0.12 + min(span_x / 6.0, 0.22)
    if abs(source[1] - target[1]) < 0.1:
        return base
    return base if source[1] > target[1] else -base


def draw_target_centric_time_series(
    ax,
    data: TimeSeriesGraphData,
    var_names: list[str],
) -> None:
    """绘制论文版目标中心时序图。"""

    ax.clear()
    style_result_axes(ax, square=False)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    if data.only_related_vars:
        vars_to_show: set[int] = data.related_var_indices
    else:
        vars_to_show: set[int] = set(range(len(var_names)))

    filtered_nodes = [
        node for node in data.nodes
        if node.var_index in vars_to_show
    ]

    node_positions = {
        (node.var_index, node.tau_back): (node.x, node.y)
        for node in filtered_nodes
    }
    node_radius = 0.14
    edge_cmap = create_diverging_colormap()
    edge_norm = Normalize(vmin=-1.0, vmax=1.0)

    strict_values = data.strict_edge_values
    if strict_values:
        sm = ScalarMappable(norm=edge_norm, cmap=edge_cmap)
        colorbar = ax.figure.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            fraction=0.035,
            pad=0.06,
        )
        colorbar.set_label("MCI", color=THEME_COLORS["text"])
        colorbar.ax.tick_params(colors=THEME_COLORS["text_muted"])
        colorbar.outline.set_edgecolor(THEME_COLORS["border"])

    for edge in data.edges:
        source_position = node_positions[(edge.source_var, edge.source_tau)]
        target_position = node_positions[(edge.target_var, edge.target_tau)]
        start_point, end_point = _compute_trimmed_points(source_position, target_position, node_radius)

        if edge.style == "strict":
            color = edge_cmap(edge_norm(edge.value))
            linestyle = "-"
            linewidth = 1.5 + abs(edge.value) * 2.2
            arrowstyle = "-|>"
        elif edge.directed:
            color = THEME_COLORS["text_muted"]
            linestyle = (0, (4, 3))
            linewidth = 1.4
            arrowstyle = "-|>"
        else:
            color = "#aeb4bb"
            linestyle = (0, (4, 3))
            linewidth = 1.2
            arrowstyle = "-"

        patch = FancyArrowPatch(
            start_point,
            end_point,
            arrowstyle=arrowstyle,
            mutation_scale=11,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            alpha=0.95,
            connectionstyle=f"arc3,rad={_edge_connection_radius(source_position, target_position)}",
            zorder=2,
        )
        ax.add_patch(patch)

    for node in filtered_nodes:
        is_target_current = node.is_current and node.is_target_row
        is_current_other = node.is_current and not node.is_target_row
        edgecolor = "#000000"
        facecolor = "#ffffff"

        circle = Circle(
            (node.x, node.y),
            radius=node_radius,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=2.0 if node.is_current else 1.7,
            zorder=3,
        )
        ax.add_patch(circle)

    tau_columns = list(range(data.tau_max, -1, -1))
    first_var = min(vars_to_show) if vars_to_show else 0
    x_positions = [node_positions[(first_var, tau)][0] for tau in tau_columns] if vars_to_show else []
    for tau_back, x_pos in zip(tau_columns, x_positions):
        ax.text(
            x_pos,
            max(node.y for node in filtered_nodes) + 0.45,
            str(tau_back),
            ha="center",
            va="bottom",
            fontsize=10,
            color=THEME_COLORS["text"],
            fontweight="600",
        )

    ax.text(
        min(x_positions) - 0.9 if x_positions else -0.9,
        max(node.y for node in filtered_nodes) + 0.45,
        "tau =",
        ha="left",
        va="bottom",
        fontsize=11,
        color=THEME_COLORS["text"],
        fontweight="600",
    )

    for var_index in sorted(vars_to_show):
        if (var_index, 0) not in node_positions:
            continue
        name = var_names[var_index]
        y_pos = node_positions[(var_index, 0)][1]
        color = THEME_COLORS["accent"] if var_index == data.target_index else THEME_COLORS["text"]
        weight = "600" if var_index == data.target_index else "400"
        ax.text(
            min(x_positions) - 0.55 if x_positions else -0.55,
            y_pos,
            name,
            ha="right",
            va="center",
            fontsize=9,
            color=color,
            fontweight=weight,
        )

    title = (
        f"目标中心论文版时序图：{var_names[data.target_index]}(t)"
        if data.only_target_edges
        else f"论文版时序图：完整时滞演化网络（目标参考：{var_names[data.target_index]}）"
    )
    if data.empty_message:
        title += f"\n{data.empty_message}"
    ax.set_title(title, fontsize=11, color=THEME_COLORS["text"], pad=14)

    xs = [node.x for node in filtered_nodes]
    ys = [node.y for node in filtered_nodes]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 0.7)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.8)


def middle_color_hex(colormap: colors.Colormap) -> str:
    """返回 colormap 中点颜色，便于测试是否为纯白。"""

    rgba = colormap(0.5)
    return colors.to_hex(rgba)


def draw_mci_bar_chart(
    ax,
    val_matrix: np.ndarray,
    p_matrix: np.ndarray,
    var_names: list[str],
    target_name: str,
    tau_min: int,
    tau_max: int,
    pc_alpha: float,
) -> None:
    """绘制每个变量对目标变量的 MCI 柱状图。

    对每个源变量，取所有滞后期中绝对值最大的 MCI 值作为代表，
    用分组柱状图同时展示正向（橙色）和负向（蓝色）影响。
    """

    if target_name not in var_names:
        raise ValueError(f"目标变量 {target_name} 不在分析结果中。")

    target_idx = var_names.index(target_name)
    source_indices = list(range(len(var_names)))

    display_names: list[str] = []
    mci_values: list[float] = []
    bar_labels: list[str] = []
    bar_colors: list[str] = []

    for src_idx in source_indices:
        name = var_names[src_idx]
        best_pos_val = 0.0
        best_neg_val = 0.0
        best_pos_label = ""
        best_neg_label = ""
        best_pos_tau = -1
        best_neg_tau = -1

        for tau in range(max(1, tau_min), tau_max + 1):
            val = float(val_matrix[src_idx, target_idx, tau])
            p_val = float(p_matrix[src_idx, target_idx, tau])
            if p_val > pc_alpha:
                continue

            if val > 0 and val > best_pos_val:
                best_pos_val = val
                best_pos_label = f"{val:.3f}"
                best_pos_tau = tau
            elif val < 0 and abs(val) > abs(best_neg_val):
                best_neg_val = val
                best_neg_label = f"{val:.3f}"
                best_neg_tau = tau

        tau0_val = float(val_matrix[src_idx, target_idx, 0])
        tau0_p = float(p_matrix[src_idx, target_idx, 0])
        if src_idx != target_idx and tau0_p <= pc_alpha:
            if tau0_val > 0 and tau0_val > best_pos_val:
                best_pos_val = tau0_val
                best_pos_label = f"{tau0_val:.3f}"
                best_pos_tau = 0
            elif tau0_val < 0 and abs(tau0_val) > abs(best_neg_val):
                best_neg_val = tau0_val
                best_neg_label = f"{tau0_val:.3f}"
                best_neg_tau = 0

        if best_pos_tau >= 0:
            display_names.append(f"{name} (+t{best_pos_tau})" if best_pos_tau > 0 else f"{name} (+t0)")
            mci_values.append(best_pos_val)
            bar_labels.append(best_pos_label)
            bar_colors.append(THEME_COLORS["accent"])

        if best_neg_tau >= 0:
            display_names.append(f"{name} (-t{best_neg_tau})" if best_neg_tau > 0 else f"{name} (-t0)")
            mci_values.append(abs(best_neg_val))
            bar_labels.append(best_neg_label)
            bar_colors.append(THEME_COLORS["negative"])

    x = np.arange(len(display_names))
    width = 0.7

    font_prop = get_chinese_font_properties()
    bars = ax.bar(x, mci_values, width, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.6)

    for bar, label_text in zip(bars, bar_labels):
        if label_text and abs(float(label_text)) > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label_text, ha="center", va="bottom", fontsize=7.5,
                    fontproperties=font_prop)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=7.5, fontproperties=font_prop)

    ax.set_ylabel("MCI", fontsize=10, fontproperties=font_prop)
    ax.set_title(f"各变量对「{target_name}」的 MCI 值", fontsize=11, fontweight="bold",
                 fontproperties=font_prop, pad=10)

    legend_elements = [
        Patch(facecolor=THEME_COLORS["accent"], alpha=0.85, edgecolor="white", linewidth=0.6, label="Positive MCI"),
        Patch(facecolor=THEME_COLORS["negative"], alpha=0.85, edgecolor="white", linewidth=0.6, label="Negative MCI")
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    style_result_axes(ax)


def draw_te_target_bar_chart(
    ax,
    te_matrix: np.ndarray,
    ndte_matrix: np.ndarray,
    var_names: list[str],
    target_name: str,
    tau_max: int,
) -> None:
    """绘制其他变量对目标变量的 TE 值柱状图（每个变量只显示最大期次的 TE 和 NDTE）。"""

    if target_name not in var_names:
        raise ValueError(f"目标变量 {target_name} 不在分析结果中。")

    target_idx = var_names.index(target_name)
    source_indices = list(range(len(var_names)))

    display_names: list[str] = []
    te_values: list[float] = []
    ndte_values: list[float] = []
    te_labels: list[str] = []
    ndte_labels: list[str] = []

    for src_idx in source_indices:
        name = var_names[src_idx]
        # 只取最大期次的值
        te_val = float(te_matrix[src_idx, target_idx, tau_max])
        ndte_val = float(ndte_matrix[src_idx, target_idx, tau_max])

        display_names.append(name)
        te_values.append(te_val)
        ndte_values.append(ndte_val)
        te_labels.append(f"{te_val:.3f}" if te_val > 0.01 else "")
        ndte_labels.append(f"{ndte_val:.3f}" if ndte_val > 0.01 else "")

    x = np.arange(len(display_names))
    width = 0.35

    font_prop = get_chinese_font_properties()
    bars_te = ax.bar(x - width / 2, te_values, width, color='#d62728', alpha=0.85, edgecolor="white", linewidth=0.6, label="TE")
    bars_ndte = ax.bar(x + width / 2, ndte_values, width, color='#1f77b4', alpha=0.85, edgecolor="white", linewidth=0.6, label="NDTE")

    for bar, label_text in zip(bars_te, te_labels):
        if label_text:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label_text, ha="center", va="bottom", fontsize=7.5,
                    fontproperties=font_prop)

    for bar, label_text in zip(bars_ndte, ndte_labels):
        if label_text:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label_text, ha="center", va="bottom", fontsize=7.5,
                    fontproperties=font_prop)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=7.5, fontproperties=font_prop)

    ax.set_ylabel("信息强度", fontsize=10, fontproperties=font_prop)
    ax.set_title(f"各变量对「{target_name}」的 TE/NDTE 值 (t{tau_max})", fontsize=11, fontweight="bold",
                 fontproperties=font_prop, pad=10)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    style_result_axes(ax)
