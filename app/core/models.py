from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ConstraintType(str, Enum):
    """高级手工边约束类型。"""

    NO_OUT = "no_out"
    NO_IN = "no_in"
    NO_LINK = "no_link"


class VariableRole(str, Enum):
    """面向用户的变量角色。

    regular:
        普通变量，按照 PCMCI 默认方式建模。
    time_driver:
        时间驱动变量，无任何入边，但可以影响普通变量。
    terrain_driver:
        地形驱动变量，无任何入边，但可以影响普通变量。
    excluded:
        从分析中排除。
    """

    REGULAR = "regular"
    TIME_DRIVER = "time_driver"
    TERRAIN_DRIVER = "terrain_driver"
    EXCLUDED = "excluded"


ROLE_LABELS = {
    VariableRole.REGULAR: "普通变量",
    VariableRole.TIME_DRIVER: "时间驱动",
    VariableRole.TERRAIN_DRIVER: "地形驱动",
    VariableRole.EXCLUDED: "排除变量",
}


ROLE_DESCRIPTIONS = {
    VariableRole.REGULAR: "按常规 PCMCI 方式建模，可接收其他变量影响，也可影响其他变量。",
    VariableRole.TIME_DRIVER: "无任何入边，可对普通变量产生滞后或同时刻影响。",
    VariableRole.TERRAIN_DRIVER: "无任何入边，可对普通变量产生滞后或同时刻影响。",
    VariableRole.EXCLUDED: "不会参与本次分析，也不会出现在结果图中。",
}


@dataclass(slots=True)
class ManualConstraintRule:
    """用户在高级设置中手动添加的一条边约束。"""

    source_name: str
    target_name: str
    constraint_type: ConstraintType
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "source_name": self.source_name,
            "target_name": self.target_name,
            "constraint_type": self.constraint_type.value,
            "description": self.description,
        }


@dataclass(slots=True)
class ResolvedConstraintRule:
    """将变量名解析成当前分析索引后的高级约束。"""

    source_index: int
    target_index: int
    source_name: str
    target_name: str
    constraint_type: ConstraintType
    description: str = ""


@dataclass(slots=True)
class AnalysisConfig:
    """PCMCI 分析参数。"""

    tau_min: int = 0
    tau_max: int = 3
    pc_alpha: float = 0.05
    quantile_range: tuple[int, int] = (25, 75)
    output_dir: str = "outputs"
    result_file_name: str = "pcmci_results.pkl"
    log_level: str = "INFO"
    default_data_path: str = ""
    manual_rules: list[ManualConstraintRule] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "pc_alpha": self.pc_alpha,
            "quantile_range": list(self.quantile_range),
            "output_dir": self.output_dir,
            "result_file_name": self.result_file_name,
            "log_level": self.log_level,
            "default_data_path": self.default_data_path,
            "manual_rules": [r.to_dict() for r in self.manual_rules],
        }


@dataclass(slots=True)
class RoleTemplate:
    """角色自动识别模板。"""

    role: VariableRole
    variable_names: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable_names": list(self.variable_names),
            "description": self.description,
        }


@dataclass(slots=True)
class AdvancedRuleTemplate:
    """高级规则示例模板，用于界面说明。"""

    name: str
    description: str
    rules: list[ManualConstraintRule] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "rules": [rule.to_dict() for rule in self.rules],
        }


@dataclass(slots=True)
class AppConfig:
    """应用配置。"""

    defaults: AnalysisConfig = field(default_factory=AnalysisConfig)
    role_templates: dict[VariableRole, RoleTemplate] = field(default_factory=dict)
    advanced_rules: dict[str, AdvancedRuleTemplate] = field(default_factory=dict)
    help_text: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "defaults": self.defaults.to_dict(),
            "role_templates": {
                role.value: template.to_dict()
                for role, template in self.role_templates.items()
            },
            "advanced_rules": {
                name: template.to_dict()
                for name, template in self.advanced_rules.items()
            },
            "help_text": dict(self.help_text),
        }


@dataclass(slots=True)
class CompiledConstraints:
    """统一约束编译结果。

    allowed_lagged:
        [source, target] 布尔矩阵，表示 tau>0 的方向是否允许存在。
    allowed_contemporaneous:
        [source, target] 布尔矩阵，表示 tau=0 时 source -> target 方向是否允许。
    incoming_blocked:
        不允许任何入边的变量索引集合，主要用于结果摘要和校验。
    lagged_outgoing / contemporaneous_outgoing:
        记录角色驱动产生的“允许向外影响”的范围，便于结果摘要说明。
    """

    link_assumptions: dict[int, dict[tuple[int, int], str]] | None
    allowed_lagged: np.ndarray
    allowed_contemporaneous: np.ndarray
    incoming_blocked: set[int] = field(default_factory=set)
    lagged_outgoing: dict[int, set[int]] = field(default_factory=dict)
    contemporaneous_outgoing: dict[int, set[int]] = field(default_factory=dict)
    role_assignments: dict[int, VariableRole] = field(default_factory=dict)
    resolved_rules: list[ResolvedConstraintRule] = field(default_factory=list)

    @property
    def has_constraints(self) -> bool:
        return bool(
            self.link_assumptions is not None
            or self.resolved_rules
            or self.incoming_blocked
        )


@dataclass(slots=True)
class AnalysisResult:
    """统一分析结果。"""

    var_names: list[str]
    tau_min: int
    tau_max: int
    pc_alpha: float
    graph: np.ndarray
    p_matrix: np.ndarray
    val_matrix: np.ndarray
    adj_matrix: np.ndarray
    raw_results: dict[str, Any]
    compiled_constraints: CompiledConstraints
    summary: dict[str, Any] = field(default_factory=dict)
    links: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "var_names": self.var_names,
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "pc_alpha": self.pc_alpha,
            "graph": self.graph,
            "p_matrix": self.p_matrix,
            "val_matrix": self.val_matrix,
            "adj_matrix": self.adj_matrix,
            "raw_results": self.raw_results,
            "compiled_constraints": self.compiled_constraints,
            "summary": self.summary,
            "links": self.links,
        }


@dataclass(slots=True)
class TEConfig:
    """TE（转移熵）分析参数。"""

    enabled: bool = False
    bins: int = 10
    k_history: int = 1
    tau_max: int = 3
    target_var: str = ""
    analyze_all: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "bins": self.bins,
            "k_history": self.k_history,
            "tau_max": self.tau_max,
            "target_var": self.target_var,
            "analyze_all": self.analyze_all,
        }


@dataclass(slots=True)
class TEResult:
    """TE（转移熵）分析结果。"""

    var_names: list[str]
    config: TEConfig
    te_matrix: np.ndarray
    ndte_matrix: np.ndarray
    significant_pairs: list[tuple[str, str, int, float, float]]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "var_names": self.var_names,
            "config": self.config.to_dict(),
            "te_matrix": self.te_matrix,
            "ndte_matrix": self.ndte_matrix,
            "significant_pairs": [
                {"source": s, "target": t, "lag": lag, "te": te, "ndte": ndte}
                for s, t, lag, te, ndte in self.significant_pairs
            ],
            "summary": self.summary,
        }
