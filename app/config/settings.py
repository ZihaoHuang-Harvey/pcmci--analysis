from __future__ import annotations

from pathlib import Path

import yaml

from app.core.models import (
    AdvancedRuleTemplate,
    AnalysisConfig,
    AppConfig,
    ManualConstraintRule,
    ConstraintType,
    RoleTemplate,
    VariableRole,
)


CONFIG_FILE_NAME = "app_settings.yaml"


def get_default_config_path() -> Path:
    """返回默认配置文件路径。"""

    return Path(__file__).resolve().parent / CONFIG_FILE_NAME


def _coerce_role(raw_value: str) -> VariableRole:
    try:
        return VariableRole(raw_value)
    except ValueError:
        return VariableRole.REGULAR


def _build_default_config() -> AppConfig:
    return AppConfig(
        defaults=AnalysisConfig(
            default_data_path="combined_all_with_dem(内插结果)_sin_cos_filtered.xlsx",
        ),
        role_templates={
            VariableRole.TIME_DRIVER: RoleTemplate(
                role=VariableRole.TIME_DRIVER,
                variable_names=["date_sin", "date_cos"],
                description="自动识别为时间驱动变量，无入边，可向外影响普通变量。",
            ),
            VariableRole.TERRAIN_DRIVER: RoleTemplate(
                role=VariableRole.TERRAIN_DRIVER,
                variable_names=["elevation_mean", "slope_mean", "aspect_mean"],
                description="自动识别为地形驱动变量，无入边，可向外影响普通变量。",
            ),
            VariableRole.REGULAR: RoleTemplate(
                role=VariableRole.REGULAR,
                variable_names=[],
                description="默认普通变量。",
            ),
        },
        advanced_rules={
            "block_target": AdvancedRuleTemplate(
                name="block_target",
                description="禁止其他变量影响目标变量，通常用于特殊先验。",
                rules=[],
            ),
            "block_source": AdvancedRuleTemplate(
                name="block_source",
                description="禁止某个变量向外影响其他变量，适合只想保留其被解释角色的场景。",
                rules=[],
            ),
            "block_specific_edge": AdvancedRuleTemplate(
                name="block_specific_edge",
                description="仅屏蔽一条特定方向关系，不影响其他边。",
                rules=[],
            ),
        },
        help_text={
            "step_data": "第 1 步：加载数据并决定哪些列不参与分析。程序只会把数值列送入 PCMCI，非数值列会自动忽略。",
            "step_roles": "第 2 步：设置变量角色。通常只需确认自动识别出的时间驱动和地形驱动是否正确；其余变量保持普通变量即可。",
            "step_run": "第 3 步：调整常用参数并运行分析。确认常用参数后即可运行 PCMCI+ 和 TE 分析。",
            "step_results": "第 4 步：查看结果摘要、因果图、时间序列图和矩阵导出。结果页会同步展示本次参数和角色设置。",
            "advanced_rules": "高级边约束面向已有明确先验的用户。若你不熟悉 no_in / no_out / no_link，保持默认不设置通常更稳妥。",
        },
    )


def load_app_config(config_path: Path | None = None) -> AppConfig:
    """从 YAML 加载配置，若文件不存在则回退默认值。"""

    path = config_path or get_default_config_path()
    default_config = _build_default_config()
    if not path.exists():
        return default_config

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults_data = raw.get("defaults", {})
    defaults = AnalysisConfig(
        tau_min=int(defaults_data.get("tau_min", default_config.defaults.tau_min)),
        tau_max=int(defaults_data.get("tau_max", default_config.defaults.tau_max)),
        pc_alpha=float(defaults_data.get("pc_alpha", default_config.defaults.pc_alpha)),
        quantile_range=tuple(defaults_data.get("quantile_range", list(default_config.defaults.quantile_range))),
        output_dir=str(defaults_data.get("output_dir", default_config.defaults.output_dir)),
        result_file_name=str(defaults_data.get("result_file_name", default_config.defaults.result_file_name)),
        log_level=str(defaults_data.get("log_level", default_config.defaults.log_level)),
        default_data_path=str(defaults_data.get("default_data_path", default_config.defaults.default_data_path)),
    )

    role_templates: dict[VariableRole, RoleTemplate] = {}
    for raw_role, template_data in (raw.get("role_templates", {}) or {}).items():
        role = _coerce_role(raw_role)
        role_templates[role] = RoleTemplate(
            role=role,
            variable_names=list(template_data.get("variable_names", []) or []),
            description=str(template_data.get("description", "")),
        )

    advanced_rules: dict[str, AdvancedRuleTemplate] = {}
    for name, template_data in (raw.get("advanced_rules", {}) or {}).items():
        rules = [
            ManualConstraintRule(
                source_name=str(item.get("source_name", "")),
                target_name=str(item.get("target_name", "")),
                constraint_type=ConstraintType(str(item.get("constraint_type", ConstraintType.NO_LINK.value))),
                description=str(item.get("description", "")),
            )
            for item in (template_data.get("rules", []) or [])
        ]
        advanced_rules[name] = AdvancedRuleTemplate(
            name=name,
            description=str(template_data.get("description", "")),
            rules=rules,
        )
    if not advanced_rules:
        advanced_rules = default_config.advanced_rules

    if not role_templates:
        role_templates = default_config.role_templates

    help_text = {
        key: str(value)
        for key, value in (raw.get("help_text", {}) or {}).items()
    }
    if not help_text:
        help_text = default_config.help_text

    return AppConfig(
        defaults=defaults,
        role_templates=role_templates,
        advanced_rules=advanced_rules,
        help_text=help_text,
    )


def save_app_config(config: AppConfig, config_path: Path | None = None) -> Path:
    """将配置写回 YAML。"""

    path = config_path or get_default_config_path()
    path.write_text(
        yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path
