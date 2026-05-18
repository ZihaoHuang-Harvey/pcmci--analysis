from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from tigramite import plotting as tp

from app.config.settings import load_app_config
from app.core.constraint_engine import ConstraintEngine
from app.core.models import (
    AnalysisConfig,
    AnalysisResult,
    AppConfig,
    ConstraintType,
    ManualConstraintRule,
    ROLE_DESCRIPTIONS,
    ROLE_LABELS,
    TEConfig,
    TEResult,
    VariableRole,
)
from app.gui.analysis_worker import AnalysisWorker
from app.gui.canvas import CanvasWidget
from app.gui.result_plotting import (
    APP_STYLESHEET,
    THEME_COLORS,
    apply_colorbar_theme,
    build_target_centric_time_series_data,
    build_upstream_graph_data,
    draw_upstream_graph,
    draw_target_centric_time_series,
    get_chinese_font_properties,
    get_tigramite_plot_kwargs,
    style_result_axes,
)


matplotlib.use("Qt5Agg")


class MainWindow(QMainWindow):
    """多方法交互因果分析平台主窗口。

    整个界面按用户操作顺序组织成 4 步：
    1. 数据导入
    2. 变量角色
    3. 参数与运行（PCMCI+ / TE 平行设置）
    4. 结果查看（PCMCI+ / TE 各自独立展示）
    """

    def __init__(self) -> None:
        super().__init__()
        self.project_root = Path(__file__).resolve().parents[2]
        self.app_config: AppConfig = load_app_config()

        self.raw_df: pd.DataFrame | None = None
        self.current_result: AnalysisResult | None = None
        self.current_result_path: Path | None = None
        self.current_te_result: TEResult | None = None
        self.worker: AnalysisWorker | None = None

        self.excluded_columns: set[str] = set()
        self.current_role_mapping: dict[str, VariableRole] = {}
        self.manual_constraints: list[ManualConstraintRule] = []

        self._build_ui()
        self._apply_theme()
        self._bind_signals()
        self._load_defaults_from_config()
        self._refresh_everything()
        self._mark_primary_actions()

    def _build_ui(self) -> None:
        self.setWindowTitle("多方法交互因果分析平台")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        self._create_menu_bar()

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        self.step_tabs = QTabWidget()
        root_layout.addWidget(self.step_tabs, stretch=1)
        self.step_tabs.addTab(self._build_data_tab(), "1. 数据导入")
        self.step_tabs.addTab(self._build_roles_tab(), "2. 变量角色")
        self.step_tabs.addTab(self._build_run_tab(), "3. 参数与运行")
        self.step_tabs.addTab(self._build_results_tab(), "4. 结果查看")

        status_group = QGroupBox("运行状态")
        status_group.setMaximumHeight(120)
        status_layout = QVBoxLayout(status_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1000)
        self.log_text.setMaximumHeight(60)
        self.log_text.setPlaceholderText("这里会显示运行日志、参数摘要和异常信息。")
        status_layout.addWidget(self.log_text, stretch=1)
        root_layout.addWidget(status_group, stretch=0)

    def _apply_theme(self) -> None:
        """统一应用浅色小米橙主题。"""

        self.setStyleSheet(APP_STYLESHEET)

    def _mark_primary_actions(self) -> None:
        """标记关键主操作按钮。"""

        for button in (
            self.load_button,
            self.step1_next_button,
            self.step2_next_button,
            self.run_analysis_button,
            self.export_result_button,
        ):
            button.setProperty("primary", True)
            button.style().unpolish(button)
            button.style().polish(button)

    def _create_menu_bar(self) -> None:
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        tools_menu = QMenu("工具(&T)", self)
        menubar.addMenu(tools_menu)

        self.ts_fitting_action = QAction("时序拟合工具", self)
        tools_menu.addAction(self.ts_fitting_action)

        help_menu = QMenu("帮助(&H)", self)
        menubar.addMenu(help_menu)

        self.user_guide_action = QAction("使用说明", self)
        self.user_guide_action.setShortcut("F1")
        help_menu.addAction(self.user_guide_action)

        self.ts_tool_guide_action = QAction("时序拟合工具说明", self)
        help_menu.addAction(self.ts_tool_guide_action)

    def _show_user_guide(self) -> None:
        guide_path = self.project_root / "docs" / "user_guide_zh.md"
        if not guide_path.exists():
            QMessageBox.warning(self, "错误", f"用户指南文件不存在：\n{guide_path}")
            return

        try:
            content = guide_path.read_text(encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"读取用户指南失败：\n{exc}")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("多方法交互因果分析平台 使用说明")
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        text_browser = QTextBrowser()
        text_browser.setMarkdown(content)
        text_browser.setOpenExternalLinks(True)
        layout.addWidget(text_browser)

        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.exec_()

    def _show_ts_tool_guide(self) -> None:
        guide_path = self.project_root / "docs" / "ts_fitting_tool_guide.md"
        if not guide_path.exists():
            QMessageBox.warning(self, "错误", f"时序拟合工具说明文件不存在：\n{guide_path}")
            return

        try:
            content = guide_path.read_text(encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"读取说明文件失败：\n{exc}")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("时序拟合工具使用说明")
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        text_browser = QTextBrowser()
        text_browser.setMarkdown(content)
        text_browser.setOpenExternalLinks(True)
        layout.addWidget(text_browser)

        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.exec_()

    def _launch_ts_fitting_tool(self) -> None:
        try:
            from app.tools.time_series_fitting_tool import MainWindow as TSMainWindow
            from PyQt5.QtCore import Qt

            self.log_text.appendPlainText("正在启动时序拟合工具...")

            self.ts_tool_window = TSMainWindow()
            self.ts_tool_window.setWindowTitle("时间序列拟合工具")
            self.ts_tool_window.setWindowModality(Qt.NonModal)
            self.ts_tool_window.show()
            self.ts_tool_window.raise_()
            self.ts_tool_window.activateWindow()

            self.log_text.appendPlainText("时序拟合工具已启动")
        except ImportError as exc:
            QMessageBox.critical(self, "错误", f"无法加载时序拟合工具：\n{exc}")
        except Exception as exc:
            import traceback
            error_msg = f"启动时序拟合工具失败：\n{exc}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "错误", error_msg)

    def _build_data_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(6)
        layout.addWidget(self._create_help_label(self.app_config.help_text.get("step_data", "")))

        file_group = QGroupBox("数据文件")
        file_layout = QHBoxLayout(file_group)
        self.file_path_edit = QPlainTextEdit()
        self.file_path_edit.setFixedHeight(38)
        self.file_path_edit.setPlaceholderText("请选择 Excel 或 CSV 文件路径")
        file_layout.addWidget(self.file_path_edit, stretch=1)
        self.browse_button = QPushButton("浏览")
        self.load_button = QPushButton("加载数据")
        file_layout.addWidget(self.browse_button)
        file_layout.addWidget(self.load_button)
        layout.addWidget(file_group)

        splitter = QSplitter(Qt.Horizontal)

        exclude_group = QGroupBox("列排除设置")
        exclude_layout = QVBoxLayout(exclude_group)
        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        exclude_layout.addWidget(QLabel("选中列后，可标记为“不参与分析”。"))
        exclude_layout.addWidget(self.column_list, stretch=1)

        exclude_buttons = QHBoxLayout()
        self.exclude_selected_button = QPushButton("排除选中列")
        self.restore_selected_button = QPushButton("恢复选中列")
        self.restore_all_button = QPushButton("恢复全部列")
        exclude_buttons.addWidget(self.exclude_selected_button)
        exclude_buttons.addWidget(self.restore_selected_button)
        exclude_buttons.addWidget(self.restore_all_button)
        exclude_layout.addLayout(exclude_buttons)

        self.data_summary_label = QLabel("尚未加载数据。")
        self.data_summary_label.setWordWrap(True)
        exclude_layout.addWidget(self.data_summary_label)
        splitter.addWidget(exclude_group)

        overview_group = QGroupBox("数据概览")
        overview_layout = QVBoxLayout(overview_group)
        self.data_overview_table = QTableWidget()
        self.data_overview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.data_overview_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.data_overview_table.setColumnCount(5)
        self.data_overview_table.setHorizontalHeaderLabels(
            ["列名", "数据类型", "是否数值", "是否参与分析", "非空值数量"]
        )
        self.data_overview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        overview_layout.addWidget(self.data_overview_table)
        self.step1_next_button = QPushButton("下一步：设置变量角色")
        overview_layout.addWidget(self.step1_next_button, alignment=Qt.AlignRight)
        splitter.addWidget(overview_group)

        splitter.setSizes([350, 750])
        layout.addWidget(splitter, stretch=1)
        return page

    def _build_roles_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(6)
        layout.addWidget(self._create_help_label(self.app_config.help_text.get("step_roles", "")))

        header_layout = QHBoxLayout()
        self.auto_detect_roles_button = QPushButton("自动识别角色")
        self.reset_regular_roles_button = QPushButton("全部设为普通变量")
        self.advanced_constraint_button = QPushButton("高级约束")
        self.advanced_constraint_button.setToolTip("设置高级边约束规则，精细控制变量间的因果方向")
        header_layout.addWidget(self.auto_detect_roles_button)
        header_layout.addWidget(self.reset_regular_roles_button)
        header_layout.addWidget(self.advanced_constraint_button)
        header_layout.addStretch(1)
        layout.addLayout(header_layout)

        self.role_summary_label = QLabel("尚未加载数据。")
        self.role_summary_label.setWordWrap(True)
        layout.addWidget(self.role_summary_label)

        self.role_table = QTableWidget()
        self.role_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.role_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.role_table.setColumnCount(4)
        self.role_table.setHorizontalHeaderLabels(["序号", "变量名", "角色", "角色说明"])
        self.role_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.role_table, stretch=1)

        nav_layout = QHBoxLayout()
        self.step2_prev_button = QPushButton("上一步：返回数据导入")
        self.step2_next_button = QPushButton("下一步：设置参数并运行")
        nav_layout.addWidget(self.step2_prev_button)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.step2_next_button)
        layout.addLayout(nav_layout)
        return page

    def _build_run_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(6)
        layout.addWidget(self._create_help_label(self.app_config.help_text.get("step_run", "")))

        algo_tabs = QTabWidget()
        layout.addWidget(algo_tabs, stretch=1)

        pcmci_page = QWidget()
        pcmci_layout = QVBoxLayout(pcmci_page)
        pcmci_layout.addWidget(self._build_pcmci_params_panel())
        algo_tabs.addTab(pcmci_page, "PCMCI+ 因果发现")

        te_page = QWidget()
        te_layout = QVBoxLayout(te_page)
        te_layout.addWidget(self._build_te_params_panel())
        algo_tabs.addTab(te_page, "TE 转移熵分析")

        run_summary_group = QGroupBox("本次运行摘要")
        run_summary_layout = QVBoxLayout(run_summary_group)
        self.run_summary_text = QPlainTextEdit()
        self.run_summary_text.setReadOnly(True)
        self.run_summary_text.setMaximumHeight(140)
        run_summary_layout.addWidget(self.run_summary_text)
        layout.addWidget(run_summary_group)

        bottom_layout = QHBoxLayout()
        self.step3_prev_button = QPushButton("上一步：返回变量角色")
        self.run_analysis_button = QPushButton("运行全部分析")
        bottom_layout.addWidget(self.step3_prev_button)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.run_analysis_button)
        layout.addLayout(bottom_layout)
        return page

    def _build_pcmci_params_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        param_group = QGroupBox("PCMCI+ 参数设置")
        param_layout = QFormLayout(param_group)
        self.tau_min_spin = QSpinBox()
        self.tau_min_spin.setRange(0, 20)
        self.tau_max_spin = QSpinBox()
        self.tau_max_spin.setRange(1, 20)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0001, 0.5)
        self.alpha_spin.setSingleStep(0.01)
        self.alpha_spin.setDecimals(4)
        self.quantile_low_spin = QSpinBox()
        self.quantile_low_spin.setRange(0, 49)
        self.quantile_high_spin = QSpinBox()
        self.quantile_high_spin.setRange(51, 100)
        param_layout.addRow("tau_min", self.tau_min_spin)
        param_layout.addRow("tau_max", self.tau_max_spin)
        param_layout.addRow("pc_alpha", self.alpha_spin)
        param_layout.addRow("归一化分位数下界", self.quantile_low_spin)
        param_layout.addRow("归一化分位数上界", self.quantile_high_spin)
        layout.addWidget(param_group)

        self.run_pcmci_button = QPushButton("运行 PCMCI+")
        self.run_pcmci_button.setMaximumWidth(200)
        layout.addWidget(self.run_pcmci_button, alignment=Qt.AlignLeft)

        layout.addStretch(1)
        return panel

    def _build_te_params_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        te_desc = QLabel("转移熵（Transfer Entropy）用于衡量两个时间序列之间的信息流动方向和强度，可作为 PCMCI+ 结果的验证与补充。")
        te_desc.setWordWrap(True)
        te_desc.setStyleSheet("color: #666; padding: 6px;")
        layout.addWidget(te_desc)

        te_group = QGroupBox("TE 参数设置")
        te_form = QFormLayout(te_group)

        self.te_bins_spin = QSpinBox()
        self.te_bins_spin.setRange(2, 50)
        self.te_bins_spin.setValue(10)
        self.te_bins_spin.setToolTip("数据离散化的分箱数量")
        te_form.addRow("分箱数", self.te_bins_spin)

        self.te_k_spin = QSpinBox()
        self.te_k_spin.setRange(1, 10)
        self.te_k_spin.setValue(1)
        self.te_k_spin.setToolTip("转移熵的历史状态长度")
        te_form.addRow("历史长度 (k)", self.te_k_spin)

        self.te_tau_max_spin = QSpinBox()
        self.te_tau_max_spin.setRange(1, 10)
        self.te_tau_max_spin.setValue(3)
        self.te_tau_max_spin.setToolTip("TE 分析的最大滞后期")
        te_form.addRow("最大滞后期", self.te_tau_max_spin)

        self.te_target_combo = QComboBox()
        self.te_target_combo.setEditable(True)
        self.te_target_combo.addItem("（分析所有变量）")
        self.te_target_combo.setToolTip("留空或选择特定变量作为 TE 分析目标")
        te_form.addRow("目标变量", self.te_target_combo)

        layout.addWidget(te_group)

        self.run_te_button = QPushButton("运行 TE")
        self.run_te_button.setMaximumWidth(200)
        layout.addWidget(self.run_te_button, alignment=Qt.AlignLeft)

        te_warning = QLabel("提示: TE 分析优先使用 pyinform；未安装时会自动改用内置离散估算。")
        te_warning.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(te_warning)

        layout.addStretch(1)
        return panel

    def _build_results_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(6)
        layout.addWidget(self._create_help_label(self.app_config.help_text.get("step_results", "")))

        self.results_summary_text = QPlainTextEdit()
        self.results_summary_text.setReadOnly(True)
        self.results_summary_text.setPlaceholderText("完成分析后，这里会显示本次设置摘要与结果保存位置。")
        self.results_summary_text.setMaximumHeight(100)
        layout.addWidget(self.results_summary_text)

        self.algo_result_tabs = QTabWidget()
        layout.addWidget(self.algo_result_tabs, stretch=1)

        pcmci_result_page = QWidget()
        pcmci_result_layout = QVBoxLayout(pcmci_result_page)
        pcmci_result_layout.addWidget(self._build_pcmci_result_panel())
        self.algo_result_tabs.addTab(pcmci_result_page, "PCMCI+ 结果")

        te_result_page = QWidget()
        te_result_layout = QVBoxLayout(te_result_page)
        te_result_layout.addWidget(self._build_te_result_panel())
        self.algo_result_tabs.addTab(te_result_page, "TE 结果")

        action_layout = QHBoxLayout()
        self.step4_prev_button = QPushButton("返回参数设置")
        self.export_result_button = QPushButton("导出 PCMCI+ 结果")
        self.export_result_button.setEnabled(False)
        action_layout.addWidget(self.step4_prev_button)
        action_layout.addStretch(1)
        action_layout.addWidget(self.export_result_button)
        layout.addLayout(action_layout)
        return page

    def _create_scalable_view(self, content_widget, base_width: int = 700, base_height: int = 500, tab_label: str = "") -> QWidget:
        """创建一个可缩放的视图容器（内容居中）。
        
        Args:
            content_widget: 要显示的内容控件
            base_width: 基础宽度
            base_height: 基础高度
            tab_label: 标签名
            
        Returns:
            包含缩放控件和内容的QWidget
        """
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(2, 2, 2, 2)
        container_layout.setSpacing(2)
        
        # 缩放控制工具栏（紧凑布局）
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        toolbar.setContentsMargins(4, 2, 4, 2)
        
        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFixedSize(24, 22)
        zoom_out_btn.setToolTip("缩小")
        toolbar.addWidget(zoom_out_btn)
        
        zoom_slider = QSlider(Qt.Horizontal)
        zoom_slider.setMinimum(50)
        zoom_slider.setMaximum(300)
        zoom_slider.setValue(100)
        zoom_slider.setMaximumWidth(120)
        zoom_slider.setToolTip("拖动调整缩放比例")
        toolbar.addWidget(zoom_slider)
        
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedSize(24, 22)
        zoom_in_btn.setToolTip("放大")
        toolbar.addWidget(zoom_in_btn)
        
        reset_btn = QPushButton("重置")
        reset_btn.setFixedSize(40, 22)
        reset_btn.setToolTip("重置为100%")
        toolbar.addWidget(reset_btn)
        
        self.zoom_value_label = QLabel("100%")
        self.zoom_value_label.setStyleSheet("color: #888; font-size: 11px;")
        self.zoom_value_label.setMinimumWidth(40)
        toolbar.addWidget(self.zoom_value_label)
        
        toolbar.addStretch()
        container_layout.addLayout(toolbar)
        
        # 滚动区域（内容居中）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        
        inner_page = QWidget()
        inner_page.setFixedSize(base_width, base_height)
        inner_layout = QVBoxLayout(inner_page)
        inner_layout.setContentsMargins(4, 4, 4, 4)
        inner_layout.setSpacing(0)
        
        content_widget.setFixedSize(base_width - 8, base_height - 8)
        inner_layout.addWidget(content_widget)
        scroll.setWidget(inner_page)
        container_layout.addWidget(scroll, stretch=1)
        
        # 缩放状态
        zoom_state = {'scale': 1.0}
        
        def apply_zoom(scale: float):
            zoom_state['scale'] = max(0.5, min(3.0, scale))
            new_w = int(base_width * zoom_state['scale'])
            new_h = int(base_height * zoom_state['scale'])
            inner_page.setFixedSize(new_w, new_h)
            content_widget.setFixedSize(new_w - 16, new_h - 16)
            self.zoom_value_label.setText(f"{int(zoom_state['scale']*100)}%")
        
        def on_zoom_in():
            zoom_slider.setValue(min(300, zoom_slider.value() + 25))
            
        def on_zoom_out():
            zoom_slider.setValue(max(50, zoom_slider.value() - 25))
            
        def on_reset():
            zoom_slider.setValue(100)
        
        def on_slider_changed(value):
            apply_zoom(value / 100.0)
        
        zoom_out_btn.clicked.connect(on_zoom_out)
        zoom_in_btn.clicked.connect(on_zoom_in)
        reset_btn.clicked.connect(on_reset)
        zoom_slider.valueChanged.connect(on_slider_changed)
        
        return container

    def _build_pcmci_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        control_group = QGroupBox("PCMCI+ 结果图控制")
        control_layout = QHBoxLayout(control_group)
        control_layout.setContentsMargins(8, 20, 8, 8)

        control_layout.addWidget(QLabel("目标变量:"))
        self.target_variable_combo = QComboBox()
        self.target_variable_combo.setEnabled(False)
        self.target_variable_combo.setMinimumWidth(120)
        control_layout.addWidget(self.target_variable_combo)

        control_layout.addSpacing(20)

        self.only_target_edges_checkbox = QCheckBox("仅指向目标")
        self.only_target_edges_checkbox.setChecked(False)
        self.only_related_vars_checkbox = QCheckBox("仅相关变量")
        self.only_related_vars_checkbox.setChecked(True)
        self.hide_historical_contemporaneous_checkbox = QCheckBox("隐藏同期边")
        self.hide_historical_contemporaneous_checkbox.setChecked(False)
        self.hide_ambiguous_edges_checkbox = QCheckBox("隐藏未决断边")
        self.hide_ambiguous_edges_checkbox.setChecked(False)
        control_layout.addWidget(self.only_target_edges_checkbox)
        control_layout.addWidget(self.only_related_vars_checkbox)
        control_layout.addWidget(self.hide_historical_contemporaneous_checkbox)
        control_layout.addWidget(self.hide_ambiguous_edges_checkbox)
        control_layout.addStretch(1)
        layout.addWidget(control_group)

        self.pcmci_result_tabs = QTabWidget()
        self.pcmci_result_tabs.setMinimumHeight(400)
        layout.addWidget(self.pcmci_result_tabs, stretch=1)

        # 因果图页面（带缩放）
        self.graph_canvas = CanvasWidget(None, width=8, height=5.5, title="PCMCI+ 因果图")
        self.graph_canvas.double_clicked.connect(self._show_figure_popup)
        graph_view = self._create_scalable_view(self.graph_canvas, base_width=560, base_height=400, tab_label="因果图")
        self.pcmci_result_tabs.addTab(graph_view, "因果图")

        # 时间序列图页面（带缩放）
        self.ts_graph_canvas = CanvasWidget(None, width=10, height=5.5, title="PCMCI+ 时间序列图")
        self.ts_graph_canvas.double_clicked.connect(self._show_figure_popup)
        ts_view = self._create_scalable_view(self.ts_graph_canvas, base_width=640, base_height=400, tab_label="时间序列图")
        self.pcmci_result_tabs.addTab(ts_view, "时间序列图")

        # 目标变量上游影响图页面（带缩放）
        self.target_graph_canvas = CanvasWidget(None, width=8, height=5.5, title="PCMCI+ 目标变量上游影响图")
        self.target_graph_canvas.double_clicked.connect(self._show_figure_popup)
        target_view = self._create_scalable_view(self.target_graph_canvas, base_width=560, base_height=400, tab_label="目标变量上游影响图")
        self.pcmci_result_tabs.addTab(target_view, "目标变量上游影响图")

        # 滞后邻接矩阵页面（带缩放）
        self.adj_table = QTableWidget()
        self.adj_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.adj_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        adj_view = self._create_scalable_view(self.adj_table, base_width=1000, base_height=450, tab_label="滞后邻接矩阵")
        self.pcmci_result_tabs.addTab(adj_view, "滞后邻接矩阵")

        # MCI 矩阵页面（带缩放）
        self.mci_table = QTableWidget()
        self.mci_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.mci_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        mci_view = self._create_scalable_view(self.mci_table, base_width=1000, base_height=450, tab_label="MCI 矩阵")
        self.pcmci_result_tabs.addTab(mci_view, "MCI 矩阵")

        return panel

    def _build_te_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        te_hint = QLabel("TE 分析结果将在启用 TE 并完成分析后显示。")
        te_hint.setWordWrap(True)
        te_hint.setStyleSheet("color: #888; padding: 6px;")
        self.te_hint_label = te_hint
        layout.addWidget(te_hint)

        self.te_result_tabs = QTabWidget()
        self.te_result_tabs.setMinimumHeight(500)
        layout.addWidget(self.te_result_tabs, stretch=1)

        # TE 结果图页面
        te_graph_scroll = QScrollArea()
        te_graph_scroll.setWidgetResizable(True)
        te_graph_page = QWidget()
        te_graph_layout = QVBoxLayout(te_graph_page)
        self.te_graph_canvas = CanvasWidget(te_graph_page, width=10, height=7, title="TE 结果图")
        self.te_graph_canvas.double_clicked.connect(self._show_figure_popup)
        te_graph_layout.addWidget(self.te_graph_canvas)
        te_graph_scroll.setWidget(te_graph_page)
        self.te_result_tabs.addTab(te_graph_scroll, "TE 结果图")

        # TE 矩阵页面
        te_matrix_scroll = QScrollArea()
        te_matrix_scroll.setWidgetResizable(True)
        te_matrix_page = QWidget()
        te_matrix_layout = QVBoxLayout(te_matrix_page)
        self.te_matrix_table = QTableWidget()
        self.te_matrix_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.te_matrix_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        te_matrix_layout.addWidget(self.te_matrix_table)
        te_matrix_scroll.setWidget(te_matrix_page)
        self.te_result_tabs.addTab(te_matrix_scroll, "TE 矩阵")

        # NDTE 矩阵页面
        te_ndte_scroll = QScrollArea()
        te_ndte_scroll.setWidgetResizable(True)
        te_ndte_page = QWidget()
        te_ndte_layout = QVBoxLayout(te_ndte_page)
        self.ndte_matrix_table = QTableWidget()
        self.ndte_matrix_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ndte_matrix_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        te_ndte_layout.addWidget(self.ndte_matrix_table)
        te_ndte_scroll.setWidget(te_ndte_page)
        self.te_result_tabs.addTab(te_ndte_scroll, "NDTE 矩阵")

        # 显著 TE 变量对页面
        te_sig_scroll = QScrollArea()
        te_sig_scroll.setWidgetResizable(True)
        te_sig_page = QWidget()
        te_sig_layout = QVBoxLayout(te_sig_page)
        self.te_table = QTableWidget()
        self.te_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.te_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.te_table.setColumnCount(5)
        self.te_table.setHorizontalHeaderLabels(["源变量", "目标变量", "滞后期", "TE值", "NDTE值"])
        self.te_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        te_sig_layout.addWidget(self.te_table)
        te_sig_scroll.setWidget(te_sig_page)
        self.te_result_tabs.addTab(te_sig_scroll, "显著 TE 变量对")

        # TE 柱状图页面
        te_bar_scroll = QScrollArea()
        te_bar_scroll.setWidgetResizable(True)
        te_bar_page = QWidget()
        te_bar_layout = QVBoxLayout(te_bar_page)
        self.te_bar_canvas = CanvasWidget(te_bar_page, width=14, height=9, title="TE 柱状图")
        self.te_bar_canvas.double_clicked.connect(self._show_figure_popup)
        te_bar_layout.addWidget(self.te_bar_canvas)
        te_bar_scroll.setWidget(te_bar_page)
        self.te_result_tabs.addTab(te_bar_scroll, "TE 柱状图")

        return panel

    def _bind_signals(self) -> None:
        self.browse_button.clicked.connect(self.select_file)
        self.load_button.clicked.connect(self.load_data)
        self.exclude_selected_button.clicked.connect(self.exclude_selected_columns)
        self.restore_selected_button.clicked.connect(self.restore_selected_columns)
        self.restore_all_button.clicked.connect(self.restore_all_columns)
        self.step1_next_button.clicked.connect(lambda: self.step_tabs.setCurrentIndex(1))

        self.auto_detect_roles_button.clicked.connect(lambda: self.auto_detect_roles(reset=True))
        self.reset_regular_roles_button.clicked.connect(self.reset_all_roles_to_regular)
        self.advanced_constraint_button.clicked.connect(self._show_advanced_constraint_dialog)
        self.step2_prev_button.clicked.connect(lambda: self.step_tabs.setCurrentIndex(0))
        self.step2_next_button.clicked.connect(lambda: self.step_tabs.setCurrentIndex(2))

        self.tau_min_spin.valueChanged.connect(self._refresh_run_summary)
        self.tau_max_spin.valueChanged.connect(self._refresh_run_summary)
        self.alpha_spin.valueChanged.connect(self._refresh_run_summary)
        self.quantile_low_spin.valueChanged.connect(self._refresh_run_summary)
        self.quantile_high_spin.valueChanged.connect(self._refresh_run_summary)
        self.te_bins_spin.valueChanged.connect(self._refresh_run_summary)
        self.te_k_spin.valueChanged.connect(self._refresh_run_summary)
        self.te_tau_max_spin.valueChanged.connect(self._refresh_run_summary)
        self.te_target_combo.currentTextChanged.connect(self._refresh_run_summary)
        self.step3_prev_button.clicked.connect(lambda: self.step_tabs.setCurrentIndex(1))
        self.run_analysis_button.clicked.connect(self.run_analysis)
        self.run_pcmci_button.clicked.connect(self.run_pcmci_only)
        self.run_te_button.clicked.connect(self.run_te_only)

        self.step4_prev_button.clicked.connect(lambda: self.step_tabs.setCurrentIndex(2))
        self.export_result_button.clicked.connect(self._show_export_dialog)
        self.algo_result_tabs.currentChanged.connect(self._update_export_button_text)
        self.target_variable_combo.currentIndexChanged.connect(self.refresh_target_dependent_graphs)
        self.only_target_edges_checkbox.toggled.connect(self.plot_time_series_graph)

        self.user_guide_action.triggered.connect(self._show_user_guide)
        self.ts_tool_guide_action.triggered.connect(self._show_ts_tool_guide)
        self.ts_fitting_action.triggered.connect(self._launch_ts_fitting_tool)
        self.only_related_vars_checkbox.toggled.connect(self.plot_time_series_graph)
        self.hide_historical_contemporaneous_checkbox.toggled.connect(self.plot_time_series_graph)
        self.hide_ambiguous_edges_checkbox.toggled.connect(self.plot_time_series_graph)

    def _load_defaults_from_config(self) -> None:
        defaults = self.app_config.defaults
        self.file_path_edit.setPlainText(defaults.default_data_path)
        self.tau_min_spin.setValue(defaults.tau_min)
        self.tau_max_spin.setValue(defaults.tau_max)
        self.alpha_spin.setValue(defaults.pc_alpha)
        self.quantile_low_spin.setValue(defaults.quantile_range[0])
        self.quantile_high_spin.setValue(defaults.quantile_range[1])

    def _refresh_everything(self) -> None:
        self._refresh_column_list()
        self._refresh_data_overview()
        self._refresh_role_table()
        self._refresh_role_summary()
        self._refresh_run_summary()
        self._refresh_target_variable_options()
        self._refresh_tab_state()

    def _get_default_target_name(self, var_names: list[str]) -> str:
        """返回结果页默认目标变量名。"""

        if "10cm_mean" in var_names:
            return "10cm_mean"
        return var_names[0] if var_names else ""

    def _refresh_target_variable_options(self) -> None:
        """刷新“目标变量上游影响图”的目标变量下拉框。"""

        var_names = self.current_result.var_names if self.current_result is not None else []
        current_name = self.target_variable_combo.currentData()

        self.target_variable_combo.blockSignals(True)
        self.target_variable_combo.clear()
        for name in var_names:
            self.target_variable_combo.addItem(name, name)

        target_name = current_name if current_name in var_names else self._get_default_target_name(var_names)
        if target_name:
            self.target_variable_combo.setCurrentIndex(self.target_variable_combo.findData(target_name))
        self.target_variable_combo.setEnabled(bool(var_names))
        self.target_variable_combo.blockSignals(False)

        self.te_target_combo.blockSignals(True)
        current_te_target = self.te_target_combo.currentData()
        self.te_target_combo.clear()
        self.te_target_combo.addItem("（分析所有变量）", "")
        for name in var_names:
            self.te_target_combo.addItem(name, name)
        if current_te_target in var_names:
            self.te_target_combo.setCurrentIndex(self.te_target_combo.findData(current_te_target))
        self.te_target_combo.setEnabled(bool(var_names))
        self.te_target_combo.blockSignals(False)

    def refresh_target_dependent_graphs(self) -> None:
        """刷新所有依赖目标变量选择的结果图。"""

        self.plot_time_series_graph()
        self.plot_target_influence_graph()

    def _create_help_label(self, text: str) -> QLabel:
        label = QLabel(text or "")
        label.setWordWrap(True)
        label.setStyleSheet(
            "QLabel {"
            f" background: {THEME_COLORS['accent_light']};"
            f" border: 1px solid {THEME_COLORS['border']};"
            f" color: {THEME_COLORS['text']};"
            " padding: 10px;"
            " border-radius: 8px;"
            "}"
        )
        return label

    def append_log(self, message: str) -> None:
        self.log_text.appendPlainText(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def select_file(self) -> None:
        file_filter = "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", file_filter)
        if file_path:
            self.file_path_edit.setPlainText(file_path)

    def load_data(self) -> None:
        file_path = self.file_path_edit.toPlainText().strip()
        if not file_path:
            QMessageBox.warning(self, "提示", "请先选择数据文件。")
            return

        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path
        if not path.exists():
            QMessageBox.warning(self, "提示", f"文件不存在：\n{file_path}")
            return

        try:
            if path.suffix.lower() == ".csv":
                self.raw_df = pd.read_csv(path)
            else:
                self.raw_df = pd.read_excel(path, engine="openpyxl")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"加载数据失败：{exc}")
            return

        self.excluded_columns.clear()
        self.current_result = None
        self.current_result_path = None
        self.current_te_result = None
        self.progress_bar.setValue(0)

        self.append_log(f"数据已加载：{len(self.raw_df)} 行，{len(self.raw_df.columns)} 列。")
        self.auto_detect_roles(reset=True)
        self._refresh_everything()
        self._clear_result_views()
        self.run_analysis_button.setEnabled(len(self.get_analysis_var_names()) >= 2)
        if hasattr(self, "run_pcmci_button"):
            self.run_pcmci_button.setEnabled(len(self.get_analysis_var_names()) >= 2)
        if hasattr(self, "run_te_button"):
            self.run_te_button.setEnabled(len(self.get_analysis_var_names()) >= 2)
        self.step_tabs.setCurrentIndex(1)

    def _get_base_numeric_df(self) -> pd.DataFrame:
        if self.raw_df is None:
            return pd.DataFrame()
        remaining = self.raw_df.drop(columns=list(self.excluded_columns), errors="ignore")
        return remaining.select_dtypes(include=[np.number]).copy()

    def _get_analysis_df(self) -> pd.DataFrame:
        numeric_df = self._get_base_numeric_df()
        excluded_vars = [
            name
            for name, role in self.current_role_mapping.items()
            if role == VariableRole.EXCLUDED and name in numeric_df.columns
        ]
        return numeric_df.drop(columns=excluded_vars, errors="ignore").copy()

    def get_analysis_var_names(self) -> list[str]:
        return self._get_analysis_df().columns.tolist()

    def _refresh_column_list(self) -> None:
        self.column_list.clear()
        if self.raw_df is None:
            return

        for column in self.raw_df.columns:
            prefix = "[已排除] " if column in self.excluded_columns else ""
            item = QListWidgetItem(prefix + column)
            item.setData(Qt.UserRole, column)
            if column in self.excluded_columns:
                item.setForeground(Qt.gray)
            self.column_list.addItem(item)

    def _refresh_data_overview(self) -> None:
        if self.raw_df is None:
            self.data_overview_table.setRowCount(0)
            self.data_summary_label.setText("尚未加载数据。")
            return

        self.data_overview_table.setRowCount(len(self.raw_df.columns))
        numeric_columns = set(self.raw_df.select_dtypes(include=[np.number]).columns)
        analysis_columns = set(self._get_base_numeric_df().columns)

        for row, column in enumerate(self.raw_df.columns):
            values = [
                column,
                str(self.raw_df[column].dtype),
                "是" if column in numeric_columns else "否",
                "是" if column in analysis_columns else "否",
                str(int(self.raw_df[column].count())),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 3 and value == "否":
                    item.setForeground(Qt.gray)
                self.data_overview_table.setItem(row, col, item)

        self.data_overview_table.resizeRowsToContents()
        analysis_df = self._get_analysis_df()
        self.data_summary_label.setText(
            f"原始列数：{len(self.raw_df.columns)}；"
            f" 数值列数：{len(numeric_columns)}；"
            f" 当前候选分析列：{len(analysis_columns)}；"
            f" 最终进入分析的变量数：{len(analysis_df.columns)}。"
        )

    def exclude_selected_columns(self) -> None:
        for item in self.column_list.selectedItems():
            self.excluded_columns.add(str(item.data(Qt.UserRole)))
        self._refresh_after_structure_change()

    def restore_selected_columns(self) -> None:
        for item in self.column_list.selectedItems():
            self.excluded_columns.discard(str(item.data(Qt.UserRole)))
        self._refresh_after_structure_change()

    def restore_all_columns(self) -> None:
        self.excluded_columns.clear()
        self._refresh_after_structure_change()

    def _refresh_after_structure_change(self) -> None:
        self._prune_roles()
        self._refresh_everything()
        self.run_analysis_button.setEnabled(len(self.get_analysis_var_names()) >= 2)

    def auto_detect_roles(self, reset: bool = True) -> None:
        numeric_names = self._get_base_numeric_df().columns.tolist()
        if reset or not self.current_role_mapping:
            self.current_role_mapping = {name: VariableRole.REGULAR for name in numeric_names}

        existing = set(numeric_names)
        for role, template in self.app_config.role_templates.items():
            if role == VariableRole.REGULAR:
                continue
            for name in template.variable_names:
                if name in existing:
                    self.current_role_mapping[name] = role

        for name in numeric_names:
            self.current_role_mapping.setdefault(name, VariableRole.REGULAR)

        self._prune_roles()
        self._refresh_everything()

    def reset_all_roles_to_regular(self) -> None:
        numeric_names = self._get_base_numeric_df().columns.tolist()
        self.current_role_mapping = {name: VariableRole.REGULAR for name in numeric_names}
        self._prune_roles()
        self._refresh_everything()

    def _prune_roles(self) -> None:
        valid_names = set(self._get_base_numeric_df().columns.tolist())
        self.current_role_mapping = {
            name: role
            for name, role in self.current_role_mapping.items()
            if name in valid_names
        }

    def _refresh_role_table(self) -> None:
        var_names = self._get_base_numeric_df().columns.tolist()
        self.role_table.setRowCount(len(var_names))

        for row, name in enumerate(var_names):
            self.role_table.setItem(row, 0, QTableWidgetItem(str(row)))
            self.role_table.setItem(row, 1, QTableWidgetItem(name))

            combo = QComboBox()
            for role in VariableRole:
                combo.addItem(ROLE_LABELS[role], role)
            combo.setCurrentIndex(combo.findData(self.current_role_mapping.get(name, VariableRole.REGULAR)))
            combo.currentIndexChanged.connect(
                lambda _index, variable_name=name, widget=combo: self._handle_role_changed(
                    variable_name,
                    widget.currentData(),
                )
            )
            self.role_table.setCellWidget(row, 2, combo)

            description = ROLE_DESCRIPTIONS[self.current_role_mapping.get(name, VariableRole.REGULAR)]
            self.role_table.setItem(row, 3, QTableWidgetItem(description))

    def _handle_role_changed(self, variable_name: str, role: VariableRole) -> None:
        self.current_role_mapping[variable_name] = role
        self._prune_roles()
        self._refresh_role_summary()
        self._refresh_run_summary()
        has_analysis_vars = len(self.get_analysis_var_names()) >= 2
        self.run_analysis_button.setEnabled(has_analysis_vars)
        if hasattr(self, "run_pcmci_button"):
            self.run_pcmci_button.setEnabled(has_analysis_vars)
        if hasattr(self, "run_te_button"):
            self.run_te_button.setEnabled(has_analysis_vars)

    def _refresh_role_summary(self) -> None:
        if self.raw_df is None:
            self.role_summary_label.setText("尚未加载数据。")
            return

        groups: dict[str, list[str]] = {}
        for name in self._get_base_numeric_df().columns.tolist():
            role = self.current_role_mapping.get(name, VariableRole.REGULAR)
            groups.setdefault(ROLE_LABELS[role], []).append(name)

        summary_parts = [f"{label}：{', '.join(names)}" for label, names in groups.items()]
        self.role_summary_label.setText(
            "；".join(summary_parts) if summary_parts else "当前没有可设置角色的数值变量。"
        )

    def _build_analysis_config(self) -> AnalysisConfig:
        defaults = self.app_config.defaults
        return AnalysisConfig(
            tau_min=self.tau_min_spin.value(),
            tau_max=self.tau_max_spin.value(),
            pc_alpha=self.alpha_spin.value(),
            quantile_range=(self.quantile_low_spin.value(), self.quantile_high_spin.value()),
            output_dir=defaults.output_dir,
            result_file_name=defaults.result_file_name,
            log_level=defaults.log_level,
            default_data_path=defaults.default_data_path,
            manual_rules=list(self.manual_constraints),
        )

    def _refresh_run_summary(self) -> None:
        analysis_vars = self.get_analysis_var_names()
        roles: dict[str, list[str]] = {}
        for name in analysis_vars:
            role = self.current_role_mapping.get(name, VariableRole.REGULAR)
            roles.setdefault(ROLE_LABELS[role], []).append(name)

        output_path = self.project_root / self.app_config.defaults.output_dir / self.app_config.defaults.result_file_name
        lines = [
            f"最终分析变量数：{len(analysis_vars)}",
            "角色划分：",
        ]
        if roles:
            for label, names in roles.items():
                lines.append(f"  - {label}：{', '.join(names)}")
        else:
            lines.append("  - 暂无")

        lines.extend(
            [
                "---",
                "PCMCI+ 参数：",
                f"  tau_min = {self.tau_min_spin.value()}",
                f"  tau_max = {self.tau_max_spin.value()}",
                f"  pc_alpha = {self.alpha_spin.value():.4f}",
                f"  归一化分位数 = ({self.quantile_low_spin.value()}, {self.quantile_high_spin.value()})",
                f"  高级边约束数 = {len(self.manual_constraints)}",
                "TE 参数：",
                f"  分箱数 = {self.te_bins_spin.value()}",
                f"  历史长度 k = {self.te_k_spin.value()}",
                f"  最大滞后期 = {self.te_tau_max_spin.value()}",
                f"  目标变量 = {self.te_target_combo.currentText()}",
                f"结果输出路径：{output_path}",
            ]
        )
        self.run_summary_text.setPlainText("\n".join(lines))

    def run_analysis(self) -> None:
        analysis_df = self._get_analysis_df()
        analysis_vars = analysis_df.columns.tolist()
        if len(analysis_vars) < 2:
            QMessageBox.warning(self, "提示", "至少需要 2 个数值变量参与分析。")
            return

        if self.tau_min_spin.value() > self.tau_max_spin.value():
            QMessageBox.warning(self, "提示", "tau_min 不能大于 tau_max。")
            return

        if self.quantile_low_spin.value() >= self.quantile_high_spin.value():
            QMessageBox.warning(self, "提示", "归一化分位数下界必须小于上界。")
            return

        config = self._build_analysis_config()
        role_mapping = {
            name: self.current_role_mapping.get(name, VariableRole.REGULAR)
            for name in analysis_vars
        }

        compiled = ConstraintEngine.compile(
            var_names=analysis_vars,
            tau_min=config.tau_min,
            tau_max=config.tau_max,
            role_mapping=role_mapping,
            manual_rules=config.manual_rules,
            logger=self.append_log,
        )

        self.progress_bar.setValue(5)
        self.run_analysis_button.setEnabled(False)
        self.current_result = None
        self.current_result_path = None
        self._clear_result_views()

        self.worker = AnalysisWorker(
            df=analysis_df,
            var_names=analysis_vars,
            config=config,
            compiled_constraints=compiled,
        )
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.append_log)
        self.worker.error_signal.connect(self.analysis_error)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.start()
        self._refresh_tab_state()

    def update_progress(self, message: str) -> None:
        self.append_log(message)
        current = self.progress_bar.value()
        if "预处理" in message:
            self.progress_bar.setValue(20)
        elif "独立性" in message:
            self.progress_bar.setValue(40)
        elif "PCMCI" in message:
            self.progress_bar.setValue(75)
        elif "矩阵" in message:
            self.progress_bar.setValue(95)
        else:
            self.progress_bar.setValue(min(95, current + 10))

    def analysis_finished(self, result: AnalysisResult) -> None:
        self.current_result = result
        self.current_result_path = self._save_result_payload(result)
        self.worker = None
        self.run_analysis_button.setEnabled(True)

        self.display_adj_matrix()
        self.display_mci_matrix()
        self._refresh_target_variable_options()
        self.plot_causal_graph()
        self.refresh_target_dependent_graphs()

        self.append_log("PCMCI+ 分析完成。")
        self.append_log(f"PCMCI+ 结果已保存到：{self.current_result_path}")

        # 运行TE分析
        self.append_log("开始 TE 分析...")
        self.progress_bar.setValue(60)
        te_summary = ""
        try:
            from app.core.analysis_service_te import TEAnalysisService
            te_config = TEConfig(
                enabled=True,
                bins=self.te_bins_spin.value(),
                k_history=self.te_k_spin.value(),
                tau_max=self.te_tau_max_spin.value(),
                target_var=self.te_target_combo.currentData() or "",
                analyze_all=self.te_target_combo.currentData() == "",
            )
            te_service = TEAnalysisService()
            te_result = te_service.run_analysis(
                df=self._get_analysis_df(),
                var_names=result.var_names,
                config=te_config,
                progress=self.update_te_progress,
                logger=self.append_log,
            )
            if te_result is not None:
                self.current_te_result = te_result
                self.display_te_results()
                self.append_log("TE 分析完成。")
                te_summary = " TE 分析已完成。"
            else:
                self.current_te_result = None
                self.te_hint_label.setText("TE 分析失败，请查看运行日志。")
                self.append_log("TE 分析失败。")
        except Exception as exc:
            self.current_te_result = None
            self.append_log(f"TE 分析出错: {exc}")
            import traceback
            self.append_log(traceback.format_exc())
            te_summary = " TE 分析出错。"

        self._refresh_results_summary()
        self._refresh_tab_state()
        self.progress_bar.setValue(100)
        self.step_tabs.setCurrentIndex(3)

        QMessageBox.information(self, "完成", f"PCMCI+ 分析已完成。{te_summary}")

    def analysis_error(self, message: str) -> None:
        self.worker = None
        self.progress_bar.setValue(0)
        self.run_analysis_button.setEnabled(True)
        self.run_pcmci_button.setEnabled(True)
        self.run_te_button.setEnabled(True)
        self.append_log(message)
        self._refresh_tab_state()
        QMessageBox.critical(self, "错误", message)

    def run_pcmci_only(self) -> None:
        analysis_df = self._get_analysis_df()
        analysis_vars = analysis_df.columns.tolist()
        if len(analysis_vars) < 2:
            QMessageBox.warning(self, "提示", "至少需要 2 个数值变量参与分析。")
            return

        if self.tau_min_spin.value() > self.tau_max_spin.value():
            QMessageBox.warning(self, "提示", "tau_min 不能大于 tau_max。")
            return

        if self.quantile_low_spin.value() >= self.quantile_high_spin.value():
            QMessageBox.warning(self, "提示", "归一化分位数下界必须小于上界。")
            return

        config = self._build_analysis_config()
        role_mapping = {
            name: self.current_role_mapping.get(name, VariableRole.REGULAR)
            for name in analysis_vars
        }

        compiled = ConstraintEngine.compile(
            var_names=analysis_vars,
            tau_min=config.tau_min,
            tau_max=config.tau_max,
            role_mapping=role_mapping,
            manual_rules=[],
            logger=self.append_log,
        )

        self.progress_bar.setValue(5)
        self.run_analysis_button.setEnabled(False)
        self.run_pcmci_button.setEnabled(False)
        self.run_te_button.setEnabled(False)
        self.current_result = None
        self.current_result_path = None
        self._clear_result_views()

        self.worker = AnalysisWorker(
            df=analysis_df,
            var_names=analysis_vars,
            config=config,
            compiled_constraints=compiled,
        )
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.append_log)
        self.worker.error_signal.connect(self.analysis_error)
        self.worker.finished_signal.connect(self.analysis_finished_pcmci_only)
        self.worker.start()
        self._refresh_tab_state()

    def analysis_finished_pcmci_only(self, result: AnalysisResult) -> None:
        self.current_result = result
        self.current_result_path = self._save_result_payload(result)
        self.worker = None
        self.run_analysis_button.setEnabled(True)
        self.run_pcmci_button.setEnabled(True)
        self.run_te_button.setEnabled(True)

        self.display_adj_matrix()
        self.display_mci_matrix()
        self._refresh_target_variable_options()
        self.plot_causal_graph()
        self.refresh_target_dependent_graphs()

        self.append_log("PCMCI+ 分析完成。")
        self.append_log(f"PCMCI+ 结果已保存到：{self.current_result_path}")

        self._refresh_results_summary()
        self._refresh_tab_state()
        self.progress_bar.setValue(100)
        self.step_tabs.setCurrentIndex(3)

        QMessageBox.information(self, "完成", "PCMCI+ 分析已完成。")

    def run_te_only(self) -> None:
        analysis_df = self._get_analysis_df()
        analysis_vars = analysis_df.columns.tolist()
        if len(analysis_vars) < 2:
            QMessageBox.warning(self, "提示", "至少需要 2 个数值变量参与分析。")
            return

        self.append_log("开始 TE 分析...")
        self.progress_bar.setValue(10)
        self.run_analysis_button.setEnabled(False)
        self.run_pcmci_button.setEnabled(False)
        self.run_te_button.setEnabled(False)

        try:
            from app.core.analysis_service_te import TEAnalysisService
            te_config = TEConfig(
                enabled=True,
                bins=self.te_bins_spin.value(),
                k_history=self.te_k_spin.value(),
                tau_max=self.te_tau_max_spin.value(),
                target_var=self.te_target_combo.currentData() or "",
                analyze_all=self.te_target_combo.currentData() == "",
            )
            te_service = TEAnalysisService()
            te_result = te_service.run_analysis(
                df=analysis_df,
                var_names=analysis_vars,
                config=te_config,
                progress=self.update_te_progress,
                logger=self.append_log,
            )
            if te_result is not None:
                self.current_te_result = te_result
                self.display_te_results()
                self.append_log("TE 分析完成。")
                self._refresh_results_summary()
                self._refresh_tab_state()
                self.progress_bar.setValue(100)
                self.step_tabs.setCurrentIndex(3)
                QMessageBox.information(self, "完成", "TE 分析已完成。")
            else:
                self.current_te_result = None
                self.te_hint_label.setText("TE 分析失败，请查看运行日志。")
                self.append_log("TE 分析失败。")
                QMessageBox.warning(self, "警告", "TE 分析失败。")
        except Exception as exc:
            self.current_te_result = None
            self.append_log(f"TE 分析出错: {exc}")
            import traceback
            self.append_log(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"TE 分析出错：\n{exc}")
        finally:
            self.run_analysis_button.setEnabled(True)
            self.run_pcmci_button.setEnabled(True)
            self.run_te_button.setEnabled(True)
            self.progress_bar.setValue(100)
            self._refresh_tab_state()

    def update_te_progress(self, message: str) -> None:
        self.append_log(message)
        current = self.progress_bar.value()
        self.progress_bar.setValue(min(95, current + 5))

    def display_te_results(self) -> None:
        if self.current_te_result is None:
            return

        te_result = self.current_te_result
        var_names = te_result.var_names
        tau_max = te_result.config.tau_max

        self.te_hint_label.setText(
            f"TE 分析完成。变量数: {len(var_names)}, "
            f"显著变量对: {len([p for p in te_result.significant_pairs if p[3] > 0])}"
        )

        self.te_matrix_table.setRowCount(len(var_names))
        self.te_matrix_table.setColumnCount(len(var_names))
        self.te_matrix_table.setHorizontalHeaderLabels(var_names)
        self.te_matrix_table.setVerticalHeaderLabels(var_names)
        max_te_matrix = np.max(te_result.te_matrix[:, :, 1:], axis=2) if tau_max >= 1 else np.zeros((len(var_names), len(var_names)))
        max_ndte_matrix = np.max(te_result.ndte_matrix[:, :, 1:], axis=2) if tau_max >= 1 else np.zeros((len(var_names), len(var_names)))
        self._plot_te_result_graph(max_te_matrix, max_ndte_matrix, var_names)

        for row in range(len(var_names)):
            for col in range(len(var_names)):
                item = QTableWidgetItem(f"{max_te_matrix[row, col]:.4f}")
                if max_te_matrix[row, col] > 0:
                    item.setBackground(QColor(THEME_COLORS["accent_light"]))
                self.te_matrix_table.setItem(row, col, item)
        self.te_matrix_table.resizeColumnsToContents()

        self.ndte_matrix_table.setRowCount(len(var_names))
        self.ndte_matrix_table.setColumnCount(len(var_names))
        self.ndte_matrix_table.setHorizontalHeaderLabels(var_names)
        self.ndte_matrix_table.setVerticalHeaderLabels(var_names)
        for row in range(len(var_names)):
            for col in range(len(var_names)):
                self.ndte_matrix_table.setItem(row, col, QTableWidgetItem(f"{max_ndte_matrix[row, col]:.4f}"))
        self.ndte_matrix_table.resizeColumnsToContents()

        self.te_table.setRowCount(len(te_result.significant_pairs))
        for row, (source, target, lag, te_val, ndte_val) in enumerate(te_result.significant_pairs):
            self.te_table.setItem(row, 0, QTableWidgetItem(str(source)))
            self.te_table.setItem(row, 1, QTableWidgetItem(str(target)))
            self.te_table.setItem(row, 2, QTableWidgetItem(str(lag)))
            self.te_table.setItem(row, 3, QTableWidgetItem(f"{te_val:.4f}"))
            self.te_table.setItem(row, 4, QTableWidgetItem(f"{ndte_val:.4f}"))

        self._plot_te_bar_graph(max_te_matrix, max_ndte_matrix, var_names)

    def _plot_te_result_graph(self, max_te_matrix: np.ndarray, max_ndte_matrix: np.ndarray, var_names: list[str]) -> None:
        fig = self.te_graph_canvas.fig
        fig.clear()
        axes = fig.subplots(1, 2)
        font_properties = get_chinese_font_properties()
        font_kwargs = {"fontproperties": font_properties} if font_properties is not None else {}
        matrices = (
            (axes[0], max_te_matrix, "TE 矩阵", "YlOrRd", "TE"),
            (axes[1], max_ndte_matrix, "NDTE 矩阵", "viridis", "NDTE"),
        )

        for axis, matrix, title, cmap, colorbar_label in matrices:
            image = axis.imshow(matrix, cmap=cmap, aspect="auto")
            axis.set_title(title, **font_kwargs)
            axis.set_xticks(range(len(var_names)))
            axis.set_yticks(range(len(var_names)))
            axis.set_xticklabels(var_names, rotation=45, ha="right")
            axis.set_yticklabels(var_names)
            axis.set_xlabel("目标变量", **font_kwargs)
            axis.set_ylabel("源变量", **font_kwargs)
            if font_properties is not None:
                for label in axis.get_xticklabels() + axis.get_yticklabels():
                    label.set_fontproperties(font_properties)
            style_result_axes(axis)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=colorbar_label)

        self.te_graph_canvas.axes = axes[0]
        apply_colorbar_theme(fig)
        self.te_graph_canvas.draw()

    def _plot_te_bar_graph(self, max_te_matrix: np.ndarray, max_ndte_matrix: np.ndarray, var_names: list[str]) -> None:
        fig = self.te_bar_canvas.fig
        fig.clear()
        axes = fig.subplots(2, 1, sharex=True)
        font_properties = get_chinese_font_properties()
        font_kwargs = {"fontproperties": font_properties} if font_properties is not None else {}

        x = np.arange(len(var_names))
        width = 0.35

        # 上子图：每个变量作为目标的 TE 和 NDTE 值（所有源变量的最大值）
        te_target_max = np.max(max_te_matrix, axis=0)
        ndte_target_max = np.max(max_ndte_matrix, axis=0)
        rects1 = axes[0].bar(x - width/2, te_target_max, width, label='TE', color='#d62728')
        rects2 = axes[0].bar(x + width/2, ndte_target_max, width, label='NDTE', color='#1f77b4')
        axes[0].set_title('每个变量作为目标的最大 TE/NDTE 值', **font_kwargs)
        axes[0].set_ylabel('信息强度', **font_kwargs)
        axes[0].legend()
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(var_names, rotation=45, ha='right', **font_kwargs)
        style_result_axes(axes[0])
        axes[0].bar_label(rects1, fmt='%.3f', padding=3, fontsize=8)
        axes[0].bar_label(rects2, fmt='%.3f', padding=3, fontsize=8)

        # 下子图：每个变量作为源的 TE 和 NDTE 值（所有目标变量的最大值）
        te_source_max = np.max(max_te_matrix, axis=1)
        ndte_source_max = np.max(max_ndte_matrix, axis=1)
        rects3 = axes[1].bar(x - width/2, te_source_max, width, label='TE', color='#d62728')
        rects4 = axes[1].bar(x + width/2, ndte_source_max, width, label='NDTE', color='#1f77b4')
        axes[1].set_title('每个变量作为源的最大 TE/NDTE 值', **font_kwargs)
        axes[1].set_xlabel('变量', **font_kwargs)
        axes[1].set_ylabel('信息强度', **font_kwargs)
        axes[1].legend()
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(var_names, rotation=45, ha='right', **font_kwargs)
        style_result_axes(axes[1])
        axes[1].bar_label(rects3, fmt='%.3f', padding=3, fontsize=8)
        axes[1].bar_label(rects4, fmt='%.3f', padding=3, fontsize=8)

        fig.tight_layout()
        self.te_bar_canvas.axes = axes[0]
        apply_colorbar_theme(fig)
        self.te_bar_canvas.draw()

    def export_te_result(self) -> None:
        if self.current_te_result is None:
            QMessageBox.warning(self, "提示", "没有可导出的 TE 分析结果。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 TE 结果",
            "",
            "Excel Files (*.xlsx);;CSV Files (*.csv)",
        )
        if not file_path:
            return

        try:
            te_result = self.current_te_result
            var_names = te_result.var_names
            max_te = np.max(te_result.te_matrix[:, :, 1:], axis=2) if te_result.config.tau_max >= 1 else np.zeros((len(var_names), len(var_names)))
            max_ndte = np.max(te_result.ndte_matrix[:, :, 1:], axis=2) if te_result.config.tau_max >= 1 else np.zeros((len(var_names), len(var_names)))

            path = Path(file_path)
            if path.suffix.lower() == ".csv":
                pd.DataFrame(max_te, index=var_names, columns=var_names).to_csv(path.parent / (path.stem + "_TE.csv"))
                pd.DataFrame(max_ndte, index=var_names, columns=var_names).to_csv(path.parent / (path.stem + "_NDTE.csv"))
            else:
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    pd.DataFrame(max_te, index=var_names, columns=var_names).to_excel(writer, sheet_name="TE矩阵")
                    pd.DataFrame(max_ndte, index=var_names, columns=var_names).to_excel(writer, sheet_name="NDTE矩阵")
                    if te_result.significant_pairs:
                        sig_df = pd.DataFrame(te_result.significant_pairs, columns=["源变量", "目标变量", "滞后期", "TE值", "NDTE值"])
                        sig_df.to_excel(writer, sheet_name="显著变量对", index=False)
            QMessageBox.information(self, "完成", f"TE 结果已导出到：\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"导出失败：{exc}")

    def _show_figure_popup(self, canvas: CanvasWidget, title: str) -> None:
        """双击图表后弹出大窗口显示。"""

        dialog = QDialog(self)
        dialog.setWindowTitle(title or "查看图表")
        dialog.resize(1200, 900)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)

        # 复制 figure，避免与原画布共享同一对象
        import pickle
        from matplotlib.patches import FancyArrowPatch
        fig_copy = pickle.loads(pickle.dumps(canvas.fig))

        # 放大弹窗中所有文字的字体大小和线条粗细
        font_scale = 1.6
        line_width_scale = 2.0
        marker_scale = 1.8
        for ax in fig_copy.axes:
            # 放大文字
            for text_obj in ax.texts:
                current_size = text_obj.get_fontsize()
                if current_size is not None and current_size > 0:
                    text_obj.set_fontsize(current_size * font_scale)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                current_size = label.get_fontsize()
                if current_size is not None and current_size > 0:
                    label.set_fontsize(current_size * font_scale)
            ax_title = ax.title
            if ax_title:
                current_size = ax_title.get_fontsize()
                if current_size is not None and current_size > 0:
                    ax_title.set_fontsize(current_size * font_scale)
            x_label = ax.xaxis.label
            y_label = ax.yaxis.label
            for axis_label in [x_label, y_label]:
                if axis_label:
                    current_size = axis_label.get_fontsize()
                    if current_size is not None and current_size > 0:
                        axis_label.set_fontsize(current_size * font_scale)

            # 放大普通线条
            for line in ax.lines:
                current_lw = line.get_linewidth()
                line.set_linewidth(max(1.0, current_lw * line_width_scale))
                # 同时放大数据点标记大小
                ms = line.get_markersize()
                if ms and ms > 0:
                    line.set_markersize(ms * marker_scale)
                mew = line.get_markeredgewidth()
                if mew and mew > 0:
                    line.set_markeredgewidth(mew * line_width_scale)

            # 放大 FancyArrowPatch (箭头) 和其他 patch
            for patch in ax.patches:
                if isinstance(patch, FancyArrowPatch):
                    lw = patch.get_linewidth()
                    patch.set_linewidth(max(1.5, lw * line_width_scale))
                    # 放大箭头头部
                    try:
                        mutation_scale = patch.get_mutation_scale()
                        if mutation_scale:
                            patch.set_mutation_scale(mutation_scale * 1.6)
                    except Exception:
                        pass
                elif hasattr(patch, 'set_linewidth'):
                    lw = patch.get_linewidth()
                    patch.set_linewidth(max(0.8, lw * line_width_scale))
                    # 尝试放大标记/符号
                    if hasattr(patch, 'set_markersize'):
                        try:
                            ms = patch.get_markersize()
                            if ms and ms > 0:
                                patch.set_markersize(ms * marker_scale)
                        except Exception:
                            pass

        popup_canvas = FigureCanvas(fig_copy)
        popup_canvas.draw()
        layout.addWidget(popup_canvas, stretch=1)

        # 底部按钮
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("保存图片")
        save_btn.clicked.connect(lambda: self._save_popup_figure(fig_copy, title))
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addStretch(1)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dialog.exec_()

    def _save_popup_figure(self, fig, default_name: str) -> None:
        """从弹窗中保存图片。"""

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", f"{default_name}.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)",
        )
        if not file_path:
            return
        try:
            fig.savefig(file_path, dpi=200, bbox_inches="tight")
            QMessageBox.information(self, "完成", f"图片已保存到：\n{file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"保存失败：{exc}")

    def _update_export_button_text(self, tab_index: int) -> None:
        """根据当前结果标签页更新导出按钮文本和启用状态。"""

        if tab_index == 0:
            # PCMCI+ 结果标签页
            self.export_result_button.setText("导出 PCMCI+ 结果")
            self.export_result_button.setEnabled(self.current_result is not None)
        elif tab_index == 1:
            # TE 结果标签页
            self.export_result_button.setText("导出 TE 结果")
            self.export_result_button.setEnabled(self.current_te_result is not None)

    def _show_advanced_constraint_dialog(self) -> None:
        """显示高级边约束设置对话框。"""

        dialog = QDialog(self)
        dialog.setWindowTitle("高级边约束设置")
        dialog.resize(700, 500)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(8)

        hint_label = QLabel(
            "高级约束允许您精细控制 PCMCI+ 分析中变量间的因果方向。\n"
            "这些约束将在角色约束之后应用，优先级更高。\n"
            "• 禁止入边 (NO_IN): 其他变量不能影响目标变量\n"
            "• 禁止出边 (NO_OUT): 该变量不能影响其他变量\n"
            "• 禁止连接 (NO_LINK): 两个变量之间无任何因果关系"
        )
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #666; padding: 6px; background: #f5f5f5; border-radius: 4px;")
        layout.addWidget(hint_label)

        var_names = list(self.current_role_mapping.keys()) if self.current_role_mapping else []

        form_layout = QHBoxLayout()
        form_layout.addWidget(QLabel("源变量:"))
        source_combo = QComboBox()
        source_combo.addItems(var_names)
        source_combo.setMinimumWidth(150)
        form_layout.addWidget(source_combo)

        form_layout.addWidget(QLabel("目标变量:"))
        target_combo = QComboBox()
        target_combo.addItems(var_names)
        target_combo.setMinimumWidth(150)
        form_layout.addWidget(target_combo)

        form_layout.addWidget(QLabel("约束类型:"))
        type_combo = QComboBox()
        type_combo.addItems(["禁止入边 (NO_IN)", "禁止出边 (NO_OUT)", "禁止连接 (NO_LINK)"])
        type_combo.setMinimumWidth(140)
        form_layout.addWidget(type_combo)

        add_btn = QPushButton("添加")
        add_btn.setMaximumWidth(60)
        add_btn.setPrimary(True)
        form_layout.addWidget(add_btn)
        form_layout.addStretch(1)
        layout.addLayout(form_layout)

        constraint_table = QTableWidget()
        constraint_table.setColumnCount(5)
        constraint_table.setHorizontalHeaderLabels(["序号", "源变量", "目标变量", "约束类型", "操作"])
        constraint_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        constraint_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        constraint_table.horizontalHeader().setSectionStretchLastSection(True)
        constraint_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        constraint_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(constraint_table, stretch=1)

        type_map = {
            "禁止入边 (NO_IN)": ConstraintType.NO_IN,
            "禁止出边 (NO_OUT)": ConstraintType.NO_OUT,
            "禁止连接 (NO_LINK)": ConstraintType.NO_LINK,
        }
        type_display = {
            ConstraintType.NO_IN: "禁止入边",
            ConstraintType.NO_OUT: "禁止出边",
            ConstraintType.NO_LINK: "禁止连接",
        }

        def refresh_table():
            constraint_table.setRowCount(len(self.manual_constraints))
            for row, rule in enumerate(self.manual_constraints):
                constraint_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
                constraint_table.setItem(row, 1, QTableWidgetItem(rule.source_name))
                constraint_table.setItem(row, 2, QTableWidgetItem(rule.target_name))
                constraint_table.setItem(row, 3, QTableWidgetItem(type_display.get(rule.constraint_type, str(rule.constraint_type))))
                del_btn = QPushButton("删除")
                del_btn.setMaximumWidth(50)
                idx = row
                del_btn.clicked.connect(lambda _, i=idx: _delete_rule(i))
                constraint_table.setCellWidget(row, 4, del_btn)

        def _add_rule():
            src = source_combo.currentText()
            tgt = target_combo.currentText()
            if not src or not tgt:
                QMessageBox.warning(dialog, "提示", "请选择源变量和目标变量。")
                return
            if src == tgt:
                QMessageBox.warning(dialog, "提示", "源变量和目标变量不能相同。")
                return
            ctype = type_map.get(type_combo.currentText())
            for existing in self.manual_constraints:
                if existing.source_name == src and existing.target_name == tgt and existing.constraint_type == ctype:
                    QMessageBox.warning(dialog, "提示", "该约束规则已存在。")
                    return
            self.manual_constraints.append(
                ManualConstraintRule(source_name=src, target_name=tgt, constraint_type=ctype)
            )
            refresh_table()

        def _delete_rule(idx):
            if 0 <= idx < len(self.manual_constraints):
                del self.manual_constraints[idx]
                refresh_table()

        add_btn.clicked.connect(_add_rule)
        refresh_table()

        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("清空全部")
        clear_btn.setToolTip("删除所有手动约束规则")
        close_btn = QPushButton("关闭")
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch(1)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        clear_btn.clicked.connect(lambda: [self.manual_constraints.clear(), refresh_table()])
        close_btn.clicked.connect(dialog.accept)

        dialog.exec_()

    def _show_export_dialog(self) -> None:
        """显示导出结果对话框，让用户选择要导出的内容。"""

        current_tab = self.algo_result_tabs.currentIndex()
        if current_tab == 0:
            self._show_pcmci_export_dialog()
        elif current_tab == 1:
            self._show_te_export_dialog()

    def _show_pcmci_export_dialog(self) -> None:
        """显示 PCMCI+ 结果导出对话框。"""

        if self.current_result is None:
            QMessageBox.warning(self, "提示", "没有可导出的 PCMCI+ 分析结果。")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("导出 PCMCI+ 结果")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        # 标题
        title_label = QLabel("请选择要导出的 PCMCI+ 结果：")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title_label)

        # 各类结果的复选框
        checkboxes: dict[str, QCheckBox] = {}
        options = [
            ("adj_matrix", "滞后邻接矩阵", True),
            ("val_matrix", "val 矩阵（条件独立性值）", False),
            ("p_matrix", "p 值矩阵", False),
            ("mci_matrix", "MCI 矩阵（跨滞后期最大值）", True),
            ("graph_image", "因果图（图片）", False),
            ("ts_graph_image", "时间序列图（图片）", False),
            ("target_graph_image", "目标变量上游影响图（图片）", False),
        ]

        for key, label, default_checked in options:
            cb = QCheckBox(label)
            cb.setChecked(default_checked)
            checkboxes[key] = cb
            layout.addWidget(cb)

        # 格式选择
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("导出格式："))
        format_combo = QComboBox()
        format_combo.addItem("Excel (.xlsx)", "xlsx")
        format_combo.addItem("CSV (.csv)", "csv")
        format_layout.addWidget(format_combo)
        layout.addLayout(format_layout)

        # 按钮
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        export_btn = QPushButton("导出")
        export_btn.setDefault(True)
        button_layout.addStretch(1)
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(export_btn)
        layout.addLayout(button_layout)

        def on_export():
            selected = {k: cb.isChecked() for k, cb in checkboxes.items()}
            if not any(selected.values()):
                QMessageBox.warning(dialog, "提示", "请至少选择一项要导出的内容。")
                return
            dialog.accept()
            self._do_export_pcmci(selected, format_combo.currentData())

        export_btn.clicked.connect(on_export)
        dialog.exec_()

    def _do_export_pcmci(self, selections: dict[str, bool], fmt: str) -> None:
        """执行 PCMCI+ 结果导出。"""

        has_data = any(selections.get(k) for k in ("adj_matrix", "val_matrix", "p_matrix", "mci_matrix"))
        has_images = any(selections.get(k) for k in ("graph_image", "ts_graph_image", "target_graph_image"))

        # 根据是否有数据文件，决定文件对话框类型
        save_dir: Path | None = None
        data_file_path: Path | None = None
        if has_data:
            file_filter = "Excel Files (*.xlsx)" if fmt == "xlsx" else "CSV Files (*.csv)"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出 PCMCI+ 数据", "", file_filter,
            )
            if not file_path:
                return
            data_file_path = Path(file_path)
            save_dir = data_file_path.parent
        elif has_images:
            dir_path = QFileDialog.getExistingDirectory(self, "选择图片保存目录")
            if not dir_path:
                return
            save_dir = Path(dir_path)
        else:
            return

        try:
            result = self.current_result
            var_names = result.var_names
            saved_files: list[str] = []

            # 导出数据矩阵
            data_dict: dict[str, pd.DataFrame] = {}
            if selections.get("adj_matrix"):
                data_dict["滞后邻接矩阵"] = pd.DataFrame(
                    result.adj_matrix, index=var_names, columns=var_names,
                )
            if selections.get("val_matrix"):
                data_dict["val矩阵"] = pd.DataFrame(
                    np.max(result.val_matrix[:, :, 1:], axis=2),
                    index=var_names, columns=var_names,
                )
            if selections.get("p_matrix"):
                data_dict["p值矩阵"] = pd.DataFrame(
                    np.min(result.p_matrix[:, :, 1:], axis=2),
                    index=var_names, columns=var_names,
                )
            if selections.get("mci_matrix"):
                data_dict["MCI矩阵"] = pd.DataFrame(
                    self._build_mci_summary_matrix(),
                    index=var_names, columns=var_names,
                )

            if data_dict and data_file_path:
                if fmt == "csv":
                    for name, df in data_dict.items():
                        csv_path = data_file_path.parent / f"{data_file_path.stem}_{name}.csv"
                        df.to_csv(csv_path)
                        saved_files.append(str(csv_path))
                else:
                    with pd.ExcelWriter(data_file_path, engine="openpyxl") as writer:
                        for name, df in data_dict.items():
                            df.to_excel(writer, sheet_name=name[:31])
                    saved_files.append(str(data_file_path))

            # 导出图片
            image_exports = {
                "graph_image": ("PCMCI+_因果图", self.graph_canvas),
                "ts_graph_image": ("PCMCI+_时间序列图", self.ts_graph_canvas),
                "target_graph_image": ("PCMCI+_目标变量上游影响图", self.target_graph_canvas),
            }
            for key, (name, canvas) in image_exports.items():
                if selections.get(key):
                    img_path = save_dir / f"{name}.png"
                    canvas.fig.savefig(str(img_path), dpi=150, bbox_inches="tight")
                    saved_files.append(str(img_path))

            QMessageBox.information(
                self, "完成",
                f"PCMCI+ 结果已导出：\n\n" + "\n".join(f"  {f}" for f in saved_files),
            )
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"导出失败：{exc}")

    def _show_te_export_dialog(self) -> None:
        """显示 TE 结果导出对话框。"""

        if self.current_te_result is None:
            QMessageBox.warning(self, "提示", "没有可导出的 TE 分析结果。")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("导出 TE 结果")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        # 标题
        title_label = QLabel("请选择要导出的 TE 结果：")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title_label)

        # 各类结果的复选框
        checkboxes: dict[str, QCheckBox] = {}
        options = [
            ("te_matrix", "TE 矩阵", True),
            ("ndte_matrix", "NDTE 矩阵", True),
            ("significant_pairs", "显著 TE 变量对", True),
            ("te_graph_image", "TE 结果图（图片）", False),
            ("te_bar_image", "TE 柱状图（图片）", False),
        ]

        for key, label, default_checked in options:
            cb = QCheckBox(label)
            cb.setChecked(default_checked)
            checkboxes[key] = cb
            layout.addWidget(cb)

        # 格式选择
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("导出格式："))
        format_combo = QComboBox()
        format_combo.addItem("Excel (.xlsx)", "xlsx")
        format_combo.addItem("CSV (.csv)", "csv")
        format_layout.addWidget(format_combo)
        layout.addLayout(format_layout)

        # 按钮
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        export_btn = QPushButton("导出")
        export_btn.setDefault(True)
        button_layout.addStretch(1)
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(export_btn)
        layout.addLayout(button_layout)

        def on_export():
            selected = {k: cb.isChecked() for k, cb in checkboxes.items()}
            if not any(selected.values()):
                QMessageBox.warning(dialog, "提示", "请至少选择一项要导出的内容。")
                return
            dialog.accept()
            self._do_export_te(selected, format_combo.currentData())

        export_btn.clicked.connect(on_export)
        dialog.exec_()

    def _do_export_te(self, selections: dict[str, bool], fmt: str) -> None:
        """执行 TE 结果导出。"""

        has_data = any(selections.get(k) for k in ("te_matrix", "ndte_matrix", "significant_pairs"))
        has_images = any(selections.get(k) for k in ("te_graph_image", "te_bar_image"))

        # 根据是否有数据文件，决定文件对话框类型
        save_dir: Path | None = None
        data_file_path: Path | None = None
        if has_data:
            file_filter = "Excel Files (*.xlsx)" if fmt == "xlsx" else "CSV Files (*.csv)"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出 TE 数据", "", file_filter,
            )
            if not file_path:
                return
            data_file_path = Path(file_path)
            save_dir = data_file_path.parent
        elif has_images:
            dir_path = QFileDialog.getExistingDirectory(self, "选择图片保存目录")
            if not dir_path:
                return
            save_dir = Path(dir_path)
        else:
            return

        try:
            te_result = self.current_te_result
            var_names = te_result.var_names
            tau_max = te_result.config.tau_max

            max_te = np.max(te_result.te_matrix[:, :, 1:], axis=2) if tau_max >= 1 else np.zeros((len(var_names), len(var_names)))
            max_ndte = np.max(te_result.ndte_matrix[:, :, 1:], axis=2) if tau_max >= 1 else np.zeros((len(var_names), len(var_names)))
            saved_files: list[str] = []

            # 准备矩阵数据
            data_dict: dict[str, pd.DataFrame] = {}
            if selections.get("te_matrix"):
                data_dict["TE矩阵"] = pd.DataFrame(max_te, index=var_names, columns=var_names)
            if selections.get("ndte_matrix"):
                data_dict["NDTE矩阵"] = pd.DataFrame(max_ndte, index=var_names, columns=var_names)
            if selections.get("significant_pairs") and te_result.significant_pairs:
                data_dict["显著变量对"] = pd.DataFrame(
                    te_result.significant_pairs,
                    columns=["源变量", "目标变量", "滞后期", "TE值", "NDTE值"],
                )

            if data_dict and data_file_path:
                if fmt == "csv":
                    for name, df in data_dict.items():
                        csv_path = data_file_path.parent / f"{data_file_path.stem}_{name}.csv"
                        df.to_csv(csv_path, index=name != "显著变量对")
                        saved_files.append(str(csv_path))
                else:
                    with pd.ExcelWriter(data_file_path, engine="openpyxl") as writer:
                        for name, df in data_dict.items():
                            df.to_excel(writer, sheet_name=name[:31], index=name != "显著变量对")
                    saved_files.append(str(data_file_path))

            # 导出图片
            image_exports = {
                "te_graph_image": ("TE_结果图", self.te_graph_canvas),
                "te_bar_image": ("TE_柱状图", self.te_bar_canvas),
            }
            for key, (name, canvas) in image_exports.items():
                if selections.get(key):
                    img_path = save_dir / f"{name}.png"
                    canvas.fig.savefig(str(img_path), dpi=150, bbox_inches="tight")
                    saved_files.append(str(img_path))

            QMessageBox.information(
                self, "完成",
                f"TE 结果已导出：\n\n" + "\n".join(f"  {f}" for f in saved_files),
            )
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"导出失败：{exc}")

    def _save_result_payload(self, result: AnalysisResult) -> Path:
        output_dir = self.project_root / self.app_config.defaults.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self.app_config.defaults.result_file_name
        with output_path.open("wb") as file_obj:
            pickle.dump(result.to_dict(), file_obj)
        return output_path

    def _refresh_results_summary(self) -> None:
        if self.current_result is None:
            self.results_summary_text.setPlainText("")
            return

        summary = self.current_result.summary
        lines = [
            f"本次分析变量数：{summary.get('var_count', len(self.current_result.var_names))}",
            f"结果文件：{self.current_result_path or '未保存'}",
            "角色摘要：",
        ]

        for label, names in (summary.get("roles", {}) or {}).items():
            lines.append(f"  - {label}：{', '.join(names)}")

        params = summary.get("parameters", {}) or {}
        lines.extend(
            [
                "参数：",
                f"  - tau_min = {params.get('tau_min', self.current_result.tau_min)}",
                f"  - tau_max = {params.get('tau_max', self.current_result.tau_max)}",
                f"  - pc_alpha = {params.get('pc_alpha', self.current_result.pc_alpha)}",
                f"  - 归一化分位数 = {params.get('quantile_range', [])}",
            ]
        )

        blocked = summary.get("incoming_blocked", []) or []
        if blocked:
            lines.append("无入边驱动变量：" + ", ".join(blocked))

        if self.current_te_result is not None:
            te_summary = self.current_te_result.summary
            lines.append("")
            lines.append("=== TE 分析结果 ===")
            lines.append(f"TE 变量对总数: {te_summary.get('total_pairs', 0)}")
            top = te_summary.get("top_contributors", [])
            if top:
                lines.append("TE 贡献排名前 5:")
                for item in top[:5]:
                    lines.append(f"  - {item['source']} -> {item['target']} (lag={item['lag']}, TE={item['te']:.4f})")

        self.results_summary_text.setPlainText("\n".join(lines))

    def _refresh_tab_state(self) -> None:
        has_data = self.raw_df is not None and not self._get_base_numeric_df().empty
        has_analysis_vars = len(self.get_analysis_var_names()) >= 2
        has_pcmci_result = self.current_result is not None
        has_te_result = self.current_te_result is not None
        worker_running = self.worker is not None and self.worker.isRunning()

        self.step_tabs.setTabEnabled(1, has_data)
        self.step_tabs.setTabEnabled(2, has_data)
        self.step_tabs.setTabEnabled(3, has_pcmci_result or has_te_result)
        self.step1_next_button.setEnabled(has_data)
        self.step2_next_button.setEnabled(has_analysis_vars)
        self.run_analysis_button.setEnabled(has_analysis_vars and not worker_running)
        self.target_variable_combo.setEnabled(has_pcmci_result and self.target_variable_combo.count() > 0)
        self.only_target_edges_checkbox.setEnabled(has_pcmci_result)
        self.only_related_vars_checkbox.setEnabled(has_pcmci_result)
        self.hide_historical_contemporaneous_checkbox.setEnabled(has_pcmci_result)
        self.hide_ambiguous_edges_checkbox.setEnabled(has_pcmci_result)

        if hasattr(self, "run_pcmci_button"):
            self.run_pcmci_button.setEnabled(has_analysis_vars and not worker_running)
        if hasattr(self, "run_te_button"):
            self.run_te_button.setEnabled(has_analysis_vars and not worker_running)

        # 更新导出按钮状态
        self._update_export_button_text(self.algo_result_tabs.currentIndex())

    def _clear_result_views(self) -> None:
        self.results_summary_text.clear()
        self.adj_table.clear()
        self.adj_table.setRowCount(0)
        self.adj_table.setColumnCount(0)
        self.mci_table.clear()
        self.mci_table.setRowCount(0)
        self.mci_table.setColumnCount(0)
        self.graph_canvas.clear_figure()
        self.ts_graph_canvas.clear_figure()
        self.target_graph_canvas.clear_figure()
        self.target_variable_combo.blockSignals(True)
        self.target_variable_combo.clear()
        self.target_variable_combo.setEnabled(False)
        self.target_variable_combo.blockSignals(False)
        self.only_target_edges_checkbox.setChecked(False)
        self.only_related_vars_checkbox.setChecked(True)
        self.hide_historical_contemporaneous_checkbox.setChecked(False)
        self.hide_ambiguous_edges_checkbox.setChecked(False)

        self.te_graph_canvas.clear_figure()
        if hasattr(self, 'te_bar_canvas'):
            self.te_bar_canvas.clear_figure()
        self.te_matrix_table.clear()
        self.te_matrix_table.setRowCount(0)
        self.te_matrix_table.setColumnCount(0)
        self.ndte_matrix_table.clear()
        self.ndte_matrix_table.setRowCount(0)
        self.ndte_matrix_table.setColumnCount(0)
        self.te_table.setRowCount(0)
        self.te_hint_label.setText("TE 分析结果将在启用 TE 并完成分析后显示。")

        self._refresh_tab_state()

    def display_adj_matrix(self) -> None:
        if self.current_result is None:
            return

        adj_matrix = self.current_result.adj_matrix
        var_names = self.current_result.var_names
        self.adj_table.setRowCount(len(var_names))
        self.adj_table.setColumnCount(len(var_names))
        self.adj_table.setHorizontalHeaderLabels(var_names)
        self.adj_table.setVerticalHeaderLabels(var_names)

        for row in range(len(var_names)):
            for col in range(len(var_names)):
                item = QTableWidgetItem(str(int(adj_matrix[row, col])))
                if adj_matrix[row, col] == 1:
                    item.setBackground(QColor(THEME_COLORS["accent_light"]))
                self.adj_table.setItem(row, col, item)

        self.adj_table.resizeColumnsToContents()

    def _build_mci_summary_matrix(self) -> np.ndarray:
        if self.current_result is None:
            return np.empty((0, 0))

        positive_taus = range(max(1, self.current_result.tau_min), self.current_result.tau_max + 1)
        tau_list = list(positive_taus)
        if not tau_list:
            size = len(self.current_result.var_names)
            return np.zeros((size, size), dtype=float)

        stacked = np.stack(
            [self.current_result.val_matrix[:, :, tau] for tau in tau_list],
            axis=2,
        )
        return np.max(stacked, axis=2)

    def display_mci_matrix(self) -> None:
        if self.current_result is None:
            return

        matrix = self._build_mci_summary_matrix()
        var_names = self.current_result.var_names
        self.mci_table.setRowCount(len(var_names))
        self.mci_table.setColumnCount(len(var_names))
        self.mci_table.setHorizontalHeaderLabels(var_names)
        self.mci_table.setVerticalHeaderLabels(var_names)

        for row in range(len(var_names)):
            for col in range(len(var_names)):
                self.mci_table.setItem(row, col, QTableWidgetItem(f"{matrix[row, col]:.4f}"))

        self.mci_table.resizeColumnsToContents()

    def export_mci_matrix(self) -> None:
        if self.current_result is None:
            QMessageBox.warning(self, "提示", "请先运行分析。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 MCI 矩阵",
            "",
            "Excel Files (*.xlsx);;CSV Files (*.csv)",
        )
        if not file_path:
            return

        try:
            matrix = self._build_mci_summary_matrix()
            df = pd.DataFrame(
                matrix,
                index=self.current_result.var_names,
                columns=self.current_result.var_names,
            )
            path = Path(file_path)
            if path.suffix.lower() == ".csv":
                df.to_csv(path)
            else:
                df.to_excel(path, engine="openpyxl")
            QMessageBox.information(self, "完成", f"MCI 矩阵已导出到：\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"导出失败：{exc}")

    def plot_causal_graph(self) -> None:
        if self.current_result is None:
            return

        self.graph_canvas.clear_figure()
        plot_kwargs = get_tigramite_plot_kwargs()
        tp.plot_graph(
            graph=self.current_result.graph,
            val_matrix=self.current_result.val_matrix,
            var_names=self.current_result.var_names,
            fig_ax=(self.graph_canvas.fig, self.graph_canvas.axes),
            figsize=(8, 8),
            curved_radius=0.16,
            show_autodependency_lags=False,
            **plot_kwargs,
        )
        style_result_axes(self.graph_canvas.axes, square=True)
        apply_colorbar_theme(self.graph_canvas.fig)
        self.graph_canvas.draw()

    def plot_time_series_graph(self) -> None:
        if self.current_result is None:
            return

        self.ts_graph_canvas.clear_figure()
        target_name = str(self.target_variable_combo.currentData() or "")
        if not target_name:
            target_name = self._get_default_target_name(self.current_result.var_names)

        graph_data = build_target_centric_time_series_data(
            graph=self.current_result.graph,
            p_matrix=self.current_result.p_matrix,
            val_matrix=self.current_result.val_matrix,
            var_names=self.current_result.var_names,
            pc_alpha=self.current_result.pc_alpha,
            target_name=target_name,
            only_target_edges=self.only_target_edges_checkbox.isChecked(),
            only_related_vars=self.only_related_vars_checkbox.isChecked(),
            hide_historical_contemporaneous=self.hide_historical_contemporaneous_checkbox.isChecked(),
            hide_ambiguous_edges=self.hide_ambiguous_edges_checkbox.isChecked(),
        )
        draw_target_centric_time_series(
            ax=self.ts_graph_canvas.axes,
            data=graph_data,
            var_names=self.current_result.var_names,
        )
        apply_colorbar_theme(self.ts_graph_canvas.fig)
        self.ts_graph_canvas.draw()

    def plot_target_influence_graph(self) -> None:
        if self.current_result is None:
            return

        target_name = str(self.target_variable_combo.currentData() or "")
        if not target_name:
            return

        self.target_graph_canvas.clear_figure()
        graph_data = build_upstream_graph_data(
            graph=self.current_result.graph,
            p_matrix=self.current_result.p_matrix,
            val_matrix=self.current_result.val_matrix,
            var_names=self.current_result.var_names,
            pc_alpha=self.current_result.pc_alpha,
            target_name=target_name,
        )
        draw_upstream_graph(
            ax=self.target_graph_canvas.axes,
            data=graph_data,
            var_names=self.current_result.var_names,
        )
        apply_colorbar_theme(self.target_graph_canvas.fig)
        self.target_graph_canvas.draw()


# 兼容旧导入路径，避免过渡期间外部模块报错。
PCMCIWindow = MainWindow
