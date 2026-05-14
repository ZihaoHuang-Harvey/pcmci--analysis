from __future__ import annotations

import matplotlib

matplotlib.use("Qt5Agg")

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from app.gui.result_plotting import THEME_COLORS


class CanvasWidget(FigureCanvas):
    """Matplotlib 画布包装。

    单独抽出这一层，是为了避免控制器里直接操作 Figure 创建细节。
    """

    def __init__(self, parent=None, width: int = 8, height: int = 6, dpi: int = 100):
        self.fig = Figure(
            figsize=(width, height),
            dpi=dpi,
            facecolor=THEME_COLORS["surface"],
            constrained_layout=True,
        )
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor(THEME_COLORS["surface_alt"])

    def clear_figure(self) -> None:
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.fig.set_facecolor(THEME_COLORS["surface"])
        self.axes.set_facecolor(THEME_COLORS["surface_alt"])
        self.draw()
