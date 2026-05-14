from __future__ import annotations

import sys
import matplotlib.pyplot as plt
import matplotlib

from PyQt5.QtWidgets import QApplication

from app.gui.main_window import MainWindow


def main() -> int:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
