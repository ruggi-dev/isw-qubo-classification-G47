import importlib
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


def test_gui_module_imports() -> None:
    module = importlib.import_module("src.qubo_project.gui")
    assert hasattr(module, "PipelineMainWindow")


def test_navigation_items_fit_without_scrollbars() -> None:
    module = importlib.import_module("src.qubo_project.gui")
    app = QApplication.instance() or QApplication([])
    window = module.PipelineMainWindow()
    try:
        window.resize(1050, 700)
        window.show()
        app.processEvents()

        navigation = window.navigation
        assert (
            navigation.verticalScrollBarPolicy()
            == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        assert (
            navigation.horizontalScrollBarPolicy()
            == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        item_rects = [
            navigation.visualItemRect(navigation.item(index))
            for index in range(navigation.count())
        ]
        assert all(
            0 <= rect.top() and rect.bottom() < navigation.viewport().height()
            for rect in item_rects
        )
    finally:
        window.close()
