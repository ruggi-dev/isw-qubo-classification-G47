import importlib
import os
import unittest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class GuiSmokeTests(unittest.TestCase):
    def test_gui_module_imports(self) -> None:
        module = importlib.import_module("src.qubo_project.gui")
        self.assertTrue(hasattr(module, "PipelineMainWindow"))

    def test_navigation_items_fit_without_scrollbars(self) -> None:
        module = importlib.import_module("src.qubo_project.gui")
        app = QApplication.instance() or QApplication([])
        window = module.PipelineMainWindow()
        window.resize(1050, 700)
        window.show()
        app.processEvents()
        self.addCleanup(window.close)

        navigation = window.navigation
        self.assertEqual(
            navigation.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertEqual(
            navigation.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        item_rects = [
            navigation.visualItemRect(navigation.item(index))
            for index in range(navigation.count())
        ]
        self.assertTrue(
            all(
                0 <= rect.top() and rect.bottom() < navigation.viewport().height()
                for rect in item_rects
            )
        )


if __name__ == "__main__":
    unittest.main()
