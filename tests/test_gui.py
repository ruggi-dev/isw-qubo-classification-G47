import importlib
import unittest


class GuiSmokeTests(unittest.TestCase):
    def test_gui_module_imports(self) -> None:
        module = importlib.import_module("src.qubo_project.gui")
        self.assertTrue(hasattr(module, "PipelineMainWindow"))


if __name__ == "__main__":
    unittest.main()
