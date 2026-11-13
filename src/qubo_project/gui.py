from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

project_root = Path(__file__).resolve().parents[2]
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

try:
    from qubo_project.feature_selection import select_features
    from qubo_project.model import predict, train
    from qubo_project.preprocessing import fit_normalize
except ModuleNotFoundError:  # pragma: no cover - fallback for package-style imports
    from src.qubo_project.feature_selection import select_features
    from src.qubo_project.model import predict, train
    from src.qubo_project.preprocessing import fit_normalize


CLASSIFIERS = [
    ("random_forest", "Random Forest"),
    ("logistic_regression", "Logistic Regression"),
    ("decision_tree", "Decision Tree"),
]

CLASSIFIER_DESCRIPTIONS = {
    "random_forest": (
        "Combines 100 decision trees. A strong general-purpose choice for "
        "tabular data that can model non-linear relationships, at the cost "
        "of being less transparent than a single tree."
    ),
    "logistic_regression": (
        "A linear classifier that estimates class probabilities. It is a "
        "simple, fast baseline; it may be less suitable when the classes "
        "cannot be separated by a roughly linear boundary."
    ),
    "decision_tree": (
        "Uses one sequence of feature-based decisions. It is relatively easy "
        "to interpret, but can overfit and may be less stable than an ensemble."
    ),
}

APP_STYLESHEET = """
QWidget {
    background-color: #e9efec;
    color: #1f302b;
    font-size: 10pt;
}
QMainWindow {
    background-color: #e9efec;
}
QFrame#sidebar {
    background-color: #dce5e0;
    border-right: 1px solid #b9c9c1;
}
QLabel#brandTitle {
    color: #24483f;
    font-size: 19pt;
    font-weight: bold;
}
QLabel#pageTitle {
    color: #223f37;
    font-size: 17pt;
    font-weight: bold;
    padding-bottom: 3px;
}
QListWidget#navigation {
    background: transparent;
    border: none;
    outline: none;
}
QListWidget#navigation::item {
    color: #334842;
    border-radius: 7px;
    padding: 10px 12px;
    margin: 2px 0;
}
QListWidget#navigation::item:hover {
    background-color: #ccd9d3;
}
QListWidget#navigation::item:selected {
    background-color: #bdcec6;
    color: #1e4036;
    font-weight: 600;
}
QPushButton {
    background-color: #426b60;
    color: #ffffff;
    border: 1px solid #426b60;
    border-radius: 6px;
    padding: 8px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #365b51;
    border-color: #365b51;
}
QPushButton:pressed {
    background-color: #2a493f;
}
QPushButton:disabled {
    background-color: #c9d5cf;
    border-color: #c9d5cf;
    color: #596961;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
QPlainTextEdit, QTextEdit, QTableView, QTableWidget {
    background-color: #fbfcfb;
    color: #1f302b;
    border: 1px solid #adbbb4;
    border-radius: 6px;
    padding: 6px;
    selection-background-color: #c9ddd5;
    selection-color: #1f302b;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus, QTextEdit:focus, QTableView:focus, QTableWidget:focus {
    border-color: #527b6e;
}
QComboBox::drop-down {
    background-color: #d3ded8;
    border-left: 1px solid #adbbb4;
    width: 25px;
}
QComboBox::drop-down:hover {
    background-color: #bdcec5;
}
QGroupBox {
    background-color: #f5f8f6;
    border: 1px solid #c5d1ca;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #304c43;
}
QHeaderView::section {
    background-color: #dce6e1;
    color: #2f433c;
    border: none;
    border-bottom: 1px solid #bdcbc3;
    padding: 8px;
    font-weight: 600;
}
QTableView, QTableWidget {
    alternate-background-color: #edf2ef;
    gridline-color: #d5dfd9;
}
QStatusBar {
    background-color: #dce6e1;
    color: #344840;
    border-top: 1px solid #b9c9c1;
}
QStatusBar::item {
    border: none;
}
QProgressBar {
    background-color: #c9d7d0;
    border: none;
    border-radius: 4px;
    max-height: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #52796e;
    border-radius: 4px;
}
QToolTip {
    background-color: #f8faf9;
    color: #1f302b;
    border: 1px solid #aebfb6;
    padding: 6px;
}
QScrollBar:vertical {
    background-color: #e2e9e5;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background-color: #aabcb2;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background-color: #8ea59a;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""

SPIN_CONTROL_STYLESHEET = """
SPIN_CONTROL::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 23px;
    background-color: #d3ded8;
    border-left: 1px solid #adbbb4;
    border-bottom: 1px solid #adbbb4;
    border-top-right-radius: 5px;
}
SPIN_CONTROL::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 23px;
    background-color: #d3ded8;
    border-left: 1px solid #adbbb4;
    border-top: 1px solid #adbbb4;
    border-bottom-right-radius: 5px;
}
SPIN_CONTROL::up-button:hover,
SPIN_CONTROL::down-button:hover {
    background-color: #bdcec5;
}
SPIN_CONTROL::up-arrow {
    image: url("SPIN_UP_ARROW_PATH");
    width: 8px;
    height: 8px;
}
SPIN_CONTROL::down-arrow {
    image: url("SPIN_DOWN_ARROW_PATH");
    width: 8px;
    height: 8px;
}
"""


def _as_display_path(path: Path | str | None) -> str:
    if path is None:
        return "Not selected"
    return str(Path(path))


class BackgroundWorker(QThread):
    finished = Signal(object)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, task: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._task = task
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            self.status.emit("Running...")
            result = self._task(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - GUI path for user errors
            self.failed.emit(str(exc))


class DatasetValidationResult:
    def __init__(self, dataframe: pd.DataFrame, warnings: list[str], summary: dict[str, Any]) -> None:
        self.dataframe = dataframe
        self.warnings = warnings
        self.summary = summary


class PipelineMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet(APP_STYLESHEET)
        self.setWindowTitle("QUBO Classification Pipeline")
        self.resize(1300, 820)
        self.setMinimumSize(1050, 700)

        self.project_root = Path(__file__).resolve().parents[2]
        self.output_dir = self.project_root / "outputs"
        self.dataset_path: Path | None = None
        self.normalized_path: Path | None = None
        self.reduced_train_path: Path | None = None
        self.reduced_test_path: Path | None = None
        self.model_path: Path | None = None
        self.predictions_path: Path | None = None
        self.training_stats: dict[str, Any] = {}
        self.prediction_stats: dict[str, Any] = {}
        self.preprocessing_stats: dict[str, Any] = {}
        self.selection_stats: dict[str, Any] = {}
        self.current_dataset: pd.DataFrame | None = None
        self.current_target: str | None = None
        self.mode: str = "Beginner"
        self.logs: list[str] = []

        self._build_ui()
        assets_dir = Path(__file__).resolve().parent / "assets"
        for control_type, spin_class in (
            (QSpinBox, "QSpinBox"),
            (QDoubleSpinBox, "QDoubleSpinBox"),
        ):
            spin_stylesheet = SPIN_CONTROL_STYLESHEET.replace(
                "SPIN_CONTROL", spin_class
            ).replace(
                "SPIN_UP_ARROW_PATH", (assets_dir / "spin-up.svg").as_posix()
            ).replace(
                "SPIN_DOWN_ARROW_PATH", (assets_dir / "spin-down.svg").as_posix()
            )
            for spin_control in self.findChildren(control_type):
                spin_control.setStyleSheet(spin_stylesheet)
        self._configure_defaults()
        self._load_recent_state()
        self.append_log("Application started.")

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(18, 24, 16, 20)
        sidebar_layout.setSpacing(12)

        title = QLabel("QUBO Pipeline")
        title.setObjectName("brandTitle")
        sidebar_layout.addWidget(title)

        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        self.navigation.setSpacing(4)
        self.navigation.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.navigation.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.navigation.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.navigation.setMinimumHeight(360)
        sections = [
            "Dashboard",
            "Dataset",
            "Preprocessing",
            "Feature Selection",
            "Training",
            "Prediction",
            "Results",
            "Settings",
        ]
        for section in sections:
            item = QListWidgetItem(section)
            item.setData(Qt.ItemDataRole.UserRole, section)
            self.navigation.addItem(item)
        self.navigation.setCurrentRow(0)
        self.navigation.currentRowChanged.connect(self._switch_page)
        sidebar_layout.addWidget(self.navigation, 1)
        sidebar_layout.addStretch(1)
        sidebar.setMinimumWidth(205)
        sidebar.setMaximumWidth(250)
        main_layout.addWidget(sidebar, 1)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_dashboard_page())
        self.stack.addWidget(self._build_dataset_page())
        self.stack.addWidget(self._build_preprocessing_page())
        self.stack.addWidget(self._build_feature_page())
        self.stack.addWidget(self._build_training_page())
        self.stack.addWidget(self._build_prediction_page())
        self.stack.addWidget(self._build_results_page())
        self.stack.addWidget(self._build_settings_page())
        main_layout.addWidget(self.stack, 5)

        self.statusBar().showMessage("Ready")
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _switch_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)

    def _create_page_layout(self, page: QWidget) -> QVBoxLayout:
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)
        return layout

    def _build_dashboard_page(self) -> QWidget:
        page = QWidget()
        layout = self._create_page_layout(page)

        title = QLabel("Project dashboard")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        summary = QTextEdit()
        summary.setReadOnly(True)
        summary.setPlainText(
            "Step-by-step binary classification workflow:\n\n"
            "1. Import CSV dataset and choose target column\n"
            "2. Validate dataset and tune preprocessing\n"
            "3. Run QUBO-based feature reduction\n"
            "4. Train a classifier and save the model\n"
            "5. Predict on the reduced test data and review evaluation metrics\n\n"
            "Beginner mode resets the available controls to recommended defaults. "
            "Advanced mode keeps your chosen values; it does not add extra controls."
        )
        layout.addWidget(summary)

        button_row = QHBoxLayout()
        next_btn = QPushButton("Start import")
        next_btn.clicked.connect(lambda: self.navigation.setCurrentRow(1))
        button_row.addWidget(next_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        layout.addStretch(1)
        return page

    def _build_dataset_page(self) -> QWidget:
        page = QWidget()
        layout = self._create_page_layout(page)

        title = QLabel("Dataset import and validation")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        form = QGridLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        dataset_label = QLabel("CSV file")
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setToolTip(
            "Path to a CSV file with a header row, numeric feature columns, "
            "and a binary target column."
        )
        browse_btn = QPushButton("Browse")
        browse_btn.setToolTip("Choose a CSV dataset from your computer.")
        browse_btn.clicked.connect(self._browse_dataset)
        form.addWidget(dataset_label, 0, 0)
        form.addWidget(self.dataset_path_edit, 0, 1)
        form.addWidget(browse_btn, 0, 2)

        target_label = QLabel("Target column")
        self.target_combo = QComboBox()
        self.target_combo.setToolTip(
            "Choose the column containing the labels to predict. It must "
            "contain only 0 and 1 and is not treated as an input feature."
        )
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        form.addWidget(target_label, 1, 0)
        form.addWidget(self.target_combo, 1, 1)
        layout.addLayout(form)

        dataset_help = QLabel(
            "Select the label column (0/1), not a feature. The target is used "
            "to train and evaluate the classifier; all other columns are "
            "candidate numeric features. Review validation warnings before "
            "continuing."
        )
        dataset_help.setWordWrap(True)
        layout.addWidget(dataset_help)

        self.dataset_summary = QLabel("No dataset selected yet.")
        layout.addWidget(self.dataset_summary)

        preview_group = QGroupBox("Dataset preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_table = QTableView()
        self.preview_table.setAlternatingRowColors(True)
        preview_layout.addWidget(self.preview_table)
        layout.addWidget(preview_group)

        warning_group = QGroupBox("Validation warnings")
        warning_layout = QVBoxLayout(warning_group)
        self.warning_box = QPlainTextEdit()
        self.warning_box.setReadOnly(True)
        warning_layout.addWidget(self.warning_box)
        layout.addWidget(warning_group)

        action_row = QHBoxLayout()
        load_btn = QPushButton("Load dataset")
        load_btn.clicked.connect(self._load_dataset)
        action_row.addWidget(load_btn)
        action_row.addStretch(1)
        next_btn = QPushButton("Next: preprocessing")
        next_btn.clicked.connect(lambda: self.navigation.setCurrentRow(2))
        action_row.addWidget(next_btn)
        layout.addLayout(action_row)
        return page

    def _build_preprocessing_page(self) -> QWidget:
        page = QWidget()
        layout = self._create_page_layout(page)

        title = QLabel("Preprocessing configuration")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        form = QGridLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        self.min_valid_spin = QDoubleSpinBox()
        self.min_valid_spin.setMinimum(0.0)
        self.min_valid_spin.setMaximum(1.0)
        self.min_valid_spin.setSingleStep(0.01)
        self.min_valid_spin.setValue(0.05)
        self.min_valid_spin.setDecimals(4)
        self.min_valid_spin.setToolTip(
            "Minimum fraction of all rows where a feature must have a "
            "non-missing, non-zero value. Range: 0 to 1. For example, 0.05 "
            "keeps features that are valid and non-zero in at least 5% of rows. "
            "Higher values drop more features."
        )
        form.addWidget(QLabel("Minimum valid proportion"), 0, 0)
        form.addWidget(self.min_valid_spin, 0, 1)
        layout.addLayout(form)

        preprocessing_help = QLabel(
            "Features below this threshold are removed. Retained feature values "
            "are z-score normalized; missing values are filled with the column "
            "mean. The target column is preserved unchanged."
        )
        preprocessing_help.setWordWrap(True)
        layout.addWidget(preprocessing_help)

        self.preprocessing_summary = QLabel("No preprocessing run yet.")
        layout.addWidget(self.preprocessing_summary)

        self.preprocessing_drops = QPlainTextEdit()
        self.preprocessing_drops.setReadOnly(True)
        self.preprocessing_drops.setPlaceholderText("Dropped columns will appear here after preprocessing.")
        layout.addWidget(self.preprocessing_drops)

        actions = QHBoxLayout()
        run_btn = QPushButton("Run preprocessing")
        run_btn.clicked.connect(self._run_preprocessing)
        actions.addWidget(run_btn)
        actions.addStretch(1)
        next_btn = QPushButton("Next: feature selection")
        next_btn.clicked.connect(lambda: self.navigation.setCurrentRow(3))
        actions.addWidget(next_btn)
        layout.addLayout(actions)
        return page

    def _build_feature_page(self) -> QWidget:
        page = QWidget()
        layout = self._create_page_layout(page)

        title = QLabel("Feature selection")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        form = QGridLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        self.perc_test_spin = QDoubleSpinBox()
        self.perc_test_spin.setRange(0.01, 0.9)
        self.perc_test_spin.setSingleStep(0.01)
        self.perc_test_spin.setValue(0.30)
        self.perc_test_spin.setDecimals(3)
        self.perc_selected_spin = QDoubleSpinBox()
        self.perc_selected_spin.setRange(0.0, 1.0)
        self.perc_selected_spin.setSingleStep(0.01)
        self.perc_selected_spin.setValue(0.20)
        self.perc_selected_spin.setDecimals(3)
        self.allowance_spin = QSpinBox()
        self.allowance_spin.setRange(0, 50)
        self.allowance_spin.setValue(1)
        self.selection_seed_spin = QSpinBox()
        self.selection_seed_spin.setRange(0, 100000)
        self.selection_seed_spin.setValue(42)
        self.perc_test_spin.setToolTip(
            "Fraction of rows reserved for testing (0.01 to 0.90). The split "
            "preserves row order: the first rows are used for training and the "
            "remaining rows for testing. Larger values leave fewer rows for training."
        )
        self.perc_selected_spin.setToolTip(
            "Desired fraction of input features to keep (0 to 1). The optimizer "
            "converts this fraction into a target feature count. Use a value "
            "above 0 if you plan to train a model."
        )
        self.allowance_spin.setToolTip(
            "How far the selected feature count may differ from the target "
            "count and still count as reached. 0 requires an exact match; "
            "larger values allow a wider difference."
        )
        self.selection_seed_spin.setToolTip(
            "Seed for the QUBO simulated-annealing search. Reusing the same "
            "seed and data makes the search reproducible."
        )

        form.addWidget(QLabel("Test set percentage"), 0, 0)
        form.addWidget(self.perc_test_spin, 0, 1)
        form.addWidget(QLabel("Feature target percentage"), 1, 0)
        form.addWidget(self.perc_selected_spin, 1, 1)
        form.addWidget(QLabel("Feature allowance"), 2, 0)
        form.addWidget(self.allowance_spin, 2, 1)
        form.addWidget(QLabel("Optimization seed"), 3, 0)
        form.addWidget(self.selection_seed_spin, 3, 1)
        layout.addLayout(form)

        selection_help = QLabel(
            "Feature selection uses correlations from the training portion only. "
            "If the requested count is not reached within the allowance, the "
            "closest result is kept and the status reports that the target was "
            "not reached. The seed helps reproduce the optimization."
        )
        selection_help.setWordWrap(True)
        layout.addWidget(selection_help)

        self.feature_summary = QLabel("No feature reduction run yet.")
        layout.addWidget(self.feature_summary)

        self.selected_features_box = QPlainTextEdit()
        self.selected_features_box.setReadOnly(True)
        self.selected_features_box.setPlaceholderText("Selected feature names will appear here.")
        layout.addWidget(self.selected_features_box)

        actions = QHBoxLayout()
        run_btn = QPushButton("Run feature selection")
        run_btn.clicked.connect(self._run_feature_selection)
        actions.addWidget(run_btn)
        actions.addStretch(1)
        next_btn = QPushButton("Next: model training")
        next_btn.clicked.connect(lambda: self.navigation.setCurrentRow(4))
        actions.addWidget(next_btn)
        layout.addLayout(actions)
        return page

    def _build_training_page(self) -> QWidget:
        page = QWidget()
        layout = self._create_page_layout(page)

        title = QLabel("Model training")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        form = QGridLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        self.classifier_combo = QComboBox()
        for key, label in CLASSIFIERS:
            self.classifier_combo.addItem(label, key)
            self.classifier_combo.setItemData(
                self.classifier_combo.count() - 1,
                CLASSIFIER_DESCRIPTIONS[key],
                Qt.ItemDataRole.ToolTipRole,
            )
        self.classifier_combo.setCurrentText("Random Forest")
        self.training_seed_spin = QSpinBox()
        self.training_seed_spin.setRange(0, 100000)
        self.training_seed_spin.setValue(42)
        self.training_seed_spin.setToolTip(
            "Controls randomized behavior where the classifier uses it. "
            "Using the same seed with the same data helps reproduce training."
        )
        form.addWidget(QLabel("Classifier"), 0, 0)
        form.addWidget(self.classifier_combo, 0, 1)
        form.addWidget(QLabel("Random seed"), 1, 0)
        form.addWidget(self.training_seed_spin, 1, 1)
        layout.addLayout(form)

        self.classifier_help = QLabel()
        self.classifier_help.setWordWrap(True)
        self.classifier_combo.currentIndexChanged.connect(
            self._update_classifier_description
        )
        self._update_classifier_description(self.classifier_combo.currentIndex())
        layout.addWidget(self.classifier_help)

        self.training_summary = QLabel("No training run yet.")
        layout.addWidget(self.training_summary)

        actions = QHBoxLayout()
        run_btn = QPushButton("Train classifier")
        run_btn.clicked.connect(self._run_training)
        actions.addWidget(run_btn)
        actions.addStretch(1)
        next_btn = QPushButton("Next: prediction")
        next_btn.clicked.connect(lambda: self.navigation.setCurrentRow(5))
        actions.addWidget(next_btn)
        layout.addLayout(actions)
        return page

    def _build_prediction_page(self) -> QWidget:
        page = QWidget()
        layout = self._create_page_layout(page)

        title = QLabel("Prediction and evaluation")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        prediction_help = QLabel(
            "Prediction uses the trained model on the reduced test set. The test "
            "CSV must have the same feature columns as the training data and "
            "include the target labels so the predictions can be evaluated. "
            "Each score is the estimated probability of class 1."
        )
        prediction_help.setWordWrap(True)
        layout.addWidget(prediction_help)

        self.evaluation_summary = QLabel("No prediction run yet.")
        layout.addWidget(self.evaluation_summary)

        self.prediction_table = QTableWidget()
        self.prediction_table.setAlternatingRowColors(True)
        layout.addWidget(self.prediction_table)

        actions = QHBoxLayout()
        run_btn = QPushButton("Run prediction")
        run_btn.clicked.connect(self._run_prediction)
        actions.addWidget(run_btn)
        actions.addStretch(1)
        next_btn = QPushButton("Next: results")
        next_btn.clicked.connect(lambda: self.navigation.setCurrentRow(6))
        actions.addWidget(next_btn)
        layout.addLayout(actions)
        return page

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        layout = self._create_page_layout(page)

        title = QLabel("Results and metrics")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        results_help = QLabel(
            "Accuracy is the share of correct predictions. Precision measures "
            "how many predicted positives are correct; recall measures how many "
            "actual positives were found; F1 balances precision and recall. "
            "ROC-AUC summarizes class separation across thresholds. In the "
            "confusion matrix, rows are actual classes and columns are predicted classes."
        )
        results_help.setWordWrap(True)
        layout.addWidget(results_help)

        self.metric_table = QTableWidget(0, 2)
        self.metric_table.setAlternatingRowColors(True)
        self.metric_table.setHorizontalHeaderLabels(["Metric", "Value"])
        layout.addWidget(self.metric_table)

        cm_group = QGroupBox("Confusion matrix")
        cm_layout = QVBoxLayout(cm_group)
        self.confusion_matrix = QTableWidget(2, 2)
        self.confusion_matrix.setAlternatingRowColors(True)
        self.confusion_matrix.setHorizontalHeaderLabels(["Pred 0", "Pred 1"])
        self.confusion_matrix.setVerticalHeaderLabels(["Actual 0", "Actual 1"])
        cm_layout.addWidget(self.confusion_matrix)
        layout.addWidget(cm_group)

        self.result_summary = QTextEdit()
        self.result_summary.setReadOnly(True)
        layout.addWidget(self.result_summary)

        actions = QHBoxLayout()
        export_btn = QPushButton("Export results")
        export_btn.setToolTip(
            "Copy the generated model and available CSV artifacts to a folder "
            "you choose."
        )
        export_btn.clicked.connect(self._export_results)
        actions.addWidget(export_btn)
        actions.addStretch(1)
        summary_btn = QPushButton("Return to dashboard")
        summary_btn.clicked.connect(lambda: self.navigation.setCurrentRow(0))
        actions.addWidget(summary_btn)
        layout.addLayout(actions)
        return page

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        layout = self._create_page_layout(page)

        title = QLabel("Settings and logs")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        form = QGridLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        output_label = QLabel("Output folder")
        self.output_path_edit = QLineEdit(str(self.output_dir))
        self.output_path_edit.setToolTip(
            "Folder where pipeline outputs are saved. It is created if it does not exist."
        )
        browse_output = QPushButton("Browse")
        browse_output.setToolTip("Choose a different output folder.")
        browse_output.clicked.connect(self._browse_output_dir)
        form.addWidget(output_label, 0, 0)
        form.addWidget(self.output_path_edit, 0, 1)
        form.addWidget(browse_output, 0, 2)

        mode_label = QLabel("Mode")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Beginner", "Advanced"])
        self.mode_combo.setItemData(
            0,
            "Resets the preprocessing threshold, feature-selection settings, "
            "classifier, and seeds to the recommended defaults.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.mode_combo.setItemData(
            1,
            "Keeps the values you set in the available controls. The current "
            "GUI does not expose additional controls in this mode.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.mode_combo.setCurrentText(self.mode)
        self.mode_combo.currentTextChanged.connect(self._set_mode)
        form.addWidget(mode_label, 1, 0)
        form.addWidget(self.mode_combo, 1, 1)
        layout.addLayout(form)

        settings_help = QLabel(
            "Beginner resets available settings to their defaults. Advanced "
            "preserves your current selections so you can adjust them manually; "
            "it does not reveal extra parameters."
        )
        settings_help.setWordWrap(True)
        layout.addWidget(settings_help)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)

        save_button = QPushButton("Save selected output folder")
        save_button.setToolTip(
            "Apply the folder shown above as the destination for pipeline outputs."
        )
        save_button.clicked.connect(self._apply_output_dir)
        layout.addWidget(save_button)
        return page

    def _configure_defaults(self) -> None:
        self.output_dir = self.project_root / "outputs"
        self.output_path_edit.setText(str(self.output_dir))
        self._apply_output_dir()

    def _load_recent_state(self) -> None:
        self.append_log("Project ready. Select a dataset to start the workflow.")

    def _browse_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV dataset",
            str(self.project_root),
            "CSV Files (*.csv)",
        )
        if not path:
            return
        self.dataset_path = Path(path)
        self.dataset_path_edit.setText(str(self.dataset_path))
        self.append_log(f"Dataset selected: {self.dataset_path}")
        self._load_dataset()

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            str(self.project_root),
        )
        if not path:
            return
        self.output_path_edit.setText(path)
        self._apply_output_dir()

    def _apply_output_dir(self) -> None:
        output_dir = Path(self.output_path_edit.text()).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.append_log(f"Output folder set to: {output_dir}")
        self.statusBar().showMessage(f"Output dir: {output_dir}")

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode == "Beginner":
            self.min_valid_spin.setValue(0.05)
            self.perc_test_spin.setValue(0.30)
            self.perc_selected_spin.setValue(0.20)
            self.allowance_spin.setValue(1)
            self.selection_seed_spin.setValue(42)
            self.training_seed_spin.setValue(42)
            self.classifier_combo.setCurrentText("Random Forest")
        self.append_log(f"Workflow mode changed to: {mode}")

    def _update_classifier_description(self, index: int) -> None:
        classifier = str(self.classifier_combo.itemData(index))
        self.classifier_help.setText(CLASSIFIER_DESCRIPTIONS[classifier])

    def _on_target_changed(self, target_name: str) -> None:
        self.current_target = target_name or None

    def _load_dataset(self) -> None:
        path = self.dataset_path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Dataset required", "Choose a CSV dataset before loading.")
            return
        try:
            dataset = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - user interaction path
            QMessageBox.critical(self, "Dataset error", f"Could not read the CSV file: {exc}")
            return

        self.current_dataset = dataset
        self.dataset_path = Path(path)
        self.target_combo.clear()
        self.target_combo.addItems([column for column in dataset.columns])
        if "target" in dataset.columns:
            self.target_combo.setCurrentText("target")
        elif dataset.columns:
            self.target_combo.setCurrentIndex(0)

        validation = self._validate_dataset(dataset, self.target_combo.currentText())
        self.warning_box.setPlainText("\n".join(validation.warnings) if validation.warnings else "No warnings.")
        self.dataset_summary.setText(
            f"Rows: {len(dataset)} | Columns: {len(dataset.columns)} | Target: {self.target_combo.currentText()}"
        )
        if validation.summary.get("missing_columns"):
            self.dataset_summary.setText(
                self.dataset_summary.text() + " | Missing values detected in: "
                + ", ".join(map(str, validation.summary["missing_columns"]))
            )
        self._render_dataframe_preview(dataset)
        self.append_log(f"Validated dataset: {self.dataset_path} ({len(dataset)} rows)")

    def _validate_dataset(self, dataset: pd.DataFrame, target_name: str) -> DatasetValidationResult:
        warnings: list[str] = []
        summary: dict[str, Any] = {
            "missing_columns": [],
            "constant_columns": [],
            "non_numeric_columns": [],
        }

        if dataset.empty:
            warnings.append("The dataset has no rows.")
        if dataset.columns.empty:
            warnings.append("The dataset has no columns.")
        if target_name not in dataset.columns:
            warnings.append(f"Target column '{target_name}' is not present in the dataset.")
            return DatasetValidationResult(dataset, warnings, summary)

        for column in dataset.columns:
            series = dataset[column]
            missing = int(series.isna().sum())
            if missing:
                summary["missing_columns"].append(column)
                warnings.append(f"Column '{column}' has {missing} missing values.")
            if series.nunique(dropna=True) <= 1:
                summary["constant_columns"].append(column)
                warnings.append(f"Column '{column}' is constant and may be uninformative.")

        for column in dataset.columns:
            if column == target_name:
                continue
            coerced = pd.to_numeric(dataset[column], errors="coerce")
            if coerced.isna().any():
                summary["non_numeric_columns"].append(column)
                warnings.append(f"Column '{column}' contains non-numeric values or invalid tokens.")

        return DatasetValidationResult(dataset, warnings, summary)

    def _render_dataframe_preview(self, dataframe: pd.DataFrame) -> None:
        preview = dataframe.head(25).copy()
        model = QStandardItemModel(preview.shape[0], preview.shape[1])
        model.setHorizontalHeaderLabels(list(preview.columns))
        for row_idx in range(preview.shape[0]):
            for col_idx in range(preview.shape[1]):
                value = preview.iloc[row_idx, col_idx]
                item = QStandardItem(str(value))
                model.setItem(row_idx, col_idx, item)
        self.preview_table.setModel(model)
        self.preview_table.resizeColumnsToContents()

    def _run_preprocessing(self) -> None:
        if self.current_dataset is None or self.current_target is None:
            QMessageBox.warning(self, "Dataset not loaded", "Load a CSV dataset before preprocessing.")
            return

        output_dir = self.output_dir / "preprocessing"
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized_path = output_dir / "normalized.csv"
        stats_path = output_dir / "preprocessing_stats.json"
        min_valid = float(self.min_valid_spin.value())

        self._set_busy(True)
        try:
            self.preprocessing_stats = fit_normalize(
                str(self.dataset_path),
                self.current_target,
                str(normalized_path),
                str(stats_path),
                min_valid,
            )
            self.normalized_path = normalized_path
            self.statusBar().showMessage("Preprocessing completed")
            self.append_log(
                f"Preprocessing finished; kept {self.preprocessing_stats.get('n_kept_features')} features."
            )
            dropped = self.preprocessing_stats.get("dropped_feature_names", [])
            self.preprocessing_drops.setPlainText("\n".join(dropped) if dropped else "No features dropped.")
            self.preprocessing_summary.setText(
                f"Rows: {self.preprocessing_stats.get('dataset_size')} | Kept: {self.preprocessing_stats.get('n_kept_features')} | Dropped: {len(dropped)}"
            )
            self._show_info_message("Preprocessing completed successfully.")
        except Exception as exc:  # pragma: no cover - GUI path for user errors
            self.append_log(f"Preprocessing error: {exc}")
            QMessageBox.critical(self, "Preprocessing failed", str(exc))
        finally:
            self._set_busy(False)

    def _run_feature_selection(self) -> None:
        if self.normalized_path is None:
            QMessageBox.warning(self, "Normalization required", "Run preprocessing before selecting features.")
            return
        output_dir = self.output_dir / "feature_selection"
        output_dir.mkdir(parents=True, exist_ok=True)
        train_path = output_dir / "reduced_train.csv"
        test_path = output_dir / "reduced_test.csv"
        optimization_path = output_dir / "optimization.csv"
        stats_path = output_dir / "feature_selection_stats.json"

        self._set_busy(True)
        try:
            self.selection_stats = select_features(
                str(self.normalized_path),
                str(train_path),
                str(test_path),
                str(optimization_path),
                str(stats_path),
                self.current_target or "target",
                percTest=float(self.perc_test_spin.value()),
                percSelected=float(self.perc_selected_spin.value()),
                allowance=int(self.allowance_spin.value()),
                seed=int(self.selection_seed_spin.value()),
            )
            self.reduced_train_path = train_path
            self.reduced_test_path = test_path
            names = self.selection_stats.get("selected_feature_names", [])
            self.selected_features_box.setPlainText("\n".join(names) if names else "No features selected.")
            self.feature_summary.setText(
                f"Selected {self.selection_stats.get('n_selected', 0)} features | Target reached: {self.selection_stats.get('target_reached', False)}"
            )
            self.append_log(
                f"Feature selection finished with {self.selection_stats.get('n_selected', 0)} selected features."
            )
            self._show_info_message("Feature reduction completed successfully.")
        except Exception as exc:  # pragma: no cover - GUI path for user errors
            self.append_log(f"Feature selection error: {exc}")
            QMessageBox.critical(self, "Feature selection failed", str(exc))
        finally:
            self._set_busy(False)

    def _run_training(self) -> None:
        if self.reduced_train_path is None:
            QMessageBox.warning(self, "Reduced training data missing", "Run feature reduction before model training.")
            return

        output_dir = self.output_dir / "model"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.joblib"
        metrics_path = output_dir / "training_metrics.json"
        classifier = self.classifier_combo.currentData()
        seed = int(self.training_seed_spin.value())

        self._set_busy(True)
        try:
            self.training_stats = train(
                str(classifier),
                str(self.reduced_train_path),
                self.current_target or "target",
                str(model_path),
                str(metrics_path),
                seed,
            )
            self.model_path = model_path
            self.training_summary.setText(
                f"Classifier: {self.training_stats.get('classifier')} | Samples: {self.training_stats.get('n_samples')} | Features: {self.training_stats.get('n_features')}"
            )
            self.append_log(f"Training complete for {classifier}.")
            self._show_info_message("Model training succeeded.")
        except Exception as exc:  # pragma: no cover - GUI path for user errors
            self.append_log(f"Training error: {exc}")
            QMessageBox.critical(self, "Training failed", str(exc))
        finally:
            self._set_busy(False)

    def _run_prediction(self) -> None:
        if self.reduced_test_path is None:
            QMessageBox.warning(self, "Reduced test set missing", "Run feature reduction before evaluation.")
            return
        if self.model_path is None:
            self.model_path = self.output_dir / "model" / "model.joblib"
        if not self.model_path.exists():
            QMessageBox.warning(self, "Model not found", "Train a model before running prediction.")
            return

        output_dir = self.output_dir / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "predictions.csv"
        stats_path = output_dir / "classification_stats.json"

        self._set_busy(True)
        try:
            self.prediction_stats = predict(
                str(self.reduced_test_path),
                self.current_target or "target",
                str(self.model_path),
                str(predictions_path),
                str(stats_path),
            )
            self.predictions_path = predictions_path
            self.evaluation_summary.setText(
                f"Accuracy: {self.prediction_stats.get('accuracy')} | ROC-AUC: {self.prediction_stats.get('roc_auc')}"
            )
            self._show_prediction_table(predictions_path)
            self._show_metrics_summary()
            self.append_log("Prediction and evaluation completed.")
            self._show_info_message("Prediction and evaluation completed successfully.")
        except Exception as exc:  # pragma: no cover - GUI path for user errors
            self.append_log(f"Prediction error: {exc}")
            QMessageBox.critical(self, "Prediction failed", str(exc))
        finally:
            self._set_busy(False)

    def _show_prediction_table(self, predictions_path: Path) -> None:
        try:
            table = pd.read_csv(predictions_path)
        except Exception:
            return
        self.prediction_table.setRowCount(len(table))
        self.prediction_table.setColumnCount(len(table.columns))
        self.prediction_table.setHorizontalHeaderLabels(list(table.columns))
        for row_idx, row in table.head(50).iterrows():
            for col_idx, value in enumerate(row):
                self.prediction_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
        self.prediction_table.resizeColumnsToContents()

    def _show_metrics_summary(self) -> None:
        if not self.prediction_stats:
            return
        self.metric_table.setRowCount(0)
        metrics = [
            ("Accuracy", self.prediction_stats.get("accuracy")),
            ("Class 0 precision", self.prediction_stats.get("class_0", {}).get("precision")),
            ("Class 0 recall", self.prediction_stats.get("class_0", {}).get("recall")),
            ("Class 0 F1", self.prediction_stats.get("class_0", {}).get("f1")),
            ("Class 1 precision", self.prediction_stats.get("class_1", {}).get("precision")),
            ("Class 1 recall", self.prediction_stats.get("class_1", {}).get("recall")),
            ("Class 1 F1", self.prediction_stats.get("class_1", {}).get("f1")),
            ("ROC-AUC", self.prediction_stats.get("roc_auc")),
        ]
        self.metric_table.setRowCount(len(metrics))
        for row_index, (name, value) in enumerate(metrics):
            self.metric_table.setItem(row_index, 0, QTableWidgetItem(str(name)))
            self.metric_table.setItem(row_index, 1, QTableWidgetItem("N/A" if value is None else str(value)))

        matrix = self.prediction_stats.get("confusion_matrix", {}).get("matrix", [[0, 0], [0, 0]])
        self.confusion_matrix.setRowCount(2)
        self.confusion_matrix.setColumnCount(2)
        for row_idx in range(2):
            for col_idx in range(2):
                item = QTableWidgetItem(str(matrix[row_idx][col_idx]))
                self.confusion_matrix.setItem(row_idx, col_idx, item)

        summary_text = (
            f"Classifier: {self.prediction_stats.get('classifier')}\n"
            f"Samples: {self.prediction_stats.get('n_samples')}\n"
            f"Positive rate: {self.prediction_stats.get('target_1_percentage')}%\n"
            f"Accuracy: {self.prediction_stats.get('accuracy')}\n"
            f"ROC-AUC: {self.prediction_stats.get('roc_auc')}"
        )
        self.result_summary.setPlainText(summary_text)

    def _export_results(self) -> None:
        destination = QFileDialog.getExistingDirectory(self, "Choose export folder", str(self.project_root))
        if not destination:
            return
        target_dir = Path(destination)
        if self.predictions_path is not None:
            shutil.copy2(self.predictions_path, target_dir / self.predictions_path.name)
        if self.model_path is not None:
            shutil.copy2(self.model_path, target_dir / self.model_path.name)
        if self.normalized_path is not None:
            shutil.copy2(self.normalized_path, target_dir / self.normalized_path.name)
        if self.reduced_train_path is not None:
            shutil.copy2(self.reduced_train_path, target_dir / self.reduced_train_path.name)
        if self.reduced_test_path is not None:
            shutil.copy2(self.reduced_test_path, target_dir / self.reduced_test_path.name)
        self.append_log(f"Exported project artifacts to {target_dir}")
        self._show_info_message(f"Artifacts exported to {target_dir}")

    def _set_busy(self, busy: bool) -> None:
        self.progress_bar.setVisible(busy)
        self.progress_bar.setRange(0, 0 if busy else 100)
        if busy:
            self.progress_bar.setValue(0)
        self.setEnabled(not busy)
        QApplication.processEvents()

    def _show_info_message(self, message: str) -> None:
        QMessageBox.information(self, "Pipeline status", message)

    def append_log(self, message: str) -> None:
        self.logs.append(message)
        if hasattr(self, "log_view"):
            self.log_view.appendPlainText(message)

    def closeEvent(self, event):
        self.append_log("Application closed.")
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = PipelineMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
