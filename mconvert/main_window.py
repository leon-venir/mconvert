"""Main window: single-file and batch structure format conversion tabs."""

import os

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QStatusBar,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from mconvert import converter, formats
from mconvert.batch import discover_input_files, resolve_output_paths
from mconvert.convert_worker import BatchConvertWorker
from mconvert.dnd import single_local_directory, single_local_file
from mconvert.space_viewer import SpaceViewerTab

WINDOW_TITLE = "mconvert - 晶体结构格式转换"


class OutputOptionsWidget(QWidget):
    """Target-format + coordinate-type + selective-dynamics controls.

    Shared by SingleFileTab and BatchTab: both need the same target-format
    combo and the same "grey out frac/dyn when the format doesn't support
    them" behaviour, and duplicating that logic in two places would only
    need to drift out of sync the day a new format is added.
    """

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.target_format_combo = QComboBox()
        for key in formats.OUTPUT_FORMATS:
            self.target_format_combo.addItem(formats.label(key), userData=key)
        self.target_format_combo.currentIndexChanged.connect(self._on_target_format_changed)
        layout.addRow("目标格式：", self.target_format_combo)

        coord_row = QHBoxLayout()
        self.frac_radio = QRadioButton("分数坐标")
        self.cart_radio = QRadioButton("笛卡尔坐标")
        self.frac_radio.setChecked(True)
        self._coord_group = QButtonGroup(self)
        self._coord_group.addButton(self.frac_radio)
        self._coord_group.addButton(self.cart_radio)
        self.frac_radio.toggled.connect(lambda _checked: self.changed.emit())
        coord_row.addWidget(self.frac_radio)
        coord_row.addWidget(self.cart_radio)
        layout.addRow("坐标类型：", coord_row)

        self.dyn_checkbox = QCheckBox("写入 selective dynamics")
        self.dyn_checkbox.toggled.connect(lambda _checked: self.changed.emit())
        layout.addRow("", self.dyn_checkbox)

        self._on_target_format_changed()

    def target_format(self) -> str:
        return self.target_format_combo.currentData()

    def frac(self) -> bool:
        return self.frac_radio.isChecked()

    def dyn(self) -> bool:
        return self.dyn_checkbox.isChecked()

    def _on_target_format_changed(self):
        target = self.target_format()
        frac_ok = formats.supports_frac_cart(target)
        dyn_ok = formats.supports_dyn(target)
        self.frac_radio.setEnabled(frac_ok)
        self.cart_radio.setEnabled(frac_ok)
        self.dyn_checkbox.setEnabled(dyn_ok)
        coord_tooltip = "" if frac_ok else "该格式坐标写法固定，此选项无效"
        self.frac_radio.setToolTip(coord_tooltip)
        self.cart_radio.setToolTip(coord_tooltip)
        self.dyn_checkbox.setToolTip("" if dyn_ok else "该格式不支持 selective dynamics")
        self.changed.emit()


class SingleFileTab(QWidget):
    """Load one structure file, choose a target format/options, preview and save."""

    structure_loaded = Signal(object)   # converter.LoadedStructure

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loaded = None
        self._input_path = None
        self._output_path_is_default = True
        self._build_ui()
        self._update_output_controls_enabled(False)
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        layout = QVBoxLayout(self)

        input_row = QHBoxLayout()
        self.choose_input_button = QPushButton("选择输入文件...")
        self.choose_input_button.clicked.connect(self._on_choose_input_file)
        self.input_path_label = QLabel("未选择文件（也可将文件拖拽到此窗口）")
        self.input_path_label.setWordWrap(True)
        input_row.addWidget(self.choose_input_button)
        input_row.addWidget(self.input_path_label, stretch=1)
        layout.addLayout(input_row)

        info_group = QGroupBox("结构信息")
        info_form = QFormLayout(info_group)
        self.detected_format_label = QLabel("-")
        self.formula_label = QLabel("-")
        self.total_atoms_label = QLabel("-")
        self.lattice_label = QLabel("-")
        info_form.addRow("识别格式：", self.detected_format_label)
        info_form.addRow("化学式：", self.formula_label)
        info_form.addRow("原子总数：", self.total_atoms_label)
        info_form.addRow("晶格参数：", self.lattice_label)
        layout.addWidget(info_group)

        output_group = QGroupBox("输出设置")
        output_group_layout = QVBoxLayout(output_group)

        self.output_options = OutputOptionsWidget()
        self.output_options.changed.connect(self._on_options_changed)
        output_group_layout.addWidget(self.output_options)

        output_path_row = QHBoxLayout()
        output_path_row.addWidget(QLabel("输出文件："))
        self.output_path_edit = QLineEdit()
        self.output_path_edit.textEdited.connect(self._on_output_path_edited)
        self.choose_output_button = QPushButton("另存为...")
        self.choose_output_button.clicked.connect(self._on_choose_output_path)
        output_path_row.addWidget(self.output_path_edit, stretch=1)
        output_path_row.addWidget(self.choose_output_button)
        output_group_layout.addLayout(output_path_row)

        layout.addWidget(output_group)

        self.preview_button = QPushButton("预览")
        self.preview_button.clicked.connect(self._on_preview)
        layout.addWidget(self.preview_button)

        self.preview_text = QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText("点击“预览”查看转换结果")
        layout.addWidget(self.preview_text, stretch=1)

        self.convert_button = QPushButton("转换并保存")
        self.convert_button.clicked.connect(self._on_convert)
        layout.addWidget(self.convert_button)

    # ------------------------------------------------------------------ #
    # input file handling
    # ------------------------------------------------------------------ #
    def _on_choose_input_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择结构文件", "", formats.all_input_dialog_filter(),
        )
        if path:
            self._load(path)

    def dragEnterEvent(self, event):
        if single_local_file(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = single_local_file(event.mimeData())
        if path is not None:
            self._load(path)

    def _load(self, path):
        try:
            loaded = converter.load_structure(path)
        except converter.ConversionError as err:
            QMessageBox.warning(self, "无法解析文件", str(err))
            self._loaded = None
            self._update_output_controls_enabled(False)
            return

        self._loaded = loaded
        self._input_path = path
        self.input_path_label.setText(path)

        summary = loaded.summary
        self.detected_format_label.setText(formats.label(summary.detected_format))
        self.formula_label.setText(summary.chemical_formula)
        self.total_atoms_label.setText(str(summary.total_atoms))
        a, b, c = summary.lattice_lengths
        alpha, beta, gamma = summary.lattice_angles
        self.lattice_label.setText(
            "a=%.4f b=%.4f c=%.4f Å  α=%.2f β=%.2f γ=%.2f°" % (a, b, c, alpha, beta, gamma)
        )

        self.preview_text.clear()
        self._output_path_is_default = True
        self._update_output_controls_enabled(True)
        self.structure_loaded.emit(loaded)

    # ------------------------------------------------------------------ #
    # output settings
    # ------------------------------------------------------------------ #
    def _update_output_controls_enabled(self, enabled):
        for widget in (
            self.output_options, self.output_path_edit, self.choose_output_button,
            self.preview_button, self.convert_button,
        ):
            widget.setEnabled(enabled)
        if enabled and self._output_path_is_default:
            self._refresh_output_defaults()

    def _on_options_changed(self):
        if self._output_path_is_default:
            self._refresh_output_defaults()

    def _refresh_output_defaults(self):
        if self._loaded is None:
            return
        directory = os.path.dirname(self._input_path) or "."
        self.output_path_edit.setText(
            os.path.join(directory, formats.default_filename(self.output_options.target_format()))
        )
        self._output_path_is_default = True

    def _on_output_path_edited(self, _text):
        self._output_path_is_default = False

    def _on_choose_output_path(self):
        target = self.output_options.target_format()
        start_path = self.output_path_edit.text() or formats.default_filename(target)
        path, _ = QFileDialog.getSaveFileName(
            self, "另存为", start_path, formats.dialog_filter(target),
        )
        if path:
            self.output_path_edit.setText(path)
            self._output_path_is_default = False

    # ------------------------------------------------------------------ #
    # preview / convert
    # ------------------------------------------------------------------ #
    def _on_preview(self):
        try:
            text = converter.preview(
                self._loaded, self.output_options.target_format(),
                frac=self.output_options.frac(), dyn=self.output_options.dyn(),
            )
        except converter.ConversionError as err:
            QMessageBox.warning(self, "预览失败", str(err))
            return
        self.preview_text.setPlainText(text)

    def _on_convert(self):
        output_path = self.output_path_edit.text().strip()
        if not output_path:
            QMessageBox.warning(self, "未指定输出文件", "请先选择输出文件路径。")
            return

        if os.path.exists(output_path):
            reply = QMessageBox.question(
                self, "文件已存在", "文件 %s 已存在，是否覆盖？" % output_path,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        try:
            converter.convert_to_file(
                self._loaded, self.output_options.target_format(), output_path,
                frac=self.output_options.frac(), dyn=self.output_options.dyn(),
            )
        except converter.ConversionError as err:
            QMessageBox.warning(self, "转换失败", str(err))
            return
        QMessageBox.information(self, "转换成功", "已保存到 %s" % output_path)


class BatchTab(QWidget):
    """Scan a folder for structure files and convert all of them in the
    background via BatchConvertWorker."""

    COLUMN_FILENAME, COLUMN_FORMAT, COLUMN_STATUS = range(3)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_root = None
        self._discovered_files = []
        self._output_dir_is_default = True
        self._worker = None
        self._build_ui()
        self._update_controls_enabled(False)
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        layout = QVBoxLayout(self)

        input_row = QHBoxLayout()
        self.choose_folder_button = QPushButton("选择输入文件夹...")
        self.choose_folder_button.clicked.connect(self._on_choose_folder)
        self.folder_label = QLabel("未选择文件夹（也可将文件夹拖拽到此窗口）")
        self.folder_label.setWordWrap(True)
        self.recursive_checkbox = QCheckBox("包含子文件夹")
        self.recursive_checkbox.setChecked(True)
        self.recursive_checkbox.stateChanged.connect(self._on_recursive_changed)
        input_row.addWidget(self.choose_folder_button)
        input_row.addWidget(self.folder_label, stretch=1)
        input_row.addWidget(self.recursive_checkbox)
        layout.addLayout(input_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["文件名", "识别格式", "转换状态"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table, stretch=1)

        output_group = QGroupBox("输出设置")
        output_group_layout = QVBoxLayout(output_group)

        self.output_options = OutputOptionsWidget()
        output_group_layout.addWidget(self.output_options)

        output_dir_row = QHBoxLayout()
        output_dir_row.addWidget(QLabel("输出目录："))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.textEdited.connect(self._on_output_dir_edited)
        self.choose_output_dir_button = QPushButton("浏览...")
        self.choose_output_dir_button.clicked.connect(self._on_choose_output_dir)
        output_dir_row.addWidget(self.output_dir_edit, stretch=1)
        output_dir_row.addWidget(self.choose_output_dir_button)
        output_group_layout.addLayout(output_dir_row)

        layout.addWidget(output_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)  # QProgressBar's own default is -1 ("no value"), which some styles render as a blank bar with no "0%" text
        layout.addWidget(self.progress_bar)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("开始批量转换")
        self.start_button.clicked.connect(self._on_start)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self._on_cancel)
        self.cancel_button.setEnabled(False)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

    # ------------------------------------------------------------------ #
    # folder selection / scanning
    # ------------------------------------------------------------------ #
    def _on_choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if path:
            self._set_input_root(path)

    def dragEnterEvent(self, event):
        if single_local_directory(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = single_local_directory(event.mimeData())
        if path is not None:
            self._set_input_root(path)

    def _set_input_root(self, path):
        self._input_root = path
        self.folder_label.setText(path)
        self._output_dir_is_default = True
        self._refresh_output_dir_default()
        self._scan()

    def _on_recursive_changed(self, _state):
        if self._input_root:
            self._scan()

    def _scan(self):
        self._discovered_files = discover_input_files(
            self._input_root, recursive=self.recursive_checkbox.isChecked(),
        )
        self.table.setRowCount(len(self._discovered_files))
        for row, path in enumerate(self._discovered_files):
            relative = os.path.relpath(path, self._input_root)
            self.table.setItem(row, self.COLUMN_FILENAME, QTableWidgetItem(relative))
            self.table.setItem(row, self.COLUMN_FORMAT, QTableWidgetItem("-"))
            self.table.setItem(row, self.COLUMN_STATUS, QTableWidgetItem("待处理"))
        self._update_controls_enabled(bool(self._discovered_files))

    # ------------------------------------------------------------------ #
    # output settings
    # ------------------------------------------------------------------ #
    def _update_controls_enabled(self, enabled):
        for widget in (
            self.output_options, self.output_dir_edit,
            self.choose_output_dir_button, self.start_button,
        ):
            widget.setEnabled(enabled)

    def _refresh_output_dir_default(self):
        if self._input_root is None:
            return
        self.output_dir_edit.setText(os.path.join(self._input_root, "converted"))
        self._output_dir_is_default = True

    def _on_output_dir_edited(self, _text):
        self._output_dir_is_default = False

    def _on_choose_output_dir(self):
        start_dir = self.output_dir_edit.text() or self._input_root or ""
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", start_dir)
        if path:
            self.output_dir_edit.setText(path)
            self._output_dir_is_default = False

    # ------------------------------------------------------------------ #
    # running the batch
    # ------------------------------------------------------------------ #
    def _on_start(self):
        if not self._discovered_files:
            QMessageBox.warning(self, "没有可转换的文件", "所选文件夹中没有找到任何文件。")
            return

        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "未指定输出目录", "请先选择输出目录。")
            return

        if os.path.isdir(output_dir) and os.listdir(output_dir):
            reply = QMessageBox.question(
                self, "输出目录非空",
                "输出目录 %s 中已有文件，同名文件会被覆盖，是否继续？" % output_dir,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        jobs = resolve_output_paths(
            self._discovered_files, self._input_root, output_dir, self.output_options.target_format(),
        )

        self._next_row = 0
        self.progress_bar.setRange(0, len(jobs))
        self.progress_bar.setValue(0)
        for row in range(self.table.rowCount()):
            self.table.item(row, self.COLUMN_STATUS).setText("待处理")

        self._set_running(True)
        self._worker = BatchConvertWorker(
            jobs, self.output_options.target_format(),
            frac=self.output_options.frac(), dyn=self.output_options.dyn(),
        )
        self._worker.item_finished.connect(self._on_item_finished)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()

    def _set_running(self, running):
        self.choose_folder_button.setEnabled(not running)
        self.recursive_checkbox.setEnabled(not running)
        self.output_options.setEnabled(not running)
        self.output_dir_edit.setEnabled(not running)
        self.choose_output_dir_button.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _on_item_finished(self, result):
        row = self._next_row
        self._next_row += 1

        format_item = self.table.item(row, self.COLUMN_FORMAT)
        format_item.setText(formats.label(result.detected_format) if result.detected_format else "-")

        status_item = self.table.item(row, self.COLUMN_STATUS)
        if result.ok:
            status_item.setText("成功")
            status_item.setForeground(QColor("green"))
        else:
            status_item.setText(result.message)
            status_item.setToolTip(result.traceback or result.message)
            status_item.setForeground(QColor("red"))

    def _on_progress(self, completed, total):
        self.progress_bar.setValue(completed)

    def _on_worker_finished(self):
        results_ok = sum(
            1 for row in range(self.table.rowCount())
            if self.table.item(row, self.COLUMN_STATUS).text() == "成功"
        )
        total = self._next_row
        self._worker = None
        self._set_running(False)
        QMessageBox.information(
            self, "批量转换完成", "共处理 %d 个文件，成功 %d 个，失败 %d 个。" % (total, results_ok, total - results_ok),
        )


class MainWindow(QMainWindow):
    """Application main window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(960, 680)

        self.single_file_tab = SingleFileTab()
        self.space_viewer_tab = SpaceViewerTab()
        self.single_file_tab.structure_loaded.connect(self.space_viewer_tab.set_structure)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.single_file_tab, "单文件转换")
        self.tabs.addTab(BatchTab(), "批量转换")
        self.tabs.addTab(self.space_viewer_tab, "实/倒空间")
        self.setCentralWidget(self.tabs)

        self.setStatusBar(QStatusBar())
