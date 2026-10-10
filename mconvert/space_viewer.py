"""Real/reciprocal-space 3D viewer tab (PyQtGraph OpenGL).

Left/centre: a rotatable, zoomable 3D view. Right: real/reciprocal switch,
display toggles and an info panel for the clicked atom or high-symmetry point.
Chemical bonds are intentionally not drawn.
"""

import colorsys
import math

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from mconvert import converter, formats
from mconvert.dnd import single_local_file
from mconvert.picking import pick_nearest

REAL, RECIPROCAL = "real", "reciprocal"

_BACKGROUND = (255, 255, 255)
_CELL_COLOR = (0.25, 0.25, 0.25, 1.0)
_ZONE_COLOR = (0.15, 0.15, 0.15, 1.0)
_PATH_COLOR = (0.85, 0.45, 0.0, 1.0)
_KPOINT_COLOR = (0.85, 0.1, 0.55, 1.0)
_HIGHLIGHT_COLOR = (1.0, 0.85, 0.0, 0.55)
_AXIS_COLORS = ((0.85, 0.1, 0.1, 1.0), (0.1, 0.6, 0.1, 1.0), (0.1, 0.2, 0.85, 1.0))

_KPOINT_PIXEL_SIZE = 14
_FOLDED_NOTE = "所选结构不是原胞：高对称点按原胞定义，因此落在所示（较小）布里渊区之外，对应能带折叠。"
_PATH_2D_NOTE = "二维材料的高对称路径目前是近似表（仅区分 4 种二维布拉维格子）。"

# CPK-style colours for common elements; anything else gets a stable colour
# derived from its atomic number.
_ELEMENT_COLORS = {
    "H": (1.0, 0.8, 0.8), "Li": (0.8, 0.5, 1.0), "B": (1.0, 0.7, 0.7),
    "C": (0.35, 0.35, 0.35), "N": (0.19, 0.31, 0.97), "O": (0.9, 0.1, 0.1),
    "F": (0.56, 0.88, 0.31), "Na": (0.67, 0.36, 0.95), "Mg": (0.54, 0.85, 0.0),
    "Al": (0.75, 0.65, 0.65), "Si": (0.94, 0.78, 0.63), "P": (1.0, 0.5, 0.0),
    "S": (0.9, 0.8, 0.1), "Cl": (0.12, 0.94, 0.12), "K": (0.56, 0.25, 0.83),
    "Ca": (0.24, 1.0, 0.0), "Ti": (0.55, 0.56, 0.6), "Fe": (0.88, 0.4, 0.2),
    "Co": (0.94, 0.56, 0.63), "Ni": (0.31, 0.82, 0.31), "Cu": (0.78, 0.5, 0.2),
    "Zn": (0.49, 0.5, 0.69), "Ga": (0.76, 0.56, 0.56), "Ge": (0.4, 0.56, 0.56),
    "As": (0.74, 0.5, 0.89), "Se": (1.0, 0.63, 0.0), "Br": (0.65, 0.16, 0.16),
    "Ag": (0.75, 0.75, 0.75), "Pt": (0.82, 0.82, 0.88), "Au": (1.0, 0.82, 0.14),
}


def _atom_color(symbol, atomic_number):
    rgb = _ELEMENT_COLORS.get(symbol)
    if rgb is None:
        rgb = colorsys.hsv_to_rgb((atomic_number * 0.137) % 1.0, 0.55, 0.85)
    return (*rgb, 1.0)


def _atom_diameter(atomic_mass):
    """Display-only size heuristic: the engine's element table carries no
    atomic radii, and mass**(1/3) roughly tracks relative atomic size."""
    mass = 12.0 if math.isnan(atomic_mass) else atomic_mass
    return 0.42 * mass ** (1.0 / 3.0)


def _qmatrix_to_numpy(matrix):
    """QMatrix4x4 exposes its data column-major."""
    return np.array(matrix.data(), dtype=float).reshape(4, 4).T


def _format_vector(values, digits=4):
    return "(" + ", ".join("%.*f" % (digits, v) for v in values) + ")"


class PickableGLView(gl.GLViewWidget):
    """GLViewWidget that also reports clicks (press and release without
    dragging) so a drag still rotates the view instead of selecting."""

    clicked = Signal(float, float)

    CLICK_TOLERANCE_PX = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._press_position = None

    def mousePressEvent(self, event):
        self._press_position = event.position()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() != Qt.LeftButton or self._press_position is None:
            return
        moved = event.position() - self._press_position
        if abs(moved.x()) <= self.CLICK_TOLERANCE_PX and abs(moved.y()) <= self.CLICK_TOLERANCE_PX:
            self.clicked.emit(event.position().x(), event.position().y())
        self._press_position = None


class SpaceViewerTab(QWidget):
    """Show a loaded structure in real space or in reciprocal space."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loaded = None
        self._real_scene = None
        self._reciprocal_scene = None
        self._mode = REAL
        self._highlight = None
        self.selected_index = None
        self._pick_points = np.zeros((0, 3))
        self._pick_radii = np.zeros(0)
        self._pick_payload = []
        self._build_ui()
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        layout = QHBoxLayout(self)

        self.view = PickableGLView()
        self.view.setBackgroundColor(_BACKGROUND)
        self.view.clicked.connect(self._on_view_clicked)
        layout.addWidget(self.view, stretch=1)

        panel = QVBoxLayout()
        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(280)
        layout.addWidget(panel_widget)

        self.choose_button = QPushButton("选择结构文件...")
        self.choose_button.clicked.connect(self._on_choose_file)
        panel.addWidget(self.choose_button)
        self.structure_label = QLabel("未加载结构（也可拖拽文件，或先在“单文件转换”中加载）")
        self.structure_label.setWordWrap(True)
        panel.addWidget(self.structure_label)

        mode_row = QHBoxLayout()
        self.real_button = QPushButton("实空间")
        self.reciprocal_button = QPushButton("倒空间")
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for button in (self.real_button, self.reciprocal_button):
            button.setCheckable(True)
            self._mode_group.addButton(button)
            mode_row.addWidget(button)
        self.real_button.setChecked(True)
        self.real_button.clicked.connect(lambda: self.set_mode(REAL))
        self.reciprocal_button.clicked.connect(lambda: self.set_mode(RECIPROCAL))
        panel.addLayout(mode_row)

        options = QGroupBox("显示")
        options_layout = QVBoxLayout(options)
        self.cell_checkbox = QCheckBox("晶胞")
        self.axes_checkbox = QCheckBox("坐标轴")
        self.path_checkbox = QCheckBox("k 路径")
        for checkbox in (self.cell_checkbox, self.axes_checkbox, self.path_checkbox):
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._rebuild_scene)
            options_layout.addWidget(checkbox)
        panel.addWidget(options)

        self.reset_button = QPushButton("重置视角")
        self.reset_button.clicked.connect(self._reset_camera)
        panel.addWidget(self.reset_button)

        info = QGroupBox("选中信息")
        info_layout = QVBoxLayout(info)
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_layout.addWidget(self.info_label)
        panel.addWidget(info, stretch=1)

        self.note_label = QLabel("")
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet("color: gray;")
        panel.addWidget(self.note_label)

        self._update_controls()

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    def _on_choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择结构文件", "", formats.all_input_dialog_filter(),
        )
        if path:
            self._load_path(path)

    def dragEnterEvent(self, event):
        if single_local_file(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = single_local_file(event.mimeData())
        if path is not None:
            self._load_path(path)

    def _load_path(self, path):
        try:
            loaded = converter.load_structure(path)
        except converter.ConversionError as err:
            QMessageBox.warning(self, "无法解析文件", str(err))
            return
        self.set_structure(loaded)

    def set_structure(self, loaded):
        """Show a structure that was already loaded via converter.load_structure()."""
        self._loaded = loaded
        self._real_scene = converter.get_real_space_scene(loaded)
        try:
            self._reciprocal_scene = converter.get_reciprocal_scene(loaded)
        except converter.ConversionError as err:
            self._reciprocal_scene = None
            QMessageBox.warning(self, "无法生成倒空间", str(err))

        summary = loaded.summary
        self.structure_label.setText("%s（%d 个原子）\n%s" % (
            summary.chemical_formula, summary.total_atoms, loaded.path))
        if self._reciprocal_scene is None and self._mode == RECIPROCAL:
            self._mode = REAL
            self.real_button.setChecked(True)
        self._update_controls()
        self._rebuild_scene()
        self._reset_camera()

    # ------------------------------------------------------------------ #
    # mode / controls
    # ------------------------------------------------------------------ #
    def set_mode(self, mode):
        if mode == RECIPROCAL and self._reciprocal_scene is None:
            return
        self._mode = mode
        (self.real_button if mode == REAL else self.reciprocal_button).setChecked(True)
        self._update_controls()
        self._rebuild_scene()
        self._reset_camera()

    def _update_controls(self):
        has_structure = self._real_scene is not None
        self.real_button.setEnabled(has_structure)
        self.reciprocal_button.setEnabled(has_structure and self._reciprocal_scene is not None)
        self.reset_button.setEnabled(has_structure)
        self.cell_checkbox.setText("晶胞" if self._mode == REAL else "布里渊区")
        self.axes_checkbox.setText("晶格矢量 a/b/c" if self._mode == REAL else "倒格矢 b1/b2/b3")
        self.path_checkbox.setVisible(self._mode == RECIPROCAL)
        self._update_note()
        self._clear_selection()

    def _update_note(self):
        notes = []
        if self._mode == RECIPROCAL and self._reciprocal_scene is not None:
            if self._reciprocal_scene.points_outside_zone:
                notes.append(_FOLDED_NOTE)
            if self._reciprocal_scene.is_2d:
                notes.append(_PATH_2D_NOTE)
        self.note_label.setText("\n".join(notes))
        self.note_label.setVisible(bool(notes))

    # ------------------------------------------------------------------ #
    # scene building
    # ------------------------------------------------------------------ #
    def _rebuild_scene(self, *_):
        self.view.clear()
        self._highlight = None
        self._pick_points = np.zeros((0, 3))
        self._pick_radii = np.zeros(0)
        self._pick_payload = []
        if self._real_scene is None:
            return
        if self._mode == REAL:
            self._build_real_scene()
        else:
            self._build_reciprocal_scene()
        self._clear_selection()

    def _add_segments(self, segments, color, width=1.5):
        points = np.asarray(segments, dtype=float).reshape(-1, 3)
        self.view.addItem(gl.GLLinePlotItem(
            pos=points, color=color, width=width, mode="lines", antialias=True,
            glOptions="translucent",
        ))

    def _add_label(self, position, text, color):
        # "translucent", not the item default "additive": additive blending
        # adds the text colour onto the background, so dark text on this
        # white background would come out invisible.
        self.view.addItem(gl.GLTextItem(
            pos=position, text=text, color=color, font=QFont("Helvetica", 12, QFont.Bold),
            glOptions="translucent",
        ))

    def _add_axes(self, vectors, names):
        for vector, name, color in zip(vectors, names, _AXIS_COLORS):
            self._add_segments([[0, 0, 0], vector], color, width=3.0)
            self._add_label(np.asarray(vector) * 1.05, name, pg.mkColor([int(c * 255) for c in color]))

    def _build_real_scene(self):
        scene = self._real_scene
        if self.cell_checkbox.isChecked():
            self._add_segments(scene.cell_edges, _CELL_COLOR)
        if self.axes_checkbox.isChecked():
            self._add_axes(scene.lattice, ("a", "b", "c"))

        if scene.atoms:
            positions = np.array([atom.cart for atom in scene.atoms])
            diameters = np.array([_atom_diameter(atom.atomic_mass) for atom in scene.atoms])
            colors = np.array([_atom_color(atom.symbol, atom.atomic_number) for atom in scene.atoms])
            self.view.addItem(gl.GLScatterPlotItem(
                pos=positions, size=diameters, color=colors, pxMode=False, glOptions="opaque",
            ))
            self._pick_points = positions
            self._pick_radii = diameters / 2.0
            self._pick_payload = list(scene.atoms)

    def _build_reciprocal_scene(self):
        scene = self._reciprocal_scene
        if self.cell_checkbox.isChecked():
            self._add_segments(scene.zone_edges, _ZONE_COLOR)
        if self.axes_checkbox.isChecked():
            self._add_axes(scene.rec_vectors[:2] if scene.is_2d else scene.rec_vectors,
                           ("b1", "b2") if scene.is_2d else ("b1", "b2", "b3"))

        by_key = {point.key: point for point in scene.kpoints}
        if self.path_checkbox.isChecked():
            segments = []
            for path in scene.kpath_segments:
                for start, end in zip(path[:-1], path[1:]):
                    segments.append([by_key[start].cart, by_key[end].cart])
            if segments:
                self._add_segments(segments, _PATH_COLOR, width=2.0)

        if scene.kpoints:
            positions = np.array([point.cart for point in scene.kpoints])
            self.view.addItem(gl.GLScatterPlotItem(
                pos=positions, size=_KPOINT_PIXEL_SIZE, color=_KPOINT_COLOR, pxMode=True,
                glOptions="opaque",
            ))
            for point in scene.kpoints:
                self._add_label(np.asarray(point.cart) + self._scene_extent() * 0.02,
                                point.label, pg.mkColor(30, 30, 30))
            self._pick_points = positions
            self._pick_radii = np.zeros(len(positions))
            self._pick_payload = list(scene.kpoints)

    # ------------------------------------------------------------------ #
    # camera
    # ------------------------------------------------------------------ #
    def _scene_bounds(self):
        if self._mode == REAL:
            scene = self._real_scene
            points = [scene.cell_edges.reshape(-1, 3)]
            if scene.atoms:
                points.append(np.array([atom.cart for atom in scene.atoms]))
        else:
            points = [self._reciprocal_scene.zone_edges.reshape(-1, 3)]
        points = np.vstack(points)
        return points.min(axis=0), points.max(axis=0)

    def _scene_extent(self):
        low, high = self._scene_bounds()
        return float(np.linalg.norm(high - low))

    def _reset_camera(self):
        if self._real_scene is None:
            return
        low, high = self._scene_bounds()
        center = (low + high) / 2.0
        extent = float(np.linalg.norm(high - low))
        self.view.opts["center"] = pg.Vector(*center)
        self.view.setCameraPosition(distance=max(extent, 1e-3) * 1.4, elevation=22, azimuth=-60)

    # ------------------------------------------------------------------ #
    # picking / info panel
    # ------------------------------------------------------------------ #
    def _on_view_clicked(self, x, y):
        if not self._pick_payload:
            return
        view_matrix, projection = self.view_matrices()
        index = pick_nearest(
            self._pick_points, self._pick_radii, view_matrix, projection,
            (x, y), (self.view.width(), self.view.height()),
        )
        if index is None:
            self._clear_selection()
        else:
            self.select(index)

    def view_matrices(self):
        """The view and projection matrices of the current camera, as 4x4 numpy arrays."""
        viewport = self.view.getViewport()
        return (_qmatrix_to_numpy(self.view.viewMatrix()),
                _qmatrix_to_numpy(self.view.projectionMatrix(viewport, viewport)))

    def select(self, index):
        """Select the atom (real space) or high-symmetry point (reciprocal) at index."""
        item = self._pick_payload[index]
        if self._mode == REAL:
            self.info_label.setText(
                "元素：%s（%s）\n序号：%d\n原子序数：%d\n原子质量：%s\n分数坐标：%s\n笛卡尔坐标 (Å)：%s" % (
                    item.symbol, item.name, item.index, item.atomic_number,
                    "未知" if math.isnan(item.atomic_mass) else "%.4f amu" % item.atomic_mass,
                    _format_vector(item.frac), _format_vector(item.cart),
                ))
            size, color = _atom_diameter(item.atomic_mass) * 1.6, _HIGHLIGHT_COLOR
            pixel_mode = False
        else:
            self.info_label.setText(
                "高对称点：%s\n分数坐标 (b1, b2, b3)：%s\n笛卡尔坐标 (1/Å)：%s" % (
                    item.label, _format_vector(item.frac), _format_vector(item.cart),
                ))
            size, color = _KPOINT_PIXEL_SIZE * 1.9, _HIGHLIGHT_COLOR
            pixel_mode = True

        self._remove_highlight()
        self._highlight = gl.GLScatterPlotItem(
            pos=np.array([self._pick_points[index]]), size=size, color=color,
            pxMode=pixel_mode, glOptions="translucent",
        )
        self.view.addItem(self._highlight)
        self.selected_index = index

    def _remove_highlight(self):
        if self._highlight is not None:
            self.view.removeItem(self._highlight)
            self._highlight = None

    def _clear_selection(self):
        self._remove_highlight()
        self.selected_index = None
        if self._real_scene is None:
            self.info_label.setText("")
        elif self._mode == REAL:
            self.info_label.setText("点击原子查看信息")
        else:
            self.info_label.setText("点击高对称点查看倒空间坐标")
