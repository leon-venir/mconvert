"""Logic tests for SpaceViewerTab.

Real OpenGL rendering cannot be exercised headlessly (the "offscreen" Qt
platform has no GL context), so these cover everything that does not need a
painted frame: scene construction, mode switching, picking through the real
camera matrices, and the info panel. Same PySide6 skip behaviour as the other
GUI tests; they also need pyqtgraph.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QMimeData, QPoint, QUrl, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from mconvert.space_viewer import RECIPROCAL, REAL, PickableGLView, SpaceViewerTab
    from mconvert import converter
    from mconvert.picking import project_points
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

SI_POSCAR = """Si fcc primitive
5.43
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
Si
2
Direct
0.0 0.0 0.0
0.25 0.25 0.25
"""

GRAPHENE_POSCAR = """graphene
1.0
2.46 0.0 0.0
-1.23 2.1304 0.0
0.0 0.0 20.0
C
2
Direct
0.3333333333 0.6666666667 0.5
0.6666666667 0.3333333333 0.5
"""


@unittest.skipUnless(AVAILABLE, "PySide6 and pyqtgraph are required")
class TestSpaceViewerTab(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tab = SpaceViewerTab()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _load(self, text):
        path = Path(self._tmp.name) / "POSCAR"
        path.write_text(text)
        loaded = converter.load_structure(str(path))
        self.tab.set_structure(loaded)
        return loaded

    def _click_on(self, point):
        view, projection = self.tab.view_matrices()
        xy, _ = project_points([point], view, projection, self.tab.view.width(), self.tab.view.height())
        self.tab._on_view_clicked(*xy[0])

    def test_controls_are_disabled_until_a_structure_is_loaded(self):
        self.assertFalse(self.tab.real_button.isEnabled())
        self.assertFalse(self.tab.reciprocal_button.isEnabled())
        self.assertEqual(self.tab.info_label.text(), "")

    def test_loading_a_structure_enables_both_modes_and_starts_in_real_space(self):
        self._load(SI_POSCAR)
        self.assertTrue(self.tab.real_button.isEnabled())
        self.assertTrue(self.tab.reciprocal_button.isEnabled())
        self.assertTrue(self.tab.real_button.isChecked())
        self.assertEqual(len(self.tab._pick_payload), 2)

    def test_clicking_an_atom_shows_its_information(self):
        self._load(SI_POSCAR)
        self._click_on(self.tab._pick_points[1])
        self.assertEqual(self.tab.selected_index, 1)
        text = self.tab.info_label.text()
        for expected in ("Si", "Silicon", "14", "序号：2"):
            self.assertIn(expected, text)

    def test_clicking_empty_space_clears_the_selection(self):
        self._load(SI_POSCAR)
        self.tab.select(0)
        self.tab._on_view_clicked(2, 2)
        self.assertIsNone(self.tab.selected_index)
        self.assertIn("点击原子", self.tab.info_label.text())

    def test_switching_to_reciprocal_space_lists_the_high_symmetry_points(self):
        self._load(SI_POSCAR)
        self.tab.set_mode(RECIPROCAL)
        labels = [point.label for point in self.tab._pick_payload]
        self.assertIn("Γ", labels)
        self.assertTrue(self.tab.reciprocal_button.isChecked())
        self.assertFalse(self.tab.note_label.isVisibleTo(self.tab))   # primitive cell, 3D: nothing to warn about

    def test_folded_and_2d_notes_appear_only_when_they_apply(self):
        self._load(GRAPHENE_POSCAR)
        self.tab.set_mode(RECIPROCAL)
        self.assertTrue(self.tab.note_label.isVisibleTo(self.tab))
        self.assertIn("二维", self.tab.note_label.text())
        self.tab.set_mode(REAL)
        self.assertFalse(self.tab.note_label.isVisibleTo(self.tab))

    def test_clicking_a_high_symmetry_point_shows_its_reciprocal_coordinates(self):
        self._load(SI_POSCAR)
        self.tab.set_mode(RECIPROCAL)
        index = next(i for i, p in enumerate(self.tab._pick_payload) if p.key == "X")
        self._click_on(self.tab._pick_points[index])
        self.assertEqual(self.tab.selected_index, index)
        point = self.tab._pick_payload[index]
        text = self.tab.info_label.text()
        self.assertIn("X", text)
        self.assertIn("%.4f" % point.frac[0], text)
        self.assertIn("%.4f" % point.cart[0], text)

    def test_switching_mode_clears_the_selection(self):
        self._load(SI_POSCAR)
        self.tab.select(0)
        self.tab.set_mode(RECIPROCAL)
        self.assertIsNone(self.tab.selected_index)

    def test_2d_structure_uses_a_polygon_zone_and_two_axes(self):
        self._load(GRAPHENE_POSCAR)
        self.tab.set_mode(RECIPROCAL)
        self.assertTrue(self.tab._reciprocal_scene.is_2d)
        self.assertEqual(len(self.tab._reciprocal_scene.zone_edges), 6)
        self.assertIn("b1", self.tab.axes_checkbox.text())

    def test_toggling_the_cell_checkbox_rebuilds_the_scene_without_losing_atoms(self):
        self._load(SI_POSCAR)
        items_with_cell = len(self.tab.view.items)
        self.tab.cell_checkbox.setChecked(False)
        self.assertLess(len(self.tab.view.items), items_with_cell)
        self.assertEqual(len(self.tab._pick_payload), 2)

    def test_dropping_a_file_loads_it(self):
        path = Path(self._tmp.name) / "POSCAR"
        path.write_text(SI_POSCAR)

        class _Event:
            def __init__(self):
                self.mime = QMimeData()
                self.mime.setUrls([QUrl.fromLocalFile(str(path))])
            def mimeData(self):
                return self.mime

        self.tab.dropEvent(_Event())
        self.assertEqual(len(self.tab._pick_payload), 2)


@unittest.skipUnless(AVAILABLE, "PySide6 and pyqtgraph are required")
class TestPickableGLView(unittest.TestCase):
    """A click selects; a drag must only rotate the camera."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.view = PickableGLView()
        self.view.resize(200, 200)
        self.clicks = []
        self.view.clicked.connect(lambda x, y: self.clicks.append((x, y)))

    def test_press_and_release_in_place_emits_clicked(self):
        QTest.mouseClick(self.view, Qt.LeftButton, Qt.NoModifier, QPoint(50, 60))
        self.assertEqual(self.clicks, [(50.0, 60.0)])

    def test_dragging_does_not_emit_clicked_but_still_rotates(self):
        azimuth = self.view.opts["azimuth"]
        QTest.mousePress(self.view, Qt.LeftButton, Qt.NoModifier, QPoint(50, 60))
        QTest.mouseMove(self.view, QPoint(90, 60))
        QTest.mouseRelease(self.view, Qt.LeftButton, Qt.NoModifier, QPoint(90, 60))
        self.assertEqual(self.clicks, [])
        self.assertNotEqual(self.view.opts["azimuth"], azimuth)

    def test_right_click_does_not_emit_clicked(self):
        QTest.mouseClick(self.view, Qt.RightButton, Qt.NoModifier, QPoint(50, 60))
        self.assertEqual(self.clicks, [])


if __name__ == "__main__":
    unittest.main()
