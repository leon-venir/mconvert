"""Tests for SpaceViewerTab and the two 3D viewports.

PainterViewport3D renders headlessly, so its tests run everywhere PySide6
does. GLViewport3D needs a real OpenGL >= 2.1 context: its tests skip under
the default 'offscreen' platform and run with e.g. QT_QPA_PLATFORM=xcb (or
wayland) on a machine with a GPU or Mesa. Same PySide6 skip behaviour as the
other GUI tests.
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
    from mconvert.space_viewer import RECIPROCAL, REAL, SpaceViewerTab
    from mconvert import viewport3d
    from mconvert.viewport3d import GLViewport3D, PainterViewport3D, Scene
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


@unittest.skipUnless(AVAILABLE, "PySide6 is required")
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

    def test_note_says_when_the_symmetry_analysis_was_skipped_for_a_large_structure(self):
        saved = converter.SYMMETRY_ATOM_LIMIT
        converter.SYMMETRY_ATOM_LIMIT = 1
        try:
            self._load(SI_POSCAR)
        finally:
            converter.SYMMETRY_ATOM_LIMIT = saved
        self.tab.set_mode(RECIPROCAL)
        self.assertTrue(self.tab.note_label.isVisibleTo(self.tab))
        self.assertIn("未做对称性分析", self.tab.note_label.text())

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
        segment_groups_with_cell = len(self.tab.view.scene.segments)
        self.tab.cell_checkbox.setChecked(False)
        self.assertLess(len(self.tab.view.scene.segments), segment_groups_with_cell)
        self.assertEqual(len(self.tab._pick_payload), 2)
        self.assertEqual(len(self.tab.view.scene.spheres), 1)

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


class ViewportTests:
    """Mouse handling and real rendered pixels, for any viewport class."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_view(self):
        raise NotImplementedError

    def setUp(self):
        self.view = self.make_view()
        self.view.resize(300, 240)
        self.clicks = []
        self.view.clicked.connect(lambda x, y: self.clicks.append((x, y)))

    def test_press_and_release_in_place_emits_clicked(self):
        QTest.mouseClick(self.view, Qt.LeftButton, Qt.NoModifier, QPoint(50, 60))
        self.assertEqual(self.clicks, [(50.0, 60.0)])

    def test_left_drag_rotates_the_camera_without_clicking(self):
        azimuth = self.view.camera.azimuth
        QTest.mousePress(self.view, Qt.LeftButton, Qt.NoModifier, QPoint(50, 60))
        QTest.mouseMove(self.view, QPoint(90, 60))
        QTest.mouseRelease(self.view, Qt.LeftButton, Qt.NoModifier, QPoint(90, 60))
        self.assertEqual(self.clicks, [])
        self.assertNotEqual(self.view.camera.azimuth, azimuth)

    def test_right_drag_pans_the_camera_without_clicking(self):
        centre = self.view.camera.center.copy()
        QTest.mousePress(self.view, Qt.RightButton, Qt.NoModifier, QPoint(50, 60))
        QTest.mouseMove(self.view, QPoint(90, 80))
        QTest.mouseRelease(self.view, Qt.RightButton, Qt.NoModifier, QPoint(90, 80))
        self.assertEqual(self.clicks, [])
        self.assertFalse(np.allclose(self.view.camera.center, centre))

    def test_right_click_does_not_emit_clicked(self):
        QTest.mouseClick(self.view, Qt.RightButton, Qt.NoModifier, QPoint(50, 60))
        self.assertEqual(self.clicks, [])

    def test_wheel_zooms(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QWheelEvent
        distance = self.view.camera.distance
        event = QWheelEvent(QPointF(50, 50), QPointF(50, 50), QPoint(0, 0), QPoint(0, 120),
                            Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        self.view.wheelEvent(event)
        self.assertLess(self.view.camera.distance, distance)

    def _render(self):
        raise NotImplementedError

    def test_empty_viewport_renders_plain_white_with_a_hint(self):
        image = self._render()
        self.assertEqual(image.pixelColor(2, 2).getRgb()[:3], (255, 255, 255))

    def test_a_sphere_is_drawn_where_the_picking_math_says_it_is(self):
        scene = Scene()
        scene.add_spheres([[0, 0, 0], [3, 0, 0]], [0.8, 0.8], [[0.9, 0.1, 0.1, 1], [0.1, 0.2, 0.9, 1]])
        self.view.set_scene(scene)
        self.view.fit_camera([-1, -1, -1], [4, 1, 1])
        view, projection = self.view.view_matrices()
        xy, _ = project_points([[0, 0, 0], [3, 0, 0]], view, projection, self.view.width(), self.view.height())
        image = self._render()
        red = image.pixelColor(int(xy[0][0]), int(xy[0][1]))
        blue = image.pixelColor(int(xy[1][0]), int(xy[1][1]))
        self.assertGreater(red.red(), 150)
        self.assertLess(red.blue(), 100)
        self.assertGreater(blue.blue(), 150)
        self.assertLess(blue.red(), 100)

    def test_a_nearer_sphere_hides_a_farther_one(self):
        scene = Scene()
        # same screen position seen from the default camera direction: put them along the view axis
        self.view.camera.azimuth, self.view.camera.elevation = 0.0, 0.0
        scene.add_spheres([[0, 0, 0], [-4, 0, 0]], [1.0, 1.0], [[0.1, 0.8, 0.1, 1], [0.9, 0.1, 0.1, 1]])
        self.view.set_scene(scene)
        self.view.camera.center = np.array([0.0, 0.0, 0.0])
        self.view.camera.distance = 20.0
        self.view.update()
        view, projection = self.view.view_matrices()
        # camera at +x looking toward -x: the sphere at x=0 is nearer than the one at x=-4
        xy, _ = project_points([[0, 0, 0]], view, projection, self.view.width(), self.view.height())
        pixel = self._render().pixelColor(int(xy[0][0]), int(xy[0][1]))
        self.assertGreater(pixel.green(), pixel.red())

    def test_lines_markers_and_labels_all_paint_something(self):
        scene = Scene()
        scene.add_segments([[[-2, 0, 0], [2, 0, 0]]], (0, 0, 0, 1), 3.0)
        scene.add_markers([[0, 2, 0]], 16, (0.85, 0.1, 0.55, 1))
        scene.add_label([0, -2, 0], "X", (0, 0, 0, 1))
        self.view.set_scene(scene)
        self.view.fit_camera([-2, -2, -1], [2, 2, 1])
        image = self._render()
        dark = sum(1 for x in range(0, image.width(), 2) for y in range(0, image.height(), 2)
                   if image.pixelColor(x, y).lightness() < 200)
        self.assertGreater(dark, 40)


@unittest.skipUnless(AVAILABLE, "PySide6 is required")
class TestPainterViewport(ViewportTests, unittest.TestCase):

    def make_view(self):
        return PainterViewport3D()

    def _render(self):
        return self.view.grab().toImage()


def _opengl_available():
    if not AVAILABLE:
        return False
    app = QApplication.instance() or QApplication([])
    return viewport3d._opengl_is_usable()


@unittest.skipUnless(_opengl_available(), "needs an OpenGL >= 2.1 context (try QT_QPA_PLATFORM=xcb)")
class TestGLViewport(ViewportTests, unittest.TestCase):

    def make_view(self):
        return GLViewport3D()

    def _render(self):
        return self.view.grabFramebuffer()

    def test_shaders_compile(self):
        self._render()
        self.assertIsNone(self.view.gl_error)

    def test_highlight_halo_surrounds_its_atom_without_hiding_it(self):
        scene = Scene()
        scene.add_spheres([[0, 0, 0]], [0.8], [[0.1, 0.2, 0.9, 1]])
        scene.highlight = (np.zeros(3), 1.4, None, (1.0, 0.85, 0.0, 0.55))
        self.view.set_scene(scene)
        self.view.fit_camera([-1, -1, -1], [1, 1, 1])
        image = self._render()     # first: the widget may be resized to its minimum size here
        view, projection = self.view.view_matrices()
        xy, depth = project_points([[0, 0, 0]], view, projection, self.view.width(), self.view.height())
        scale = image.width() / self.view.width()          # high-DPI screens grab more pixels than widget units
        centre = (int(xy[0][0] * scale), int(xy[0][1] * scale))
        atom = image.pixelColor(*centre)
        self.assertGreater(atom.blue(), atom.red() + 60)                        # atom still shows
        ring_radius = int(1.1 * projection[1, 1] * self.view.height() / 2.0 / depth[0] * scale)
        ring = image.pixelColor(centre[0] + ring_radius, centre[1])
        self.assertGreater(ring.red(), ring.blue() + 40)                        # yellowish halo

    def test_tens_of_thousands_of_atoms_render(self):
        rng = np.random.default_rng(0)
        count = 50000
        scene = Scene()
        scene.add_spheres(rng.uniform(-30, 30, (count, 3)), np.full(count, 0.4),
                          np.tile([0.2, 0.6, 0.3, 1.0], (count, 1)))
        self.view.set_scene(scene)
        self.view.fit_camera([-30] * 3, [30] * 3)
        image = self._render()
        coloured = sum(1 for x in range(0, image.width(), 3) for y in range(0, image.height(), 3)
                       if image.pixelColor(x, y).green() > image.pixelColor(x, y).red() + 30)
        self.assertGreater(coloured, 500)


@unittest.skipUnless(AVAILABLE, "PySide6 is required")
class TestViewportChoice(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_renderer_can_be_forced(self):
        for name, cls in (("painter", PainterViewport3D), ("opengl", GLViewport3D)):
            os.environ["MCONVERT_RENDERER"] = name
            try:
                self.assertIsInstance(viewport3d.create_viewport(), cls)
            finally:
                del os.environ["MCONVERT_RENDERER"]

    def test_falls_back_to_painter_when_opengl_is_unusable(self):
        saved = viewport3d._opengl_usable
        viewport3d._opengl_usable = False
        try:
            self.assertIsInstance(viewport3d.create_viewport(), PainterViewport3D)
        finally:
            viewport3d._opengl_usable = saved

    def test_offscreen_platform_without_opengl_is_detected(self):
        if QApplication.instance().platformName() == "offscreen":
            self.assertFalse(viewport3d._opengl_is_usable())


if __name__ == "__main__":
    unittest.main()
