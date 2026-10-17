"""The 3D viewport of the real/reciprocal-space tab.

Two interchangeable widgets draw the same Scene:

  * GLViewport3D -- OpenGL 2.1 (GLSL 1.20) through Qt's own GL classes, so no
    PyOpenGL/pyqtgraph is needed. Atoms are ray-traced sphere impostors: one
    quad per atom, shaded per pixel in the fragment shader, uploaded to the GPU
    once per scene. Rotating or zooming only redraws, which keeps tens of
    thousands of atoms interactive.
  * PainterViewport3D -- QPainter, no GPU at all. Used automatically when the
    machine offers less than OpenGL 2.1 (some virtual machines and remote
    desktops), and for headless tests. Fine for up to a few thousand atoms.

create_viewport() picks one; set MCONVERT_RENDERER=painter (or opengl) to
override the choice.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QGuiApplication, QMatrix4x4, QOffscreenSurface,
    QOpenGLContext, QPainter, QPen, QRadialGradient, QSurfaceFormat,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from mconvert.camera import Camera
from mconvert.picking import project_points

Color = Tuple[float, float, float, float]

_BACKGROUND = QColor(255, 255, 255)
_EMPTY_HINT_COLOR = QColor(150, 150, 150)
_EMPTY_HINT = "未加载结构"


def _qcolor(rgba: Color) -> QColor:
    return QColor.fromRgbF(*rgba)


@dataclass
class Scene:
    """What the viewport draws, in world coordinates."""
    segments: List[Tuple[np.ndarray, Color, float]] = field(default_factory=list)   # ((N,2,3), rgba, width px)
    spheres: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)  # (N,3), radii (N,), rgba (N,4)
    markers: List[Tuple[np.ndarray, float, Color]] = field(default_factory=list)    # (N,3), diameter px, rgba
    labels: List[Tuple[np.ndarray, str, Color]] = field(default_factory=list)       # (3,), text, rgba
    # translucent halo behind one point: (position, world radius or None, pixel diameter or None, rgba)
    highlight: Optional[Tuple[np.ndarray, Optional[float], Optional[float], Color]] = None

    def add_segments(self, segments, color, width=1.5):
        self.segments.append((np.asarray(segments, dtype=float).reshape(-1, 2, 3), color, width))

    def add_spheres(self, positions, radii, colors):
        self.spheres.append((np.asarray(positions, float).reshape(-1, 3), np.asarray(radii, float),
                             np.asarray(colors, float).reshape(-1, 4)))

    def add_markers(self, positions, diameter_px, color):
        self.markers.append((np.asarray(positions, float).reshape(-1, 3), float(diameter_px), color))

    def add_label(self, position, text, color):
        self.labels.append((np.asarray(position, float), text, color))


class _InteractionMixin:
    """Camera, scene and mouse handling shared by both viewports. The concrete
    class is a QWidget and declares the `clicked(x, y)` signal."""

    CLICK_TOLERANCE_PX = 4

    def _init_interaction(self):
        self.camera = Camera()
        self.scene = None
        self._press_position = None
        self._last_position = None
        self._dragged = False
        self.setMinimumSize(320, 240)
        self.setMouseTracking(False)

    def set_scene(self, scene):
        self.scene = scene
        self._scene_changed()
        self.update()

    def _scene_changed(self):
        pass

    def fit_camera(self, low, high):
        self.camera.fit(low, high)
        self.update()

    def view_matrices(self):
        """The current view and projection matrices as 4x4 numpy arrays."""
        aspect = self.width() / max(self.height(), 1)
        return self.camera.view_matrix(), self.camera.projection_matrix(aspect)

    def mousePressEvent(self, event):
        self._press_position = self._last_position = event.position()
        self._dragged = False
        event.accept()

    def mouseMoveEvent(self, event):
        if self._last_position is None:
            return
        delta = event.position() - self._last_position
        self._last_position = event.position()
        total = event.position() - self._press_position
        if abs(total.x()) > self.CLICK_TOLERANCE_PX or abs(total.y()) > self.CLICK_TOLERANCE_PX:
            self._dragged = True
        if event.buttons() & Qt.LeftButton:
            self.camera.orbit(delta.x(), delta.y())
        elif event.buttons() & (Qt.RightButton | Qt.MiddleButton):
            self.camera.pan(delta.x(), delta.y(), max(self.height(), 1))
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._press_position is not None and not self._dragged:
            self.clicked.emit(event.position().x(), event.position().y())
        self._press_position = self._last_position = None

    def wheelEvent(self, event):
        self.camera.zoom(event.angleDelta().y() / 120.0)
        self.update()
        event.accept()

    def _paint_labels(self, painter):
        """Text labels, projected onto the screen; shared by both renderers."""
        view, projection = self.view_matrices()
        painter.setFont(QFont("Helvetica", 11, QFont.Bold))
        for position, text, color in self.scene.labels:
            screen, depth = project_points([position], view, projection, self.width(), self.height())
            if depth[0] > 0:
                painter.setPen(_qcolor(color))
                painter.drawText(QPointF(screen[0, 0] + 4, screen[0, 1] - 4), text)


# ---------------------------------------------------------------------- #
# QPainter renderer (no GPU)
# ---------------------------------------------------------------------- #
class PainterViewport3D(_InteractionMixin, QWidget):
    """Left-drag rotates, right/middle-drag pans, the wheel zooms; a click
    (press and release without dragging) emits clicked(x, y). Atoms are shaded
    discs, depth-sorted (painter's algorithm)."""

    clicked = Signal(float, float)
    renderer_name = "painter"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_interaction()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), _BACKGROUND)
        if self.scene is None:
            painter.setPen(_EMPTY_HINT_COLOR)
            painter.drawText(self.rect(), Qt.AlignCenter, _EMPTY_HINT)
        else:
            self._paint_scene(painter)
        painter.end()

    def _paint_scene(self, painter):
        view, projection = self.view_matrices()
        width, height = self.width(), self.height()
        pixels_per_world = projection[1, 1] * height / 2.0
        draw_calls = []   # (depth, callable); drawn far to near

        for segments, color, line_width in self.scene.segments:
            screen, depth = project_points(segments.reshape(-1, 3), view, projection, width, height)
            screen, depth = screen.reshape(-1, 2, 2), depth.reshape(-1, 2)
            pen = QPen(_qcolor(color), line_width, Qt.SolidLine, Qt.RoundCap)
            for ends, ends_depth in zip(screen, depth):
                if np.all(ends_depth > 0):
                    draw_calls.append((ends_depth.mean(), self._line_call(pen, ends)))

        for positions, radii, colors in self.scene.spheres:
            screen, depth = project_points(positions, view, projection, width, height)
            for center, z, radius, rgba in zip(screen, depth, radii, colors):
                if z > 0:
                    draw_calls.append((z, self._sphere_call(center, radius * pixels_per_world / z, rgba)))

        for positions, diameter, color in self.scene.markers:
            screen, depth = project_points(positions, view, projection, width, height)
            for center, z in zip(screen, depth):
                if z > 0:
                    draw_calls.append((z, self._marker_call(center, diameter / 2.0, color)))

        if self.scene.highlight is not None:
            position, world_radius, pixel_diameter, color = self.scene.highlight
            screen, depth = project_points([position], view, projection, width, height)
            if depth[0] > 0:
                radius = pixel_diameter / 2.0 if world_radius is None else world_radius * pixels_per_world / depth[0]
                # a hair farther than its point, so the halo is painted just before it
                draw_calls.append((depth[0] * 1.000001, self._halo_call(screen[0], radius, color)))

        for _depth, call in sorted(draw_calls, key=lambda item: -item[0]):
            call(painter)

        self._paint_labels(painter)

    @staticmethod
    def _line_call(pen, ends):
        return lambda painter: (painter.setPen(pen),
                                painter.drawLine(QPointF(*ends[0]), QPointF(*ends[1])))

    @staticmethod
    def _sphere_call(center, radius, rgba):
        def draw(painter):
            base = _qcolor(rgba)
            gradient = QRadialGradient(QPointF(center[0] - radius * 0.35, center[1] - radius * 0.35), radius * 1.3)
            gradient.setColorAt(0.0, base.lighter(150))
            gradient.setColorAt(0.55, base)
            gradient.setColorAt(1.0, base.darker(150))
            painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(base.darker(190), 0.8))
            painter.drawEllipse(QPointF(*center), radius, radius)
        return draw

    @staticmethod
    def _marker_call(center, radius, rgba):
        def draw(painter):
            base = _qcolor(rgba)
            painter.setBrush(QBrush(base))
            painter.setPen(QPen(base.darker(170), 1.0))
            painter.drawEllipse(QPointF(*center), radius, radius)
        return draw

    @staticmethod
    def _halo_call(center, radius, rgba):
        def draw(painter):
            painter.setBrush(QBrush(_qcolor(rgba)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(*center), radius, radius)
        return draw


# ---------------------------------------------------------------------- #
# OpenGL 2.1 renderer
# ---------------------------------------------------------------------- #
# Raw GL enums: Qt's wrapper classes do not export them as Python constants.
_GL_COLOR_BUFFER_BIT, _GL_DEPTH_BUFFER_BIT = 0x4000, 0x0100
_GL_DEPTH_TEST, _GL_BLEND, _GL_MULTISAMPLE = 0x0B71, 0x0BE2, 0x809D
_GL_VERTEX_PROGRAM_POINT_SIZE, _GL_POINT_SPRITE = 0x8642, 0x8861
_GL_ZERO, _GL_ONE, _GL_SRC_ALPHA, _GL_ONE_MINUS_SRC_ALPHA = 0, 1, 0x0302, 0x0303
_GL_LINES, _GL_TRIANGLES, _GL_POINTS = 0x0001, 0x0004, 0x0000
_GL_FLOAT = 0x1406
_FLOAT_BYTES = 4

_SPHERE_VERTEX_SHADER = """
#version 120
uniform mat4 u_view;
uniform mat4 u_proj;
attribute vec3 a_center;
attribute float a_radius;
attribute vec4 a_color;
attribute vec2 a_corner;
varying vec3 v_eye;
varying vec3 v_center;
varying float v_radius;
varying vec4 v_color;
void main() {
    vec3 center = (u_view * vec4(a_center, 1.0)).xyz;
    // a camera-facing quad pushed one radius toward the viewer: it covers the
    // sphere's whole silhouette, the fragment shader cuts out the true outline
    vec3 corner = center + vec3(a_corner * a_radius, a_radius);
    v_eye = corner;
    v_center = center;
    v_radius = a_radius;
    v_color = a_color;
    gl_Position = u_proj * vec4(corner, 1.0);
}
"""

_SPHERE_FRAGMENT_SHADER = """
#version 120
uniform mat4 u_proj;
uniform float u_flat;      // 1.0: translucent halo on the far side of the sphere
varying vec3 v_eye;
varying vec3 v_center;
varying float v_radius;
varying vec4 v_color;
void main() {
    // exact ray / sphere intersection in eye space (the eye is the origin)
    vec3 ray = normalize(v_eye);
    float b = dot(ray, v_center);
    float disc = b * b - (dot(v_center, v_center) - v_radius * v_radius);
    if (disc < 0.0) discard;
    float root = sqrt(disc);
    vec3 hit = ray * (u_flat > 0.5 ? b + root : b - root);
    vec3 normal = normalize(hit - v_center);

    vec4 clip = u_proj * vec4(hit, 1.0);
    gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;

    if (u_flat > 0.5) {
        gl_FragColor = v_color;
        return;
    }
    vec3 light = normalize(vec3(-0.45, 0.55, 0.7));
    float diffuse = max(dot(normal, light), 0.0);
    float specular = pow(max(dot(reflect(-light, normal), -ray), 0.0), 24.0);
    float rim = smoothstep(0.0, 0.35, normal.z);          // darker toward the outline
    vec3 color = v_color.rgb * (0.32 + 0.68 * diffuse) * mix(0.55, 1.0, rim) + 0.28 * specular;
    gl_FragColor = vec4(color, v_color.a);
}
"""

_POINT_VERTEX_SHADER = """
#version 120
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_size;
attribute vec3 a_pos;
void main() {
    gl_Position = u_proj * u_view * vec4(a_pos, 1.0);
    gl_PointSize = u_size;
}
"""

_POINT_FRAGMENT_SHADER = """
#version 120
uniform vec4 u_color;
uniform float u_outline;   // 1.0: dark outline ring, 0.0: plain disc
void main() {
    float r = length(gl_PointCoord * 2.0 - 1.0);
    if (r > 1.0) discard;
    vec3 color = u_color.rgb;
    if (u_outline > 0.5 && r > 0.78) color *= 0.55;
    gl_FragColor = vec4(color, u_color.a);
}
"""

_LINE_VERTEX_SHADER = """
#version 120
uniform mat4 u_view;
uniform mat4 u_proj;
attribute vec3 a_pos;
void main() {
    gl_Position = u_proj * u_view * vec4(a_pos, 1.0);
}
"""

_LINE_FRAGMENT_SHADER = """
#version 120
uniform vec4 u_color;
void main() {
    gl_FragColor = u_color;
}
"""

# quad corners of one sphere impostor: two triangles
_QUAD_CORNERS = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]], dtype=np.float32)


def _sphere_vertices(positions, radii, colors):
    """Interleaved float32 vertices (center3, radius, rgba, corner2), 6 per sphere."""
    count = len(positions)
    data = np.empty((count, 6, 10), dtype=np.float32)
    data[:, :, 0:3] = positions[:, None, :]
    data[:, :, 3] = radii[:, None]
    data[:, :, 4:8] = colors[:, None, :]
    data[:, :, 8:10] = _QUAD_CORNERS[None, :, :]
    return data.reshape(-1, 10)


def _make_program(vertex_source, fragment_source):
    from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram
    program = QOpenGLShaderProgram()
    ok = program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex_source)
    ok = ok and program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment_source)
    ok = ok and program.link()
    if not ok:
        raise RuntimeError("shader error: " + program.log())
    return program


def _opengl_is_usable():
    """True if this machine gives Qt an OpenGL >= 2.1 context. A throwaway
    context is made on an offscreen surface; platforms without GL (Qt's
    'offscreen' test platform, OpenGL 1.1 virtual machines) fail the check."""
    app = QGuiApplication.instance()
    if app is None or app.platformName() == "minimal":
        return False
    try:
        from PySide6.QtOpenGL import QOpenGLFunctions_2_1
        context = QOpenGLContext()
        surface = QOffscreenSurface()
        surface.create()
        if not (context.create() and context.makeCurrent(surface)):
            return False
        try:
            if context.format().version() < (2, 1):
                return False
            return bool(QOpenGLFunctions_2_1().initializeOpenGLFunctions())
        finally:
            context.doneCurrent()
    except Exception:
        return False


class GLViewport3D(_InteractionMixin, QOpenGLWidget):
    """Same controls as PainterViewport3D, drawn on the GPU."""

    clicked = Signal(float, float)
    renderer_name = "opengl"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_interaction()
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        self.setFormat(fmt)
        self.gl_error = None
        self._gl = None
        self._programs = {}
        self._batches = []        # (kind, QOpenGLBuffer, vertex count, params)
        self._scratch = None
        self._scene_dirty = True

    # -- scene upload ---------------------------------------------------- #
    def _scene_changed(self):
        self._scene_dirty = True

    def initializeGL(self):
        from PySide6.QtOpenGL import QOpenGLFunctions_2_1
        try:
            self._gl = QOpenGLFunctions_2_1()
            if not self._gl.initializeOpenGLFunctions():
                raise RuntimeError("OpenGL 2.1 is not available")
            self._programs = {
                "sphere": _make_program(_SPHERE_VERTEX_SHADER, _SPHERE_FRAGMENT_SHADER),
                "point": _make_program(_POINT_VERTEX_SHADER, _POINT_FRAGMENT_SHADER),
                "line": _make_program(_LINE_VERTEX_SHADER, _LINE_FRAGMENT_SHADER),
            }
        except Exception as err:   # keep the GUI alive; show the reason in the view
            self.gl_error = str(err)

    def _new_buffer(self, array):
        from PySide6.QtOpenGL import QOpenGLBuffer
        buffer = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        buffer.create()
        buffer.bind()
        data = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        buffer.allocate(data, len(data))
        buffer.release()
        return buffer

    def _release_batches(self):
        for _kind, buffer, _count, _params in self._batches:
            buffer.destroy()
        self._batches = []

    def _upload_scene(self):
        self._release_batches()
        self._scene_dirty = False
        scene = self.scene
        if scene is None:
            return
        for segments, color, width in scene.segments:
            if len(segments):
                self._batches.append(("line", self._new_buffer(segments.reshape(-1, 3)),
                                      len(segments) * 2, (color, width)))
        positions = [p for p, _, _ in scene.spheres if len(p)]
        if positions:
            vertices = _sphere_vertices(
                np.vstack(positions),
                np.concatenate([r for p, r, _ in scene.spheres if len(p)]),
                np.vstack([c for p, _, c in scene.spheres if len(p)]))
            self._batches.append(("sphere", self._new_buffer(vertices), len(vertices), None))
        for marker_positions, diameter, color in scene.markers:
            if len(marker_positions):
                self._batches.append(("point", self._new_buffer(marker_positions),
                                      len(marker_positions), (color, diameter)))

    # -- drawing ---------------------------------------------------------- #
    @staticmethod
    def _matrix(array):
        return QMatrix4x4([float(v) for v in np.asarray(array).ravel()])

    def _bind_program(self, name, view, projection):
        program = self._programs[name]
        program.bind()
        program.setUniformValue(program.uniformLocation(b"u_view"), self._matrix(view))
        program.setUniformValue(program.uniformLocation(b"u_proj"), self._matrix(projection))
        return program

    def _attach(self, program, buffer, layout, stride_floats):
        """Point the program's attributes at `buffer`: layout is (name, float offset, size)."""
        buffer.bind()
        for name, offset, size in layout:
            location = program.attributeLocation(name)
            program.enableAttributeArray(location)
            program.setAttributeBuffer(location, _GL_FLOAT, offset * _FLOAT_BYTES, size,
                                       stride_floats * _FLOAT_BYTES)

    def _detach(self, program, buffer, names):
        for name in names:
            program.disableAttributeArray(program.attributeLocation(name))
        buffer.release()
        program.release()

    _SPHERE_LAYOUT = (("a_center", 0, 3), ("a_radius", 3, 1), ("a_color", 4, 4), ("a_corner", 8, 2))

    def _draw_batch(self, kind, buffer, count, params, view, projection, ratio):
        gl = self._gl
        if kind == "line":
            color, width = params
            program = self._bind_program("line", view, projection)
            program.setUniformValue(program.uniformLocation(b"u_color"), *[float(c) for c in color])
            gl.glLineWidth(max(width * ratio, 1.0))
            self._attach(program, buffer, (("a_pos", 0, 3),), 3)
            gl.glDrawArrays(_GL_LINES, 0, count)
            self._detach(program, buffer, ("a_pos",))
        elif kind == "sphere":
            program = self._bind_program("sphere", view, projection)
            program.setUniformValue1f(program.uniformLocation(b"u_flat"), 0.0)
            self._attach(program, buffer, self._SPHERE_LAYOUT, 10)
            gl.glDrawArrays(_GL_TRIANGLES, 0, count)
            self._detach(program, buffer, [name for name, _, _ in self._SPHERE_LAYOUT])
        else:
            color, diameter = params
            self._draw_points(buffer, count, color, diameter * ratio, 1.0, view, projection)

    def _draw_points(self, buffer, count, color, size_px, outline, view, projection):
        program = self._bind_program("point", view, projection)
        program.setUniformValue(program.uniformLocation(b"u_color"), *[float(c) for c in color])
        program.setUniformValue1f(program.uniformLocation(b"u_size"), float(size_px))
        program.setUniformValue1f(program.uniformLocation(b"u_outline"), float(outline))
        self._attach(program, buffer, (("a_pos", 0, 3),), 3)
        self._gl.glDrawArrays(_GL_POINTS, 0, count)
        self._detach(program, buffer, ("a_pos",))

    def _draw_highlight(self, view, projection, ratio):
        position, world_radius, pixel_diameter, color = self.scene.highlight
        gl = self._gl
        gl.glEnable(_GL_BLEND)
        # keep the framebuffer's alpha at 1: a translucent window would show the desktop through the halo
        gl.glBlendFuncSeparate(_GL_SRC_ALPHA, _GL_ONE_MINUS_SRC_ALPHA, _GL_ZERO, _GL_ONE)
        gl.glDepthMask(False)
        if world_radius is not None:
            vertices = _sphere_vertices(np.asarray([position], float), np.asarray([world_radius]),
                                        np.asarray([color], float))
            buffer = self._new_buffer(vertices)
            program = self._bind_program("sphere", view, projection)
            program.setUniformValue1f(program.uniformLocation(b"u_flat"), 1.0)
            self._attach(program, buffer, self._SPHERE_LAYOUT, 10)
            gl.glDrawArrays(_GL_TRIANGLES, 0, len(vertices))
            self._detach(program, buffer, [name for name, _, _ in self._SPHERE_LAYOUT])
        else:
            buffer = self._new_buffer(np.asarray([position], float))
            self._draw_points(buffer, 1, color, pixel_diameter * ratio, 0.0, view, projection)
        buffer.destroy()
        gl.glDepthMask(True)
        gl.glDisable(_GL_BLEND)

    def paintGL(self):
        gl = self._gl
        if gl is None or self.gl_error is not None:
            self._paint_message(self.gl_error or "OpenGL 初始化失败")
            return
        if self._scene_dirty:
            self._upload_scene()

        gl.glClearColor(_BACKGROUND.redF(), _BACKGROUND.greenF(), _BACKGROUND.blueF(), 1.0)
        gl.glClear(_GL_COLOR_BUFFER_BIT | _GL_DEPTH_BUFFER_BIT)
        if self.scene is None:
            self._paint_message(_EMPTY_HINT)
            return

        view, projection = self.view_matrices()
        ratio = self.devicePixelRatioF()
        gl.glEnable(_GL_DEPTH_TEST)
        gl.glEnable(_GL_MULTISAMPLE)
        gl.glEnable(_GL_VERTEX_PROGRAM_POINT_SIZE)
        gl.glEnable(_GL_POINT_SPRITE)   # makes gl_PointCoord valid in a compatibility context
        try:
            for kind, buffer, count, params in self._batches:
                self._draw_batch(kind, buffer, count, params, view, projection, ratio)
            if self.scene.highlight is not None:
                self._draw_highlight(view, projection, ratio)
        finally:
            gl.glDisable(_GL_DEPTH_TEST)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._paint_labels(painter)
        painter.end()

    def _paint_message(self, text):
        painter = QPainter(self)
        painter.setPen(_EMPTY_HINT_COLOR)
        painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, text)
        painter.end()


_opengl_usable = None


def create_viewport(parent=None):
    """The best available viewport: OpenGL when the machine offers 2.1 or
    newer, otherwise the QPainter one. MCONVERT_RENDERER=painter|opengl forces
    a choice."""
    global _opengl_usable
    forced = os.environ.get("MCONVERT_RENDERER", "").lower()
    if forced == "painter":
        return PainterViewport3D(parent)
    if forced == "opengl":
        return GLViewport3D(parent)
    if _opengl_usable is None:
        _opengl_usable = _opengl_is_usable()
    return GLViewport3D(parent) if _opengl_usable else PainterViewport3D(parent)
