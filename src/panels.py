"""The contents of the viewer's docks. Each panel builds and owns its
widgets and the state that only it needs; the Viewer connects to their
widgets and signals and applies the results to the scene."""

import numpy as np
from qtpy import QtCore, QtWidgets

from gui_helpers import RAY_COLOR_MODES, wrap_label
from meshing import MESHER_CAPABILITIES
from qt_helpers import (
    add_labeled_row,
    add_row,
    ask_color,
    clear_layout,
    make_button,
    make_checkbox,
    make_double_spin,
    make_elided_combo,
    make_note_label,
    make_swatch_button,
    repopulate_combo,
    set_option_enabled,
    set_swatch_color,
)
from scene_objects import DEFAULT_MARKER_SIZES, DEFAULT_RAY_COLORS, RULER_COLOR
from widgets import ColorBarWidget

# The meshers offered in the dock, in display order.
MESHER_KEYS = ("mc", "dc", "brep")

MESHER_DESCRIPTIONS = {
    "mc": (
        "Marching Cubes: samples the signed distance field on a uniform grid. "
        "Fast, but curved surfaces are limited by the chosen resolution."
    ),
    "dc": (
        "Dual Contouring: builds an adaptive octree from the point cloud. "
        "Better at preserving sharp edges/corners than Marching Cubes."
    ),
    "brep": (
        "BREP (exact CAD): uses OpenCASCADE boundary representation for exact, "
        "resolution-independent geometry. Most accurate, but can be slower for "
        "complex boolean operations."
    ),
}

# The marching-cubes grid resolutions offered in the dock.
RESOLUTIONS = (16, 32, 64, 128, 256, 512)
DEFAULT_RESOLUTION = 64

DEFLECTION_TOOLTIP = (
    "How closely the brep mesher's triangles must follow the exact curved "
    "surface, in metres. Smaller values hug curves more closely (more "
    "triangles, slower to build); larger values are coarser but faster."
)

DEFAULT_SWATCH_COLOR = "#b6b6b6"


def _panel_layout(panel):
    layout = QtWidgets.QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    return layout


class SettingsPanel(QtWidgets.QWidget):
    """File loading, what to draw, the mcrun --trace status, and the view
    and export buttons."""

    def __init__(self):
        super().__init__()
        self.open_file_button = make_button(
            "Open File...", "Open a McStas instrument (shortcut: Ctrl+O)"
        )
        self.material_group_checkbox = make_checkbox(
            "Group by material",
            True,
            "Combine components that share a material into a single "
            "rendered object (trimesh concatenation, not a boolean fusion).",
        )
        self.vacuum_checkbox = make_checkbox(
            "Hide vacuum",
            True,
            "Hides volumes whose material is 'vacuum'/'Vacuum' or "
            "'exit'/'Exit' (McStas treats 'exit' as vacuum too).",
        )
        self.pygen_checkbox = make_checkbox(
            "Force mcstas-pygen preprocessing",
            False,
            "Translate a .instr input through the real McStas front-end "
            "(mcstas-pygen) instead of mcstasscript's lightweight .instr "
            "reader. mcstas-pygen already runs automatically whenever that "
            "lightweight reader raises an error - this instead forces it "
            "for every load, for instruments the lightweight reader "
            "mis-parses without raising an error. Requires mcstas-pygen "
            "on PATH (ships with the McStas install).",
        )
        self.components_checkbox = make_checkbox(
            "Show McStas components",
            True,
            "Draw every non-Union component the way McStas's own MCDISPLAY "
            "draws it (no priority cutting). Compiles and runs the "
            "instrument with mcrun --trace, so it needs a working McStas "
            "install and compiler.",
        )
        self.arms_checkbox = make_checkbox(
            "Show Arms", False, "Arms only mark coordinate frames."
        )
        self.rays_checkbox = make_checkbox(
            "Show neutron rays",
            False,
            "Trace neutrons through the instrument with mcrun --trace and "
            "draw their paths. Options are in the Neutron Rays panel.",
        )
        self.grid_checkbox = make_checkbox("Show grid", True)
        self.gizmo_checkbox = make_checkbox("Show gizmo", True)
        self.axes_checkbox = make_checkbox("Show xyz axes", True)
        self.edges_checkbox = make_checkbox(
            "Show edges", True, "Draw the sharp edges of the Union meshes as lines."
        )
        self.trace_status_label = make_note_label()
        self.reset_view_button = make_button("Reset view", "Refit the camera (shortcut: R)")
        self.fit_instrument_button = make_button(
            "Fit whole instrument",
            "Fit the camera to every McStas component, not just the Union geometry.",
        )
        self.export_stl_button = make_button(
            "Export STL...",
            "Export the visible Union meshes and McStas components as a "
            "single .stl file, cut by the clipping plane if enabled. "
            "McStas lines are exported as thin tubes.",
        )

        layout = _panel_layout(self)
        for widget in (
            self.open_file_button,
            self.material_group_checkbox,
            self.vacuum_checkbox,
            self.pygen_checkbox,
            self.components_checkbox,
            self.arms_checkbox,
            self.rays_checkbox,
            self.grid_checkbox,
            self.gizmo_checkbox,
            self.axes_checkbox,
            self.edges_checkbox,
            self.trace_status_label,
            self.reset_view_button,
            self.fit_instrument_button,
            self.export_stl_button,
        ):
            layout.addWidget(widget)

    def set_trace_status(self, text, error=False):
        """Show text under the checkboxes: grey, or red (last 6 lines
        only) for an error."""
        if error:
            text = "\n".join(text.strip().splitlines()[-6:])
        self.trace_status_label.setStyleSheet("color: #c0392b;" if error else "color: gray;")
        self.trace_status_label.setText(text)


def format_ruler_text(points):
    """The ruler readout below the point coordinates, for zero, one or two
    world points."""
    if not points:
        return "Tick Measure distances and click two points on the geometry, or type their coordinates."
    if len(points) == 1:
        return "Click or type a second point."
    delta = np.asarray(points[1]) - np.asarray(points[0])
    return (
        f"Distance: {np.linalg.norm(delta):.4f} m\n"
        f"dx, dy, dz: ({delta[0]:.4f}, {delta[1]:.4f}, {delta[2]:.4f}) m"
    )


class RulerPanel(QtWidgets.QWidget):
    """Measures the distance between two points, given in world coordinates
    (m). While measure_checkbox is checked the Viewer adds clicked points
    with add_point; a third click starts a new measurement. The coordinates
    can also be typed or adjusted by hand. points_edited() is emitted
    whenever the points change, color_changed(hex) when the colour does."""

    points_edited = QtCore.Signal()
    color_changed = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.color = RULER_COLOR
        self.point_set = [False, False]
        self.measure_checkbox = make_checkbox(
            "Measure distances",
            False,
            "Click two points on the geometry to measure the distance "
            "between them. Dragging still rotates the view.",
        )
        self.point_spins = [
            [
                make_double_spin(-1e3, 1e3, 0.0, decimals=5, step=0.001, tooltip=f"Point {i + 1} {axis} (m)")
                for axis in "xyz"
            ]
            for i in range(2)
        ]
        self.readout_label = make_note_label()
        self.readout_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.clear_button = make_button("Clear", "Remove the measured points")
        self.color_button = make_swatch_button(14, self.color, "Click to change the ruler's colour.", self._pick_color)

        layout = _panel_layout(self)
        layout.addWidget(self.measure_checkbox)
        # One column per point, its x, y and z stacked like a vector.
        grid = QtWidgets.QGridLayout()
        for row, axis in enumerate("xyz", 1):
            grid.addWidget(QtWidgets.QLabel(f"{axis} (m)"), row, 0)
        for i, spins in enumerate(self.point_spins):
            grid.addWidget(QtWidgets.QLabel(f"Point {i + 1}"), 0, i + 1, QtCore.Qt.AlignmentFlag.AlignCenter)
            for row, spin in enumerate(spins, 1):
                spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
                grid.addWidget(spin, row, i + 1)
                spin.valueChanged.connect(lambda _value, i=i: self._on_point_typed(i))
        layout.addLayout(grid)
        layout.addWidget(self.readout_label)
        add_row(layout, self.clear_button, QtWidgets.QLabel("Colour"), self.color_button, stretch=True)
        self.show_points([])

    def points(self):
        """The points that are set, Point 1 first."""
        return [
            np.array([spin.value() for spin in spins])
            for spins, is_set in zip(self.point_spins, self.point_set)
            if is_set
        ]

    def add_point(self, point):
        """Fill the first point that isn't set, or start a new measurement
        at Point 1 if both are."""
        index = self.point_set.index(False) if False in self.point_set else 0
        if index == 0:
            self.point_set = [False, False]
        self._set_spins(index, point)
        self.point_set[index] = True
        self.points_edited.emit()

    def clear(self):
        self.point_set = [False, False]
        for i in range(2):
            self._set_spins(i, (0.0, 0.0, 0.0))
        self.points_edited.emit()

    def show_points(self, points):
        self.readout_label.setText(format_ruler_text(points))

    def _set_spins(self, index, point):
        for spin, value in zip(self.point_spins[index], point):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    def _on_point_typed(self, index):
        self.point_set[index] = True
        self.points_edited.emit()

    def _pick_color(self):
        hex_color = ask_color(self, self.color, "Ruler colour")
        if hex_color is None:
            return
        self.color = hex_color
        set_swatch_color(self.color_button, hex_color)
        self.color_changed.emit(hex_color)


class ParametersPanel(QtWidgets.QWidget):
    """One text field per instrument parameter. Emits edited() when a
    value changes; values maps parameter name to the text typed (empty
    means use the default)."""

    edited = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.values = {}
        self.edits = {}
        self.names = None

        form_widget = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(form_widget)
        self.empty_label = make_note_label("No instrument loaded.")

        # Scrollable, and capped at a fixed height rather than growing with
        # the parameter count - an instrument can have many parameters, and
        # this keeps the rest of the panels from being pushed off-screen.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        scroll.setMaximumHeight(220)

        layout = _panel_layout(self)
        layout.addWidget(self.empty_label)
        layout.addWidget(scroll)

    def rebuild(self, parameters):
        """One labelled field per (name, type, default) in parameters,
        showing its default as the placeholder. Values typed earlier are
        kept by name."""
        names = [name for name, _, _ in parameters]
        if names == self.names:
            return
        self.names = names
        while self.form.rowCount():
            self.form.removeRow(0)
        self.edits = {}
        self.values = {n: v for n, v in self.values.items() if n in names}
        for name, param_type, default in parameters:
            edit = QtWidgets.QLineEdit(self.values.get(name, ""))
            edit.setPlaceholderText(default if default is not None else "required")
            edit.setToolTip(
                f"{param_type} {name}"
                + (f" (default {default})" if default is not None else " (no default)")
                + ". Leave empty to use the default."
            )
            edit.editingFinished.connect(lambda n=name, e=edit: self._on_edited(n, e.text()))
            self.form.addRow(name, edit)
            self.edits[name] = edit
        self.empty_label.setText("This instrument has no parameters.")
        self.empty_label.setVisible(not names)

    def _on_edited(self, name, text):
        if self.values.get(name, "").strip() == text.strip():
            return
        self.values[name] = text.strip()
        self.edited.emit()


class MesherPanel(QtWidgets.QWidget):
    """Mesher choice, marching-cubes resolution and brep deflection. Emits
    changed() when any of them changes; options the selected mesher
    doesn't use are disabled."""

    changed = QtCore.Signal()

    def __init__(self, mesher="brep"):
        super().__init__()
        layout = _panel_layout(self)

        self.mesher_box = QtWidgets.QComboBox()
        for key in MESHER_KEYS:
            label = key
            if not MESHER_CAPABILITIES[key]["incremental_rebuild"]:
                label += " (full rebuild only)"
            self.mesher_box.addItem(label, key)
        self.mesher_box.setCurrentIndex(self.mesher_box.findData(mesher))
        add_labeled_row(layout, "Mesher", self.mesher_box)

        self.description_label = make_note_label(MESHER_DESCRIPTIONS.get(mesher, ""), italic=True)
        layout.addWidget(self.description_label)

        self.resolution_box = QtWidgets.QComboBox()
        for res in RESOLUTIONS:
            self.resolution_box.addItem(str(res), res)
        self.resolution_box.setCurrentIndex(self.resolution_box.findData(DEFAULT_RESOLUTION))
        self.resolution_label = add_labeled_row(layout, "Resolution", self.resolution_box)

        self.deflection_spin = make_double_spin(
            0.0001, 10.0, 0.01, decimals=4, step=0.001, tooltip=DEFLECTION_TOOLTIP
        )
        self.deflection_label = add_labeled_row(layout, "Surface deflection", self.deflection_spin)
        self.deflection_label.setToolTip(DEFLECTION_TOOLTIP)

        self.update_capabilities()
        self.mesher_box.currentIndexChanged.connect(self._on_mesher_changed)
        self.resolution_box.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.deflection_spin.valueChanged.connect(lambda *_: self.changed.emit())

    def mesher(self):
        return self.mesher_box.currentData()

    def resolution(self):
        return self.resolution_box.currentData()

    def deflection(self):
        return self.deflection_spin.value()

    def update_capabilities(self):
        mesher = self.mesher()
        caps = MESHER_CAPABILITIES[mesher]
        unused_tip = f"Not used by the '{mesher}' mesher."
        set_option_enabled(
            (self.resolution_box, self.resolution_label),
            caps["resolution"],
            "" if caps["resolution"] else unused_tip,
        )
        set_option_enabled(
            (self.deflection_spin, self.deflection_label),
            caps["deflection"],
            DEFLECTION_TOOLTIP if caps["deflection"] else unused_tip,
        )

    def _on_mesher_changed(self):
        self.description_label.setText(MESHER_DESCRIPTIONS.get(self.mesher(), ""))
        self.update_capabilities()
        self.changed.emit()


class ClippingPanel(QtWidgets.QWidget):
    """The clipping plane: on/off, coordinate system, axis, side and
    position. clip holds the current settings as the dict clipping.py
    expects; changed() is emitted whenever it changes."""

    changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.clip = {
            "enable": False,
            "axis": "Z",
            "mode": "Above",
            "position": 0.0,
            "frame": None,
        }
        layout = _panel_layout(self)

        self.enable_checkbox = make_checkbox("Enable clipping")
        layout.addWidget(self.enable_checkbox)

        self.frame_combo = make_elided_combo()
        self.frame_combo.addItem("World", None)
        self.frame_combo.setToolTip(
            "Axis and position are taken in this component's own coordinate "
            "system (its AT/ROTATED frame), e.g. the sample's Arm."
        )
        add_labeled_row(layout, "Coordinate system", self.frame_combo)

        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItems(["X", "Y", "Z"])
        add_labeled_row(layout, "Axis", self.axis_combo)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Above", "Below"])
        add_labeled_row(layout, "Mode", self.mode_combo)

        self.position_spin = make_double_spin(-1e6, 1e6, 0.0, decimals=5, step=0.01)
        add_labeled_row(layout, "Position", self.position_spin)

        self.enable_checkbox.stateChanged.connect(self._on_changed)
        self.axis_combo.currentTextChanged.connect(self._on_changed)
        self.frame_combo.currentIndexChanged.connect(self._on_changed)
        self.mode_combo.currentTextChanged.connect(self._on_changed)
        self.position_spin.valueChanged.connect(self._on_changed)

    def set_frames(self, names):
        """Offer "World" plus each component name as coordinate system,
        keeping the selection if that component still exists."""
        combo = self.frame_combo
        if list(names) == [combo.itemData(i) for i in range(1, combo.count())]:
            return
        if repopulate_combo(combo, "World", names):
            self._on_changed()

    def _on_changed(self, *_):
        self.clip["enable"] = self.enable_checkbox.isChecked()
        self.clip["axis"] = self.axis_combo.currentText()
        self.clip["mode"] = self.mode_combo.currentText()
        self.clip["position"] = self.position_spin.value()
        self.clip["frame"] = self.frame_combo.currentData()
        self.changed.emit()


class CutPanel(QtWidgets.QWidget):
    """Collapsing the gap between two components: on/off, the 'From' and
    'To' components and the distance to keep between them. changed() is
    emitted whenever any of them changes."""

    changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        layout = _panel_layout(self)

        self.enable_checkbox = make_checkbox(
            "Collapse gap",
            tooltip=(
                "Move every component up to and including 'From' along the "
                "line to 'To' until only 'Keep' metres separate them, and "
                "hide the components in between. Shapes and angles on each "
                "side stay true; only rays crossing the gap (drawn dashed) "
                "are shortened. STL export keeps true positions but leaves "
                "the hidden components out."
            ),
        )
        layout.addWidget(self.enable_checkbox)

        self.from_combo = make_elided_combo()
        self.to_combo = make_elided_combo()
        for text, combo in (("From", self.from_combo), ("To", self.to_combo)):
            combo.addItem("-", None)
            add_labeled_row(layout, text, combo)

        self.keep_spin = make_double_spin(
            0.0, 1e6, 2.0, decimals=3, step=0.1,
            tooltip="Distance left between 'From' and 'To'.",
        )
        self.keep_spin.setSuffix(" m")
        add_labeled_row(layout, "Keep", self.keep_spin)

        self.info_label = make_note_label()
        layout.addWidget(self.info_label)

        self.enable_checkbox.stateChanged.connect(self._on_changed)
        self.from_combo.currentIndexChanged.connect(self._on_changed)
        self.to_combo.currentIndexChanged.connect(self._on_changed)
        self.keep_spin.valueChanged.connect(self._on_changed)

    def set_components(self, names):
        """Offer each component name, keeping the selections that still
        exist."""
        changed = False
        for combo in (self.from_combo, self.to_combo):
            if list(names) != [combo.itemData(i) for i in range(1, combo.count())]:
                changed |= repopulate_combo(combo, "-", names)
        if changed:
            self._on_changed()

    def selection(self):
        """(start, end, keep), or None while switched off or unset."""
        start, end = self.from_combo.currentData(), self.to_combo.currentData()
        if not self.enable_checkbox.isChecked() or start is None or end is None:
            return None
        return start, end, self.keep_spin.value()

    def _on_changed(self, *_):
        self.changed.emit()


class RaysPanel(QtWidgets.QWidget):
    """Options for the traced neutron rays.

    Signals: rerun_requested() for the Re-run button, or 700 ms after the
    ray count or seed last changed; style_changed() when the colouring or
    the "reaching" filter changes; visibility_changed() for the show/mark
    checkboxes; color_changed(key, hex) and marker_size_changed(key, size)
    for the "ray", "teleport", "scatter" and "absorb" styles, which are
    kept in colors and marker_sizes.
    """

    rerun_requested = QtCore.Signal()
    style_changed = QtCore.Signal()
    visibility_changed = QtCore.Signal()
    color_changed = QtCore.Signal(str, str)
    marker_size_changed = QtCore.Signal(str, float)

    def __init__(self):
        super().__init__()
        self.colors = dict(DEFAULT_RAY_COLORS)
        self.marker_sizes = dict(DEFAULT_MARKER_SIZES)
        self.color_buttons = {}
        self.size_spins = {}
        layout = _panel_layout(self)

        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setRange(1, 100_000_000)
        self.count_spin.setValue(50)
        self.count_spin.setToolTip(
            "Trace mode is single-threaded and verbose - keep this small; "
            "McStas itself has no lower limit worth mentioning."
        )
        add_labeled_row(layout, "Number of rays", self.count_spin)

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setSpecialValueText("random")
        self.seed_spin.setToolTip("0 picks a new random seed on every run.")
        add_labeled_row(layout, "Seed", self.seed_spin)

        self.rerun_button = make_button("Re-run", "Trace a new set of rays.", self._request_rerun)
        add_row(layout, self.rerun_button)

        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItems(list(RAY_COLOR_MODES))
        add_labeled_row(layout, "Colour by", self.color_combo)

        self.colorbar = ColorBarWidget()
        self.colorbar.hide()
        layout.addWidget(self.colorbar)

        # Label above the combo, not beside it, so a long selected/eliding
        # entry doesn't push the row (and the panel) wider.
        layout.addWidget(QtWidgets.QLabel("Only rays reaching"))
        self.reaching_combo = make_elided_combo()
        self.reaching_combo.addItem("any component", None)
        layout.addWidget(self.reaching_combo)

        self.line_checkbox = make_checkbox(
            "Show rays",
            True,
            "The ordinary path each ray follows. The colour applies when "
            "colouring by Uniform.",
        )
        add_row(layout, self.line_checkbox, self._color_button("ray"), stretch=True)

        self.teleport_checkbox = make_checkbox(
            "Show teleports",
            True,
            "The jump a restore_neutron monitor (e.g. PSD_monitor) causes: "
            "it detects a ray, then restores its pre-detection state.",
        )
        add_row(layout, self.teleport_checkbox, self._color_button("teleport"), stretch=True)

        self.scatter_checkbox = make_checkbox(
            "Mark scatterings",
            True,
            "Points where a ray changed direction. Union volume boundary "
            "crossings are not marked.",
        )
        add_row(
            layout,
            self.scatter_checkbox,
            self._color_button("scatter"),
            self._marker_size_spin("scatter"),
            stretch=True,
        )

        self.absorb_checkbox = make_checkbox("Mark absorptions", True)
        add_row(
            layout,
            self.absorb_checkbox,
            self._color_button("absorb"),
            self._marker_size_spin("absorb"),
            stretch=True,
        )

        self.info_label = make_note_label()
        layout.addWidget(self.info_label)

        self.rerun_timer = QtCore.QTimer(self)
        self.rerun_timer.setSingleShot(True)
        self.rerun_timer.setInterval(700)
        self.rerun_timer.timeout.connect(self._request_rerun)

        self.count_spin.valueChanged.connect(self.rerun_timer.start)
        self.seed_spin.valueChanged.connect(self.rerun_timer.start)
        self.color_combo.currentIndexChanged.connect(lambda *_: self.style_changed.emit())
        self.reaching_combo.currentIndexChanged.connect(lambda *_: self.style_changed.emit())
        for checkbox in (
            self.line_checkbox,
            self.teleport_checkbox,
            self.scatter_checkbox,
            self.absorb_checkbox,
        ):
            checkbox.stateChanged.connect(lambda *_: self.visibility_changed.emit())

    def ncount(self):
        return self.count_spin.value()

    def seed(self):
        """The chosen seed, or None for a random one."""
        return self.seed_spin.value() or None

    def set_component_names(self, names):
        """Offer "any component" plus names in the "Only rays reaching"
        combo, keeping the selection if it is still offered."""
        repopulate_combo(self.reaching_combo, "any component", names)

    def _request_rerun(self):
        self.rerun_timer.stop()
        self.rerun_requested.emit()

    def _color_button(self, key):
        """A small colour button that picks the colour of one object in
        this panel ("ray", "teleport", "scatter" or "absorb")."""
        button = make_swatch_button(
            14, self.colors[key], "Click to change the colour.", lambda: self._pick_color(key)
        )
        self.color_buttons[key] = button
        return button

    def _marker_size_spin(self, key):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(1, 50)
        spin.setSingleStep(1)
        spin.setDecimals(1)
        spin.setSuffix(" px")
        spin.setValue(self.marker_sizes[key])
        spin.setToolTip("Marker size.")
        spin.valueChanged.connect(lambda value: self._set_marker_size(key, value))
        self.size_spins[key] = spin
        return spin

    def _pick_color(self, key):
        hex_color = ask_color(self, self.colors[key], "Colour")
        if hex_color is None:
            return
        self.colors[key] = hex_color
        set_swatch_color(self.color_buttons[key], hex_color)
        self.color_changed.emit(key, hex_color)

    def _set_marker_size(self, key, size):
        self.marker_sizes[key] = size
        self.marker_size_changed.emit(key, size)


class CountsPanel(QtWidgets.QWidget):
    """Loading a run folder's detector counts, their colour range, and one
    checkbox per loaded detector. visibility maps detector name to whether
    its checkbox is ticked; detector_toggled(name, checked) is emitted
    when one is toggled."""

    detector_toggled = QtCore.Signal(str, bool)

    def __init__(self):
        super().__init__()
        self.visibility = {}
        self.checkboxes = {}
        layout = _panel_layout(self)

        self.load_button = make_button(
            "Load counts folder...",
            "A McStas run's output folder (containing mccode.sim). Every "
            "detector in it that can be placed in 3D - a Union logger/"
            "abs_logger, an ordinary monitor (PSD_monitor, Monitor_nD, ...), "
            "or an event-list logger with x/y/z columns - is drawn at its "
            "own position: a colour-mapped plane for 2D data, a coloured "
            "line for 1D.",
        )
        layout.addWidget(self.load_button)

        self.show_checkbox = make_checkbox("Show detector counts", True)
        layout.addWidget(self.show_checkbox)

        range_tooltip = "Editable - overrides the automatic min/max for every visible logger's plane."
        self.min_spin = make_double_spin(-1e12, 1e12, 0.0, decimals=4, tooltip=range_tooltip)
        self.max_spin = make_double_spin(-1e12, 1e12, 0.0, decimals=4, tooltip=range_tooltip)
        add_row(layout, QtWidgets.QLabel("Colour range"), self.min_spin, self.max_spin)

        self.auto_range_button = make_button(
            "Auto range",
            "Reset the colour range to the min/max of the currently visible loggers.",
        )
        layout.addWidget(self.auto_range_button)

        self.colorbar = ColorBarWidget()
        self.colorbar.hide()
        layout.addWidget(self.colorbar)

        self.rows_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.rows_layout)

        self.info_label = make_note_label("No counts loaded.")
        layout.addWidget(self.info_label)

    def value_range(self):
        return self.min_spin.value(), self.max_spin.value()

    def set_range(self, vmin, vmax):
        """Set the colour range spin boxes without emitting valueChanged,
        widening their limits if needed."""
        for spin, value in ((self.min_spin, vmin), (self.max_spin, vmax)):
            spin.blockSignals(True)
            spin.setRange(min(-1e12, value * 2 - 1), max(1e12, value * 2 + 1))
            spin.setValue(value)
            spin.blockSignals(False)

    def rebuild_rows(self, counts):
        """One checkbox per {name: LoggerCounts} in counts."""
        clear_layout(self.rows_layout)
        self.checkboxes.clear()

        for name, lc in counts.items():
            axes = lc.axis1 if lc.axis2 is None else f"{lc.axis1}, {lc.axis2}"
            bins = "x".join(str(n) for n in lc.grid.shape)
            cb = make_checkbox(
                wrap_label(f"{name} ({lc.kind})"),
                self.visibility.get(name, True),
                f"{name}: axes ({axes}), {bins} bins, total {lc.total:.4g}",
            )
            cb.toggled.connect(lambda checked, n=name: self._on_toggled(n, checked))
            add_row(self.rows_layout, cb)
            self.checkboxes[name] = cb

    def _on_toggled(self, name, checked):
        self.visibility[name] = checked
        self.detector_toggled.emit(name, checked)


class GeometryPanel(QtWidgets.QWidget):
    """The Visible Geometries list: a filter, Show/Hide all, and one row
    (checkbox and colour swatch) per Union geometry and per McStas
    component.

    union_visibility and component_visibility map a row's name to whether
    its checkbox is ticked. Signals: union_toggled(name, checked) and
    component_toggled(name, checked) when a row is toggled,
    union_color_requested(name) and component_color_requested(name) when
    its swatch is clicked.
    """

    union_toggled = QtCore.Signal(str, bool)
    component_toggled = QtCore.Signal(str, bool)
    union_color_requested = QtCore.Signal(str)
    component_color_requested = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.union_checkboxes = {}
        self.union_color_buttons = {}
        self.union_visibility = {}
        self.component_checkboxes = {}
        self.component_color_buttons = {}
        self.component_visibility = {}
        self.show_components = True

        layout = QtWidgets.QVBoxLayout(self)

        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filter...")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self.apply_filter)
        layout.addWidget(self.filter_edit)

        self.show_all_button = make_button("Show all", on_click=lambda: self.set_all_visible(True))
        self.hide_all_button = make_button("Hide all", on_click=lambda: self.set_all_visible(False))
        add_row(layout, self.show_all_button, self.hide_all_button)

        rows_widget = QtWidgets.QWidget()
        rows_layout = QtWidgets.QVBoxLayout(rows_widget)
        self.union_header = QtWidgets.QLabel("<b>Union components</b>")
        self.union_header.hide()
        rows_layout.addWidget(self.union_header)
        self.union_layout = QtWidgets.QVBoxLayout()
        rows_layout.addLayout(self.union_layout)
        self.component_header = QtWidgets.QLabel("<b>McStas components</b>")
        self.component_header.hide()
        rows_layout.addWidget(self.component_header)
        self.component_layout = QtWidgets.QVBoxLayout()
        rows_layout.addLayout(self.component_layout)
        rows_layout.addStretch()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rows_widget)
        layout.addWidget(scroll)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def set_union_rows(self, names, color_of):
        """One row per Union geometry name; color_of(name) gives its
        swatch colour or None."""
        self._rebuild_rows(
            names,
            color_of,
            self.union_layout,
            self.union_checkboxes,
            self.union_color_buttons,
            self.union_visibility,
            self.union_toggled,
            self.union_color_requested,
        )
        self.union_header.setVisible(bool(names))
        self.apply_filter()

    def set_component_rows(self, names, color_of):
        """One row per McStas component name; color_of(name) gives its
        swatch colour or None."""
        self._rebuild_rows(
            names,
            color_of,
            self.component_layout,
            self.component_checkboxes,
            self.component_color_buttons,
            self.component_visibility,
            self.component_toggled,
            self.component_color_requested,
        )
        self.component_header.setVisible(bool(names) and self.show_components)
        self.apply_filter()

    def set_components_shown(self, shown):
        """Show or hide the whole McStas components section."""
        self.show_components = shown
        self.component_header.setVisible(shown and bool(self.component_checkboxes))
        self.apply_filter()

    def set_color(self, name, hex_color, component=False):
        buttons = self.component_color_buttons if component else self.union_color_buttons
        button = buttons.get(name)
        if button is not None:
            set_swatch_color(button, hex_color)

    def apply_filter(self, *_):
        text = self.filter_edit.text().strip().lower()
        for checkboxes, buttons, section_visible in (
            (self.union_checkboxes, self.union_color_buttons, True),
            (self.component_checkboxes, self.component_color_buttons, self.show_components),
        ):
            for name, cb in checkboxes.items():
                match = section_visible and text in name.lower()
                cb.setVisible(match)
                button = buttons.get(name)
                if button is not None:
                    button.setVisible(match)

    def set_all_visible(self, visible):
        for cb in self.union_checkboxes.values():
            cb.setChecked(visible)
        for cb in self.component_checkboxes.values():
            cb.setChecked(visible)

    @staticmethod
    def _rebuild_rows(
        names, color_of, layout, checkboxes, color_buttons, visibility, toggled, color_requested
    ):
        clear_layout(layout)
        checkboxes.clear()
        color_buttons.clear()

        def on_toggled(name, checked):
            visibility[name] = checked
            toggled.emit(name, checked)

        for name in names:
            row = QtWidgets.QHBoxLayout()
            cb = make_checkbox(wrap_label(name), visibility.get(name, True), name)
            cb.toggled.connect(lambda checked, n=name: on_toggled(n, checked))
            row.addWidget(cb, 1)
            button = make_swatch_button(
                20,
                color_of(name) or DEFAULT_SWATCH_COLOR,
                f"Change colour for '{name}'",
                lambda n=name: color_requested.emit(n),
            )
            row.addWidget(button)
            layout.addLayout(row)
            checkboxes[name] = cb
            color_buttons[name] = button
