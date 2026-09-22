import sys
import time
from pathlib import Path
import traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import trimesh
import pygfx as gfx
from qtpy import QtWidgets, QtCore, QtGui
from rendercanvas.qt import QRenderWidget
from pygfx.utils.viewport import Viewport
from preprocess import preprocess
from signed_distance_functions import build_sdfs
from meshing import build_all_meshes, build_mesh, MESHER_CAPABILITIES, DEFAULT_BREP_DEFLECTION
from brep import build_many_brep_meshes
from bounding_box import compute_all_world_bboxes, component_dependency_signature
from gui_helpers import group_meshes_by_material, is_vacuum_material, assign_default_color
import argparse

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

# ============================================================
# Geometry generation
# ============================================================


def rebuild_mesh(
    meshes,
    name,
    union_geometries,
    world_matrices,
    sdfs,
    final_sdfs,
    res,
    clip,
    mesher,
    deflection=DEFAULT_BREP_DEFLECTION,
    world_bboxes=None,
):
    meshes[name] = build_mesh(
        union_geometries[name],
        union_geometries,
        world_matrices,
        sdfs,
        final_sdfs,
        res,
        clip,
        mesher=mesher,
        deflection=deflection,
        world_bboxes=world_bboxes,
    )
    return meshes


def compute_mesh_data(
    input_file,
    clip,
    meshes=None,
    dependencies=None,
    mesher="brep",
    force_remesh=False,
    res=64,
    verbose=True,
    group_by_material=False,
    deflection=DEFAULT_BREP_DEFLECTION,
    force_pygen=False,
):
    """Everything generate_group() needs to do that doesn't touch pygfx/Qt:
    preprocessing, SDF/BREP mesh building (with dependency-based incremental
    rebuild), and material grouping/vacuum tagging. Returns plain,
    picklable data (trimesh objects + dicts), so this is safe to run in a
    worker *process* rather than just a thread - some of the geometry
    kernel calls it makes (OpenCASCADE, via pythonocc-core) don't release
    the GIL, so running them on a QThread still freezes the GUI."""
    instr, world_matrices, union_geometries = preprocess(
        input_file,
        verbose=False,
        force_pygen=force_pygen,
    )
    world_bboxes = compute_all_world_bboxes(union_geometries, world_matrices)
    # The "brep" mesher never touches sdfs/final_sdfs, so skip building them.
    if mesher == "brep":
        final_sdfs, sdfs = {}, {}
    else:
        print("Building sdfs")
        final_sdfs, sdfs = build_sdfs(
            union_geometries, world_matrices, clip, world_bboxes=world_bboxes
        )
    print("Building meshes")

    new_dependencies = {
        name: component_dependency_signature(name, union_geometries, world_bboxes)
        for name in union_geometries
    }

    if meshes is None or force_remesh:
        meshes = build_all_meshes(
            union_geometries,
            world_matrices,
            sdfs,
            final_sdfs,
            res,
            clip,
            export=False,
            verbose=False,
            mesher=mesher,
            deflection=deflection,
            world_bboxes=world_bboxes,
        )
    else:
        changed_names = [
            name for name in new_dependencies
            if new_dependencies[name] != (dependencies or {}).get(name)
        ]
        if verbose:
            for name in changed_names:
                print(f"Rebuilding {name}")
        if mesher == "brep":
            # Each changed component's boolean-cut chain is independent -
            # fan them out across worker processes instead of rebuilding
            # one at a time (see build_many_brep_meshes).
            meshes.update(
                build_many_brep_meshes(
                    changed_names, union_geometries, world_matrices, clip,
                    verbose, deflection=deflection, world_bboxes=world_bboxes,
                )
            )
        else:
            for name in changed_names:
                meshes = rebuild_mesh(
                    meshes,
                    name,
                    union_geometries,
                    world_matrices,
                    sdfs,
                    final_sdfs,
                    res,
                    clip,
                    mesher,
                    deflection=deflection,
                    world_bboxes=world_bboxes,
                )
    dependencies = new_dependencies

    # One entry per component, or per material if grouped (trimesh
    # concatenation, not a boolean fusion). Vacuum is not filtered here -
    # Viewer.apply_geometry_visibility hides it client-side instead.
    if group_by_material:
        render_meshes = group_meshes_by_material(union_geometries, meshes)
        geometry_is_vacuum = {material: is_vacuum_material(material) for material in render_meshes}
    else:
        render_meshes = {
            name: meshes[name]
            for name in union_geometries
            if meshes.get(name) is not None
        }
        geometry_is_vacuum = {
            name: is_vacuum_material(getattr(union_geometries[name], "material_string", None))
            for name in render_meshes
        }

    return (render_meshes, geometry_is_vacuum, meshes, dependencies)


def build_gfx_group(render_meshes, geometry_is_vacuum, colors):
    """Wrap compute_mesh_data()'s plain trimesh output into pygfx objects.
    Cheap (no geometry kernel calls) - safe to run on the GUI thread or a
    plain QThread."""
    print("Defining group")
    group = gfx.Group()
    group.geometry_meshes = {}
    group.geometry_is_vacuum = geometry_is_vacuum
    group.geometry_trimeshes = render_meshes

    for key, mesh in render_meshes.items():
        assign_default_color(colors, key)

        gfx_mesh = gfx.Mesh(
            gfx.geometry_from_trimesh(mesh),
            gfx.MeshStandardMaterial(
                color=colors[key],
                metalness=0,
                roughness=0.8,
            ),
        )

        group.add(gfx_mesh)
        group.geometry_meshes[key] = gfx_mesh

    return group


def generate_group(
    input_file,
    clip,
    colors={},
    meshes=None,
    dependencies=None,
    mesher="brep",
    force_remesh=False,
    res=64,
    verbose=True,
    group_by_material=False,
    deflection=DEFAULT_BREP_DEFLECTION,
    force_pygen=False,
):
    render_meshes, geometry_is_vacuum, meshes, dependencies = compute_mesh_data(
        input_file,
        clip,
        meshes=meshes,
        dependencies=dependencies,
        mesher=mesher,
        force_remesh=force_remesh,
        res=res,
        verbose=verbose,
        group_by_material=group_by_material,
        deflection=deflection,
        force_pygen=force_pygen,
    )
    group = build_gfx_group(render_meshes, geometry_is_vacuum, colors)
    return (group, meshes, dependencies)


def make_coordinate_axes(
    length=10.0,
    tick_spacing=1.0,
    tick_size=0.1,
    show_labels=True,
):
    group = gfx.Group()

    axes = [
        ("X", (1, 0, 0), np.array([1, 0, 0])),
        ("Y", (0, 1, 0), np.array([0, 1, 0])),
        ("Z", (0, 0, 1), np.array([0, 0, 1])),
    ]

    for label, color, axis in axes:
        # main axis line
        positions = np.array([[0, 0, 0], axis * length], dtype=np.float32)
        line = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(thickness=2.0, color=color),
        )
        group.add(line)

        # ticks
        n_ticks = int(length / tick_spacing)
        for i in range(1, n_ticks + 1):
            p = axis * i * tick_spacing
            perp = np.array([0, 1, 0]) if axis[0] else np.array([1, 0, 0])

            tick = gfx.Line(
                gfx.Geometry(
                    positions=np.array(
                        [p - perp * tick_size, p + perp * tick_size],
                        dtype=np.float32,
                    )
                ),
                gfx.LineMaterial(thickness=1.0, color=color),
            )
            group.add(tick)

        # label
        if show_labels:
            text = gfx.Text(
                text=label,
                font_size=0.5,
                material=gfx.materials.TextMaterial(color=color),
            )
            text.local.position = axis * (length + tick_size * 4)
            group.add(text)

    return group


# ============================================================
# Camera fitting
# ============================================================


def fit_camera_to_scene(camera, controller, scene, scale=2.0):
    print(scene)
    bbox = scene.get_world_bounding_box()

    if bbox is None:
        return
    bmin = bbox[0]
    bmax = bbox[1]
    center = (bmin + bmax) / 2.0
    extent = bmax - bmin
    radius = np.linalg.norm(extent) * 0.5
    direction = np.array([1.0, 1.0, 0.7])
    direction /= np.linalg.norm(direction)
    distance = radius * scale
    position = center + direction * distance
    camera.local.position = position
    camera.look_at(center)
    camera.depth_range = (
        max(0.01, distance - radius * 4),
        1e6,
    )
    controller.target = center


def recentre_controller(controller, group):
    """Move the orbit pivot to the group's centre, without touching the
    camera's position or zoom."""
    if group is None:
        return
    bbox = group.get_world_bounding_box()
    if bbox is None:
        return
    controller.target = (bbox[0] + bbox[1]) / 2.0


# ============================================================
# Background mesh building
# ============================================================


class MeshBuildWorker(QtCore.QObject):
    """Off-loads a mesh rebuild so the GUI thread stays responsive.

    The actual geometry-kernel work (compute_mesh_data) runs in a worker
    *process*, not just this QThread: pythonocc-core's OpenCASCADE calls
    (used by the "brep" mesher) don't release the GIL, so a long boolean
    operation running on a plain QThread would still starve the main
    thread's event loop and freeze the window. Blocking here on
    future.result() is safe because waiting on inter-process I/O releases
    the GIL, unlike the geometry-kernel call itself. Building the pygfx
    objects from the returned trimesh data is cheap and stays on this
    thread.
    """

    finished = QtCore.Signal(object, object, object)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        process_pool,
        input_file,
        clip,
        colors,
        meshes,
        dependencies,
        mesher,
        res,
        force_remesh,
        group_by_material,
        deflection,
        force_pygen,
    ):
        super().__init__()
        self.process_pool = process_pool
        self.input_file = input_file
        self.clip = clip
        self.colors = colors
        self.meshes = meshes
        self.dependencies = dependencies
        self.mesher = mesher
        self.res = res
        self.force_remesh = force_remesh
        self.group_by_material = group_by_material
        self.deflection = deflection
        self.force_pygen = force_pygen

    def run(self):
        try:
            future = self.process_pool.submit(
                compute_mesh_data,
                self.input_file,
                self.clip,
                meshes=self.meshes,
                dependencies=self.dependencies,
                mesher=self.mesher,
                force_remesh=self.force_remesh,
                res=self.res,
                group_by_material=self.group_by_material,
                deflection=self.deflection,
                force_pygen=self.force_pygen,
            )
            render_meshes, geometry_is_vacuum, meshes, dependencies = future.result()
            group = build_gfx_group(render_meshes, geometry_is_vacuum, self.colors)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(group, meshes, dependencies)


# ============================================================
# Main window
# ============================================================


class Viewer(QtWidgets.QMainWindow):
    def __init__(self, input_file=None):
        super().__init__()
        self.setWindowTitle("Union Viewer")
        self.resize(1400, 900)
        self.colors = {}
        self.input_file = input_file
        self.start_input = bool(input_file)
        self.last_mtime = None
        self.current_group = None
        self.dependencies = None
        self.meshes = None
        self.mesher = "brep"

        # ----------------------------------------------------
        # Async mesh rebuild state
        # ----------------------------------------------------
        # Geometry-kernel work runs in this pool, not just a QThread - see
        # MeshBuildWorker's docstring for why (pythonocc doesn't release
        # the GIL, so a thread alone still freezes the GUI).
        self._mesh_process_pool = ProcessPoolExecutor(max_workers=1)
        self._reload_busy = False
        self._reload_pending = False
        self._reload_pending_force = False
        self._reload_thread = None
        self._reload_worker = None
        self._pending_fit_camera = False

        # ----------------------------------------------------
        # Render widget
        # ----------------------------------------------------
        self.canvas = QRenderWidget()
        self.setCentralWidget(self.canvas)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        # ----------------------------------------------------
        # Scene
        # ----------------------------------------------------
        self.scene = gfx.Scene()

        # soft global light
        ambient = gfx.AmbientLight(intensity=0.6)
        self.scene.add(ambient)
        # main light (like sun)
        key = gfx.DirectionalLight(intensity=1.5)
        key.local.position = (1, 1, 1)
        key.look_at((0, 0, 0))
        self.scene.add(key)

        # main light (like sun)
        key2 = gfx.DirectionalLight(intensity=1.5)
        key2.local.position = (-1, -1, -1)
        key2.look_at((0, 0, 0))
        self.scene.add(key2)
        # optional fill light (soft opposite side)
        fill = gfx.DirectionalLight(intensity=0.5)
        fill.local.position = (-1, 1, -1)
        fill.look_at((0, 0, 0))
        self.scene.add(fill)
        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------
        self.camera = gfx.PerspectiveCamera(35)
        self.camera.local.position = (0, 1, 10)
        self.camera.look_at((0, 0, 0))
        self.controller = gfx.OrbitController(
            self.camera,
            register_events=self.renderer,
        )
        self.controller.target = (0, 0, 0)
        self.scene.add(make_coordinate_axes(length=1000, tick_spacing=1000))

        self.grid = gfx.GridHelper(size=100, divisions=100, thickness=1)
        self.scene.add(self.grid)

        self.gizmo_viewport = Viewport(self.renderer, (0, 0, 120, 120))
        self.gizmo_scene = gfx.Scene()
        self.gizmo = make_coordinate_axes(length=1.0, tick_spacing=0.5, tick_size=0.05)
        self.gizmo_scene.add(self.gizmo)
        self.gizmo_camera = gfx.PerspectiveCamera(50, 1)
        self.gizmo_camera.local.position = (0, 0, 4)

        # ----------------------------------------------------
        # Open file shortcut (the "Open File..." button lives in the
        # Settings dock; this just keeps Ctrl+O working)
        # ----------------------------------------------------
        self.open_file_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Open, self
        )
        self.open_file_shortcut.activated.connect(self.open_file)
        # ----------------------------------------------------
        # Loading indicator (status bar)
        # ----------------------------------------------------
        self._loading_base_pixmap = self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_BrowserReload
        ).pixmap(16, 16)
        self._loading_spin_angle = 0
        # A rotated 16x16 pixmap's bounding box grows to ~22x22 for any
        # angle that isn't a multiple of 90 degrees (QPixmap.transformed()
        # enlarges the pixmap to fit the rotated content). Painting into a
        # fixed-size canvas instead of using that pixmap's own size keeps
        # the label's geometry constant every tick - otherwise the label
        # (and the status bar layout around it) resizes ~17 times/sec as
        # the icon spins, which is what actually caused the flicker.
        self._loading_icon_size = 24

        self.loading_icon_label = QtWidgets.QLabel()
        self.loading_icon_label.setFixedSize(
            self._loading_icon_size, self._loading_icon_size
        )
        self.loading_icon_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.loading_icon_label.setPixmap(self._loading_base_pixmap)
        self._loading_start_time = None
        self.loading_text_label = QtWidgets.QLabel("Building meshes...")
        # Fixed width sized for the longest plausible elapsed time, so the
        # growing digit count doesn't resize the label (and reflow the
        # status bar) every tick the way the un-fixed-size spinner icon
        # used to - see _spin_loading_icon()'s canvas-painting comment.
        self.loading_text_label.setFixedWidth(
            self.loading_text_label.fontMetrics().horizontalAdvance(
                "Building meshes for 9999.99 seconds"
            )
        )

        self.statusBar().addWidget(self.loading_icon_label)
        self.statusBar().addWidget(self.loading_text_label)
        self.loading_icon_label.hide()
        self.loading_text_label.hide()

        self.loading_spin_timer = QtCore.QTimer(self)
        self.loading_spin_timer.setInterval(60)
        self.loading_spin_timer.timeout.connect(self._spin_loading_icon)
        # ----------------------------------------------------
        # Render timer
        # ----------------------------------------------------
        self.render_timer = QtCore.QTimer()
        self.render_timer.timeout.connect(self.animate)
        self.render_timer.start(16)
        # ----------------------------------------------------
        # File watcher timer
        # ----------------------------------------------------
        self.watch_timer = QtCore.QTimer()
        self.watch_timer.timeout.connect(self.check_file_update)
        self.watch_timer.start(100)

        # ----------------------------------------------------
        # Clipping state
        # ----------------------------------------------------

        self.clip_enable = False
        self.clip_axis = "Z"
        self.clip_mode = "Above"
        self.clip_position = 0.0
        self.clip = {
            "enable": self.clip_enable,
            "axis": self.clip_axis,
            "mode": self.clip_mode,
            "position": self.clip_position,
        }

        # Docks are added in this order (Settings, Mesher options, Clipping,
        # Visible geometries) so they stack top-to-bottom in the left panel.

        # ----------------------------------------------------
        # Settings dock
        # ----------------------------------------------------

        settings_dock, settings_layout = self._add_dock("Settings")

        self.open_file_button = QtWidgets.QPushButton("Open File...")
        self.open_file_button.setToolTip("Open a McStas instrument (shortcut: Ctrl+O)")
        self.open_file_button.clicked.connect(self.open_file)
        settings_layout.addWidget(self.open_file_button)

        self.material_group_checkbox = QtWidgets.QCheckBox("Group by material")
        self.material_group_checkbox.setChecked(True)
        self.material_group_checkbox.setToolTip(
            "Combine components that share a material into a single "
            "rendered object (trimesh concatenation, not a boolean fusion)."
        )
        settings_layout.addWidget(self.material_group_checkbox)

        self.vacuum_checkbox = QtWidgets.QCheckBox("Hide vacuum")
        self.vacuum_checkbox.setChecked(True)
        self.vacuum_checkbox.setToolTip(
            "Hides volumes whose material is 'vacuum'/'Vacuum' or "
            "'exit'/'Exit' (McStas treats 'exit' as vacuum too)."
        )
        settings_layout.addWidget(self.vacuum_checkbox)

        self.pygen_checkbox = QtWidgets.QCheckBox("Force mcstas-pygen preprocessing")
        self.pygen_checkbox.setChecked(False)
        self.pygen_checkbox.setToolTip(
            "Translate a .instr input through the real McStas front-end "
            "(mcstas-pygen) instead of mcstasscript's lightweight .instr "
            "reader. mcstas-pygen already runs automatically whenever that "
            "lightweight reader raises an error - this instead forces it "
            "for every load, for instruments the lightweight reader "
            "mis-parses without raising an error. Requires mcstas-pygen "
            "on PATH (ships with the McStas install)."
        )
        settings_layout.addWidget(self.pygen_checkbox)

        self.reset_view_button = QtWidgets.QPushButton("Reset view")
        self.reset_view_button.setToolTip("Refit the camera (shortcut: R)")
        self.reset_view_button.clicked.connect(self.reset_view)
        settings_layout.addWidget(self.reset_view_button)
        self.reset_view_shortcut = QtGui.QShortcut(QtGui.QKeySequence("R"), self)
        self.reset_view_shortcut.activated.connect(self.reset_view)

        self.export_stl_button = QtWidgets.QPushButton("Export STL...")
        self.export_stl_button.setToolTip(
            "Export the currently visible meshes as a single .stl file."
        )
        self.export_stl_button.clicked.connect(self.export_stl)
        settings_layout.addWidget(self.export_stl_button)

        settings_layout.addStretch()
        settings_dock.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )

        # ----------------------------------------------------
        # Mesher options dock
        # ----------------------------------------------------

        mesher_dock, mesher_options_layout = self._add_dock("Mesher Options")

        mesher_selector_layout = QtWidgets.QHBoxLayout()
        mesher_selector_layout.addWidget(QtWidgets.QLabel("Mesher"))
        self.mesher_box = QtWidgets.QComboBox()
        for key in MESHER_KEYS:
            label = key
            if not MESHER_CAPABILITIES[key]["incremental_rebuild"]:
                label += " (full rebuild only)"
            self.mesher_box.addItem(label, key)
        self.mesher_box.setCurrentIndex(self.mesher_box.findData(self.mesher))
        mesher_selector_layout.addWidget(self.mesher_box)
        mesher_options_layout.addLayout(mesher_selector_layout)

        self.mesher_description_label = QtWidgets.QLabel(
            MESHER_DESCRIPTIONS.get(self.mesher, "")
        )
        self.mesher_description_label.setWordWrap(True)
        self.mesher_description_label.setStyleSheet("color: gray; font-style: italic;")
        mesher_options_layout.addWidget(self.mesher_description_label)

        resolution_layout = QtWidgets.QHBoxLayout()
        self.resolution_label = QtWidgets.QLabel("Resolution")
        self.res_val = QtWidgets.QComboBox()
        self.res_val.addItem("16", 16)
        self.res_val.addItem("32", 32)
        self.res_val.addItem("64", 64)
        self.res_val.addItem("128", 128)
        self.res_val.addItem("256", 256)
        self.res_val.addItem("512", 512)
        self.res_val.setCurrentIndex(2)
        resolution_layout.addWidget(self.resolution_label)
        resolution_layout.addWidget(self.res_val)
        mesher_options_layout.addLayout(resolution_layout)

        deflection_layout = QtWidgets.QHBoxLayout()
        self.deflection_label = QtWidgets.QLabel("Surface deflection")
        self.deflection_val = QtWidgets.QDoubleSpinBox()
        self.deflection_val.setDecimals(4)
        self.deflection_val.setRange(0.0001, 10.0)
        self.deflection_val.setSingleStep(0.001)
        self.deflection_val.setValue(0.01)
        deflection_layout.addWidget(self.deflection_label)
        deflection_layout.addWidget(self.deflection_val)
        mesher_options_layout.addLayout(deflection_layout)

        mesher_options_layout.addStretch()
        mesher_dock.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )

        # ----------------------------------------------------
        # Clipping dock
        # ----------------------------------------------------

        clipping_dock, clipping_layout = self._add_dock("Clipping")

        self.clip_checkbox = QtWidgets.QCheckBox("Enable clipping")
        clipping_layout.addWidget(self.clip_checkbox)

        axis_layout = QtWidgets.QHBoxLayout()
        axis_layout.addWidget(QtWidgets.QLabel("Axis"))
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItems(["X", "Y", "Z"])
        axis_layout.addWidget(self.axis_combo)
        clipping_layout.addLayout(axis_layout)

        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("Mode"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Above", "Below"])
        mode_layout.addWidget(self.mode_combo)
        clipping_layout.addLayout(mode_layout)

        position_layout = QtWidgets.QHBoxLayout()
        position_layout.addWidget(QtWidgets.QLabel("Position"))
        self.slice_val = QtWidgets.QDoubleSpinBox()
        self.slice_val.setDecimals(5)
        self.slice_val.setRange(-1e6, 1e6)
        self.slice_val.setSingleStep(0.01)
        self.slice_val.setValue(0.0)
        position_layout.addWidget(self.slice_val)
        clipping_layout.addLayout(position_layout)

        clipping_layout.addStretch()
        clipping_dock.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )

        # ----------------------------------------------------
        # Geometry visibility dock
        # ----------------------------------------------------

        self.geometry_checkboxes = {}
        self.geometry_color_buttons = {}
        self.geometry_visibility = {}

        self.geometry_dock = QtWidgets.QDockWidget("Visible Geometries", self)
        geometry_dock_widget = QtWidgets.QWidget()
        geometry_dock_layout = QtWidgets.QVBoxLayout(geometry_dock_widget)

        self.geometry_filter = QtWidgets.QLineEdit()
        self.geometry_filter.setPlaceholderText("Filter...")
        self.geometry_filter.setClearButtonEnabled(True)
        self.geometry_filter.textChanged.connect(self.on_geometry_filter_changed)
        geometry_dock_layout.addWidget(self.geometry_filter)

        show_hide_layout = QtWidgets.QHBoxLayout()
        self.show_all_button = QtWidgets.QPushButton("Show all")
        self.hide_all_button = QtWidgets.QPushButton("Hide all")
        self.show_all_button.clicked.connect(
            lambda: self.set_all_geometry_visibility(True)
        )
        self.hide_all_button.clicked.connect(
            lambda: self.set_all_geometry_visibility(False)
        )
        show_hide_layout.addWidget(self.show_all_button)
        show_hide_layout.addWidget(self.hide_all_button)
        geometry_dock_layout.addLayout(show_hide_layout)

        self.geometry_widget = QtWidgets.QWidget()
        self.geometry_layout = QtWidgets.QVBoxLayout(self.geometry_widget)
        self.geometry_layout.addStretch()

        geometry_scroll = QtWidgets.QScrollArea()
        geometry_scroll.setWidgetResizable(True)
        geometry_scroll.setWidget(self.geometry_widget)
        geometry_dock_layout.addWidget(geometry_scroll)

        self.geometry_dock.setWidget(geometry_dock_widget)
        geometry_dock_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
            self.geometry_dock,
        )

        # ----------------------------------------------------
        # Signals
        # ----------------------------------------------------

        self.clip_checkbox.stateChanged.connect(self.on_clip_changed)
        self.material_group_checkbox.stateChanged.connect(self.on_material_group_changed)
        self.vacuum_checkbox.stateChanged.connect(self.on_vacuum_changed)
        self.mesher_box.currentIndexChanged.connect(self.on_mesher_changed)
        self.pygen_checkbox.stateChanged.connect(self.on_pygen_changed)
        self.axis_combo.currentTextChanged.connect(self.on_clip_changed)
        self.mode_combo.currentTextChanged.connect(self.on_clip_changed)
        self.slice_val.valueChanged.connect(self.on_clip_changed)
        self.res_val.currentIndexChanged.connect(self.on_res_changed)
        self.deflection_val.valueChanged.connect(self.on_deflection_changed)

        self.update_mesher_capability_ui()

    def _add_dock(self, title):
        """Create a left-docked QDockWidget titled `title` and return
        (dock, layout) for the caller to populate."""
        dock = QtWidgets.QDockWidget(title, self)
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        dock.setWidget(widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        return dock, layout

    # ========================================================
    # Open file dialog
    # ========================================================

    def open_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open McStas File",
            "",
            "McStas Files (*.instr *.py);;All Files (*)",
        )
        if not filename:
            return
        self.input_file = filename
        self.last_mtime = Path(filename).stat().st_mtime
        self._pending_fit_camera = True
        self.reload_meshes()

    def apply_geometry_visibility(self):
        """Recompute mesh.visible for every row from the per-row checkbox
        state and the "Hide vacuum" filter. Never rebuilds geometry - safe
        to call on its own whenever either input changes."""
        if self.current_group is None:
            return
        hide_vacuum = self.vacuum_checkbox.isChecked()
        is_vacuum = self.current_group.geometry_is_vacuum
        for key, mesh in self.current_group.geometry_meshes.items():
            visible = self.geometry_visibility.get(key, True)
            if hide_vacuum and is_vacuum.get(key, False):
                visible = False
            mesh.visible = visible

    def on_geometry_visibility_changed(self, name, mesh, checked):
        self.geometry_visibility[name] = checked
        self.apply_geometry_visibility()

    def _clear_geometry_layout_item(self, item):
        """Delete whatever a takeAt() handed back: a widget, or a nested
        row layout of [checkbox, colour button]."""
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            row = item.layout()
            while row.count():
                self._clear_geometry_layout_item(row.takeAt(0))

    def rebuild_geometry_panel(self):
        while self.geometry_layout.count() > 1:
            self._clear_geometry_layout_item(self.geometry_layout.takeAt(0))

        self.geometry_checkboxes.clear()
        self.geometry_color_buttons.clear()

        if self.current_group is None:
            return

        for name, mesh in self.current_group.geometry_meshes.items():
            visible = self.geometry_visibility.get(name, True)

            row = QtWidgets.QHBoxLayout()

            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(visible)
            cb.toggled.connect(
                lambda checked, n=name, m=mesh: self.on_geometry_visibility_changed(
                    n, m, checked
                )
            )
            row.addWidget(cb, 1)

            color_button = QtWidgets.QPushButton()
            color_button.setFixedSize(20, 20)
            color_button.setToolTip(f"Change colour for '{name}'")
            color_button.clicked.connect(
                lambda checked=False, n=name: self.pick_color(n)
            )
            self._set_swatch_color(color_button, self.colors.get(name, "#b6b6b6"))
            row.addWidget(color_button)

            self.geometry_layout.insertLayout(
                self.geometry_layout.count() - 1,
                row,
            )

            self.geometry_checkboxes[name] = cb
            self.geometry_color_buttons[name] = color_button

        self.on_geometry_filter_changed(self.geometry_filter.text())

    def on_geometry_filter_changed(self, text):
        text = text.strip().lower()
        for name, cb in self.geometry_checkboxes.items():
            match = text in name.lower()
            cb.setVisible(match)
            button = self.geometry_color_buttons.get(name)
            if button is not None:
                button.setVisible(match)

    def set_all_geometry_visibility(self, visible):
        for cb in self.geometry_checkboxes.values():
            cb.setChecked(visible)

    def _set_swatch_color(self, button, hex_color):
        button.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #888;")

    def pick_color(self, key):
        if self.current_group is None or key not in self.current_group.geometry_meshes:
            return
        current = QtGui.QColor(self.colors.get(key, "#b6b6b6"))
        color = QtWidgets.QColorDialog.getColor(current, self, f"Colour for '{key}'")
        if not color.isValid():
            return
        hex_color = color.name()
        self.colors[key] = hex_color
        self.current_group.geometry_meshes[key].material.color = hex_color
        button = self.geometry_color_buttons.get(key)
        if button is not None:
            self._set_swatch_color(button, hex_color)

    # ========================================================
    # Export STL
    # ========================================================

    def export_stl(self):
        if self.current_group is None or not self.current_group.geometry_meshes:
            QtWidgets.QMessageBox.warning(
                self, "Export STL", "No geometry loaded to export."
            )
            return

        trimeshes = self.current_group.geometry_trimeshes
        parts = [
            trimeshes[key]
            for key, mesh in self.current_group.geometry_meshes.items()
            if mesh.visible and trimeshes.get(key) is not None
        ]
        if not parts:
            QtWidgets.QMessageBox.warning(
                self, "Export STL", "No visible geometry to export."
            )
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export STL", "", "STL Files (*.stl)"
        )
        if not filename:
            return
        if not filename.lower().endswith(".stl"):
            filename += ".stl"

        try:
            combined = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
            combined.export(filename)
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self, "Export STL", f"Failed to export STL:\n{e}"
            )
            return

        QtWidgets.QMessageBox.information(
            self, "Export STL", f"Exported {len(parts)} mesh(es) to:\n{filename}"
        )

    # ========================================================
    # Reload geometry (asynchronous)
    # ========================================================

    def reload_meshes(self, force_reload=False):
        if self.input_file is None:
            return
        # dc has no working incremental rebuild path (build_mesh returns
        # None for it), so always fully rebuild while it's selected.
        if not MESHER_CAPABILITIES[self.mesher]["incremental_rebuild"]:
            force_reload = True

        if self._reload_busy:
            self._reload_pending = True
            self._reload_pending_force = self._reload_pending_force or force_reload
            return

        self._reload_busy = True
        self._reload_pending = False
        self._reload_pending_force = False
        self._set_loading(True)
        print("Building meshes...")

        thread = QtCore.QThread(self)
        worker = MeshBuildWorker(
            self._mesh_process_pool,
            self.input_file,
            dict(self.clip),
            self.colors,
            self.meshes,
            self.dependencies,
            self.mesher,
            self.res_val.currentData(),
            force_reload,
            self.material_group_checkbox.isChecked(),
            self.deflection_val.value(),
            self.pygen_checkbox.isChecked(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_reload_finished)
        worker.failed.connect(self._on_reload_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_reload_thread_finished)

        self._reload_thread = thread
        self._reload_worker = worker
        thread.start()

    def _on_reload_finished(self, new_group, meshes, dependencies):
        if self.current_group is not None:
            self.scene.remove(self.current_group)
        self.dependencies = dependencies
        self.meshes = meshes

        self.current_group = new_group

        self.scene.add(self.current_group)

        self.rebuild_geometry_panel()
        self.apply_geometry_visibility()
        recentre_controller(self.controller, self.current_group)
        print("Reload complete")

        if self._pending_fit_camera:
            self._pending_fit_camera = False
            fit_camera_to_scene(
                self.camera,
                self.controller,
                self.current_group,
            )

    def _on_reload_failed(self, error_text):
        print("Mesh rebuild failed:")
        print(error_text)

    def _on_reload_thread_finished(self):
        self._reload_thread = None
        self._reload_worker = None
        self._reload_busy = False
        self._set_loading(False)

        if self._reload_pending:
            pending_force = self._reload_pending_force
            self._reload_pending = False
            self._reload_pending_force = False
            self.reload_meshes(force_reload=pending_force)

    # ========================================================
    # Loading indicator
    # ========================================================

    def _spin_loading_icon(self):
        self._loading_spin_angle = (self._loading_spin_angle + 30) % 360

        size = self._loading_icon_size
        canvas = QtGui.QPixmap(size, size)
        canvas.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(canvas)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.translate(size / 2, size / 2)
        painter.rotate(self._loading_spin_angle)
        base = self._loading_base_pixmap
        painter.drawPixmap(
            QtCore.QPointF(-base.width() / 2, -base.height() / 2), base
        )
        painter.end()

        self.loading_icon_label.setPixmap(canvas)

        elapsed = time.time() - self._loading_start_time
        self.loading_text_label.setText(f"Building meshes for {elapsed:.2f} seconds")

    def _set_loading(self, is_loading):
        self.loading_icon_label.setVisible(is_loading)
        self.loading_text_label.setVisible(is_loading)
        if is_loading:
            self._loading_start_time = time.time()
            self.loading_text_label.setText("Building meshes for 0.00 seconds")
            self.loading_spin_timer.start()
        else:
            self.loading_spin_timer.stop()
            self.loading_icon_label.setPixmap(self._loading_base_pixmap)

    # ========================================================
    # Watch file changes
    # ========================================================

    def check_file_update(self):
        if self.input_file is None:
            return
        try:
            new_mtime = Path(self.input_file).stat().st_mtime
            if new_mtime != self.last_mtime:
                self.last_mtime = new_mtime
                print("File changed -> rebuilding")
                if self.start_input:
                    self._pending_fit_camera = True
                    self.start_input = False
                self.reload_meshes()

        except Exception as e:
            print(e)

    # ========================================================
    # Clip settings changed
    # ========================================================

    def on_clip_changed(self):
        self.clip["enable"] = self.clip_checkbox.isChecked()
        self.clip["axis"] = self.axis_combo.currentText()
        self.clip["mode"] = self.mode_combo.currentText()
        self.clip["position"] = self.slice_val.value()
        # Global cut, not part of any component's dependency signature.
        self.reload_meshes(force_reload=True)

    def on_mesher_changed(self):
        self.mesher = self.mesher_box.currentData()
        self.mesher_description_label.setText(
            MESHER_DESCRIPTIONS.get(self.mesher, "")
        )
        self.update_mesher_capability_ui()
        self.reload_meshes(force_reload=True)

    def on_material_group_changed(self):
        self.reload_meshes()

    def on_pygen_changed(self):
        # Switches which parser builds the McStas_instr entirely, so
        # nothing from a previous load can be trusted as unchanged.
        self.reload_meshes(force_reload=True)

    def on_vacuum_changed(self):
        self.apply_geometry_visibility()

    def on_res_changed(self):
        # Global mesher setting, not part of any dependency signature.
        self.reload_meshes(force_reload=True)

    def on_deflection_changed(self):
        self.reload_meshes(force_reload=True)

    # ========================================================
    # Mesher capability reflection
    # ========================================================

    def update_mesher_capability_ui(self):
        caps = MESHER_CAPABILITIES[self.mesher]

        res_used = caps["resolution"]
        self.res_val.setEnabled(res_used)
        self.resolution_label.setEnabled(res_used)
        tip = "" if res_used else f"Not used by the '{self.mesher}' mesher."
        self.res_val.setToolTip(tip)
        self.resolution_label.setToolTip(tip)

        deflection_used = caps["deflection"]
        self.deflection_val.setEnabled(deflection_used)
        self.deflection_label.setEnabled(deflection_used)
        tip = "" if deflection_used else f"Not used by the '{self.mesher}' mesher."
        self.deflection_val.setToolTip(tip)
        self.deflection_label.setToolTip(tip)

    # ========================================================
    # Reset view
    # ========================================================

    def reset_view(self):
        target = self.current_group if self.current_group is not None else self.scene
        fit_camera_to_scene(self.camera, self.controller, target)

    # ========================================================
    # Render loop
    # ========================================================

    def animate(self):
        self.gizmo.local.rotation = self.camera.local.rotation
        self.renderer.render(self.scene, self.camera)
        w, h = self.canvas.get_logical_size()
        s = 160
        self.gizmo_viewport.rect = (10, h - s - 10, s, s)
        self.gizmo_viewport.render(self.gizmo_scene, self.gizmo_camera)

        self.canvas.request_draw()

    # ========================================================
    # Shutdown
    # ========================================================

    def closeEvent(self, event):
        self._mesh_process_pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


# ============================================================
# Main
# ============================================================


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", help="Input mcstas file, can either be mcstasscript or mcstas"
    )
    return parser


if __name__ == "__main__":
    parser = parse()
    args = parser.parse_args()
    input_file = args.input_file
    app = QtWidgets.QApplication(sys.argv)

    app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar)

    viewer = Viewer(input_file=input_file)
    viewer.show()
    sys.exit(app.exec())
