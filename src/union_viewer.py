import argparse
import sys
from pathlib import Path
import traceback
import numpy as np
import trimesh
import pygfx as gfx
import pylinalg as la
from qtpy import QtWidgets, QtCore, QtGui
from rendercanvas.qt import QRenderWidget
from pygfx.utils.viewport import Viewport
from clipping import resolve_clip_frame
from meshing import MESHER_CAPABILITIES
from gui_helpers import (
    component_color_key,
    clip_planes,
    instrument_param_args,
    clip_mesh,
    RAY_COLOR_MODES,
    rays_reaching,
    BEFORE,
    GAP,
    cut_sides,
    cut_shift,
    cut_rays,
)
import logger_output
from pipeline import NO_CLIP, compute_counts_data, compute_mesh_data, compute_trace_data
from scene_objects import (
    GRID_SPACING,
    MAX_GRID_DIVISIONS,
    add_default_lights,
    build_component_group,
    build_counts_group,
    build_cut_marker,
    CUT_MARKER_MIN_SIZE,
    build_gfx_group,
    build_ray_group,
    build_ruler_group,
    grid_size_for_bbox,
    make_coordinate_axes,
    pick_world_point,
)
from background import JobRunner
from qt_helpers import ask_color
from camera import fit_camera_to_scene, recentre_controller, update_camera_depth_range
from widgets import CollapsibleTitleBar, LoadingIndicator
from panels import (
    ClippingPanel,
    CountsPanel,
    CutPanel,
    GeometryPanel,
    MesherPanel,
    ParametersPanel,
    RaysPanel,
    RulerPanel,
    SettingsPanel,
)

# ============================================================
# Main window
# ============================================================


class Viewer(QtWidgets.QMainWindow):
    def __init__(self, input_file=None, settings=None):
        super().__init__()
        self.settings = settings or QtCore.QSettings("unviz", "union_viewer")
        self.setWindowTitle("Union Viewer")
        self.resize(1400, 900)
        self.colors = {}
        self.input_file = input_file
        self.start_input = bool(input_file)
        self.last_mtime = None
        self.current_group = None
        self.dependencies = None
        self.meshes = None

        # Collapsed gap: (sides, shift) while active (see apply_cut), else None.
        self.cut = None
        self.cut_marker = None
        self.component_order = []
        self.geometry_members = {}

        # ----------------------------------------------------
        # Background jobs
        # ----------------------------------------------------
        # Meshing, mcrun --trace and counts loading each get their own
        # worker process, so a slow mcrun compile never delays the meshes,
        # a mesher/clip change never reruns McStas, and loading a large
        # logger file never delays either.
        self.mesh_jobs = JobRunner(
            self, self._on_reload_finished, self._on_reload_failed, self._on_mesh_jobs_idle
        )
        self.trace_jobs = JobRunner(self, self._on_trace_finished, self._on_trace_failed)
        self.counts_jobs = JobRunner(
            self, self._on_counts_finished, self._on_counts_failed, self._on_counts_jobs_idle
        )
        self._reload_pending_force = False
        self._pending_fit_camera = False

        self.trace_group = None
        self.trace_rays = None
        self.ray_group = None

        # ----------------------------------------------------
        # Detector counts
        # ----------------------------------------------------
        self.counts = {}              # {name: logger_output.LoggerCounts}
        self.counts_group = None
        self.counts_meshes = {}       # {name: gfx.Mesh}
        self.counts_textures = {}     # {name: gfx.Texture}
        self.counts_run_folder = None
        self._loading_counts_folder = None

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
        add_default_lights(self.scene)
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

        self.axes = make_coordinate_axes(length=1000, tick_spacing=1000)
        self.scene.add(self.axes)

        self.grid = None
        self._resize_grid(None)

        self.ruler_points = []
        self.ruler_group = None
        self._ruler_press = None
        self.renderer.add_event_handler(self.on_ruler_pointer, "pointer_down", "pointer_up")

        self.gizmo_viewport = Viewport(self.renderer, (0, 0, 120, 120))
        self.gizmo_scene = gfx.Scene()
        self.gizmo = make_coordinate_axes(length=1.0, tick_spacing=0.5, tick_size=0.05)
        self.gizmo_scene.add(self.gizmo)
        self.gizmo_camera = gfx.PerspectiveCamera(50, 1)
        self.gizmo_camera.local.position = (0, 0, 4)

        # ----------------------------------------------------
        # Keyboard shortcuts (Ctrl+O opens a file, R resets the view; the
        # matching buttons live in the Settings dock)
        # ----------------------------------------------------
        self.open_file_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Open, self
        )
        self.open_file_shortcut.activated.connect(self.open_file)
        self.reset_view_shortcut = QtGui.QShortcut(QtGui.QKeySequence("R"), self)
        self.reset_view_shortcut.activated.connect(self.reset_view)

        self.loading_indicator = LoadingIndicator(self)
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

        self.world_matrices = {}

        # ----------------------------------------------------
        # Docks, added in this order so they stack top-to-bottom in the
        # left panel
        # ----------------------------------------------------
        self.settings_panel = SettingsPanel()
        self.params_panel = ParametersPanel()
        self.mesher_panel = MesherPanel()
        self.clipping_panel = ClippingPanel()
        self.cut_panel = CutPanel()
        self.ruler_panel = RulerPanel()
        self.rays_panel = RaysPanel()
        self.counts_panel = CountsPanel()
        self.geometry_panel = GeometryPanel()

        self._add_dock("Settings", self.settings_panel, collapsed=False)
        self._add_dock("Instrument Parameters", self.params_panel, stretch=False)
        self._add_dock("Mesher Options", self.mesher_panel)
        self._add_dock("Clipping", self.clipping_panel)
        self._add_dock("Collapse Gap", self.cut_panel)
        self._add_dock("Ruler", self.ruler_panel)
        self._add_dock("Neutron Rays", self.rays_panel)
        self._add_dock("Detector Counts", self.counts_panel)
        self.geometry_dock = self._add_dock("Visible Geometries", self.geometry_panel)
        QtCore.QTimer.singleShot(0, self.rebalance_docks)

        # ----------------------------------------------------
        # Signals
        # ----------------------------------------------------

        settings = self.settings_panel
        settings.open_file_button.clicked.connect(self.open_file)
        settings.reset_view_button.clicked.connect(self.reset_view)
        settings.fit_instrument_button.clicked.connect(self.fit_whole_instrument)
        settings.export_stl_button.clicked.connect(self.export_stl)
        settings.material_group_checkbox.stateChanged.connect(self.on_material_group_changed)
        settings.vacuum_checkbox.stateChanged.connect(self.on_vacuum_changed)
        settings.pygen_checkbox.stateChanged.connect(self.on_pygen_changed)
        settings.components_checkbox.stateChanged.connect(self.on_components_changed)
        settings.arms_checkbox.stateChanged.connect(self.apply_component_visibility)
        settings.rays_checkbox.stateChanged.connect(self.on_rays_changed)
        settings.edges_checkbox.stateChanged.connect(self.apply_edges_visibility)
        settings.grid_checkbox.toggled.connect(self.apply_grid_visibility)
        for checkbox, attr in (
            (settings.gizmo_checkbox, "gizmo"),
            (settings.axes_checkbox, "axes"),
        ):
            checkbox.toggled.connect(
                lambda on, attr=attr: setattr(getattr(self, attr), "visible", on)
            )

        self.params_panel.edited.connect(self.on_params_edited)
        self.mesher_panel.changed.connect(self.on_mesher_options_changed)
        self.clipping_panel.changed.connect(self.apply_clipping)
        self.cut_panel.changed.connect(self.apply_cut)
        self.ruler_panel.clear_button.clicked.connect(self.clear_ruler)
        self.ruler_panel.points_edited.connect(self.on_ruler_points_edited)
        self.ruler_panel.color_changed.connect(lambda _color: self.on_ruler_points_edited())

        rays = self.rays_panel
        rays.rerun_requested.connect(self.rerun_rays)
        rays.style_changed.connect(self.rebuild_ray_group)
        rays.visibility_changed.connect(self.apply_ray_visibility)
        rays.color_changed.connect(self.on_ray_color_changed)
        rays.marker_size_changed.connect(self.on_ray_marker_size_changed)

        counts = self.counts_panel
        counts.load_button.clicked.connect(self.load_counts_folder)
        counts.show_checkbox.stateChanged.connect(self.apply_counts_visibility)
        counts.min_spin.valueChanged.connect(self.on_counts_range_changed)
        counts.max_spin.valueChanged.connect(self.on_counts_range_changed)
        counts.auto_range_button.clicked.connect(self.reset_counts_range)
        counts.detector_toggled.connect(self.on_counts_visibility_changed)

        geometry = self.geometry_panel
        geometry.union_toggled.connect(self.on_geometry_visibility_changed)
        geometry.component_toggled.connect(self.on_component_visibility_changed)
        geometry.union_color_requested.connect(self.pick_color)
        geometry.component_color_requested.connect(self.pick_component_color)

        geometry.set_components_shown(settings.components_checkbox.isChecked())
        rays.setEnabled(settings.rays_checkbox.isChecked())

    def _add_dock(self, title, panel, collapsed=True, stretch=True):
        """Put panel in a collapsible, left-docked QDockWidget titled
        `title` (collapsed at first unless collapsed=False) that only takes
        the height its contents need. With stretch=True, extra space goes
        below the panel. Returns the dock."""
        dock = QtWidgets.QDockWidget(title, self)
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addWidget(panel)
        if stretch:
            layout.addStretch()
        dock.setWidget(widget)
        dock.setTitleBarWidget(
            CollapsibleTitleBar(dock, self.settings, self.rebalance_docks, collapsed)
        )
        dock.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        return dock

    def rebalance_docks(self):
        """Hand the height collapsed docks free up to the expanded ones:
        each expanded dock gets its natural height and the Visible
        Geometries dock takes whatever is left."""
        docks = [
            d for d in self.findChildren(QtWidgets.QDockWidget)
            if d.isVisibleTo(self)
            and not d.isFloating()
            and self.dockWidgetArea(d) == QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
        ]
        if not docks:
            return
        sizes = [
            d.titleBarWidget().sizeHint().height() if d.titleBarWidget().collapsed
            else d.sizeHint().height()
            for d in docks
        ]
        if self.geometry_dock in docks and not self.geometry_dock.titleBarWidget().collapsed:
            sizes[docks.index(self.geometry_dock)] = self.height()
        self.resizeDocks(docks, sizes, QtCore.Qt.Orientation.Vertical)

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
        self.reload_trace()

    def apply_geometry_visibility(self):
        """Recompute mesh.visible for every row from the per-row checkbox
        state and the "Hide vacuum" filter. Never rebuilds geometry - safe
        to call on its own whenever either input changes."""
        if self.current_group is None:
            return
        hide_vacuum = self.settings_panel.vacuum_checkbox.isChecked()
        is_vacuum = self.current_group.geometry_is_vacuum
        for key, mesh in self.current_group.geometry_meshes.items():
            visible = self.geometry_panel.union_visibility.get(key, True)
            if hide_vacuum and is_vacuum.get(key, False):
                visible = False
            if self.in_gap(self.geometry_owner(key)):
                visible = False
            mesh.visible = visible

    def apply_edges_visibility(self):
        if self.current_group is None:
            return
        show = self.settings_panel.edges_checkbox.isChecked()
        for edges in self.current_group.edge_lines:
            edges.visible = show

    def on_geometry_visibility_changed(self, name, checked):
        self.apply_geometry_visibility()

    def rebuild_geometry_panel(self):
        names = [] if self.current_group is None else list(self.current_group.geometry_meshes)
        self.geometry_panel.set_union_rows(names, self.colors.get)

    def rebuild_component_panel(self):
        names = [] if self.trace_group is None else list(self.trace_group.component_objects)
        self.geometry_panel.set_component_rows(
            names, lambda name: self.colors.get(component_color_key(name))
        )

    def on_component_visibility_changed(self, name, checked):
        self.apply_component_visibility()

    def apply_component_visibility(self):
        if self.trace_group is None:
            return
        self.trace_group.visible = self.settings_panel.components_checkbox.isChecked()
        show_arms = self.settings_panel.arms_checkbox.isChecked()
        for name, obj in self.trace_group.component_objects.items():
            visible = self.geometry_panel.component_visibility.get(name, True)
            if self.trace_group.component_is_arm[name] and not show_arms:
                visible = False
            if self.in_gap(name):
                visible = False
            obj.visible = visible

    def apply_clipping(self):
        planes = clip_planes(self.resolved_clip())
        for group in (self.current_group, self.trace_group, self.ray_group, self.counts_group):
            if group is None:
                continue
            for obj in group.iter():
                if getattr(obj, "material", None) is not None:
                    # Planes are in world space: shift each one with its
                    # object, so a component moved by a collapsed gap is cut
                    # where its true geometry is.
                    # ponytail: rays before a collapsed gap are moved per
                    # point, so they are still cut by the unshifted plane.
                    offset = np.asarray(obj.world.position)
                    obj.material.clipping_planes = [
                        (a, b, c, d - float(np.dot((a, b, c), offset)))
                        for a, b, c, d in planes
                    ]

    def _pick_panel_color(self, name, color_key, objects, component=False):
        """Ask for a new colour for the Visible Geometries row `name`, then
        store it in self.colors[color_key] and apply it to objects'
        materials and the row's swatch button."""
        hex_color = ask_color(self, self.colors.get(color_key, "#b6b6b6"), f"Colour for '{name}'")
        if hex_color is None:
            return
        self.colors[color_key] = hex_color
        for obj in objects:
            obj.material.color = hex_color
        self.geometry_panel.set_color(name, hex_color, component=component)

    def pick_component_color(self, name):
        if self.trace_group is None or name not in self.trace_group.component_objects:
            return
        self._pick_panel_color(
            name,
            component_color_key(name),
            self.trace_group.component_objects[name].children,
            component=True,
        )

    def pick_color(self, key):
        if self.current_group is None or key not in self.current_group.geometry_meshes:
            return
        self._pick_panel_color(key, key, [self.current_group.geometry_meshes[key]])

    # ========================================================
    # Instrument parameters and clip frame
    # ========================================================

    def resolved_clip(self):
        return resolve_clip_frame(self.clipping_panel.clip, self.world_matrices)

    def on_params_edited(self):
        self.reload_meshes()
        self.reload_trace()

    # ========================================================
    # Export STL
    # ========================================================

    def _export_parts(self):
        """(Union meshes, McStas component meshes, names that failed) for
        what is currently visible, each cut by the clip plane as shown."""
        clip = self.resolved_clip()
        failed = []

        def clipped(name, mesh):
            try:
                return clip_mesh(mesh, clip)
            except Exception:
                traceback.print_exc()
                failed.append(name)
                return None

        union_parts = []
        if self.current_group is not None:
            trimeshes = self.current_group.geometry_trimeshes
            for key, obj in self.current_group.geometry_meshes.items():
                if obj.visible and trimeshes.get(key) is not None:
                    mesh = clipped(key, trimeshes[key])
                    if mesh is not None:
                        union_parts.append(mesh)
        component_parts = []
        if self.trace_group is not None and self.trace_group.visible:
            for name, obj in self.trace_group.component_objects.items():
                if not obj.visible:
                    continue
                mesh = self.trace_group.components[name].export_mesh()
                if mesh is not None:
                    mesh = clipped(name, mesh)
                if mesh is not None:
                    component_parts.append(mesh)
        return union_parts, component_parts, failed

    def export_stl(self):
        if self.current_group is None and self.trace_group is None:
            QtWidgets.QMessageBox.warning(
                self, "Export STL", "No geometry loaded to export."
            )
            return

        try:
            union_parts, component_parts, failed = self._export_parts()
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self, "Export STL", f"Failed to prepare the export:\n{e}"
            )
            return
        parts = union_parts + component_parts
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
            self,
            "Export STL",
            f"Exported {len(union_parts)} Union mesh(es) and "
            f"{len(component_parts)} McStas component(s) to:\n{filename}"
            + (f"\n\nLeft out (could not be clipped): {', '.join(failed)}" if failed else ""),
        )

    # ========================================================
    # Reload geometry (asynchronous)
    # ========================================================

    def reload_meshes(self, force_reload=False):
        if self.input_file is None:
            return
        # dc has no working incremental rebuild path (build_mesh returns
        # None for it), so always fully rebuild while it's selected.
        if not MESHER_CAPABILITIES[self.mesher_panel.mesher()]["incremental_rebuild"]:
            force_reload = True

        if self.mesh_jobs.busy:
            self._reload_pending_force = self._reload_pending_force or force_reload
            self.mesh_jobs.run_when_idle(self._reload_pending_meshes)
            return

        self._reload_pending_force = False
        self.loading_indicator.set_loading(True)
        print("Building meshes...")

        colors = self.colors
        self.mesh_jobs.start(
            compute_mesh_data,
            self.input_file,
            NO_CLIP,
            meshes=self.meshes,
            dependencies=self.dependencies,
            mesher=self.mesher_panel.mesher(),
            force_remesh=force_reload,
            res=self.mesher_panel.resolution(),
            group_by_material=self.settings_panel.material_group_checkbox.isChecked(),
            deflection=self.mesher_panel.deflection(),
            force_pygen=self.settings_panel.pygen_checkbox.isChecked(),
            param_values=dict(self.params_panel.values),
            # Building the pygfx objects is cheap, so it stays on the job's
            # QThread instead of crossing the process boundary.
            post=lambda data: (build_gfx_group(data[0], data[1], colors), *data[2:]),
        )

    def _reload_pending_meshes(self):
        self.reload_meshes(force_reload=self._reload_pending_force)

    def _resize_grid(self, group):
        """(Re)build the floor grid to cover the loaded geometry's footprint.

        GridHelper bakes its size and division count into its vertex data at
        construction time, so there's no in-place resize - the old grid is
        replaced instead.
        """
        bbox = group.get_world_bounding_box() if group is not None else None
        size = grid_size_for_bbox(bbox)
        divisions = min(MAX_GRID_DIVISIONS, max(10, round(size / GRID_SPACING)))
        visible = True
        if self.grid is not None:
            visible = self.grid.visible
            self.scene.remove(self.grid)
        self.grid = gfx.GridHelper(size=size, divisions=divisions, thickness=1)
        self.grid.visible = visible
        self.scene.add(self.grid)

    def apply_grid_visibility(self):
        """The floor grid follows its checkbox, but is hidden while a gap is
        collapsed: distances across the gap are no longer true."""
        self.grid.visible = self.settings_panel.grid_checkbox.isChecked() and self.cut is None

    def _on_reload_finished(self, result, elapsed):
        new_group, meshes, dependencies, instrument_info = result
        self.params_panel.rebuild(instrument_info["parameters"])
        self.world_matrices = {
            name: np.array(M) for name, M in instrument_info["world_matrices"].items()
        }
        self.component_order = instrument_info["component_order"]
        self.geometry_members = instrument_info["geometry_members"]
        self.clipping_panel.set_frames(list(self.world_matrices))
        self.cut_panel.set_components(self.component_order)
        if self.current_group is not None:
            self.scene.remove(self.current_group)
        self.dependencies = dependencies
        self.meshes = meshes

        self.current_group = new_group

        self.scene.add(self.current_group)
        self.apply_clipping()

        self.rebuild_geometry_panel()
        self.apply_geometry_visibility()
        self.apply_edges_visibility()
        self._resize_grid(self.current_group)
        recentre_controller(self.controller, self.current_group)
        if self.counts:
            self.rebuild_counts_group()
        self.apply_cut()
        print("Reload complete")

        if self._pending_fit_camera:
            self._pending_fit_camera = False
            fit_camera_to_scene(
                self.camera,
                self.controller,
                self.current_group,
            )

    def _on_reload_failed(self, traceback_text, summary):
        print("Mesh rebuild failed:")
        print(traceback_text)

    def _on_mesh_jobs_idle(self):
        self.loading_indicator.set_loading(False)

    # ========================================================
    # McStas trace (components and rays)
    # ========================================================

    def rays_enabled(self):
        return self.settings_panel.rays_checkbox.isChecked()

    def trace_enabled(self):
        return self.settings_panel.components_checkbox.isChecked() or self.rays_enabled()

    def reload_trace(self):
        if self.input_file is None or not self.trace_enabled():
            return
        if self.trace_jobs.busy:
            self.trace_jobs.run_when_idle(self.reload_trace)
            return
        params = instrument_param_args(self.params_panel.values)

        self.settings_panel.set_trace_status("Running mcrun --trace...")

        colors = self.colors
        self.trace_jobs.start(
            compute_trace_data,
            self.input_file,
            self.settings_panel.pygen_checkbox.isChecked(),
            params,
            self.rays_panel.ncount() if self.rays_enabled() else 0,
            self.rays_panel.seed(),
            post=lambda data: (build_component_group(data[0], colors), data[1]),
        )

    def _on_trace_finished(self, result, elapsed):
        group, rays = result
        if self.trace_group is not None:
            self.scene.remove(self.trace_group)
        self.trace_group = group
        self.scene.add(group)
        self.rebuild_component_panel()
        self.place_cut_objects()
        self.apply_component_visibility()
        status = f"{len(group.component_objects)} McStas components"
        self.set_trace_rays(rays)
        if rays is not None:
            status += f", {rays.n_rays} rays"
        self.apply_clipping()
        self.settings_panel.set_trace_status(f"{status} ({elapsed:.1f} s)")

    def _on_trace_failed(self, traceback_text, summary):
        print(traceback_text, end="", file=sys.stderr)
        print("McStas trace failed:")
        print(summary)
        self.settings_panel.set_trace_status(
            "mcrun FAILED. ENSURE THAT INSTRUMENT COMPILES IN ORDER TO "
            "VISUALIZE THE MCSTAS COMPONENTS",
            error=True,
        )

    def on_components_changed(self):
        self.geometry_panel.set_components_shown(
            self.settings_panel.components_checkbox.isChecked()
        )
        if self.trace_group is None:
            self.reload_trace()
        else:
            self.apply_component_visibility()

    def on_rays_changed(self):
        self.rays_panel.setEnabled(self.rays_enabled())
        if not self.rays_enabled():
            self.apply_ray_visibility()
        elif self.trace_rays is None:
            self.reload_trace()
        else:
            self.rebuild_ray_group()

    def rerun_rays(self):
        if self.rays_enabled():
            self.reload_trace()

    def set_trace_rays(self, rays):
        self.trace_rays = rays
        self.rays_panel.set_component_names([] if rays is None else rays.component_names)
        self.rebuild_ray_group()

    def rebuild_ray_group(self):
        if self.ray_group is not None:
            self.scene.remove(self.ray_group)
            self.ray_group = None
        rays = self.trace_rays
        if rays is None:
            self.rays_panel.info_label.setText("")
            self.rays_panel.colorbar.hide()
            self.rebalance_docks()
            return
        chosen = rays_reaching(rays, self.rays_panel.reaching_combo.currentData())
        mode = self.rays_panel.color_combo.currentText()
        cut_mask = None
        if self.cut is not None:
            rays, cut_mask = cut_rays(rays, *self.cut)
        self.ray_group, value_range = build_ray_group(
            rays, chosen, mode, self.rays_panel.colors, self.rays_panel.marker_sizes, cut_mask
        )
        self.scene.add(self.ray_group)
        self.apply_ray_visibility()
        self.apply_clipping()

        self.rays_panel.info_label.setText(f"Showing {len(chosen)} of {rays.n_rays} rays.")
        if value_range is not None:
            label, unit = RAY_COLOR_MODES[mode]
            suffix = f" {unit}" if unit else ""
            self.rays_panel.colorbar.set_range(
                f"{label} {value_range[0]:.4g}{suffix}",
                f"{value_range[1]:.4g}{suffix}",
            )
            self.rays_panel.colorbar.show()
        else:
            self.rays_panel.colorbar.hide()
        # The colorbar appearing/disappearing changes this panel's natural
        # height - give the docks a chance to reclaim or yield that space.
        self.rebalance_docks()

    def on_ray_color_changed(self, key, hex_color):
        group = self.ray_group
        if group is None:
            return
        obj = {
            "teleport": group.teleport_line,
            "scatter": group.scatter_points,
            "absorb": group.absorb_points,
        }.get(key)
        if key == "ray":
            if self.rays_panel.color_combo.currentText() == "Uniform":
                self.rebuild_ray_group()
        elif obj is not None:
            obj.material.color = hex_color

    def on_ray_marker_size_changed(self, key, size):
        points = getattr(self.ray_group, f"{key}_points", None)
        if points is not None:
            points.material.size = size

    def apply_ray_visibility(self):
        if self.ray_group is None:
            return
        self.ray_group.visible = self.settings_panel.rays_checkbox.isChecked()
        for obj, checkbox in (
            (self.ray_group.ray_line, self.rays_panel.line_checkbox),
            (self.ray_group.cut_line, self.rays_panel.line_checkbox),
            (self.ray_group.teleport_line, self.rays_panel.teleport_checkbox),
            (self.ray_group.scatter_points, self.rays_panel.scatter_checkbox),
            (self.ray_group.absorb_points, self.rays_panel.absorb_checkbox),
        ):
            if obj is not None:
                obj.visible = checkbox.isChecked()

    # ========================================================
    # Detector counts
    # ========================================================

    def load_counts_folder(self):
        if self.input_file is None:
            QtWidgets.QMessageBox.warning(
                self, "Detector Counts",
                "Open an instrument first - counts are matched to it by component name.",
            )
            return
        if self.counts_jobs.busy:
            QtWidgets.QMessageBox.information(
                self, "Detector Counts", "Already loading a counts folder - please wait."
            )
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select a McStas run folder (containing mccode.sim)"
        )
        if not folder:
            return

        self.counts_panel.load_button.setEnabled(False)
        self.counts_panel.info_label.setText(f"Loading counts from '{folder}'...")

        self._loading_counts_folder = folder
        self.counts_jobs.start(
            compute_counts_data, folder, self.input_file, self.settings_panel.pygen_checkbox.isChecked()
        )

    def _on_counts_finished(self, counts, elapsed):
        folder = self._loading_counts_folder
        self.counts = counts
        self.counts_run_folder = folder
        self.counts_panel.visibility = {name: True for name in counts}
        if counts:
            self.counts_panel.info_label.setText(f"{len(counts)} detector(s) loaded from '{folder}'.")
        else:
            self.counts_panel.info_label.setText(
                f"No spatially-placeable detector output found in '{folder}'."
            )
        self.counts_panel.rebuild_rows(counts)
        self.reset_counts_range()

    def _on_counts_failed(self, traceback_text, error_text):
        folder = self._loading_counts_folder
        print(traceback_text, end="", file=sys.stderr)
        print("Loading detector counts failed:")
        print(error_text)
        self.counts_panel.info_label.setText(f"Could not load counts from '{folder}': {error_text}")
        QtWidgets.QMessageBox.warning(
            self, "Detector Counts", f"Could not load counts from '{folder}':\n{error_text}"
        )

    def _on_counts_jobs_idle(self):
        self.counts_panel.load_button.setEnabled(True)

    def on_counts_visibility_changed(self, name, checked):
        self.apply_counts_visibility()

    def apply_counts_visibility(self):
        if self.counts_group is None:
            return
        self.counts_group.visible = self.counts_panel.show_checkbox.isChecked()
        for name, mesh in self.counts_meshes.items():
            mesh.visible = self.counts_panel.visibility.get(name, True) and not self.in_gap(name)

    def reset_counts_range(self):
        """Set the colour range to the min/max of the currently visible
        loggers (or every loaded logger, if none are individually toggled
        off yet), then rebuild the planes to match."""
        visible = [lc for name, lc in self.counts.items() if self.counts_panel.visibility.get(name, True)]
        grids = [lc.grid for lc in (visible or self.counts.values())]
        vmax = max((float(grid.max()) for grid in grids), default=1.0)
        vmin = 0.0
        if vmax <= vmin:
            vmax = vmin + 1.0

        self.counts_panel.set_range(vmin, vmax)
        self.on_counts_range_changed()

    def on_counts_range_changed(self, *_args):
        vmin, vmax = self.counts_panel.value_range()
        if vmax <= vmin or not self.counts:
            self.counts_panel.colorbar.hide()
            return
        if not self.counts_meshes:
            self.rebuild_counts_group()
            return
        for name, mesh in self.counts_meshes.items():
            lc = self.counts[name]
            texture = self.counts_textures.get(name)
            if texture is not None:
                texture.set_data(logger_output.texture_image(lc.grid, vmin, vmax))
            else:
                mesh.geometry.colors.set_data(logger_output.line_vertex_colors(lc.grid, vmin, vmax))
        self.counts_panel.colorbar.set_range(f"{vmin:.4g}", f"{vmax:.4g}")
        self.counts_panel.colorbar.show()
        self.rebalance_docks()

    def rebuild_counts_group(self):
        if self.counts_group is not None:
            self.scene.remove(self.counts_group)
        self.counts_group, self.counts_meshes, self.counts_textures = None, {}, {}
        if not self.counts:
            return
        vmin, vmax = self.counts_panel.value_range()
        self.counts_group, self.counts_meshes, self.counts_textures = build_counts_group(
            self.counts, self.world_matrices, vmin, vmax
        )
        self.scene.add(self.counts_group)
        self.place_cut_objects()
        self.apply_counts_visibility()
        self.apply_clipping()
        self.counts_panel.colorbar.set_range(f"{vmin:.4g}", f"{vmax:.4g}")
        self.counts_panel.colorbar.show()
        self.rebalance_docks()

    # ========================================================
    # Collapse gap
    # ========================================================

    def in_gap(self, name):
        return self.cut is not None and self.cut[0].get(name) == GAP

    def cut_offset(self, name):
        if self.cut is not None and self.cut[0].get(name) == BEFORE:
            return self.cut[1]
        return np.zeros(3)

    def geometry_owner(self, key):
        """The Union component a geometry-panel row (a component, or a
        material grouping several) is placed with."""
        return self.geometry_members.get(key, [key])[0]

    def place_cut_objects(self):
        """Move every Union mesh, McStas component and detector plane by
        its collapsed-gap offset (zero when no gap is collapsed)."""
        if self.current_group is not None:
            for key, mesh in self.current_group.geometry_meshes.items():
                mesh.local.position = self.cut_offset(self.geometry_owner(key))
        if self.trace_group is not None:
            for name, obj in self.trace_group.component_objects.items():
                obj.local.position = self.cut_offset(name)
        for name, mesh in self.counts_meshes.items():
            mesh.local.position = self.cut_offset(name)

    def apply_cut(self):
        selection = self.cut_panel.selection()
        self.cut = None
        info = ""
        if selection is not None:
            start, end, keep = selection
            order = self.component_order
            if order.index(start) >= order.index(end):
                info = "'To' must come after 'From'."
            elif start not in self.world_matrices or end not in self.world_matrices:
                info = "Could not place 'From' or 'To'."
            else:
                shift = cut_shift(self.world_matrices, start, end, keep)
                self.cut = (cut_sides(order, start, end), shift)
                info = f"Removed {np.linalg.norm(shift):.4g} m."

        self.place_cut_objects()
        self.apply_geometry_visibility()
        self.apply_component_visibility()
        self.apply_counts_visibility()
        self.rebuild_ray_group()
        self.rebuild_cut_marker(selection)
        self.apply_grid_visibility()
        self.apply_clipping()
        self.cut_panel.info_label.setText(info)

    def rebuild_cut_marker(self, selection):
        if self.cut_marker is not None:
            self.scene.remove(self.cut_marker)
            self.cut_marker = None
        if self.cut is None:
            return
        _, end, keep = selection
        shift = self.cut[1]
        removed = np.linalg.norm(shift)
        if removed == 0:
            return
        direction = shift / removed
        end_position = self.world_matrices[end][:3, 3]
        # ponytail: marker sized by keep, not by the geometry around it.
        self.cut_marker = build_cut_marker(
            end_position - direction * keep / 2,
            direction,
            max(keep, CUT_MARKER_MIN_SIZE),
            f"-{removed:.4g} m",
        )
        self.scene.add(self.cut_marker)

    def fit_whole_instrument(self):
        if self.trace_group is None or not self.trace_group.visible:
            self.reset_view()
            return
        fit_camera_to_scene(self.camera, self.controller, self.trace_group)

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
                self.reload_trace()

        except Exception as e:
            print(e)

    # ========================================================
    # Settings changed
    # ========================================================

    def on_mesher_options_changed(self):
        # Global mesher settings, not part of any component's dependency
        # signature, so everything is remeshed.
        self.reload_meshes(force_reload=True)

    def on_material_group_changed(self):
        self.reload_meshes()

    def on_pygen_changed(self):
        # Switches which parser builds the McStas_instr entirely, so
        # nothing from a previous load can be trusted as unchanged.
        self.reload_meshes(force_reload=True)
        self.reload_trace()

    def on_vacuum_changed(self):
        self.apply_geometry_visibility()

    # ========================================================
    # Ruler
    # ========================================================

    def on_ruler_pointer(self, event):
        """While measuring, a left click (press and release without
        dragging) on a mesh adds a ruler point."""
        if not self.ruler_panel.measure_checkbox.isChecked() or event.button != 1:
            return
        if event.type == "pointer_down":
            self._ruler_press = (event.x, event.y)
            return
        press, self._ruler_press = self._ruler_press, None
        if press is None or np.hypot(event.x - press[0], event.y - press[1]) > 4:
            return
        pick_info = self.renderer.get_pick_info((event.x, event.y))
        w, h = self.renderer.logical_size
        ndc = (2 * event.x / w - 1, 1 - 2 * event.y / h)
        point = pick_world_point(pick_info, self.camera, ndc, self._union_surface(pick_info))
        if point is None:
            return
        self.ruler_panel.add_point(point)

    def _union_surface(self, pick_info):
        """(vertices, faces) of the Union mesh a pick hit, which is drawn
        RENDER_OFFSET outward from it, or None for anything else."""
        if self.current_group is None:
            return None
        for key, mesh in self.current_group.geometry_meshes.items():
            if pick_info.get("world_object") is mesh:
                source = self.current_group.geometry_trimeshes[key]
                return source.vertices, source.faces
        return None

    def set_ruler_points(self, points):
        self.ruler_points = list(points)
        if self.ruler_group is not None:
            self.scene.remove(self.ruler_group)
            self.ruler_group = None
        if self.ruler_points:
            self.ruler_group = build_ruler_group(self.ruler_points, self.ruler_panel.color)
            self.scene.add(self.ruler_group)
        self.ruler_panel.show_points(self.ruler_points)

    def on_ruler_points_edited(self):
        self.set_ruler_points(self.ruler_panel.points())

    def clear_ruler(self):
        self.ruler_panel.clear()

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
        self.gizmo.local.rotation = la.quat_inv(self.camera.local.rotation)
        self.renderer.render(self.scene, self.camera, flush=False)
        _, h = self.canvas.get_logical_size()
        s = 160
        self.gizmo_viewport.rect = (10, h - s - 10, s, s)
        self.gizmo_viewport.render(self.gizmo_scene, self.gizmo_camera, flush=True)
        update_camera_depth_range(self.camera, self.controller.target)

        self.canvas.request_draw()

    # ========================================================
    # Shutdown
    # ========================================================

    def closeEvent(self, event):
        for jobs in (self.mesh_jobs, self.trace_jobs, self.counts_jobs):
            jobs.shutdown()
        super().closeEvent(event)


# ============================================================
# Main
# ============================================================


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        "--input_file",
        dest="input_file",
        help="Input mcstas file, can either be mcstasscript or mcstas",
    )
    return parser


def launch(input_file=None):
    """Create the application and open the interactive Union viewer."""
    app = QtWidgets.QApplication([sys.argv[0]])

    app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar)

    viewer = Viewer(input_file=input_file)
    viewer.show()
    return app.exec()


def main(argv=None):
    parser = parse()
    args = parser.parse_args(argv)
    return launch(args.input_file)


if __name__ == "__main__":
    raise SystemExit(main())
