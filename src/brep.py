from OCC.Core.BRepPrimAPI import (
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakeBox, 
        BRepPrimAPI_MakeSphere, 
        BRepPrimAPI_MakeCone,
        BRepPrimAPI_MakeHalfSpace)
from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.gp import gp_Pln
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_Sewing,
)
from OCC.Core.BRepAlgoAPI import (
    BRepAlgoAPI_Common,
    BRepAlgoAPI_Cut,
    BRepAlgoAPI_Fuse,
)
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.TopAbs import TopAbs_SHELL
from OCC.Core.TopoDS import Shell as topods_Shell
import trimesh
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from preprocess import box_dimensions
from bounding_box import compute_all_world_bboxes, overlapping


def _mesh_to_brep(mesh, name, verbose=False):
    """Convert a world-space trimesh.Trimesh into a genuine OCC solid (one
    real BRep face per triangle, sewn into a shell and closed into a
    solid), rather than a raw faceted STL import. A Union_mesh component
    can be used both as the shape being meshed and as a higher-priority
    cutter/mask against another component (see build_single_brep_mesh), so
    it needs to behave like any other geometry under BRepAlgoAPI_Cut/Common
    - a bare faceted shape doesn't reliably support that."""
    sewing = BRepBuilderAPI_Sewing()
    n_faces = 0
    for tri in mesh.faces:
        p0, p1, p2 = (gp_Pnt(*mesh.vertices[i]) for i in tri)
        polygon = BRepBuilderAPI_MakePolygon(p0, p1, p2, True)
        if not polygon.IsDone():
            # Degenerate (zero-area/collinear) triangle - skip it rather
            # than let it break the sewn shell.
            if verbose:
                print(f"Skipping degenerate triangle on mesh '{name}'")
            continue
        face = BRepBuilderAPI_MakeFace(polygon.Wire()).Face()
        sewing.Add(face)
        n_faces += 1

    if n_faces == 0:
        raise ValueError(f"Mesh '{name}' has no valid triangles to build a BRep from")

    sewing.Perform()
    shape = sewing.SewedShape()

    if shape.ShapeType() != TopAbs_SHELL:
        # Sewing didn't produce a single closed shell (e.g. a non-watertight
        # mesh) - fall back to the sewn shape as-is rather than raising, so
        # a slightly-open mesh still degrades gracefully.
        if verbose:
            print(f"Warning: mesh '{name}' did not sew into a closed shell; "
                  f"using the sewn shape as-is.")
        return shape

    solid_maker = BRepBuilderAPI_MakeSolid(topods_Shell(shape))
    if not solid_maker.IsDone():
        if verbose:
            print(f"Warning: could not close mesh '{name}' into a solid; "
                  f"using the sewn shell as-is.")
        return shape
    return solid_maker.Solid()


def _rectangle_wire(center, x_dir, y_dir, width, height):
    """A closed rectangular wire of the given size, centred on `center` and
    spanned by the (unit, orthogonal) directions x_dir/y_dir. Corners are
    emitted in a fixed order relative to those directions so that two such
    wires stacked along z traverse the same way round and the loft between
    them does not come out twisted."""
    half_x = 0.5 * width * x_dir
    half_y = 0.5 * height * y_dir

    polygon = BRepBuilderAPI_MakePolygon(
        gp_Pnt(*(center - half_x - half_y)),
        gp_Pnt(*(center + half_x - half_y)),
        gp_Pnt(*(center + half_x + half_y)),
        gp_Pnt(*(center - half_x + half_y)),
        True,
    )
    return polygon.Wire()


def _build_box_brep(comp, center, x_dir, y_dir, z_dir):
    """Build a Union_box solid, which is a plain cuboid unless xwidth2 or
    yheight2 taper it into a rectangular frustum."""
    x1, y1, x2, y2 = box_dimensions(comp)
    zdepth = float(comp.zdepth)

    if x1 == x2 and y1 == y2:
        corner = (
            center
            - 0.5 * x1 * x_dir
            - 0.5 * y1 * y_dir
            - 0.5 * zdepth * z_dir
        )

        axis = gp_Ax2(
            gp_Pnt(*corner),
            gp_Dir(*z_dir),  # local Z
            gp_Dir(*x_dir),  # local X
        )

        return BRepPrimAPI_MakeBox(axis, x1, y1, zdepth).Shape()

    # BRepPrimAPI_MakeBox cannot express a taper, so loft between the two
    # end faces instead. isSolid=True gives a closed solid (which the
    # BRepAlgoAPI_Cut/Common/Fuse calls downstream need) and ruled=True keeps
    # the four side faces planar, which is what McStas's own box intersection
    # code assumes - it stores one fixed normal per side (Union_box.comp,
    # normal_vectors[2..5]).
    loft = BRepOffsetAPI_ThruSections(True, True)
    loft.AddWire(_rectangle_wire(center - 0.5 * zdepth * z_dir, x_dir, y_dir, x1, y1))
    loft.AddWire(_rectangle_wire(center + 0.5 * zdepth * z_dir, x_dir, y_dir, x2, y2))
    loft.Build()

    if not loft.IsDone():
        raise RuntimeError(
            f"Could not build tapered box '{comp.name}' "
            f"(xwidth={x1}, yheight={y1}, xwidth2={x2}, yheight2={y2}, "
            f"zdepth={zdepth})"
        )

    return loft.Shape()


def build_comp_brep(comp, world_matrices):
    name = comp.name
    c_type = comp.component_name
    mat = world_matrices[name]
    pnt = mat[:3, 3]
    x_dir = mat[:3, 0]
    y_dir = mat[:3, 1]
    z_dir = mat[:3, 2]
    axis = gp_Ax2(gp_Pnt(*pnt), gp_Dir(*x_dir), gp_Dir(*z_dir))
    if c_type == "Union_sphere":
        brep = BRepPrimAPI_MakeSphere(axis, comp.radius).Shape()
    elif c_type == "Union_box":
        brep = _build_box_brep(comp, pnt, x_dir, y_dir, z_dir)

    elif c_type == "Union_cylinder":
        pnt = pnt - comp.yheight/2 * y_dir
        axis = gp_Ax2(gp_Pnt(*pnt), gp_Dir(*y_dir))
        brep = BRepPrimAPI_MakeCylinder(axis, comp.radius, comp.yheight).Shape()
    elif c_type == "Union_cone":
        pnt = pnt - comp.yheight / 2 * y_dir
        axis = gp_Ax2(gp_Pnt(*pnt), gp_Dir(*y_dir))
        brep = BRepPrimAPI_MakeCone(
            axis, comp.radius_bottom, comp.radius_top, comp.yheight
        ).Shape()
    elif c_type == "Union_mesh":
        mesh = trimesh.load_mesh(comp.filename.strip('"'))
        if comp.coordinate_scale is None:
            comp.coordinate_scale = 1e-3
        mesh.apply_scale(float(comp.coordinate_scale))
        mesh.apply_transform(mat)
        brep = _mesh_to_brep(mesh, name)
    return brep

def get_mask_comps(focus_comp, union_geometries):
    mask_comps = []
    mask_setting = "No"
    for name,comp in union_geometries.items():
        if comp.mask_string is None: 
            continue
        if focus_comp.name in comp.mask_string:
            mask_comps.append(comp)
    for mask in mask_comps:
        if mask.mask_setting == "All":
            mask_setting = mask.mask_setting
    if mask_setting == "No" and len(mask_comps)>0:
        mask_setting = "Any"
    return mask_comps, mask_setting

def intersect_with_masks(shape, mask_comps, mask_setting):
    if mask_setting == "All":
        # C ∩ M1 ∩ M2 ∩ ... ∩ Mn
        result_shape = shape

        for mask_comp in mask_comps:
            common = BRepAlgoAPI_Common(result_shape, mask_comp)

            if not common.IsDone():
                raise RuntimeError(
                    f"Boolean intersection failed for mask "
                    f"{mask_comp.name!r}"
                )

            result_shape = common.Shape()
        return result_shape

    elif mask_setting == "Any":
        # C ∩ (M1 ∪ M2 ∪ ... ∪ Mn)
        combined_mask = mask_comps[0]

        for mask_comp in mask_comps[1:]:
            fuse = BRepAlgoAPI_Fuse(combined_mask, mask_comp)

            if not fuse.IsDone():
                raise RuntimeError(
                    f"Boolean union failed for mask "
                    f"{mask_comp.name!r}"
                )

            combined_mask = fuse.Shape()

        common = BRepAlgoAPI_Common(shape, combined_mask)

        if not common.IsDone():
            raise RuntimeError(
                "Boolean intersection with combined ANY mask failed"
            )

        return common.Shape()
    return shape



def build_mask_comps(mask_comps, world_matrices):
    breps = []
    for i, comp in enumerate(mask_comps):
        brep = build_comp_brep(comp, world_matrices)
        breps.append(brep)

    return breps

def build_higher_priorities(higher_priorities, world_matrices):
    breps = []
    for i, comp in enumerate(higher_priorities):
        brep = build_comp_brep(comp, world_matrices)
        breps.append(brep)
    return breps


def subtract_higher_priorities(comp, prio_breps):
    for prio in prio_breps:
        # Two-shape constructor already performs the cut - no .Build() needed.
        cut = BRepAlgoAPI_Cut(comp, prio)
        if not cut.IsDone():
            raise RuntimeError("Boolean cut failed")
        comp = cut.Shape()
    return comp




def clip_component(shape, clip):
    if not clip["enable"]:
        return shape

    axis_name = clip["axis"].upper()
    position = float(clip["position"])
    mode = clip["mode"]

    if axis_name == "X":
        normal = np.array([1.0, 0.0, 0.0])
    elif axis_name == "Y":
        normal = np.array([0.0, 1.0, 0.0])
    elif axis_name == "Z":
        normal = np.array([0.0, 0.0, 1.0])
    else:
        raise ValueError(f"Unknown clip axis: {axis_name}")

    # plane point
    plane_point = np.zeros(3)

    if axis_name == "X":
        plane_point[0] = position
    elif axis_name == "Y":
        plane_point[1] = position
    elif axis_name == "Z":
        plane_point[2] = position

    # select side to keep
    if mode == "Above":
        keep_point = plane_point + normal
    elif mode == "Below":
        keep_point = plane_point - normal
    else:
        raise ValueError(f"Unknown clip mode: {mode}")

    plane = gp_Pln(
        gp_Pnt(*plane_point),
        gp_Dir(*normal)
    )

    plane_face = BRepBuilderAPI_MakeFace(plane).Face()

    halfspace = BRepPrimAPI_MakeHalfSpace(
        plane_face,
        gp_Pnt(*keep_point)
    ).Solid()

    result = BRepAlgoAPI_Common(
        shape,
        halfspace
    )

    if not result.IsDone():
        raise RuntimeError("Clip operation failed")

    return result.Shape()


def build_single_brep_mesh(
    comp, union_geometries, world_matrices, clip, verbose, deflection=0.01,
    world_bboxes=None,
):
    if hasattr(comp, 'mask_string'):
        if comp.mask_string != None:
            if verbose:
                print(f"{comp.name} is a mask, and is therefore not meshed.")
            return None
    if world_bboxes is None:
        world_bboxes = compute_all_world_bboxes(union_geometries, world_matrices)
    # Only a higher-priority component whose world bbox actually overlaps
    # this one can change the result of subtracting it.
    candidates = [
        x.name for n, x in union_geometries.items() if x.priority > comp.priority
    ]
    higher_priority = [
        union_geometries[n] for n in overlapping(comp.name, candidates, world_bboxes)
    ]
    res_comp = build_comp_brep(comp, world_matrices)
    mask_comps, mask_setting = get_mask_comps(comp, union_geometries)
    mask_comps = build_mask_comps(mask_comps, world_matrices)
    prio_breps = build_higher_priorities(higher_priority, world_matrices)
    res_comp = subtract_higher_priorities(res_comp, prio_breps)
    res_comp = intersect_with_masks(res_comp, mask_comps, mask_setting)
    res_comp = clip_component(res_comp, clip)

    BRepMesh_IncrementalMesh(res_comp, deflection).Perform()
    vertices = []
    faces = []

    vertex_offset = 0

    explorer = TopExp_Explorer(res_comp, TopAbs_FACE)

    while explorer.More():
        face = explorer.Current()

        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)

        if triangulation is None:
            explorer.Next()
            continue

        trsf = loc.Transformation()

        for i in range(1, triangulation.NbNodes() + 1):
            p = triangulation.Node(i).Transformed(trsf)

            vertices.append([p.X(), p.Y(), p.Z()])

        # triangles
        for i in range(1, triangulation.NbTriangles() + 1):
            tri = triangulation.Triangle(i)

            n1, n2, n3 = tri.Get()
            if face.Orientation() == TopAbs_REVERSED:
                n2, n3 = n3, n2

            faces.append(
                [
                    vertex_offset + n1 - 1,
                    vertex_offset + n2 - 1,
                    vertex_offset + n3 - 1,
                ]
            )

        vertex_offset += triangulation.NbNodes()

        explorer.Next()
    if len(vertices) == 0 or len(faces) == 0:
        print(f"WARNING: empty mesh for {comp.name}")
        return None
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return mesh


_CLASS_SPEC_ATTRS = (
    "parameter_names", "parameter_defaults", "parameter_types",
    "parameter_units", "parameter_comments", "category", "line_limit",
)


def _component_class_spec(comp):
    """Plain-data (str/list/dict/int only) snapshot of the attributes
    mcstasscript's own dynamic class construction assigns - see
    _register_component_classes for why this has to be plain data rather
    than the Component instance itself."""
    return {attr: getattr(comp, attr) for attr in _CLASS_SPEC_ATTRS}


def _register_component_classes(class_specs):
    """Runs once in each freshly spawned worker process, before it accepts
    any real task. mcstasscript represents each Union_* component as a
    class it creates *dynamically* at parse time (McStas_instr's
    _create_component_instance), registering it into
    mcstasscript.interface.instr's own module globals so a Component
    instance of that type can be pickled - but that registration only
    happens in whichever process actually parses an instrument file. A
    worker spawned by this pool never does that itself, so without this,
    unpickling a Component instance sent to it fails with an AttributeError
    ("module ... has no attribute 'Union_box'"). This recreates an
    equivalent class, matching mcstasscript's own construction, from a
    plain-data spec built by _component_class_spec - it can't take a live
    sample Component instead, because that instance would itself need to be
    pickled to reach this initializer, hitting the same problem one level
    earlier (unpickling the process's own spawn payload).
    """
    import mcstasscript.interface.instr as msi

    for component_name, spec in class_specs.items():
        if hasattr(msi, component_name):
            continue
        input_dict = {key: None for key in spec["parameter_names"]}
        input_dict.update(spec)
        setattr(msi, component_name, type(component_name, (msi.Component,), input_dict))


# Reused across calls rather than recreated per call, so the worker
# processes are spawned once (lazily, on first submit) and stay warm across
# reloads instead of paying process-startup cost every time. Rebuilt (see
# below) if a later call needs a component type its workers don't know
# about yet - e.g. after switching to a different instrument.
_mesh_build_pool = None
_mesh_build_pool_component_types = frozenset()


def _get_mesh_build_pool(union_geometries):
    global _mesh_build_pool, _mesh_build_pool_component_types

    sample_components = {}
    for comp in union_geometries.values():
        sample_components.setdefault(comp.component_name, comp)
    needed_types = frozenset(sample_components)

    if _mesh_build_pool is not None and not needed_types.issubset(
        _mesh_build_pool_component_types
    ):
        _mesh_build_pool.shutdown(wait=True)
        _mesh_build_pool = None

    if _mesh_build_pool is None:
        class_specs = {
            name: _component_class_spec(comp)
            for name, comp in sample_components.items()
        }
        _mesh_build_pool = ProcessPoolExecutor(
            max_workers=os.cpu_count() or 1,
            initializer=_register_component_classes,
            initargs=(class_specs,),
        )
        _mesh_build_pool_component_types = needed_types

    return _mesh_build_pool


def build_many_brep_meshes(
    names, union_geometries, world_matrices, clip, verbose, deflection=0.01,
    world_bboxes=None, max_workers=None,
):
    """Build the brep mesh for each of `names` (a subset of, or all of,
    union_geometries). Every component's build is independent of every
    other component's - build_single_brep_mesh only ever reads *other*
    components' cheaply-rebuilt primitive shapes, never another
    component's finished mesh - so this fans them out across worker
    processes instead of building one at a time. Real OS processes, not
    threads: pythonocc-core/OpenCASCADE calls don't release the GIL, so a
    thread pool would just serialize on it and gain nothing.
    """
    if world_bboxes is None:
        world_bboxes = compute_all_world_bboxes(union_geometries, world_matrices)

    names = list(names)
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = min(max_workers, len(names))

    if max_workers <= 1:
        return {
            name: build_single_brep_mesh(
                union_geometries[name], union_geometries, world_matrices, clip,
                verbose, deflection=deflection, world_bboxes=world_bboxes,
            )
            for name in names
        }

    pool = _get_mesh_build_pool(union_geometries)
    futures = {
        pool.submit(
            build_single_brep_mesh,
            union_geometries[name], union_geometries, world_matrices, clip,
            verbose, deflection, world_bboxes,
        ): name
        for name in names
    }
    meshes_dict = {}
    for future in as_completed(futures):
        meshes_dict[futures[future]] = future.result()
    return meshes_dict


def build_brep_meshes(
    union_geometries, world_matrices, clip, verbose, deflection=0.01,
    world_bboxes=None, max_workers=None,
):
    return build_many_brep_meshes(
        union_geometries.keys(), union_geometries, world_matrices, clip, verbose,
        deflection=deflection, world_bboxes=world_bboxes, max_workers=max_workers,
    )
