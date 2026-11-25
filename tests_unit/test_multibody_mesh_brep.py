"""A Union_mesh file holding several separate bodies must become real solids,
so it can be cut from, or cut out of, another component."""

import sys
import unittest
from pathlib import Path

import trimesh
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import brep  # noqa: E402


def volume(shape):
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass()


class TestMultibodyMeshBrep(unittest.TestCase):
    def test_two_bodies_become_two_solids_that_cut(self):
        left = trimesh.creation.box(extents=(1, 1, 1))
        left.apply_translation((-1, 0, 0))
        right = trimesh.creation.box(extents=(1, 1, 1))
        right.apply_translation((1, 0, 0))
        shape = brep._mesh_to_brep(trimesh.util.concatenate([left, right]), "pair")

        n_solids = 0
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            n_solids += 1
            explorer.Next()
        self.assertEqual(n_solids, 2)

        slab = brep._mesh_to_brep(trimesh.creation.box(extents=(4, 1, 0.5)), "slab")
        cut = BRepAlgoAPI_Cut(slab, shape)
        self.assertTrue(cut.IsDone())
        self.assertAlmostEqual(volume(cut.Shape()), 4 * 0.5 - 2 * 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
