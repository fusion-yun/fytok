import unittest
import pathlib
from spdm.core.field import Field
from spdm.mesh.mesh_rectilinear import RectilinearMesh

from fytok.utils.logger import logger
from fytok.modules.equilibrium import Equilibrium

pwd = pathlib.Path(__file__).parent.as_posix()


class TestFileGEQdsk(unittest.TestCase):

    def test_plugin(self):
        class FEQ(Equilibrium, plugin_name="feq"):
            pass

        self.assertIn("fytok.modules.equilibrium.feq", Equilibrium._plugin_registry)

    def test_create(self):
        eq = Equilibrium()
        self.assertEqual(eq.__class__.code.default_value.get("name", None), "fy_eq")

    def test_field(self):
        equilibrium = Equilibrium(f"file+geqdsk://{pwd}/data/geqdsk.txt#equilibrium")
        time_slice = equilibrium.time_slice
        current = time_slice.current
        profiles_2d = current.profiles_2d
        self.assertIsInstance(profiles_2d.psi, Field)

    def test_mesh(self):
        equilibrium = Equilibrium(f"file+geqdsk://{pwd}/data/geqdsk.txt#equilibrium")

        profiles_2d = equilibrium.time_slice.current.profiles_2d

        self.assertIsInstance(profiles_2d.grid, RectilinearMesh)

        self.assertIs(profiles_2d.grid, profiles_2d.psi.mesh)


if __name__ == "__main__":
    unittest.main()
