import unittest
import pathlib
from spdm.utils.logger import logger
from spdm.core.entry import open_entry


pwd = pathlib.Path(__file__).parent.as_posix()


class TestFileGEQdsk(unittest.TestCase):
    def test_read(self):

        entry = open_entry(f"{pwd}/data/geqdsk.txt#equilibrium", format="file+geqdsk")

        eq = entry.child("time_slice/0/profiles_2d").get()

        self.assertEqual(eq["grid_type"]["name"], "rectangular")


if __name__ == "__main__":
    unittest.main()
