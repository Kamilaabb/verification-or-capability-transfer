import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("gv_domain_tables", Path(__file__).resolve().parents[1]/"scripts/gv_domain_tables.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class EventTests(unittest.TestCase):
    def test_eight_cells_and_direction(self):
        cells = {"000":2,"001":3,"010":5,"011":7,"100":11,"101":13,"110":17,"111":19}
        rows = [tuple(map(int,k)) for k,n in cells.items() for _ in range(n)]
        g,v,gv = map(list,zip(*rows))
        r = mod.measure(g,v,gv,"V")
        self.assertEqual(r["events"],cells)
        self.assertEqual(r["GV_minus_G"]["rows"],3+7-11-17)
        self.assertEqual(r["GV_minus_baseline2"]["rows"],3+13-5-17)
        reverse = mod.measure(v,g,gv,"G")
        self.assertEqual(reverse["events"]["010"],11)
        self.assertEqual(reverse["GV_minus_baseline2"],r["GV_minus_baseline2"])

    def test_reject_incomplete(self):
        with self.assertRaises(ValueError): mod.measure([1],[1,0],[1],"V")
        with self.assertRaises(ValueError): mod.measure([1],[None],[1],"V")
