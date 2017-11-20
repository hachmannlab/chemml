import unittest
from data.utilities.generators.PhaseDiagramCompositionEntryGenerator import \
    PhaseDiagramCompositionEntryGenerator

class testPhaseDiagramCompositionEntryGenerator(unittest.TestCase):
    def setUp(self):
        self.pg = PhaseDiagramCompositionEntryGenerator()

    def tearDown(self):
        self.pg = None

    def test_generate_alloy_compositions(self):
        self.pg.set_even_spacing(True)
        self.pg.set_size(5)
        self.pg.set_order(1, 3)

        comps = self.pg.generate_alloy_compositions()
        self.assertEquals(1, len(comps[1]))
        self.assertEquals(3, len(comps[2]))
        self.assertEquals(3, len(comps[3]))

        self.pg.set_order(3, 3)
        comps = self.pg.generate_alloy_compositions()
        self.assertEquals(1, len(comps))

    def test_generate_crystal_compositions(self):
        self.pg.set_even_spacing(False)
        self.pg.set_size(4)
        self.pg.set_order(1, 3)

        comps = self.pg.generate_crystal_compositions()
        self.assertEquals(1, len(comps[1]))
        self.assertEquals(5, len(comps[2]))
        self.assertEquals(4, len(comps[3]))

        # Try making entries.
        self.pg.set_elements_by_index([0, 1, 2])
        entries = self.pg.generate_entries()
        self.assertEquals(3 * 1 + 3 * 5 + 4, len(entries))

        self.pg.set_elements_by_index([0, 1, 2, 3])
        entries = self.pg.generate_entries()
        self.assertEquals(4 * 1 + 6 * 5 + 4 * 4, len(entries))

        self.pg.set_order(2, 3)
        comps = self.pg.generate_crystal_compositions()
        self.assertEquals(2, len(comps))