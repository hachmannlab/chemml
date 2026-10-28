from chemml.chem.magpie_python.data.utilities.generators.PhaseDiagramCompositionEntryGenerator \
    import PhaseDiagramCompositionEntryGenerator

class TestPhaseDiagramCompositionEntryGenerator:
    def setup_method(self):
        self.pg = PhaseDiagramCompositionEntryGenerator()

    def teardown_method(self):
        self.pg = None

    def test_generate_alloy_compositions(self):
        self.pg.set_even_spacing(True)
        self.pg.set_size(5)
        self.pg.set_order(1, 3)

        comps = self.pg.generate_alloy_compositions()
        assert 1 == len(comps[1])
        assert 3 == len(comps[2])
        assert 3 == len(comps[3])

        self.pg.set_order(3, 3)
        comps = self.pg.generate_alloy_compositions()
        assert 1 == len(comps)

    def test_generate_crystal_compositions(self):
        self.pg.set_even_spacing(False)
        self.pg.set_size(4)
        self.pg.set_order(1, 3)

        comps = self.pg.generate_crystal_compositions()
        assert 1 == len(comps[1])
        assert 5 == len(comps[2])
        assert 4 == len(comps[3])

        # Try making entries.
        self.pg.set_elements_by_index([0, 1, 2])
        entries = self.pg.generate_entries()
        assert 3 * 1 + 3 * 5 + 4 == len(entries)

        self.pg.set_elements_by_index([0, 1, 2, 3])
        entries = self.pg.generate_entries()
        assert 4 * 1 + 6 * 5 + 4 * 4 == len(entries)

        self.pg.set_order(2, 3)
        comps = self.pg.generate_crystal_compositions()
        assert 2 == len(comps)