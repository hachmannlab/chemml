# -*- coding: utf-8 -*-
from chemml.chem.magpie_python import APEAttributeGenerator
from chemml.chem.magpie_python import CompositionEntry
from chemml.chem.magpie_python.data.materials.util.LookUpData import LookUpData

class TestAPEAttributeGenerator:

    def test_APECalculator(self):
        # Test ideal icosahedron.
        assert abs(1.0 - APEAttributeGenerator.compute_APE(n_neighbors=12, center_radius=0.902113, neigh_eff_radius=1.0)) <= 1e-06
        assert round(abs(1.0 - APEAttributeGenerator.compute_APE(radii=[0.902113, 1.0], center_type=0, shell_types=[0, 12])), 7) == 0

        # Make sure overpacked in less than 1.
        # using the conventional from 10.1038/ncomms9123.
        assert APEAttributeGenerator.compute_APE(radii=[0.902113, 1.0], center_type=0, shell_types=[1, 11]) < 1.0

    def test_cluster_finder(self):
        # Test unary system.
        clusters = APEAttributeGenerator.find_efficiently_packed_clusters([
            1.0], 0.05)
        assert 1 == len(clusters)
        assert 2 == len(clusters[0])
        assert 13 == clusters[0][0][0]
        assert 14 == clusters[0][1][0]

        # Test binary system.
        radii = [1.0, 0.902113]
        clusters = APEAttributeGenerator.find_efficiently_packed_clusters(
            radii, 0.01)
        assert 2 == len(clusters)

        # Make sure all clusters actually have |APE - 1| below 0.05.
        for i in range(len(radii)):
            for c in clusters[i]:
                ape = APEAttributeGenerator.compute_APE(radii=radii,
                                    center_type=i, shell_types=c)
                assert abs(ape - 1) < 0.01

        # Test quinary system.
        radii = [1.0, 0.902113, 1.1, 1.2, 0.7]
        clusters = APEAttributeGenerator.find_efficiently_packed_clusters(
            radii, 0.01)
        assert 5 == len(clusters)

        # Make sure all clusters actually have |APE - 1| below 0.05.
        for i in range(len(radii)):
            for c in clusters[i]:
                ape = APEAttributeGenerator.compute_APE(radii=radii,
                                    center_type=i, shell_types=c)
                assert abs(ape - 1) < 0.01

    def test_composition(self):
        # Make a fake cluster output.
        clusters = []
        elements = [0, 1]

        # Central type 0.
        l1 = [[12, 0], [5, 5]]
        clusters.append(l1)

        # Central type 1.
        l2 = [[2, 5]]
        clusters.append(l2)

        # Run the conversion.
        comps = APEAttributeGenerator.compute_cluster_compositions(elements,
                                                                   clusters)
        assert 3 == len(comps)
        assert CompositionEntry(composition='H') in comps
        assert CompositionEntry(composition='H6He5') in comps
        assert CompositionEntry(composition='H2He6') in comps

    def test_optimal_solver(self):
        # Get the radii lookup table.
        radii = LookUpData.load_property("MiracleRadius")

        # Find the best Cu cluster.
        entry = CompositionEntry(composition="Cu")
        ape = APEAttributeGenerator.determine_optimal_APE(28, entry, radii)
        assert abs(0.976006 / 1.0 - ape) <= 1e-06

        # Find the best Cu-centered Cu64.3Zr35.7.
        entry = CompositionEntry(composition="Cu64.3Zr35.7")
        ape = APEAttributeGenerator.determine_optimal_APE(28, entry, radii)
        assert abs(0.902113 / 0.916870416 - ape) <= 1e-06

    def test_attribute_generation(self):
        # Make a fake list of CompositionEntry's.
        entries = [CompositionEntry(composition="Cu64.3Zr35.7")]

        # Make attribute generator.
        aag = APEAttributeGenerator()

        # Set options.
        aag.set_packing_threshold(0.01)
        aag.set_n_nearest_to_eval([1, 3])

        # Compute features.
        features = aag.generate_features(entries)

        # Test results.
        assert 4 == features.size
        assert features.size == features.values.size
        assert features.values[0][0] < features.values[0][1]
        assert abs(0.979277669 - features.values[0][2]) <= 1e-06
        assert abs(0.020722331 - features.values[0][3]) <= 1e-06

    def test_range_finder(self):
        # Equal sized spheres.
        l,r = APEAttributeGenerator.get_cluster_range([1.0, 1.0], 0.03)
        assert 13 == l
        assert 13 == r

        l, r = APEAttributeGenerator.get_cluster_range([1.0, 1.0], 0.05)
        assert 13 == l
        assert 14 == r

        l, r = APEAttributeGenerator.get_cluster_range([1.0, 1.0], 0.1)
        assert 12 == l
        assert 14 == r

        # Unequal spheres.
        l, r = APEAttributeGenerator.get_cluster_range([1.0, 1.2], 0.03)
        assert 11 == l
        assert 16 == r

        l, r = APEAttributeGenerator.get_cluster_range([1.0, 1.2], 0.05)
        assert 10 == l
        assert 17 == r

        l, r = APEAttributeGenerator.get_cluster_range([1.0, 1.2], 0.1)
        assert 10 == l
        assert 18 == r

        # Unequal spheres, extra and order reverse.
        l, r = APEAttributeGenerator.get_cluster_range([1.4, 1.0, 1.2], 0.03)
        assert 9 == l
        assert 20 == r

        l, r = APEAttributeGenerator.get_cluster_range([1.4, 1.0, 1.2], 0.05)
        assert 9 == l
        assert 20 == r

        l, r = APEAttributeGenerator.get_cluster_range([1.4, 1.0, 1.2], 0.1)
        assert 9 == l
        assert 21 == r

    def test_many_types(self):
        # This test is based on 6 as the max # of types.
        assert 6 == APEAttributeGenerator.max_n_types

        # Make a 6 and 6+1 component alloy.
        entries = [CompositionEntry(composition="ScTiHfZrCrMo"),
                   CompositionEntry(composition="ScTiHfZrCrMoP0.9")]

        aag = APEAttributeGenerator()

        features = aag.generate_features(entries)
        feat_values = features.values

        # Make sure the features are not identical.
        for i in range(feat_values[0].size):
            assert abs(feat_values[0][i] - feat_values[1][i]) > 1e-06