import types
import numpy as np
import pandas as pd
from ....data.materials.CompositionEntry import CompositionEntry
from ....data.materials.util.LookUpData import LookUpData

class ElementPairPropertyAttributeGenerator:
    """Class to generate attributes based on the properties of constituent
    binary systems.
    Computes the minimum, maximum and range of all pairs in
    the material, and the fraction-weighted mean and variance of all pairs.
    Variance is defined as the mean absolute deviation from the mean over all
    pairs. If an entry has only one element, the value of NaN is used for all
    attributes.

    Attributes
    ----------
    elemental_pair_properties : list
        Elemental properties to be associated with this class for the generation of
        features.
    pair_lookup-data : dict
        Dictionary containing the property name as the key and a list of floats as
        the value.

    """

    elemental_pair_properties = []
    pair_lookup_data = {}

    def load_pair_lookup_data(self):
        """Function to load the property values into `self.lookup_data` for the
        computation of features.

        """

        self.pair_lookup_data = LookUpData.load_pair_properties(
            self.elemental_pair_properties)

    def add_elemental_pair_property(self, property):
        """Function to add an elemental pair property to be used to compute
        features.

        Parameters
        ----------
        property : str
            Property to be added.

        """

        if property not in self.elemental_pair_properties:
            self.elemental_pair_properties.append(property)

    def add_elemental_pair_properties(self, properties):
        """Function to provide a list of elemental pair properties to be used to
        compute features.

        Parameters
        ----------
        properties : array-like
            Properties to be included. A list of strings containing property
            names.

        """

        for prop in properties:
            self.add_elemental_pair_property(prop)

    def remove_elemental_pair_property(self, property):
        """Function to remove an elemental pair property from the list of
        elemental properties.

        Parameters
        ----------
        property : str
            Property to be removed.

        """

        if property in self.elemental_pair_properties:
            self.elemental_pair_properties.remove(property)

    def remove_elemental_pair_properties(self, properties):
        """Function to remove a list of elemental pair properties from the
        list of elemental properties.

        Parameters
        ----------
        properties : array-like
            Properties to be removed. A list of strings containing property
            names.

        """

        for prop in properties:
            self.remove_elemental_pair_property(prop)

    def generate_features(self, entries):
        """Function to generate features as mentioned in the class description.

        Parameters
        ----------
        entries : array-like
            Compositions for which features are to be generated. A list of
            CompositionEntry's.

        Returns
        ----------
        features : DataFrame
            Features for the given entries. Pandas data frame containing the
            names and values of the descriptors.

        Raises
        ------
        ValueError
            If no elemental properties are set.
            If input is not of type list.
            If items in the list are not CompositionEntry instances.

        """

        # Initialize lists of feature values and headers for pandas data frame.
        feat_values = []
        feat_headers = []

        # Make sure that there is at least one elemental pair property provided.
        if not self.elemental_pair_properties:
            raise ValueError("No elemental property is set. Add at least one "
                             "property to compute meaningful descriptors.")

        # If the dictionary containing the property values is empty,
        # load values into it.
        if not self.pair_lookup_data:
            self.load_pair_lookup_data()

        # Raise exception if input argument is not of type list of
        # Composition Entry's.
        if not isinstance(entries, list):
            raise ValueError("Argument should be of type list of "
                             "CompositionEntry's")
        elif (entries and not isinstance(entries[0], CompositionEntry)):
            raise ValueError("Argument should be of type list of "
                             "CompositionEntry's")

        # Insert header names here.
        n_statistics = 5
        for prop in self.elemental_pair_properties:
            feat_headers.append("binary_max_" + prop)
            feat_headers.append("binary_min_" + prop)
            feat_headers.append("binary_range_" + prop)
            feat_headers.append("binary_mean_" + prop)
            feat_headers.append("binary_variance_" + prop)

        for entry in entries:
            tmp_list = []
            elem_ids = entry.get_element_ids()
            elem_fractions = entry.get_element_fractions()

            if len(elem_fractions) == 1:
                for i in range(n_statistics):
                    tmp_list.append(np.nan)
                feat_values.append(tmp_list)
                continue

            pair_weights = []
            for i in range(len(elem_fractions)):
                for j in range(i):
                    pair_weights.append(elem_fractions[i]*elem_fractions[j])

            total_sum = sum(pair_weights)
            for i in range(len(pair_weights)):
                pair_weights[i] /= total_sum

            # Look up values for each pair property.
            for prop in self.elemental_pair_properties:
                tmp_prop = []

                for i in range(len(elem_fractions)):
                    e_i = elem_ids[i]
                    for j in range(i):
                        e_j = elem_ids[j]
                        idx_1 = max(e_i, e_j)
                        idx_2 = min(e_i, e_j)
                        tmp_prop.append(self.pair_lookup_data[prop][idx_1][
                                            idx_2])

                max_ = max(tmp_prop)
                min_= min(tmp_prop)
                range_ = max_ - min_
                mean_ = np.average(tmp_prop, weights=pair_weights)
                variance_ = np.average([abs(x - mean_) for x in tmp_prop],
                                      weights=pair_weights)
                tmp_list.append(max_)
                tmp_list.append(min_)
                tmp_list.append(range_)
                tmp_list.append(mean_)
                tmp_list.append(variance_)


            feat_values.append(tmp_list)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        return features
