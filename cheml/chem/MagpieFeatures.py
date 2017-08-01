# TODO: Write unit tests.
# TODO: Implement feature extraction parts of Magpie in python.
# TODO: Allow multiple types of input file, not just csv.
import sys
import os
import csv
import numpy as np

def getFeatures(composition_file_str, verbose=False, file_type='csv'):
    """
    This function computes the features as described in the article
    Ward, L., et. al.. npj Computational Materials. 2 (2016), 16028
    and using Magpie (https://bitbucket.org/wolverton/magpie/). The
    function reads the element compositions from the input file
    composition_file_str located in the datasets subdirectory
    within the magpie directory. The function creates a script file
    that is used as input for magpie and saves it in the examples
    subdirectory within the magpie directory. Then it runs magpie
    with the input script file and generates corresponding features.
    Features and feature headers are saved to a file as well as returned as
    numpy arrays.

    :parameter composition_file_str:
    Name of the input file that contains the compositions of materials
    in a csv format. The file itself should be of a specific format.
    The first row contains the header information followed by rows
    containing the composition information of the materials. The last
    column contains dummy values just because Magpie requires that it
    be present. Once this function is implemented completely in python
    the dummy column need not exist as part of the input file. An
    example of the contents of a composition_file_str is given below:

    name dummy_column
    Fe,2.0,O,3.0 1.0
    Na,1.0,Cl,1.0 1.0
    Sc,4.0,Si,1.0,Al,1.0,C,1.0,N,1.0, 1.0

    :parameter verbose:
    Flag to be used mainly for debugging purposes as it prints out a lot of
    information. Default value is False.

    :parameter file_type:
    String that denotes the type of file used as input for the function,
    i.e., the type of composition_file_str. Default is 'csv'.

    :return: feature_headers, feature_values
    Feature headers and values as numpy arrays
    """

    # Add comments to explain what the script file does
    script_str = "// This script shows to import data sets in Magpie\n"
    script_str += "// and generate features. Lines starting with '//' are " \
                  "comments.\n\n"

    # Load data set
    script_str += "// Load in a dataset of compounds\n"
    script_str += "data = new data.materials.CompositionDataset\n"
    composition_file_path = "../datasets/{}".format(composition_file_str)
    script_str += "data import {}\n\n".format(composition_file_path)

    # Define where to find elemental property data
    script_str += "// Define where to find elemental property data\n"
    script_str += "data attributes properties directory Lookup Data/\n"

    # Select which set of elemental properties to use for features
    script_str += "// Select which set of elemental properties to use for " \
                  "attributes\n"
    script_str += "data attributes properties add set general\n\n"

    # Generate new features
    script_str += "// Generate new attributes\n"
    script_str += "data attributes generate\n"

    # Save features to csv file
    split_str = ".{}".format(file_type)
    tmp_words = composition_file_str.split(split_str)
    composition_file_name = tmp_words[0]
    features_str = composition_file_name + "_featuresPy"
    script_str += "save data {} csv\n\n".format(features_str)
    script_str += "exit"

    if (verbose):
        print "Magpie input script file contents:"
        print script_str

    # Create input script file for magpie
    input_str = "../examples/{}.in".format(composition_file_name)
    try:
        script_file = open(input_str, 'w')
    except IOError:
        print "Error writing to file {}\n".format(script_str)
    script_file.write(script_str)
    script_file.close()

    # Run magpie
    cmd_str = "java -jar ../dist/Magpie.jar ../examples/{}.in".format(composition_file_name)

    if (verbose):
        print "Java command:"
        print cmd_str

    ret_value = os.system(cmd_str)
    if (ret_value != 0):
        print "Something went wrong with the execution of the command:"
        print cmd_str
        sys.exit(1)

    # Read features from features file
    feature_file_name = features_str + ".csv"
    f_headers = []
    f_values = []
    flag = False
    with open(feature_file_name) as f1:
        file_reader = csv.reader(f1)
        for row in file_reader:
            if (not flag):
                f_headers.append(list(row))
                flag = True
            else:
                f_values.append(list(row))

    feature_headers = np.asarray(f_headers)
    feature_values = np.asarray(f_values)
    return (feature_headers, feature_values)

# Include unit tests here
if __name__ == "__main__":
    f1_headers, f1_values = getFeatures("example.csv",verbose=True)
    feat_headers = ['NComp', 'Comp_L2Norm', 'Comp_L3Norm', 'Comp_L5Norm',
                    'Comp_L7Norm', 'Comp_L10Norm', 'mean_Number',
                    'maxdiff_Number', 'dev_Number', 'max_Number',
                    'min_Number', 'most_Number', 'mean_MendeleevNumber',
                    'maxdiff_MendeleevNumber', 'dev_MendeleevNumber',
                    'max_MendeleevNumber', 'min_MendeleevNumber',
                    'most_MendeleevNumber', 'mean_AtomicWeight',
                    'maxdiff_AtomicWeight', 'dev_AtomicWeight',
                    'max_AtomicWeight', 'min_AtomicWeight',
                    'most_AtomicWeight', 'mean_MeltingT', 'maxdiff_MeltingT',
                    'dev_MeltingT', 'max_MeltingT', 'min_MeltingT',
                    'most_MeltingT', 'mean_Column', 'maxdiff_Column',
                    'dev_Column', 'max_Column', 'min_Column', 'most_Column',
                    'mean_Row', 'maxdiff_Row', 'dev_Row', 'max_Row',
                    'min_Row', 'most_Row', 'mean_CovalentRadius',
                    'maxdiff_CovalentRadius', 'dev_CovalentRadius',
                    'max_CovalentRadius', 'min_CovalentRadius',
                    'most_CovalentRadius', 'mean_Electronegativity',
                    'maxdiff_Electronegativity', 'dev_Electronegativity',
                    'max_Electronegativity', 'min_Electronegativity',
                    'most_Electronegativity', 'mean_NsValence',
                    'maxdiff_NsValence', 'dev_NsValence', 'max_NsValence',
                    'min_NsValence', 'most_NsValence', 'mean_NpValence',
                    'maxdiff_NpValence', 'dev_NpValence', 'max_NpValence',
                    'min_NpValence', 'most_NpValence', 'mean_NdValence',
                    'maxdiff_NdValence', 'dev_NdValence', 'max_NdValence',
                    'min_NdValence', 'most_NdValence', 'mean_NfValence',
                    'maxdiff_NfValence', 'dev_NfValence', 'max_NfValence',
                    'min_NfValence', 'most_NfValence', 'mean_NValance',
                    'maxdiff_NValance', 'dev_NValance', 'max_NValance',
                    'min_NValance', 'most_NValance', 'mean_NsUnfilled',
                    'maxdiff_NsUnfilled', 'dev_NsUnfilled', 'max_NsUnfilled',
                    'min_NsUnfilled', 'most_NsUnfilled', 'mean_NpUnfilled',
                    'maxdiff_NpUnfilled', 'dev_NpUnfilled', 'max_NpUnfilled',
                    'min_NpUnfilled', 'most_NpUnfilled', 'mean_NdUnfilled',
                    'maxdiff_NdUnfilled', 'dev_NdUnfilled', 'max_NdUnfilled',
                    'min_NdUnfilled', 'most_NdUnfilled', 'mean_NfUnfilled',
                    'maxdiff_NfUnfilled', 'dev_NfUnfilled', 'max_NfUnfilled',
                    'min_NfUnfilled', 'most_NfUnfilled', 'mean_NUnfilled',
                    'maxdiff_NUnfilled', 'dev_NUnfilled', 'max_NUnfilled',
                    'min_NUnfilled', 'most_NUnfilled', 'mean_GSvolume_pa',
                    'maxdiff_GSvolume_pa', 'dev_GSvolume_pa',
                    'max_GSvolume_pa', 'min_GSvolume_pa', 'most_GSvolume_pa',
                    'mean_GSbandgap', 'maxdiff_GSbandgap', 'dev_GSbandgap',
                    'max_GSbandgap', 'min_GSbandgap', 'most_GSbandgap',
                    'mean_GSmagmom', 'maxdiff_GSmagmom', 'dev_GSmagmom',
                    'max_GSmagmom', 'min_GSmagmom', 'most_GSmagmom',
                    'mean_SpaceGroupNumber', 'maxdiff_SpaceGroupNumber',
                    'dev_SpaceGroupNumber', 'max_SpaceGroupNumber',
                    'min_SpaceGroupNumber', 'most_SpaceGroupNumber',
                    'frac_sValence', 'frac_pValence', 'frac_dValence',
                    'frac_fValence', 'CanFormIonic', 'MaxIonicChar',
                    'MeanIonicChar']
