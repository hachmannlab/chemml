########################### INPUT #####################################
import pandas as pd
data = pd.read_csv('benchmarks/homo_dump/sample_50/data_NOsmi_50.csv',
                   sep = None,
                   skiprows = 0,
                   header = [0,1])
target = pd.read_csv('benchmarks/homo_dump/sample_50/homo_50.csv',
                     sep = None,
                     skiprows = 0,
                     header = 0)
#######################################################################

########################### OUTPUT ####################################
from cheml import initialization
output_directory, log_file, error_file, tmp_folder = initialization.output(output_directory = 'ChemML.out',
                                                                           logfile = 'log.txt',
                                                                           errorfile = 'error.txt')
#######################################################################

########################### MISSING_VALUES ############################
from cheml import preprocessing
rata, target = preprocessing.missing_values(method = 'mean',
                                            string_as_null = True,
                                            inf_as_null = True,
                                            missing_values = "False")
#######################################################################

