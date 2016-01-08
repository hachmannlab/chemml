#!/usr/bin/env python

PROGRAM_NAME = "CheML"
PROGRAM_VERSION = "v0.0.1"
REVISION_DATE = "2015-06-23"
AUTHORS = "Johannes Hachmann (hachmann@buffalo.edu) and Mojtaba Haghighatlari (mojtabah@buffalo.edu)"
CONTRIBUTORS = " "
DESCRIPTION = "ChemML is a machine learning and informatics program suite for the chemical and materials sciences."

# Version history timeline (move to CHANGES periodically):
# v0.0.1 (2015-06-02): complete refactoring of original CheML code in new package format


###################################################################################################
#TODO:
# -restructure more general functions into modules
###################################################################################################

import sys
import os
import time
import copy
import argparse
from lxml import objectify, etree

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
"""*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*

 									CheML FUNCTIONS 		

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#"""
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*									  

def _status_parser(fis, element):
    for sub_element in element.iterchildren():
        if sub_element.attrib['status']=='on':
            fis.append(sub_element.tag)
#         elif sub_element.attrib['status']=='sub':
#             _status_parser(fis, sub_element)
    return fis
    
def main(SCRIPT_NAME):
    """main:
        Driver of ChemML
    """
    global cmls
    global pyscript
    global imports

    time_start = time.time()
    # TODO: add banner
    ## SCRIPT PARSER
    doc = etree.parse(SCRIPT_NAME)
    cmls = etree.tostring(doc) 	# ChemML script : cmls
    #print cmls
    cmls = objectify.fromstring(cmls)
    objectify.deannotate(cmls)
    etree.cleanup_namespaces(cmls)
    print "\n"
    print(objectify.dump(cmls))
    fis = []
    fis = _status_parser(fis, cmls)

    ## CHECK SCRIPT'S REQUIREMENTS    
    functions = [element.attrib['function'] for element in cmls.iterchildren()]
    if "INPUT" not in functions:
        raise RuntimeError("cheml requires input data")
        # TODO: check typical error names		

    ## PYTHON SCRIPT
    if "OUTPUT" in functions:
        output_ind = functions.index("OUTPUT")
        pyscript_file = cmls[fis[output_ind]].filename_pyscript.pyval
    else:
        pyscript_file = "CheML_PyScript.py"
    pyscript = open(pyscript_file,'w',0)
    imports = []    

    ## implementing orders
    functions = {'INPUT'                : INPUT,
                 'OUTPUT'               : OUTPUT,
                 'MISSING_VALUES'       : MISSING_VALUES,
                 'StandardScaler'       : StandardScaler,
                 'MinMaxScaler'         : MinMaxScaler,
                 'MaxAbsScaler'         : MaxAbsScaler,
                 'RobustScaler'         : RobustScaler 
                }

    for fi in fis:
        if cmls[fi].attrib['function'] not in functions:
            raise NameError("name %s is not defined"%cmls[fi].attrib['function'])
        functions[cmls[fi].attrib['function']](fi)
    print "\n"
    print "NOTES:"
    print "* The python script with name '%s' has been stored in the current directory."\
        %pyscript_file
    print "** list of required 'package: module's in the python script:", imports
    print "\n"

    return 0    #successful termination of program
    
##################################################################################################

def write_split(line):
    """(write_split):
        Write the invoked line of python code in multiple lines.
    """ 
    pran_ind = line.index('(')
    function = line[:pran_ind+1]
    options = line[pran_ind+1:].split(';')
    spaces = len(function)
    lines = [function + options[0]+','] + [' ' * spaces + options[i] +',' for i in range(1,len(options)-1)] + [' ' * spaces + options[-1]]
    
    for text in lines:
        pyscript.write(text+'\n')

##################################################################################################

def block(state, function):
    """(block):
        Sign begin and end of a function.
    """
    secondhalf = 71-len(function)-2-27 
    if state == 'begin':
        pyscript.write('#'*27 + ' ' + function + '\n')
    if state == 'end':
        pyscript.write('#'*27 + '\n')
        pyscript.write('\n') 
	
##################################################################################################

def INPUT(fi):
    """(INPUT):
		Read input files.
		pandas.read_csv: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    block ('begin', 'INPUT' )
    pyscript.write("import pandas as pd\n")
    imports.append("pandas")
    line = "data = pd.read_csv('%s';sep = %s;skiprows = %s;header = %s)"\
        %(cmls[fi].data_path, cmls[fi].data_delimiter,cmls[fi].data_skiprows,\
        cmls[fi].data_header)
    write_split(line)
    line = "target = pd.read_csv('%s';sep = %s;skiprows = %s;header = %s)"\
        %(cmls[fi].target_path, cmls[fi].target_delimiter,\
        cmls[fi].target_skiprows,cmls[fi].target_header)
    write_split(line) 	
    block ('end', 'INPUT' )
    
									###################
    
def OUTPUT(fi):
    """(OUTPUT):
		Open output files.
    """
    block ('begin', 'OUTPUT' )
    if "cheml: initialization" not in imports:
        pyscript.write("from cheml import initialization\n")
        imports.append("cheml: initialization")
    line = "output_directory, log_file, error_file = initialization.output(output_directory = '%s';logfile = '%s';errorfile = '%s')"\
        %(cmls[fi].path, cmls[fi].filename_logfile, cmls[fi].filename_errorfile)
    write_split(line)
    block ('end', 'OUTPUT')
    
									###################

def MISSING_VALUES(fi):
    """(MISSING_VALUES):
		Handle missing values.
    """
    block ('begin', 'MISSING_VALUES')
    if "cheml: preprocessing" not in imports:
        pyscript.write("from cheml import preprocessing\n")
        imports.append("cheml: preprocessing")
    line = """missval = preprocessing.missing_values(strategy = '%s';string_as_null = %s;inf_as_null = %s;missing_values = %s)"""\
        %(cmls[fi].strategy,cmls[fi].string_as_null,cmls[fi].inf_as_null,cmls[fi].missing_values)
    write_split(line)
    line = """data = missval.fit(data)"""
    pyscript.write(line + '\n')
    line = """target = missval.fit(target)"""
    pyscript.write(line + '\n')
    if cmls[fi].strategy in ['zero', 'ignore', 'interpolate']:
        line = """data, target = missval.transform(data, target)"""
        pyscript.write(line + '\n')
    elif cmls[fi].strategy in ['mean', 'median', 'most_frequent']:
        if "sklearn: Imputer" not in imports:
            pyscript.write("from sklearn.preprocessing import Imputer\n")
            imports.append("sklearn: Imputer")
        line = """imp = Imputer(strategy = '%s';missing_values = 'NaN';axis = 0;verbose = 0;copy = True)"""\
            %(cmls[fi].strategy)
        write_split(line)
        line = """imp_data, data = preprocessing.Imputer_dataframe(imputer = imp, df = data)"""
        pyscript.write(line + '\n')
        line = """imp_target, target = preprocessing.Imputer_dataframe(imputer = imp, df = target)"""
        pyscript.write(line + '\n')
    block ('end', 'MISSING_VALUES')
				
									###################

def StandardScaler(fi):
    """(StandardScaler):
		http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
    """
    block ('begin', 'StandardScaler')
    if "cheml: preprocessing" not in imports:
        pyscript.write("from cheml import preprocessing\n")
        imports.append("cheml: preprocessing")    
    if "sklearn: StandardScaler" not in imports:
        pyscript.write("from sklearn.preprocessing import StandardScaler\n")
        imports.append("sklearn: StandardScaler")

    line = """scaler = StandardScaler(copy = %s;with_mean = %s;with_std = %s)"""\
        %(cmls[fi].copy,cmls[fi].with_mean,cmls[fi].with_std)
    write_split(line)
    line = """data_scaler, data = preprocessing.Scaler_dataframe(scaler = scaler, df = data)"""
    pyscript.write(line + '\n')
    line = """target_scaler, target = preprocessing.Scaler_dataframe(scaler = scaler, df = target)"""
    pyscript.write(line + '\n')
    block ('end', 'StandardScaler')
				
									###################

def MinMaxScaler(fi):
    """(MinMaxScaler):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler    
    """
    block ('begin', 'MinMaxScaler')
    if "cheml: preprocessing" not in imports:
        pyscript.write("from cheml import preprocessing\n")
        imports.append("cheml: preprocessing")    
    if "sklearn: MinMaxScaler" not in imports:
        pyscript.write("from sklearn.preprocessing import MinMaxScaler\n")
        imports.append("sklearn: MinMaxScaler")

    line = """min_max_scaler = MinMaxScaler(feature_range = %s;copy = %s)"""\
        %(cmls[fi].feature_range,cmls[fi].copy)
    write_split(line)
    line = """data_min_max_scaler, data = preprocessing.Scaler_dataframe(scaler = min_max_scaler, df = data)"""
    pyscript.write(line + '\n')
    line = """target_min_max_scaler, target = preprocessing.Scaler_dataframe(scaler = min_max_scaler, df = target)"""
    pyscript.write(line + '\n')
    block ('end', 'MinMaxScaler')
				
									###################

def MaxAbsScaler(fi):
    """(MaxAbsScaler):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler    
    """
    block ('begin', 'MaxAbsScaler')
    if "cheml: preprocessing" not in imports:
        pyscript.write("from cheml import preprocessing\n")
        imports.append("cheml: preprocessing")    
    if "sklearn: MaxAbsScaler" not in imports:
        pyscript.write("from sklearn.preprocessing import MaxAbsScaler\n")
        imports.append("sklearn: MaxAbsScaler")

    line = """max_abs_scaler = MaxAbsScaler(copy = %s)"""\
        %(cmls[fi].copy)
    pyscript.write(line + '\n')
    line = """data_min_max_scaler, data = preprocessing.Scaler_dataframe(scaler = max_abs_scaler, df = data)"""
    pyscript.write(line + '\n')
    line = """target_min_max_scaler, target = preprocessing.Scaler_dataframe(scaler = max_abs_scaler, df = target)"""
    pyscript.write(line + '\n')
    block ('end', 'MaxAbsScaler')

									###################

def RobustScaler(fi):
    """(RobustScaler):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler    
    """
    block ('begin', 'RobustScaler')
    if "cheml: preprocessing" not in imports:
        pyscript.write("from cheml import preprocessing\n")
        imports.append("cheml: preprocessing")    
    if "sklearn: RobustScaler" not in imports:
        pyscript.write("from sklearn.preprocessing import RobustScaler\n")
        imports.append("sklearn: RobustScaler")

    line = """robust_scaler = RobustScaler(with_centering = %s;with_scaling = %s;copy = %s)"""\
        %(cmls[fi].with_centering,cmls[fi].with_scaling,cmls[fi].copy)
    write_split(line)
    line = """data_robust_scaler, data = preprocessing.Scaler_dataframe(scaler = robust_scaler, df = data)"""
    pyscript.write(line + '\n')
    line = """target_robust_scaler, target = preprocessing.Scaler_dataframe(scaler = robust_scaler, df = target)"""
    pyscript.write(line + '\n')
    block ('end', 'RobustScaler')

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
"""*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*

 									  CheML PySCRIPT 		
																						
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#"""
#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*					

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="ChemML will be started by specifying a script file as a todo list")
    parser.add_argument("-i", type=str, required=True, help="input directory: must include the script file name and its format")                    		
    args = parser.parse_args()            		
    SCRIPT_NAME = args.i      
    main(SCRIPT_NAME)   #numbering of sys.argv is only meaningful if it is launched as main
    
else:
    sys.exit("Sorry, must run as driver...")



								  


	


