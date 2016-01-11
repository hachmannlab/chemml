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
from sct_utils import isfloat, islist, istuple, isnpdot

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
                 'RobustScaler'         : RobustScaler,
                 'Normalizer'           : Normalizer,
                 'Binarizer'            : Binarizer,
                 'OneHotEncoder'        : OneHotEncoder,
                 'PolynomialFeatures'   : PolynomialFeatures,
                 'FunctionTransformer'  : FunctionTransformer
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

def gen_skl_preprocessing(fi, skl_funct, cml_funct, frames):
    if cml_funct:
        if "cheml: %s"%cml_funct not in imports:
            pyscript.write("from cheml import %s\n"%cml_funct)
            imports.append("cheml: %s"%cml_funct)    
    if "sklearn: %s"%skl_funct not in imports:
        pyscript.write("from sklearn.preprocessing import %s\n"%skl_funct)
        imports.append("sklearn: %s"%skl_funct)
    
    line = "%s_%s = %s(" %(skl_funct,'API',skl_funct)
    param_count = 0
    for parameter in cmls[fi].iterchildren():
        param_count += 1
        if parameter == 'None' or isfloat(parameter) or islist(parameter) or istuple(parameter) or isnpdot(parameter):
            line += """;%s = %s"""%(parameter.tag,cmls[fi][parameter.tag])
        else:
            line += """;%s = "%s" """%(parameter.tag,parameter)
    line += ')'
    line = line.replace('(;','(')
    
    if param_count > 1 :
        write_split(line)
    else:
        pyscript.write(line + '\n')
    
    for frame in frames:
        line = """%s_%s_%s, %s = preprocessing.transformer_dataframe(transformer = %s_%s;df = %s)"""\
            %(skl_funct,'API',frame,frame,skl_funct,'API',frame)
        write_split(line)


##################################################################################################

def INPUT(fi):
    """(INPUT):
		Read input files.
		pandas.read_csv: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    block ('begin', 'Import' )
    pyscript.write("import numpy as np\n")
    imports.append("numpy")
    block ('end', 'Import' )
    
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
    skl_funct = 'StandardScaler'
    cml_funct = 'preprocessing'
    frames=['data']  # ,'target'
    
    block ('begin', '%s'%skl_funct )   
    gen_skl_preprocessing(fi, skl_funct, cml_funct, frames)    
    block ('end', '%s'%skl_funct )
				
									###################

def MinMaxScaler(fi):
    """(MinMaxScaler):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler    
    """
    skl_funct = 'MinMaxScaler'
    cml_funct = 'preprocessing'
    frames=['data']  # ,'target'
    
    block ('begin', '%s'%skl_funct )   
    gen_skl_preprocessing(fi, skl_funct, cml_funct, frames)    
    block ('end', '%s'%skl_funct )
				
									###################

def MaxAbsScaler(fi):
    """(MaxAbsScaler):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler    
    """
    skl_funct = 'MaxAbsScaler'
    cml_funct = 'preprocessing'
    frames=['data']  # ,'target'
    
    block ('begin', '%s'%skl_funct )   
    gen_skl_preprocessing(fi, skl_funct, cml_funct, frames)    
    block ('end', '%s'%skl_funct )

									###################

def RobustScaler(fi):
    """(RobustScaler):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler    
    """
    skl_funct = 'RobustScaler'
    cml_funct = 'preprocessing'
    frames=['data']  # ,'target'
    
    block ('begin', '%s'%skl_funct )   
    gen_skl_preprocessing(fi, skl_funct, cml_funct, frames)    
    block ('end', '%s'%skl_funct )

									###################

def Normalizer(fi):
    """(Normalizer):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer    
    """
    skl_funct = 'Normalizer'
    cml_funct = 'preprocessing'
    frames=['data']  # ,'target'
    
    block ('begin', '%s'%skl_funct )   
    gen_skl_preprocessing(fi, skl_funct, cml_funct, frames)    
    block ('end', '%s'%skl_funct )

									###################

def Binarizer(fi):
    """(Binarizer):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer    
    """
    skl_funct = 'Binarizer'
    cml_funct = 'preprocessing'
    frames=['data']  # ,'target'
    
    block ('begin', '%s'%skl_funct )   
    gen_skl_preprocessing(fi, skl_funct, cml_funct, frames)    
    block ('end', '%s'%skl_funct )

									###################

def OneHotEncoder(fi):
    """(OneHotEncoder):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html    
    """
    skl_funct = 'OneHotEncoder'
    cml_funct = 'preprocessing'
    frames=['data']  # ,'target'
    
    block ('begin', '%s'%skl_funct )   
    gen_skl_preprocessing(fi, skl_funct, cml_funct, frames)    
    block ('end', '%s'%skl_funct )

									###################

def PolynomialFeatures(fi):
    """(PolynomialFeatures):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures   
    """
    skl_funct = 'PolynomialFeatures'
    cml_funct = 'preprocessing'
    frames=['data']  # ,'target'
    
    block ('begin', '%s'%skl_funct )   
    gen_skl_preprocessing(fi, skl_funct, cml_funct, frames)    
    block ('end', '%s'%skl_funct )

									###################

def FunctionTransformer(fi):
    """(FunctionTransformer):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer   
    """
    skl_funct = 'FunctionTransformer'
    cml_funct = 'preprocessing'
    if cmls[fi].pass_y :
        frames=['data','target']  # ,'target'
    else:
        frames=['data']  # ,'target'
    
    block ('begin', '%s'%skl_funct )   
    gen_skl_preprocessing(fi, skl_funct, cml_funct, frames)    
    block ('end', '%s'%skl_funct )

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



								  


	


