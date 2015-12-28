#!/usr/bin/env python

PROGRAM_NAME = "CheML"
PROGRAM_VERSION = "v0.0.1"
REVISION_DATE = "2015-06-23"
AUTHORS = "Johannes Hachmann (hachmann@buffalo.edu) and Mojtaba Haghighatlari (mojtabah@buffalo.edu)"
CONTRIBUTORS = """ """
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

def _status_parser(todo_order, element):
    for sub_element in element.iterchildren():
        if sub_element.attrib['status']=='on':
            todo_order.append(sub_element.tag)
        elif sub_element.attrib['status']=='sub':
            _status_parser(todo_order, sub_element)
    return todo_order
    
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
    todo_order = []
    _status_parser(todo_order, cmls)

    ## CHECK SCRIPT'S REQUIREMENTS    
    if "INPUT" not in todo_order or cmls.INPUT.data_path == "enter data path" :
        raise RuntimeError("cheml requires input data")
        # TODO: check typical error names		

    ## PYTHON SCRIPT
    if "OUTPUT" in todo_order:
        pyscript_file = cmls.OUTPUT.filename_pyscript.pyval
    else:
        pyscript_file = "ChemML_PyScript.py"
    pyscript = open(pyscript_file,'w',0)
    imports = []    

    ## implementing orders
    functions = {'INPUT'       : INPUT,
                 'OUTPUT'      : OUTPUT,
                 'MISSING_VALUES'     : MISSING_VALUES 
                }

    for order in todo_order:
        if order not in functions:
            raise NameError("name %s is not defined"%order)
        functions[order]()
    print "\n"
    print "NOTE:"
    print "* The python script with name '%s' has been stored in the current directory."\
     %cmls.OUTPUT.filename_pyscript.pyval
    print "** list of required 'packge: module' in the python script:", imports
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
        pyscript.write('#'*27 + ' ' + function + ' ' + '#'*secondhalf + '\n')
    if state == 'end':
        pyscript.write('#'*71 + '\n')
        pyscript.write('\n') 
	
##################################################################################################

def INPUT():
    """(INPUT):
		Read input files.
		pandas.read_csv: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    block ('begin', 'INPUT' )
    pyscript.write("import pandas as pd\n")
    imports.append("pandas")
    line = "data = pd.read_csv('%s';sep = %s;skiprows = %s;header = %s)"\
        %(cmls.INPUT.data_path, cmls.INPUT.data_delimiter,cmls.INPUT.data_skiprows,\
        cmls.INPUT.data_header)
    write_split(line)
    line = "target = pd.read_csv('%s';sep = %s;skiprows = %s;header = %s)"\
        %(cmls.INPUT.target_path, cmls.INPUT.target_delimiter,\
        cmls.INPUT.target_skiprows,cmls.INPUT.target_header)
    write_split(line) 	
    block ('end', 'INPUT' )
    
									###################
    
def OUTPUT():
    """(OUTPUT):
		Open output files.
    """
    block ('begin', 'OUTPUT' )
    if "cheml: initialization" not in imports:
        pyscript.write("from cheml import initialization\n")
        imports.append("cheml: initialization")
    line = "output_directory, log_file, error_file, tmp_folder = initialization.output(output_directory = '%s';logfile = '%s';errorfile = '%s')"\
    %(cmls.OUTPUT.path, cmls.OUTPUT.filename_logfile, cmls.OUTPUT.filename_errorfile)
    write_split(line)
    block ('end', 'OUTPUT' )
    
									###################

def MISSING_VALUES():
    """(MISSING_VALUES):
		Handle missing values.
    """
    block ('begin', 'MISSING_VALUES' )
    if "cheml: preprocessing" not in imports:
        pyscript.write("from cheml import preprocessing\n")
        imports.append("cheml: preprocessing")
    line = """data, target = preprocessing.missing_values(method = '%s';string_as_null = %s;inf_as_null = %s;missing_values = "%s")"""%(cmls.PREPROCESSING.MISSING_VALUES.method.pyval,cmls.PREPROCESSING.MISSING_VALUES.string_as_null.pyval,cmls.PREPROCESSING.MISSING_VALUES.inf_as_null.pyval,cmls.PREPROCESSING.MISSING_VALUES.missing_values.pyval)
    write_split(line)    
    block ('end', 'MISSING_VALUES' )


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



								  


	


