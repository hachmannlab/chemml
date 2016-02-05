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
from sct_utils import isfloat, islist, istuple, isnpdot, std_datetime_str

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
"""*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*

 									CheML FUNCTIONS 		

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#"""
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*									  

def _block_finder(script, blocks={}, item=-1):
    for line in script:
        if '##' in line:
            item += 1    
            blocks[item] = [line]
            continue
        elif '#' not in line and '%' in line:
            blocks[item].append(line)
            continue
    return blocks

def _functions(line):
    if '%' in line:
        function = line[line.index('##')+2:line.index('%')].strip()
    else:
        function = line[line.index('##')+2:].strip()
    return function

def _options(blocks, item=-1):
    cmls = []
    for i in xrange(len(blocks)):
        item += 1
        block = blocks[i]
        cmls.append({"function": _functions(block[0]),
                     "parameters": {}})
        for line in block:
            while '%%' in line:
                line = line[line.index('%%')+2:].strip()
                if '%' in line:
                    args = line[:line.index('%')].strip()
                else:
                    args = line.strip()
                param = args[:args.index('=')].strip()
                val = args[args.index('=')+1:].strip()
                cmls[item]['parameters'][param] = "%s"%val
    return cmls

def _print_out(cmls):
    item = 0
    for block in cmls:
        item+=1
        line = '%s\n' %(block['function'])
        line = line.rstrip("\n")
        print '%i'%item+' '*(4-len(str(item)))+'function = '+line
        for param in block['parameters']:
            line = '%s = %s\n'%(param,block['parameters'][param])
            line = line.rstrip("\n")
            print '        '+line 

def _sub_function(block,line):
    line = line.split('__')
    imp = line[0]
    block['sub_function'] = line[0].split('.')[-1]
    block['sub_parameters'] = {}
    for arg in line[1:]:
        param = arg.split('=')[0].strip()
        val = arg.split('=')[1].strip()
        block['sub_parameters'][param] = "%s"%val
    return imp 

def main(SCRIPT_NAME):
    """main:
        Driver of ChemML
    """
    global cmls
    global cmlnb
    global it
    it = -1
    
    script = open(SCRIPT_NAME,'r')
    script = script.readlines()
    blocks = _block_finder(script)
    cmls = _options(blocks)
    _print_out(cmls)
    
    ## CHECK SCRIPT'S REQUIREMENTS    
    called_functions = [block["function"] for block in cmls]
    if "INPUT" not in called_functions:
        raise RuntimeError("cheml requires input data")
        # TODO: check typical error names		

    ## PYTHON SCRIPT
    if "OUTPUT" in called_functions:
        output_ind = called_functions.index("OUTPUT")
        pyscript_file = cmls[output_ind]['parameters']['filename_pyscript'][1:-1]
    else:
        pyscript_file = "CheML_PyScript.py"
    cmlnb = {"blocks": [],
             "date": std_datetime_str('date'),
             "time": std_datetime_str('time'),
             "file_name": pyscript_file,
             "version": "1.1.0",
             "imports": []
            }
    
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
                 'FunctionTransformer'  : FunctionTransformer,
                 'VarianceThreshold'    : VarianceThreshold,
                 'SelectKBest'          : SelectKBest,
                 'SelectPercentile'     : SelectPercentile,
                 'SelectFpr'            : SelectFpr,
                 'SelectFdr'            : SelectFdr,
                 'SelectFwe'            : SelectFwe,
                 'RFE'                  : RFE,
                 'RFECV'                : RFECV,
                 'SelectFromModel'      : SelectFromModel,
                 'Trimmer'              : Trimmer,
                 'Uniformer'            : Uniformer,
                 'PCA'                  : PCA,
                 'KernelPCA'            : KernelPCA,
                 'RandomizedPCA'        : RandomizedPCA,
                 'LDA'                  : LDA
                 
                }

    for block in cmls:
        if block['function'] not in functions:
            raise NameError("name %s is not defined"%block['function'])
        else:
            it += 1
            cmlnb["blocks"].append({"function": block['function'],
                                    "imports": [],
                                    "source": []
                                    })
            functions[block['function']](block)
    
    ## write files
    pyscript = open(pyscript_file,'w',0)
    for block in cmlnb["blocks"]:
        pyscript.write(banner('begin', block["function"]))
        for line in block["imports"]:
            pyscript.write(line)
        pyscript.write('\n')
        for line in block["source"]:
            pyscript.write(line)
        pyscript.write(banner('end', block["function"]))
        pyscript.write('\n')
        
    print "\n"
    print "NOTES:"
    print "* The python script with name '%s' has been stored in the current directory."\
        %pyscript_file
    print "** list of required 'package: module's in the python script:"
    for item in cmlnb["imports"]:
        line = '    ' + item + '\n'
        line = line.rstrip("\n")
        print line
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
    lines = [function + options[0]+',\n'] + [' ' * spaces + options[i] +',\n' for i in range(1,len(options)-1)] + [' ' * spaces + options[-1]+'\n']
    return lines

##################################################################################################

def banner(state, function):
    """(banner):
        Sign begin and end of a function.
    """
    secondhalf = 71-len(function)-2-27 
    if state == 'begin':
        line = '#'*27 + ' ' + function + '\n'
        return line
    if state == 'end':
        line = '#'*27 + '\n'
        return line 
	
##################################################################################################

def handle_imports(called_imports):
    """
    called_imports: list of strings
    strings ex.:
    1- "cheml.preprocessing.missing_values" ==> from cheml.preprocessing import missing_values
    2- "numpy as np" ==> import numpy as np
    3- "sklearn.feature_selection as sklfs" ==> import sklearn.feature_selection as sklfs
    """
    for item in called_imports:
        if ' as ' in item:
            item = item.split(' as ')
            if item[0] not in cmlnb["imports"]:
                cmlnb["blocks"][it]["imports"].append("import %s as %s\n"%(item[0],item[-1]))
                cmlnb["imports"].append("%s"%item[0])
        elif '.' in item:
            item = item.split('.')
            if "%s: %s"%(item[0],item[-1]) not in cmlnb["imports"]:
                dir = '.'.join(item[:-1])
                cmlnb["blocks"][it]["imports"].append("from %s import %s\n"%(dir,item[-1]))
                cmlnb["imports"].append("%s: %s"%(item[0],item[-1]))

##################################################################################################

def handle_API(block, function = False):
    """
    make a class object with input arguments
    """
    if function:
        line = "%s_%s = %s(" %(function,'API',function)
    else:
        line = "%s_%s = %s(" %(block["function"],'API',block["function"])
    param_count = 0
    for parameter in block["parameters"]:
        param_count += 1
        line += """;%s = %s"""%(parameter,block["parameters"][parameter])
    line += ')'
    line = line.replace('(;','(')
    
    if param_count > 1 :
        cmlnb["blocks"][it]["source"] += write_split(line)
    else:
        cmlnb["blocks"][it]["source"].append(line + '\n')

##################################################################################################

def handle_subAPI(block):
    """
    make a sub-class object in another class
    """
    line = "%s_%s = %s(" %(block["sub_function"],'API',block["sub_function"])
    param_count = 0
    for parameter in block["sub_parameters"]:
        param_count += 1
        line += """;%s = %s"""%(parameter,block["sub_parameters"][parameter])
    line += ')'
    line = line.replace('(;','(')
    
    if param_count > 1 :
        cmlnb["blocks"][it]["source"] += write_split(line)
    else:
        cmlnb["blocks"][it]["source"].append(line + '\n')

##################################################################################################

def handle_transform(block, interface, function = False, which_df = 'data'):
    """
    calls related cheml class to deal with dataframe and API
    
    Parameters:
    -----------
    block: list of strings
        block of parameters for called class
    
    interface: string
        cheml class
    
    function: string
        name of main class/function
    
    which_df: string
        the data frames in the action, including:
            - data: input and output are only data  
            - target: input and output are only target
            - both: input is both of data and target, but output is only data
            - multiple: input and output are both of data and target
    """
    if which_df == 'data':
        if function:
            line = "data = %s(transformer = %s_API;df = data)"\
                %(interface, function)
        else:
            line = "data = %s(transformer = %s_API;df = data)"\
                %(interface, block["function"])
        cmlnb["blocks"][it]["source"] += write_split(line)
        
    elif which_df == 'target':
        if function:
            line = "target = %s(transformer = %s_API;df = target)"\
                %(interface, function)
        else:
            line = "target = %s(transformer = %s_API;df = target)"\
                %(interface, block["function"])
        cmlnb["blocks"][it]["source"] += write_split(line)
    
    elif which_df == 'both':
        if function:
            line = "data = %s(transformer = %s_API;df = data;tf = target)"\
                %(interface, function)
        else:
            line = "data = %s(transformer = %s_API;df = data;tf = target)"\
                %(interface, block["function"])
        cmlnb["blocks"][it]["source"] += write_split(line)
 
    elif which_df == 'multiple':
        if function:
            line = "data, target = %s(transformer = %s_API;df = data;tf = target)"\
                %(interface, function)
        else:
            line = "data, target = %s(transformer = %s_API;df = data;tf = target)"\
                %(interface, block["function"])
        cmlnb["blocks"][it]["source"] += write_split(line)
   
##################################################################################################

def handle_simple_transform(block, sub_function, function = False, which_df = 'data'):
    """
    calls related cheml class to deal with dataframe and API
    
    Parameters:
    -----------
    block: list of strings
        block of parameters for called class
        
    function: string
        name of main class/function
    
    which_df: string
        the data frames in the action, including:
            - data: input and output are only data  
            - target: input and output are only target
            - both: input is both of data and target, but output is only data
            - multiple: input and output are both of data and target
    """
    if which_df == 'data':
        if function:
            line = "data = %s.%s(data)"\
                %(function, sub_function)
        else:
            line = "data = %s.%s(data)"\
                %(block["function"], sub_function)
        cmlnb["blocks"][it]["source"].append(line + '\n')
        
    elif which_df == 'target':
        if function:
            line = "target = %s.%s(target)"\
                %(function, sub_function)
        else:
            line = "target = %s.%s(target)"\
                %(block["function"], sub_function)
        cmlnb["blocks"][it]["source"].append(line + '\n')
    
    elif which_df == 'both':
        if function:
            line = "data = %s.%s(data, target)"\
                %(function, sub_function)
        else:
            line = "data = %s.%s(data, target)"\
                %(block["function"], sub_function)
        cmlnb["blocks"][it]["source"].append(line + '\n')
 
    elif which_df == 'multiple':
        if function:
            line = "data, target = %s.%s(data, target)"\
                %(function, sub_function)
        else:
            line = "data, target = %s.%s(data, target)"\
                %(block["function"], sub_function)
        cmlnb["blocks"][it]["source"].append(line + '\n')
   
##################################################################################################

def INPUT(block):
    """(INPUT):
		Read input files.
		pandas.read_csv: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    handle_imports(["numpy as np","pandas as pd"])
    
    line = "data = pd.read_csv(%s;sep = %s;skiprows = %s;header = %s)"\
        %(block["parameters"]["data_path"], block["parameters"]["data_delimiter"],\
        block["parameters"]["data_skiprows"], block["parameters"]["data_header"])
    cmlnb["blocks"][it]["source"] += write_split(line)
    
    line = "target = pd.read_csv(%s;sep = %s;skiprows = %s;header = %s)"\
        %(block["parameters"]["target_path"], block["parameters"]["target_delimiter"],\
        block["parameters"]["target_skiprows"],block["parameters"]["target_header"])
    cmlnb["blocks"][it]["source"] += write_split(line) 	
									
									###################
def OUTPUT(block):
    """(OUTPUT):
		Open output files.
    """
    handle_imports(["cheml.initialization.output"])
    line = "output_directory, log_file, error_file = output(output_directory = %s;logfile = %s;errorfile = %s)"\
        %(block["parameters"]["path"], block["parameters"]["filename_logfile"],\
        block["parameters"]["filename_errorfile"])
    cmlnb["blocks"][it]["source"] += write_split(line)
									
									###################
def MISSING_VALUES(block):
    """(MISSING_VALUES):
		Handle missing values.
    """
    handle_imports(["cheml.preprocessing.missing_values"])
    handle_API(block, function = 'missing_values')
    line = """data = missing_values_API.fit(data)"""
    cmlnb["blocks"][it]["source"].append(line + '\n')
    line = """target = missing_values_API.fit(target)"""
    cmlnb["blocks"][it]["source"].append(line + '\n')
    if block["parameters"]["strategy"][1:-1] in ['zero', 'ignore', 'interpolate']:
        line = """data, target = missing_values_API.transform(data, target)"""
        cmlnb["blocks"][it]["source"].append(line + '\n')
    elif block["parameters"]["strategy"][1:-1] in ['mean', 'median', 'most_frequent']:
        handle_imports(["sklearn.preprocessing.Imputer","cheml.preprocessing.Imputer_dataframe"])
        handle_API(block, function = 'Imputer')
        handle_transform(block, interface = 'Imputer_dataframe' , function = 'Imputer', which_df = 'data')
        handle_transform(block, interface = 'Imputer_dataframe' , function = 'Imputer', which_df = 'target')
    									
									###################
def StandardScaler(block):
    """(StandardScaler):
		http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
    """
    handle_imports(["sklearn.preprocessing.StandardScaler","cheml.preprocessing.transformer_dataframe"])
    handle_API(block, function = 'StandardScaler')
    handle_transform(block, interface = 'transformer_dataframe', function = 'StandardScaler', which_df = 'data')
 									
									###################
def MinMaxScaler(block):
    """(MinMaxScaler):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler    
    """
    handle_imports(["sklearn.preprocessing.MinMaxScaler","cheml.preprocessing.transformer_dataframe"])
    handle_API(block, function = 'MinMaxScaler')
    handle_transform(block, interface = 'transformer_dataframe', function = 'MinMaxScaler', which_df = 'data')
									
									###################
def MaxAbsScaler(block):
    """(MaxAbsScaler):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler    
    """
    handle_imports(["sklearn.preprocessing.MaxAbsScaler","cheml.preprocessing.transformer_dataframe"])
    handle_API(block, function = 'MaxAbsScaler')
    handle_transform(block, interface = 'transformer_dataframe', function = 'MaxAbsScaler', which_df = 'data')
									
									###################
def RobustScaler(block):
    """(RobustScaler):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler    
    """
    handle_imports(["sklearn.preprocessing.RobustScaler","cheml.preprocessing.transformer_dataframe"])
    handle_API(block, function = 'RobustScaler')
    handle_transform(block, interface = 'transformer_dataframe', function = 'RobustScaler', which_df = 'data')

									###################
def Normalizer(block):
    """(Normalizer):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer    
    """
    handle_imports(["sklearn.preprocessing.Normalizer","cheml.preprocessing.transformer_dataframe"])
    handle_API(block, function = 'Normalizer')
    handle_transform(block, interface = 'transformer_dataframe', function = 'Normalizer', which_df = 'data')

									###################
def Binarizer(block):
    """(Binarizer):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer    
    """
    handle_imports(["sklearn.preprocessing.Binarizer","cheml.preprocessing.transformer_dataframe"])
    handle_API(block, function = 'Binarizer')
    handle_transform(block, interface = 'transformer_dataframe', function = 'Binarizer', which_df = 'data')

									###################
def OneHotEncoder(block):
    """(OneHotEncoder):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html    
    """
    handle_imports(["sklearn.preprocessing.OneHotEncoder","cheml.preprocessing.transformer_dataframe"])
    handle_API(block, function = 'OneHotEncoder')
    handle_transform(block, interface = 'transformer_dataframe', function = 'OneHotEncoder', which_df = 'data')

									###################
def PolynomialFeatures(block):
    """(PolynomialFeatures):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures   
    """
    handle_imports(["sklearn.preprocessing.PolynomialFeatures","cheml.preprocessing.transformer_dataframe"])
    handle_API(block, function = 'PolynomialFeatures')
    handle_transform(block, interface = 'transformer_dataframe', function = 'PolynomialFeatures', which_df = 'data')

									###################
def FunctionTransformer(block):
    """(FunctionTransformer):
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer   
    """
    handle_imports(["sklearn.preprocessing.FunctionTransformer","cheml.preprocessing.transformer_dataframe"])
    handle_API(block, function = 'FunctionTransformer')    
    if block["parameters"]["pass_y"]=='True' :
        handle_transform(block, interface = 'transformer_dataframe', function = 'FunctionTransformer', which_df = 'data')
    else:
        frames=['data']

									###################
def VarianceThreshold(block):
    """(VarianceThreshold):
        http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold    
    """
    handle_imports(["sklearn.feature_selection.VarianceThreshold","cheml.preprocessing.selector_dataframe"])
    handle_API(block, function = 'VarianceThreshold')
    handle_transform(block, interface = 'selector_dataframe', function = 'VarianceThreshold', which_df = 'both')

									###################
def SelectKBest(block):
    """(SelectKBest):
        http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest   
    """
    if "sklearn: %s"%block["parameters"]["score_func"] not in cmlnb["imports"]:
        handle_imports(["sklearn.feature_selection.SelectKBest","cheml.preprocessing.selector_dataframe",
        "sklearn.feature_selection.%s"%block["parameters"]["score_func"]])
    else:
        handle_imports(["sklearn.feature_selection.SelectKBest","cheml.preprocessing.selector_dataframe"])
    handle_API(block, function = 'SelectKBest')
    handle_transform(block, interface = 'selector_dataframe', function = 'SelectKBest', which_df = 'both')

									###################
def SelectPercentile(block):
    """(SelectPercentile):
        http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile  
    """
    if "sklearn: %s"%block["parameters"]["score_func"] not in cmlnb["imports"]:
        handle_imports(["sklearn.feature_selection.SelectPercentile","cheml.preprocessing.selector_dataframe",
        "sklearn.feature_selection.%s"%block["parameters"]["score_func"]])
    else:
        handle_imports(["sklearn.feature_selection.SelectPercentile","cheml.preprocessing.selector_dataframe"])
    handle_API(block, function = 'SelectPercentile')
    handle_transform(block, interface = 'selector_dataframe', function = 'SelectPercentile', which_df = 'both')

									###################
def SelectFpr(block):
    """(SelectFpr):
        http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html#sklearn.feature_selection.SelectFpr 
    """
    if "sklearn: %s"%block["parameters"]["score_func"] not in cmlnb["imports"]:
        handle_imports(["sklearn.feature_selection.SelectFpr","cheml.preprocessing.selector_dataframe",
        "sklearn.feature_selection.%s"%block["parameters"]["score_func"]])
    else:
        handle_imports(["sklearn.feature_selection.SelectFpr","cheml.preprocessing.selector_dataframe"])
    handle_API(block, function = 'SelectFpr')
    handle_transform(block, interface = 'selector_dataframe', function = 'SelectFpr', which_df = 'both')

									###################
def SelectFdr(block):
    """(SelectFdr):
        http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html#sklearn.feature_selection.SelectFdr 
    """
    if "sklearn: %s"%block["parameters"]["score_func"] not in cmlnb["imports"]:
        handle_imports(["sklearn.feature_selection.SelectFdr","cheml.preprocessing.selector_dataframe",
        "sklearn.feature_selection.%s"%block["parameters"]["score_func"]])
    else:
        handle_imports(["sklearn.feature_selection.SelectFdr","cheml.preprocessing.selector_dataframe"])
    handle_API(block, function = 'SelectFdr')
    handle_transform(block, interface = 'selector_dataframe', function = 'SelectFdr', which_df = 'both')

									###################
def SelectFwe(block):
    """(SelectFwe):
        http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html#sklearn.feature_selection.SelectFwe
    """
    if "sklearn: %s"%block["parameters"]["score_func"] not in cmlnb["imports"]:
        handle_imports(["sklearn.feature_selection.SelectFwe","cheml.preprocessing.selector_dataframe",
        "sklearn.feature_selection.%s"%block["parameters"]["score_func"]])
    else:
        handle_imports(["sklearn.feature_selection.SelectFwe","cheml.preprocessing.selector_dataframe"])
    handle_API(block, function = 'SelectFwe')
    handle_transform(block, interface = 'selector_dataframe', function = 'SelectFwe', which_df = 'both')

									###################
def RFE(block):
    """(RFE):
        http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE
    """
    imp = _sub_function(block,block["parameters"]["estimator"])
    handle_imports(["sklearn.feature_selection.RFE","cheml.preprocessing.selector_dataframe",imp])
    handle_subAPI(block)
    block["parameters"]["estimator"] = "%s_%s" %(block["sub_function"],'API')
    handle_API(block, function = 'RFE')
    handle_transform(block, interface = 'selector_dataframe', function = 'RFE', which_df = 'both')

									###################
def RFECV(block):
    """(RFECV):
        http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV
    """
    imp = _sub_function(block,block["parameters"]["estimator"])
    handle_imports(["sklearn.feature_selection.RFECV","cheml.preprocessing.selector_dataframe",imp])
    handle_subAPI(block)
    block["parameters"]["estimator"] = "%s_%s" %(block["sub_function"],'API')
    handle_API(block, function = 'RFECV')
    handle_transform(block, interface = 'selector_dataframe', function = 'RFECV', which_df = 'both')

									###################
def SelectFromModel(block):
    """(SelectFromModel):
        http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel
    """
    imp = _sub_function(block,block["parameters"]["estimator"])
    handle_imports(["sklearn.feature_selection.SelectFromModel","cheml.preprocessing.selector_dataframe",imp])
    handle_subAPI(block)
    if block["parameters"]["prefit"] == 'True':
        line = "%s_%s = %s_%s.fit(data, target)" %(block["sub_function"],'API',block["sub_function"],'API')
        cmlnb["blocks"][it]["source"].append(line + '\n')
    block["parameters"]["estimator"] = "%s_%s" %(block["sub_function"],'API')
    handle_API(block, function = 'SelectFromModel')
    handle_transform(block, interface = 'selector_dataframe', function = 'SelectFromModel', which_df = 'both')

									###################
def Trimmer(block):
    """(Trimmer):
    
    """
    handle_imports(["cheml.initializtion.Trimmer"])
    handle_API(block, function = 'Trimmer')
    handle_simple_transform(block, sub_function = 'fit_transform', function = 'Trimmer_API', which_df = 'both')

									###################
def Uniformer(block):
    """(Uniformer):
    
    """
    handle_imports(["cheml.initializtion.Uniformer"])
    handle_API(block, function = 'Uniformer')
    handle_simple_transform(block, sub_function = 'fit_transform', function = 'Uniformer_API', which_df = 'both')

									###################
def PCA(block):
    """(PCA):
        http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA    
    """
    handle_imports(["sklearn.decomposition.PCA"])
    handle_API(block, function = 'PCA')
    handle_simple_transform(block, sub_function = 'fit_transform', function = 'PCA_API', which_df = 'data')
    line = "data = pd.DataFrame(data)"
    cmlnb["blocks"][it]["source"].append(line + '\n')

									###################
def KernelPCA(block):
    """(KernelPCA):
        http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA    
    """
    handle_imports(["sklearn.decomposition.KernelPCA"])
    handle_API(block, function = 'KernelPCA')
    handle_simple_transform(block, sub_function = 'fit_transform', function = 'KernelPCA_API', which_df = 'data')
    line = "data = pd.DataFrame(data)"
    cmlnb["blocks"][it]["source"].append(line + '\n')
    
									###################
def RandomizedPCA(block):
    """(RandomizedPCA):
        http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.RandomizedPCA.html#sklearn.decomposition.RandomizedPCA   
    """
    handle_imports(["sklearn.decomposition.RandomizedPCA"])
    handle_API(block, function = 'RandomizedPCA')
    handle_simple_transform(block, sub_function = 'fit_transform', function = 'RandomizedPCA_API', which_df = 'data')
    line = "data = pd.DataFrame(data)"
    cmlnb["blocks"][it]["source"].append(line + '\n')

									###################
def LDA(block):
    """(LDA):
        http://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html#sklearn.lda.LDA   
    """
    handle_imports(["sklearn.lda.LDA"])
    handle_API(block, function = 'LDA')
    handle_simple_transform(block, sub_function = 'fit_transform', function = 'LDA_API', which_df = 'data')
    line = "data = pd.DataFrame(data)"
    cmlnb["blocks"][it]["source"].append(line + '\n')

									###################
def LDA(block):
    """(LDA):
        http://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html#sklearn.lda.LDA   
    """
    handle_imports(["sklearn.lda.LDA"])
    handle_API(block, function = 'LDA')
    handle_simple_transform(block, sub_function = 'fit_transform', function = 'LDA_API', which_df = 'data')
    line = "data = pd.DataFrame(data)"
    cmlnb["blocks"][it]["source"].append(line + '\n')

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
"""*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
 
 									  CheML PySCRIPT		
																						
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#"""
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*					

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="ChemML will be started by specifying a script file as a todo list")
    parser.add_argument("-i", type=str, required=True, help="input directory: must include the script file name and its format")                    		
    args = parser.parse_args()            		
    SCRIPT_NAME = args.i      
    main(SCRIPT_NAME)   #numbering of sys.argv is only meaningful if it is launched as main
    
else:
    sys.exit("Sorry, must run as driver...")



								  


	


