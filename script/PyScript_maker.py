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


def _block_finder(script):
    blocks={}
    item=-1
    a = [i for i,line in enumerate(script) if '#' in line]
    b = [i for i,line in enumerate(script) if '###' in line]
    if len(b)%2 != 0:
        msg = "one of the super functions has not been wrapped with '###' properly."
        raise ValueError(msg)
    for ind in b[1::2]:
        script[ind] = ''
    for i,j in zip(b,b[1:])[::2]:
        a = [ind for ind in a if ind<=i or ind>j]
    inactive = [a.index(i) for i in a if '##' not in script[i]]
    for i in xrange(len(a)-1):
        blocks[i] = script[a[i]:a[i+1]]
    blocks[len(a)-1] = script[a[-1]:]
    return blocks

def _functions(line):
    if '>' in line:
        function = line[line.index('##')+2:line.index('>')].strip()
    else:
        function = line[line.index('##')+2:].strip()
    return function

def _parameters(block):
    parameters = {}
    for line in block:
        while '>>' in line:
            line = line[line.index('>>')+2:].strip()
            if '>' in line:
                args = line[:line.index('>')].strip()
            else:
                args = line.strip()
            param = args[:args.index('=')].strip()
            val = args[args.index('=')+1:].strip()
            parameters[param] = "%s"%val
    return parameters

def _superfunctions(line):
    if '#' in line[line.index('###')+3:]:
        function = line[line.index('###')+3:line.index('#')].strip()
    else:
        function = line[line.index('###')+3:].strip()
    return function

def _superparameters(block):
    parameters = []
    block[0] = block[0][block[0].index('###')+3:]
    sub_blocks = _block_finder(block)
    for i in xrange(len(sub_blocks)):
        blk = sub_blocks[i]
        if '##' in blk[0]:
            parameters.append({"function": _functions(blk[0]),
                               "parameters": _parameters(blk)})
        else:
            continue        
    return parameters
    
def _options(blocks):
    cmls = []
    for item in xrange(len(blocks)):
        block = blocks[item]
        if '###' in block[0]:
            cmls.append({"function": _superfunctions(block[0]),
                         "parameters": _superparameters(block)})
        elif '##' in block[0]:
            cmls.append({"function": _functions(block[0]),
                         "parameters": _parameters(block)})
    return cmls

def _print_out(cmls):
    item = 0
    for block in cmls:
        item+=1
        line = '%s\n' %(block['function'])
        line = line.rstrip("\n")
        print '%i'%item+' '*(4-len(str(item)))+'function = '+line
        if type(block['parameters']) == dict:
            for param in block['parameters']:
                line = '%s = %s\n'%(param,block['parameters'][param])
                line = line.rstrip("\n")
                print '        '+line 
        elif type(block['parameters']) == list:
            for sub_block in block['parameters']:
                line = '%s\n' %(sub_block['function'])
                line = line.rstrip("\n")
                print '     *'+' '*2+'function = '+line
                for param in sub_block['parameters']:
                    line = '%s = %s\n'%(param,sub_block['parameters'][param])
                    line = line.rstrip("\n")
                    print '            '+line 
        else:
            msg = "script parser can not recognize the format of script"
            raise ValueError(msg)
            
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
                 'Dragon'               : Dragon,
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
                 'LDA'                  : LDA,
                 'SupervisedLearning_regression' : SupervisedLearning_regression
                 
                }

    for block in cmls:
        if block['function'] not in functions:
            msg = "name %s is not defined"%block['function']
            raise NameError(msg)
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

def handle_API(block, function = False, ignore = []):
    """
    make a class object with input arguments
    """
    if function:
        line = "%s_%s = %s(" %(function,'API',function)
    else:
        line = "%s_%s = %s(" %(block["function"],'API',block["function"])
    param_count = 0
    for parameter in block["parameters"]:
        if parameter in ignore:
            continue
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

def handle_funct_API(block, inputs, outputs, function, ignore = []):
    """
    call a function with all of its parameters.
    """
    if outputs:
       line = "%s = " %outputs
    else:
        line = ""
        
    if inputs:
        line += "%s(%s" %(function,inputs)
    else:
        line += "%s(" %(function)

    param_count = 0
    for parameter in block["parameters"]:
        if parameter in ignore:
            continue
        param_count += 1
        line += """;%s = %s"""%(parameter,block["parameters"][parameter])
    line += ')'
    if not inputs:
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

def Dragon(block):
    """(Dragon):
        http://www.talete.mi.it/help/dragon_help/index.html?script_file.htm
    """
    handle_imports(["cheml.chem.dragon"])
    handle_funct_API(block, inputs=False, outputs="dragon_API", function="dragon", ignore = ["script"])
    line = "dragon_API.script_wizard(script = %s)"%block["parameters"]["script"]
    cmlnb["blocks"][it]["source"].append(line + '\n')
    line = "dragon_API.run()"
    cmlnb["blocks"][it]["source"].append(line + '\n')
    line = "data_path = dragon_API.drs"
    cmlnb["blocks"][it]["source"].append(line + '\n')

   
									###################
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
    """(LinearDiscriminantAnalysis):
        http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis   
    """
    handle_imports(["sklearn.discriminant_analysis.LinearDiscriminantAnalysis"])
    handle_API(block, function = 'LinearDiscriminantAnalysis')
    handle_simple_transform(block, sub_function = 'fit_transform', function = 'LinearDiscriminantAnalysis_API', which_df = 'data')
    line = "data = pd.DataFrame(data)"
    cmlnb["blocks"][it]["source"].append(line + '\n')

									###################
def SupervisedLearning_regression(block):
    """(SupervisedLearning_regression):
        The regression full package   
    """
    sub_functions = {'split'            : split,
                     'cross_validation' : cross_validation,
                     'scaler'           : scaler,
                     'learner'          : learner,
                     'metrics'          : 0
#                      'plot'             : plot,
#                      'save'             : save
                     }
    for sub_block in block['parameters']:
        if sub_block['function'] not in sub_functions:
            msg = "subfunction %s in the SupervisedLearning_regression is not defined"%sub_block['function']
            raise NameError(msg)
        elif sub_block['function'] == 'metrics':
            continue
        else:
            sub_functions[sub_block['function']](block, sub_block)    
           
									#*****************#
									
def split(block, sub_block):
    line = '\n# split'
    cmlnb["blocks"][it]["source"].append(line + '\n')
    if sub_block['parameters']['module'][1:-1] == 'sklearn':
        if sub_block['parameters']['method'][1:-1] == 'train_test_split':
            handle_imports(["sklearn.cross_validation.train_test_split"])
            outputs = 'data_train, data_test, target_train, target_test'
            inputs = 'data;target'
            handle_funct_API(sub_block, inputs=inputs, outputs=outputs, function = 'train_test_split', ignore = ['module','method'])
        else:
            msg = "Enter a valid sklearn function for split."
            raise NameError(msg)
    else:
        msg = "Enter a valid module name for split."
        raise NameError(msg)

									#*****************#
									
def cross_validation(block, sub_block):
    line = '\n# cross_validation'
    cmlnb["blocks"][it]["source"].append(line + '\n')
    if sub_block['parameters']['module'][1:-1] == 'sklearn':
        if sub_block['parameters']['method'][1:-1] == 'K-fold':
            if not sub_block['parameters'].has_key('n'):
                sub_block['parameters']['n'] = 'len(data)'
            handle_imports(["sklearn.cross_validation.K-fold"])
            outputs = 'CV_indices'
            handle_funct_API(sub_block, inputs=False, outputs=outputs, function = 'K-fold', ignore = ['module','method'])
        elif sub_block['parameters']['method'][1:-1] == 'StratifiedKFold':
            if not sub_block['parameters'].has_key('y'):
                sub_block['parameters']['y'] = target
            handle_imports(["sklearn.cross_validation.StratifiedKFold"])
            outputs = 'CV_indices'
            handle_funct_API(sub_block, inputs=False, outputs=outputs, function = 'StratifiedKFold', ignore = ['module','method'])
        elif sub_block['parameters']['method'][1:-1] == 'LabelKFold':
            if not sub_block['parameters'].has_key('labels'):
                sub_block['parameters']['labels'] = target
            handle_imports(["sklearn.cross_validation.LabelKFold"])
            outputs = 'CV_indices'
            handle_funct_API(sub_block, inputs=False, outputs=outputs, function = 'LabelKFold', ignore = ['module','method'])
        elif sub_block['parameters']['method'][1:-1] == 'LeaveOneOut':
            if not sub_block['parameters'].has_key('n'):
                sub_block['parameters']['n'] = 'len(data)'
            handle_imports(["sklearn.cross_validation.LeaveOneOut"])
            outputs = 'CV_indices'
            handle_funct_API(sub_block, inputs=False, outputs=outputs, function = 'LeaveOneOut', ignore = ['module','method'])
        elif sub_block['parameters']['method'][1:-1] == 'LeavePOut':
            if not sub_block['parameters'].has_key('n'):
                sub_block['parameters']['n'] = 'len(data)'
            handle_imports(["sklearn.cross_validation.LeavePOut"])
            outputs = 'CV_indices'
            handle_funct_API(sub_block, inputs=False, outputs=outputs, function = 'LeavePOut', ignore = ['module','method'])
        elif sub_block['parameters']['method'][1:-1] == 'LeaveOneLabelOut':
            if not sub_block['parameters'].has_key('labels'):
                sub_block['parameters']['labels'] = target
            handle_imports(["sklearn.cross_validation.LeaveOneLabelOut"])
            outputs = 'CV_indices'
            handle_funct_API(sub_block, inputs=False, outputs=outputs, function = 'LeaveOneLabelOut', ignore = ['module','method'])
        elif sub_block['parameters']['method'][1:-1] == 'LeavePLabelOut':
            if not sub_block['parameters'].has_key('labels'):
                sub_block['parameters']['labels'] = target
            handle_imports(["sklearn.cross_validation.LeavePLabelOut"])
            outputs = 'CV_indices'
            handle_funct_API(sub_block, inputs=False, outputs=outputs, function = 'LeavePLabelOut', ignore = ['module','method'])        
        elif sub_block['parameters']['method'][1:-1] == 'ShuffleSplit':
            if not sub_block['parameters'].has_key('n'):
                sub_block['parameters']['n'] = 'len(data)'
            handle_imports(["sklearn.cross_validation.ShuffleSplit"])
            outputs = 'CV_indices'
            handle_funct_API(sub_block, inputs=False, outputs=outputs, function = 'ShuffleSplit', ignore = ['module','method'])
        elif sub_block['parameters']['method'][1:-1] == 'LabelShuffleSplit':
            if not sub_block['parameters'].has_key('labels'):
                sub_block['parameters']['labels'] = target
            handle_imports(["sklearn.cross_validation.LabelShuffleSplit"])
            outputs = 'CV_indices'
            handle_funct_API(sub_block, inputs=False, outputs=outputs, function = 'LabelShuffleSplit', ignore = ['module','method'])
        elif sub_block['parameters']['method'][1:-1] == 'PredefinedSplit':
            handle_imports(["sklearn.cross_validation.PredefinedSplit"])
            outputs = 'CV_indices'
            handle_funct_API(sub_block, inputs=False, outputs=outputs, function = 'PredefinedSplit', ignore = ['module','method'])
        else:
            msg = "Enter a valid sklearn function for cross_validation."
            raise NameError(msg)
    else:
        msg = "Enter a valid module name for cross_validation."
        raise NameError(msg)

									#*****************#
									
def scaler(block, sub_block):
    line = '\n# scaler'
    cmlnb["blocks"][it]["source"].append(line + '\n')
    if sub_block['parameters']['module'][1:-1] == 'sklearn':
        if sub_block['parameters']['method'][1:-1] == 'StandardScaler':
            handle_imports(["sklearn.preprocessing.StandardScaler"])
            handle_API(sub_block, function = 'StandardScaler', ignore = ['module','method'])
        elif sub_block['parameters']['method'][1:-1] == 'MinMaxScaler':
            handle_imports(["sklearn.preprocessing.MinMaxScaler"])
            handle_API(sub_block, function = 'MinMaxScaler', ignore = ['module','method'])
        elif sub_block['parameters']['method'][1:-1] == 'MaxAbsScaler':
            handle_imports(["sklearn.preprocessing.MaxAbsScaler"])
            handle_API(sub_block, function = 'MaxAbsScaler', ignore = ['module','method'])
        elif sub_block['parameters']['method'][1:-1] == 'RobustScaler':
            handle_imports(["sklearn.preprocessing.RobustScaler"])
            handle_API(sub_block, function = 'RobustScaler', ignore = ['module','method'])
        elif sub_block['parameters']['method'][1:-1] == 'Normalizer':
            handle_imports(["sklearn.preprocessing.Normalizer"])
            handle_API(sub_block, function = 'Normalizer', ignore = ['module','method'])
        else:
            msg = "Enter a valid sklearn function for scaler."
            raise NameError(msg)
    else:
        msg = "Enter a valid module name for scaler."
        raise NameError(msg)

									#*****************#

def learner(block, sub_block):
    line = '\n# learner'
    cmlnb["blocks"][it]["source"].append(line + '\n')
    if sub_block['parameters']['module'][1:-1] == 'sklearn':
        if sub_block['parameters']['method'][1:-1] == 'LinearRegression':
            handle_imports(["sklearn.linear_model.LinearRegression"])
            handle_API(sub_block, function = 'LinearRegression', ignore = ['module','method'])
            handle_regression_sklearn(block, learner_API = 'LinearRegression_API')
        elif sub_block['parameters']['method'][1:-1] == 'Ridge':
            handle_imports(["sklearn.linear_model.Ridge"])
            handle_API(sub_block, function = 'Ridge', ignore = ['module','method'])
            handle_regression_sklearn(block, learner_API = 'Ridge_API')
        elif sub_block['parameters']['method'][1:-1] == 'Lasso':
            handle_imports(["sklearn.linear_model.Lasso"])
            handle_API(sub_block, function = 'Lasso', ignore = ['module','method'])
            handle_regression_sklearn(block, learner_API = 'Lasso_API')
        elif sub_block['parameters']['method'][1:-1] == 'ElasticNet':
            handle_imports(["sklearn.linear_model.ElasticNet"])
            handle_API(sub_block, function = 'ElasticNet', ignore = ['module','method'])
            handle_regression_sklearn(block, learner_API = 'ElasticNet_API')
        elif sub_block['parameters']['method'][1:-1] == 'LassoLars':
            handle_imports(["sklearn.linear_model.LassoLars"])
            handle_API(sub_block, function = 'LassoLars', ignore = ['module','method'])
            handle_regression_sklearn(block, learner_API = 'LassoLars_API')
        elif sub_block['parameters']['method'][1:-1] == 'SVR':
            handle_imports(["sklearn.svm.SVR"])
            handle_API(sub_block, function = 'SVR', ignore = ['module','method'])
            handle_regression_sklearn(block, learner_API = 'SVR_API')
        elif sub_block['parameters']['method'][1:-1] == 'NuSVR':
            handle_imports(["sklearn.svm.NuSVR"])
            handle_API(sub_block, function = 'NuSVR', ignore = ['module','method'])
            handle_regression_sklearn(block, learner_API = 'NuSVR_API')
        elif sub_block['parameters']['method'][1:-1] == 'LinearSVR':
            handle_imports(["sklearn.svm.LinearSVR"])
            handle_API(sub_block, function = 'LinearSVR', ignore = ['module','method'])
            handle_regression_sklearn(block, learner_API = 'LinearSVR_API')

def handle_regression_sklearn(block, learner_API):
    order = [ sb['function'] for sb in block['parameters'] ]
    if 'split' in order:
        line = '\n# result: split'
        cmlnb["blocks"][it]["source"].append(line + '\n')
        # scale
        if 'scaler' in order:
            scaler_API = block['parameters'][order.index('scaler')]['parameters']['method'][1:-1] + '_API'
            line = '%s.fit(data_train)'%scaler_API
            cmlnb["blocks"][it]["source"].append(line + '\n')
            line = 'data_train = ' + scaler_API +'.transform(data_train)'
            cmlnb["blocks"][it]["source"].append(line + '\n')
            line = 'data_test = ' + scaler_API +'.transform(data_test)'
            cmlnb["blocks"][it]["source"].append(line + '\n')
        # train
        line = '%s.fit(data_train, target_train)'%learner_API
        cmlnb["blocks"][it]["source"].append(line + '\n')
        # metrics
        if 'metrics' in order:
            metrics_items = block['parameters'][order.index('metrics')]['parameters'].items()
            order_metrics = [item[0] for item in metrics_items if item[1]=='True']
            metrics(order_metrics, learner_API, style='split')

    if 'cross_validation' in order:
        line = '\n# result: cross_validation'
        cmlnb["blocks"][it]["source"].append(line + '\n')
        if 'metrics' in order:
            metrics_items = block['parameters'][order.index('metrics')]['parameters'].items()
            order_metrics = [item[0] for item in metrics_items if item[1]=='True']
            CV_metrics = {'training':{}, 'test':{}}
            for metric in order_metrics:
                CV_metrics['training'][metric] = []
                CV_metrics['test'][metric] = []
            line = "CV_metrics = %s"%str(CV_metrics)
            cmlnb["blocks"][it]["source"].append(line + '\n')

        line = "for train_index, test_index in CV_indices:"
        cmlnb["blocks"][it]["source"].append(line + '\n')
        line = "    data_train = data.iloc[train_index,:]"
        cmlnb["blocks"][it]["source"].append(line + '\n')
        line = "    target_train = target.iloc[train_index,:]"
        cmlnb["blocks"][it]["source"].append(line + '\n')
        line = "    data_test = target.iloc[test_index,:]"
        cmlnb["blocks"][it]["source"].append(line + '\n')
        line = "    target_test = target.iloc[test_index,:]"
        cmlnb["blocks"][it]["source"].append(line + '\n')
        # scale
        if 'scaler' in order:
            scaler_API = block['parameters'][order.index('scaler')]['parameters']['method'][1:-1] + '_API'
            line = '    %s.fit(data_train)'%scaler_API
            cmlnb["blocks"][it]["source"].append(line + '\n')
            line = '    data_train = ' + scaler_API +'.transform(data_train)'
            cmlnb["blocks"][it]["source"].append(line + '\n')
            line = '    data_test = ' + scaler_API +'.transform(data_test)'
            cmlnb["blocks"][it]["source"].append(line + '\n')                 
        # train
        line = '    %s.fit(data_train, target_train)'%learner_API
        cmlnb["blocks"][it]["source"].append(line + '\n')
        # metrics
        if 'metrics' in order:
            metrics(order_metrics, learner_API, style='cross_validation')
 
def metrics(order_metrics, learner_API, style):
    metrics_all = ['r2_score',
                   'explained_variance_score',
                   'mean_absolute_error',
                   'median_absolute_error',
                   'mean_squared_error',
                   'root_mean_squared_error']
    if style == 'split':
        line = 'target_train_pred = %s.predict(data_train)'%learner_API
        cmlnb["blocks"][it]["source"].append(line + '\n')
        line = 'target_test_pred = %s.predict(data_test)'%learner_API
        cmlnb["blocks"][it]["source"].append(line + '\n')
        line = "split_metrics = {'training':{}, 'test':{}}"
        cmlnb["blocks"][it]["source"].append(line + '\n')
        for order in order_metrics:
            if order not in metrics_all:
                msg = "Not a valid metrics type."
                raise ValueError(msg)
            elif order != 'root_mean_squared_error':
                handle_imports(['sklearn.metrics.%s'%order])
                line = "split_metrics['training']['%s'] = %s(target_train, target_train_pred)"%(order,order)
                cmlnb["blocks"][it]["source"].append(line + '\n') 
                line = "split_metrics['test']['%s'] = %s(target_test, target_test_pred)"%(order,order)
                cmlnb["blocks"][it]["source"].append(line + '\n') 
            elif order == 'root_mean_squared_error':
                handle_imports(['sklearn.metrics.mean_squared_error'])
                line = "split_metrics['training']['%s'] = np.sqrt(%s(target_train, target_train_pred))"%('mean_squared_error','mean_squared_error')
                cmlnb["blocks"][it]["source"].append(line + '\n') 
                line = "split_metrics['test']['%s'] = np.sqrt(%s(target_test, target_test_pred))"%('mean_squared_error','mean_squared_error')
                cmlnb["blocks"][it]["source"].append(line + '\n') 
    elif style == 'cross_validation':
        line = '    target_train_pred = %s.predict(data_train)'%learner_API
        cmlnb["blocks"][it]["source"].append(line + '\n')
        line = '    target_test_pred = %s.predict(data_test)'%learner_API
        cmlnb["blocks"][it]["source"].append(line + '\n')
        for order in order_metrics:
            if order not in metrics_all:
                msg = "Not a valid metrics type."
                raise ValueError(msg)
            elif order != 'root_mean_squared_error':
                handle_imports(['sklearn.metrics.%s'%order])
                line = "    CV_metrics['training']['%s'].append(%s(target_train, target_train_pred))"%(order,order)
                cmlnb["blocks"][it]["source"].append(line + '\n') 
                line = "    CV_metrics['test']['%s'].append(%s(target_test, target_test_pred))"%(order,order)
                cmlnb["blocks"][it]["source"].append(line + '\n') 
            elif order == 'root_mean_squared_error':
                handle_imports(['sklearn.metrics.mean_squared_error'])
                line = "    CV_metrics['training']['%s'].append(np.sqrt(%s(target_train, target_train_pred)))"%('mean_squared_error','mean_squared_error')
                cmlnb["blocks"][it]["source"].append(line + '\n') 
                line = "    CV_metrics['test']['%s'].append(np.sqrt(%s(target_test, target_test_pred)))"%('mean_squared_error','mean_squared_error')
                cmlnb["blocks"][it]["source"].append(line + '\n') 

									###################
      

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



								  


	


