#!/usr/bin/env python

PROGRAM_NAME = "CheML"
PROGRAM_VERSION = "v0.0.1"
REVISION_DATE = "2015-06-23"
AUTHORS = "Johannes Hachmann (hachmann@buffalo.edu) and Mojtaba Haghighatlari (mojtabah@buffalo.edu)"
CONTRIBUTORS = " "
DESCRIPTION = "ChemML is a machine learning and informatics program suite for the chemical and materials sciences."

import sys
import os
import time
import copy
import argparse
import warnings

from .scikit_learn import Sklearn_Base
from .cheml import Cheml_Base
from .sct_utils import isfloat, islist, istuple, isnpdot, std_datetime_str

def value(string):
    try:
        return eval(string)
    except NameError:
        return string

class Parser(object):
    """
    script: list of strings
        A list of lines in the cheml script file.
    """
    def __init__(self, script):
        self.script = script

    def fit(self):
        """
        The main funtion for parsing cheml script.
        It starts with finding blocks and then runs other functions.

        :return:
        cmls: cheml script
        """
        blocks={}
        it = -1
        for line in self.script:
            if '##' in line:
                it += 1
                blocks[it] = [line]
                continue
            elif '#' not in line and ('<' in line or '>' in line):
                blocks[it].append(line)
                continue

        cmls = self._options(blocks)
        ImpOrder,CompGraph = self.transform(cmls)
        self._print_out(cmls)
        return cmls, ImpOrder, CompGraph

    def _functions(self, line):
        if '<' in line:
            function = line[line.index('##')+2:line.index('<')].strip()
        elif '>' in line:
            function = line[line.index('##')+2:line.index('>')].strip()
        else:
            function = line[line.index('##')+2:].strip()
        return function

    def _parameters(self, block,item):
        parameters = {}
        send = {}
        recv = {}
        for line in block:
            while '<<' in line:
                line = line[line.index('<<')+2:].strip()
                if '<' in line:
                    args = line[:line.index('<')].strip()
                else:
                    args = line.strip()
                param = args[:args.index('=')].strip()
                val = args[args.index('=')+1:].strip()
                parameters[param] = value(val) #val #"%s"%val
            while '>>' in line:
                line = line[line.index('>>') + 2:].strip()
                if '>' in line:
                    args = line[:line.index('>')].strip()
                else:
                    args = line.strip()
                arg = args.split()
                if len(arg) == 2:
                    var, id = arg
                    send[(var, int(id))] = item
                elif len(arg) == 1:
                    recv[('recv%i'%item,int(arg[0]))] = item
                else:
                    msg = 'wrong format of send and receive in block #%i at %s (send: >> var id; recv: >> id)'%(item+1,args)
                    raise ValueError(msg)


        return parameters, send, recv

    def _options(self, blocks):
        cmls = []
        for item in xrange(len(blocks)):
            block = blocks[item]
            function = self._functions(block[0])
            parameters, send, recv = self._parameters(block,item)
            cmls.append({"SuperFunction": function,
                         "parameters": parameters,
                         "send": send,
                         "recv": recv})
        return cmls

    def _print_out(self, cmls):
        item = 0
        for block in cmls:
            item+=1
            line = '%s\n' %(block['SuperFunction'])
            line = line.rstrip("\n")
            print '%i'%item+' '*(4-len(str(item)))+'Task: '+line
            line = '<<<<<<<'
            line = line.rstrip("\n")
            print '        ' + line
            if len(block['parameters']) > 0 :
                for param in block['parameters']:
                    line = '%s = %s\n'%(param,block['parameters'][param])
                    line = line.rstrip("\n")
                    print '        '+line
            else:
                line = ' :no parameter passed: set to defaul values if available'
                line = line.rstrip("\n")
                print '        ' + line
            line = '>>>>>>>'
            line = line.rstrip("\n")
            print '        ' + line
            if len(block['send']) > 0:
                for param in block['send']:
                    line = '%s -> send\n' %str(param)
                    line = line.rstrip("\n")
                    print '        ' + line
            else:
                line = ' :no send:'
                line = line.rstrip("\n")
                print '        ' + line
            if len(block['recv']) > 0:
                for param in block['recv']:
                    line = '%s -> recv\n' %str(param)
                    line = line.rstrip("\n")
                    print '        ' + line
            else:
                line = ' :no receive:'
                line = line.rstrip("\n")
                print '        ' + line
            line = ''
            line = line.rstrip("\n")
            print '        ' + line

    def transform(self, cmls):
        """
        goals:
            - collect all sends and receives
            - check send and receive format.
            - make the computational graph
            - find the order of implementation of functions based on sends and receives

        :param cmls:
        :return implementation order:
        """
        send_all = {}
        recv_all = {}
        for block in cmls:
            send_all.update(block['send'])
            recv_all.update(block['recv'])
        # check send and recv
        if len(send_all) != len(recv_all):
            msg = 'not an equal number of send and receive has been provided'
            raise ValueError(msg)
        send_ids = [k[1] for k,v in send_all.items()]
        recv_ids = [k[1] for k,v in recv_all.items()]
        for id in send_ids:
            if send_ids.count(id)>1:
                msg = 'identified non unique send id (id#%i)'%id
                raise NameError(msg)
        for id in recv_ids:
            if recv_ids.count(id)>1:
                msg = 'identified non unique receive id (id#%i)'%id
                raise NameError(msg)
        if len(set(send_ids) - set(recv_ids))>0:
            msg = 'missing pairs of send and receive id'
            raise ValueError(msg)

        # make graph
        reformat_send = {k[1]:[v,-1,k[0]] for k,v in send_all.items()}
        for k, v in recv_all.items():
            reformat_send[k[1]][1] = v
            reformat_send[k[1]] = tuple(reformat_send[k[1]])
        CompGraph = tuple(reformat_send.values())

        # find orders
        ids_sent = []
        ImpOrder = []
        inf_checker = 0
        while len(ImpOrder)<len(cmls):
            inf_checker +=1
            for i in range(len(cmls)):
                if i not in ImpOrder:
                    ids_recvd = [k[1] for k,v in cmls[i]['recv'].items()]
                    if len(ids_recvd) == 0:
                        ids_sent += [k[1] for k,v in cmls[i]['send'].items()]
                        ImpOrder.append(i)
                    elif len(set(ids_recvd) - set(ids_sent))==0:
                        ids_sent += [k[1] for k,v in cmls[i]['send'].items()]
                        ImpOrder.append(i)
            if  inf_checker > len(cmls):
                msg = 'Your design of send and receive tokens makes a loop of interdependencies. We believe that you can avoid such loops with setting only one received input per input type.'
                raise IOError(msg)
        return tuple(ImpOrder),CompGraph

class order(object):
    """
    main driver

    :return:
    API
    """
    def __init__(self, cmls):
        self.send_recv = ()
        for block in cmls:
            self.SuperFunction = block['SuperFunction']
            self.parameters = block['parameters']
            self.send = block['send']
            self._checker()
            self.call()

    def _checker(self):
        """
        check for key parameters like module and function and
        any possible typo in other params.
        """
        # check super function
        legal_superfunctions = ['DataRepresentation','Input','Output','Preprocessor','FeatureSelection','FeatureTransformation','Divider','Regression','Classification','Evaluation','Visualization','Optimizer']
        if self.SuperFunction not in legal_superfunctions:
            msg = '%s is not a valid task'%self.SuperFunction
            raise NameError(msg)
        # check modules
        legal_modules = ['cheml','sklearn']
        if 'module' in self.parameters:
            self.module = self.parameters.pop('module')
            if self.module not in legal_modules:
                msg = 'The only available modules in this version are: %s'%str(legal_modules)
                raise NameError(msg)
        else:
            msg = 'no module name passed. Always determine the module and the function name in parameters'
            raise NameError(msg)
        # check function
        legal_functions = {'cheml':['RDKitFingerprint','Dragon','CoulombMatrix','BagofBonds','File','Merge','Split','SaveFile','settings','MissingValues','Trimmer','Uniformer','TBFS','NN_MLP_PSGD','NN_MLP_DSGD','NN_MLP_Theano','NN_MLP_Tensorflow','SVR','GA_Binary','GA_Real'],
                           'sklearn': ['PolynomialFeatures','Imputer','StandardScaler','MinMaxScaler','MaxAbsScaler','RobustScaler','Normalizer','Binarizer','OneHotEncoder','VarianceThreshold','SelectKBest','SelectPercentile','SelectFpr','SelectFdr','SelectFwe','RFE','RFECV','SelectFromModel','PCA','KernelPCA','RandomizedPCA','LDA','','','','','']}
        if 'function' in self.parameters:
            self.function = self.parameters.pop('function')
            if self.function not in legal_functions[self.module]:
                msg = 'The only available functions in the passed module are: %s'%str(legal_functions[self.module])
                raise NameError(msg)
        else:
            msg = 'no function name passed. Always determine the module and the function name in parameters'
            raise NameError(msg)
        # check send
        send_recv = {}
        for token, receiver in self.send.items():
            self.send_recv += ({receiver:(self.function,token)},)
        return 0

    def call(self):
        if self.module == 'cheml':
            Cheml_Base(self.function, self.parameters, self.send)
        elif self.module == 'sklearn':
            Sklearn_Base(self.function, self.parameters, self.send)

    def transform(self):
        pass



def main(script):
    """main:
        Driver of ChemML
    """
    global cmls
    global cmlnb
    global it
    it = -1
    cmls, ImpOrder,CompGraph = Parser(script).fit()
    order(cmls)

    ## CHECK SCRIPT'S REQUIREMENTS
    called_functions = [block["function"] for block in cmls]
    input_functions = [funct for funct in ["INPUT","Dragon","RDKFP","CoulombMatrix"] if funct in called_functions]
    if len(input_functions)==0:
        raise RuntimeError("cheml requires input data")
    elif len(input_functions)>1:
        msg = "more than one input functions are available!"
        warnings.warn(msg,Warning)

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
             "run": "# how to run: python",
             "version": "1.1.0",
             "imports": []
            }
    
    ## implementing orders
    functions = {'INPUT'                : INPUT,
                 'Dragon'               : Dragon,
                 'RDKFP'                : RDKFP,
                 'CoulombMatrix'        : CoulombMatrix,
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
                 'SupervisedLearning_regression' : SupervisedLearning_regression,
                 'slurm_script'         : slurm_script
                 
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
	

   

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*


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
    script = open(SCRIPT_NAME, 'r')
    script = script.readlines()
    main(script)   #numbering of sys.argv is only meaningful if it is launched as main
    
# else:
    # sys.exit("Sorry, must run as driver...")


