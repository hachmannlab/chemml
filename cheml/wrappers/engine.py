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
import inspect

import .scikit_learn as skl
import .cheml_wrapper as cml
#todo: use utils subdirectory instead
from .sct_utils import isint, value, std_datetime_str

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
                    a, b = arg
                    if isint(b) and not isint(a):
                        send[(a, int(b))] = item
                    elif isint(a) and not isint(b):
                        recv[(b, int(a))] = item
                    else:
                        msg = 'wrong format of send and receive in block #%i at %s (send: >> var id; recv: >> id var)' % (item+1, args)
                        raise IOError(msg)
                else:
                    msg = 'wrong format of send and receive in block #%i at %s (send: >> var id; recv: >> id var)'%(item+1,args)
                    raise IOError(msg)
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
                line = ' :no parameter passed: set to default values if available'
                line = line.rstrip("\n")
                print '        ' + line
            line = '>>>>>>>'
            line = line.rstrip("\n")
            print '        ' + line
            if len(block['send']) > 0:
                for param in block['send']:
                    line = '%s -> send (id=%i)\n' %(param[0],param[1])
                    line = line.rstrip("\n")
                    print '        ' + line
            else:
                line = ' :nothing to send:'
                line = line.rstrip("\n")
                print '        ' + line
            if len(block['recv']) > 0:
                for param in block['recv']:
                    line = '%s <- recv (id=%i)\n' %(param[0],param[1])
                    line = line.rstrip("\n")
                    print '        ' + line
            else:
                line = ' :nothing to receive:'
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
            msg = 'number of send and receive is not equal'
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
        reformat_send = {k[1]:[v,k[0]] for k,v in send_all.items()}
        for k, v in recv_all.items():
            reformat_send[k[1]] += [v,k[0]]
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

class BASE(object):
    def __init__(self, CompGraph):
        self.graph = CompGraph
        self.send = {}      # {(iblock,token):[value,count]}
        self.requirements = []
        self.start_time = time.time()
        self.block_time = 0
        self.date = std_datetime_str('date')
        self.time = std_datetime_str('time')
        print 'initialized'

class Wrapper(object):
    """
    Todo: documentation
    """
    def __init__(self, cmls, ImpOrder, CompGraph):
        self.Base = BASE(CompGraph)
        self.ImpOrder = ImpOrder
        self.cmls = cmls
        self._checker()

    def _checker(self):
        """
        check for key parameters like module and function and
        any possible typo in other params.
        """
        # get params
        legal_superfunctions = ['DataRepresentation','Input','Output','Preprocessor','FeatureSelection','FeatureTransformation','Divider','Regression','Classification','Evaluation','Visualization','Optimizer']
        legal_modules = {'cheml':['RDKitFingerprint','Dragon','CoulombMatrix','BagofBonds','File','Merge','Split','SaveFile','settings','MissingValues','Trimmer','Uniformer','TBFS','NN_MLP_PSGD','NN_MLP_DSGD','NN_MLP_Theano','NN_MLP_Tensorflow','SVR','GA_Binary','GA_Real'],
                         'sklearn': ['PolynomialFeatures','Imputer','StandardScaler','MinMaxScaler','MaxAbsScaler','RobustScaler','Normalizer','Binarizer','OneHotEncoder','VarianceThreshold','SelectKBest','SelectPercentile','SelectFpr','SelectFdr','SelectFwe','RFE','RFECV','SelectFromModel','PCA','KernelPCA','RandomizedPCA','LDA','','','','',''],
                         'mlpy':[]}

        # run over graph
        for iblock, block in enumerate(self.cmls):
            # check super function
            SuperFunction = block['SuperFunction']
            if SuperFunction not in legal_superfunctions:
                msg = '%s is not a valid task' %SuperFunction
                raise NameError(msg)
            # check parameters
            parameters = block['parameters']
            if 'module' not in parameters:
                msg = "Task %s (task#%i): no 'module' name found" % (SuperFunction, iblock + 1)
                raise NameError(msg)
            if 'function' not in parameters:
                msg = "Task %s (task#%i): no 'function' name found" % (SuperFunction, iblock + 1)
                raise NameError(msg)
            # check module and function
            module = block['parameters']['module']
            function = block['parameters']['function']
            if module not in legal_modules:
                msg = 'Task %s (task#%i): not a valid module passed' % (SuperFunction, iblock + 1)
                raise NameError(msg)
            elif function in legal_modules[module]:
                msg = 'Task %s (task#%i): not a valid function passed' % (SuperFunction, iblock + 1)
                raise NameError(msg)
        return 'The input file is in a correct format.'

    def call(self):
        for iblock in self.ImpOrder:
            SuperFunction = self.cmls[iblock]['SuperFunction']
            parameters = block['parameters']
            module = parameters.pop('module')
            function = parameters.pop('function')
            if module == 'sklearn':
                # check methods0
                legal_functions = [klass[0] for klass in inspect.getmembers(skl)]
                if S
                if function not in legal_functions:
                    msg = "function name '%s' in module '%s' is not a valid method"%(function,module)
                    raise NameError(msg)
                cml_interface = [klass[1] for klass in inspect.getmembers(skl) if klass[0]==function][0]
                cmli = cml_interface(self.Base,parameters,iblock)
                cmli.run()


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
    cmls, ImpOrder, CompGraph = Parser(script).fit()
    wrapper = Wrapper(cmls, ImpOrder, CompGraph)
    wrapper.call()
# else:
    # sys.exit("Sorry, must run as driver...")


