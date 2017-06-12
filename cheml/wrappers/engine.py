#!/usr/bin/env python

import sys
import os
import time
import copy
import argparse
import warnings
import inspect
import shutil

from .pandas import pdw
from .cheml import cmlw
from .sklearn import sklw
from .tf import tfw

from ..utils import isint, value, std_datetime_str, tot_exec_time_str
from .base import LIBRARY

def banner():
    PROGRAM_NAME = "ChemML"
    PROGRAM_VERSION = "v1.3.1"
    REVISION_DATE = "2017-01-03"
    AUTHORS = ["Johannes Hachmann (hachmann@buffalo.edu)","Mojtaba Haghighatlari (mojtabah@buffalo.edu)"]
    CONTRIBUTORS = " "
    DESCRIPTION = "ChemML is a machine learning and informatics program suite for the chemical and materials sciences."
    str = []
    str.append("=================================================")
    str.append(PROGRAM_NAME + " " + PROGRAM_VERSION + " (" + REVISION_DATE + ")")
    for AUTHOR in AUTHORS:
        str.append(AUTHOR)
    str.append("=================================================")
    str.append(time.ctime())
    str.append("")
    # str.append(DESCRIPTION)
    # str.append("")

    print
    for line in str:
        print line

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
        check_block = False
        for line in self.script:
            if '##' in line:
                it += 1
                blocks[it] = [line]
                check_block = True
                continue
            elif '#' in line:
                check_block = False
            elif check_block and ('<' in line or '>' in line):
                blocks[it].append(line)
                continue

        cmls = self._options(blocks)
        ImpOrder,CompGraph = self.transform(cmls)
        tmp_str =  'Input File: \n'
        print tmp_str
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
            tmp_str =  '%i'%item+' '*(4-len(str(item)))+'Task: '+line
            print tmp_str
            line = '<<<<<<<'
            line = line.rstrip("\n")
            tmp_str = '        ' + line
            print tmp_str
            if len(block['parameters']) > 0 :
                for param in block['parameters']:
                    line = '%s = %s\n'%(param,block['parameters'][param])
                    line = line.rstrip("\n")
                    tmp_str =  '        '+line
                    print tmp_str
            else:
                line = ' :no parameter passed: set to default values if available'
                line = line.rstrip("\n")
                tmp_str =  '        ' + line
                print tmp_str
            line = '>>>>>>>'
            line = line.rstrip("\n")
            tmp_str =  '        ' + line
            print tmp_str
            if len(block['send']) > 0:
                for param in block['send']:
                    line = '%s -> send (id=%i)\n' %(param[0],param[1])
                    line = line.rstrip("\n")
                    tmp_str =  '        ' + line
                    print tmp_str
            else:
                line = ' :nothing to send:'
                line = line.rstrip("\n")
                tmp_str =  '        ' + line
                print tmp_str
            if len(block['recv']) > 0:
                for param in block['recv']:
                    line = '%s <- recv (id=%i)\n' %(param[0],param[1])
                    line = line.rstrip("\n")
                    tmp_str = '        ' + line
                    print tmp_str
            else:
                line = ' :nothing to receive:'
                line = line.rstrip("\n")
                tmp_str = '        ' + line
                print tmp_str
            line = ''
            line = line.rstrip("\n")
            tmp_str = '        ' + line
            print tmp_str

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
        send_all = []
        recv_all = []
        for block in cmls:
            send_all += block['send'].items()
            recv_all += block['recv'].items()
        # check send and recv
        if len(send_all) > len(recv_all):
            msg = '@cehml script - number of sent tokens must be less or equal to number of received tokens'
            raise ValueError(msg)
        send_ids = [k[1] for k,v in send_all]
        recv_ids = [k[1] for k,v in recv_all]
        for id in send_ids:
            if send_ids.count(id)>1:
                msg = 'identified non unique send id (id#%i)'%id
                raise NameError(msg)
        if set(send_ids) != set(recv_ids):
            msg = 'missing pairs of send and receive id'
            raise ValueError(msg)

        # make graph
        reformat_send = {k[1]:[v,k[0]] for k,v in send_all}
        CompGraph = tuple([tuple(reformat_send[k[1]]+[v,k[0]]) for k,v in recv_all])

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
                    elif len(set(ids_recvd) - set(ids_sent)) == 0:
                        ids_sent += [k[1] for k,v in cmls[i]['send'].items()]
                        ImpOrder.append(i)
            if  inf_checker > len(cmls):
                msg = 'Your design of send and receive tokens makes a loop of interdependencies. You can avoid such loops by designing your workflow hierarchichally'
                raise IOError(msg)
        return tuple(ImpOrder),CompGraph

class BASE(object):
    def __init__(self, CompGraph):
        self.graph = CompGraph
        self.graph_info = {}
        self.send = {}      # {(iblock,token):[value,count]}
        self.requirements = ['pandas']
        self.start_time = time.time()
        self.block_time = 0
        self.date = std_datetime_str('date')
        self.time = std_datetime_str('time')
        self.InputScript = ''
        self.output_directory = '.'
        self.log = []
        self.cheml_type = {'descriptor':[], 'interpreter':[], 'input':[], 'output':[],
                           'selector':[],   'transformer':[], 'regressor':[],
                           'preprocessor':[],   'divider':[], 'postprocessor':[],
                           'classifier':[],   'evaluator':[], 'visualizer':[], 'optimizer':[]}

class Wrapper(LIBRARY):
    """
    Todo: documentation
    """
    def __init__(self, cmls, ImpOrder, CompGraph, InputScript, output_directory):
        self.Base = BASE(CompGraph)
        self.Base.InputScript = InputScript
        self.Base.output_directory = output_directory
        self.ImpOrder = ImpOrder
        self.cmls = cmls
        tmp_str = "=================================================\n"
        print tmp_str
        self._checker()

    def _checker(self):
        """
        check for key parameters like module and function and
        any possible typo in other params.
        """
        # get params
        legal_superfunctions = ['DataRepresentation','Script','Input','Output','Preprocessor','FeatureSelection','FeatureTransformation','Divider','Regression','Classification','Postprocessor','Evaluation','Visualization','Optimizer']

        # run over graph
        for iblock, block in enumerate(self.cmls):
            # check super function
            SuperFunction = block['SuperFunction']
            if SuperFunction not in legal_superfunctions:
                msg = '@Task #%i(%s): %s is not a valid task' %(iblock + 1, SuperFunction,SuperFunction)
                raise NameError(msg)
            # check parameters
            parameters = block['parameters']
            if 'host' not in parameters:
                msg = "@Task #%i(%s): no host name found" % (iblock + 1, SuperFunction)
                raise NameError(msg)
            if 'function' not in parameters:
                msg = "@Task #%i(%s): no function name found" % (iblock + 1, SuperFunction)
                raise NameError(msg)
            # check host and function
            host_function = (block['parameters']['host'], block['parameters']['function'])
            if not self.manual(host_function = host_function):
                msg = '@Task #%i(%s): not a valid (host,function) passed: %s' % (iblock + 1, SuperFunction, str(host_function) )
                raise NameError(msg)
        return 'The input file is in a correct format.'

    def call(self):
        self.refs = {}
        for iblock in self.ImpOrder:
            task = self.cmls[iblock]['SuperFunction']
            parameters = self.cmls[iblock]['parameters']
            host = parameters.pop('host')
            function = parameters.pop('function')
            start_time = time.time()
            tmp_str =  "======= block#%i: (%s, %s)" % (iblock + 1, host, function)
            print tmp_str
            tmp_str = "| run ...\n"
            print tmp_str
            if host == 'sklearn':
                # check methods
                legal_functions = [klass[0] for klass in inspect.getmembers(sklw)]
                if task == 'Define_Model':
                    pass
                elif function not in legal_functions:
                    msg = "function name '%s' in module '%s' is not available/valid"%(function,host)
                    raise NameError(msg)
                if task=='Define_Model':
                    self.references(host, function)  # check references
                    self.Base.graph_info[iblock] = (host, function)
                    cml_interface = [klass[1] for klass in inspect.getmembers(sklw) if klass[0] == 'regression'][0]
                    cmli = cml_interface(self.Base, parameters, iblock,task,function,host)
                    cmli.run()
                else:
                    self.references(host,function) # check references
                    self.Base.graph_info[iblock] = (host, function)
                    cml_interface = [klass[1] for klass in inspect.getmembers(sklw) if klass[0]==function][0]
                    cmli = cml_interface(self.Base,parameters,iblock,task,function,host)
                    cmli.run()
            elif host == 'cheml':
                # check methods
                legal_functions = [klass[0] for klass in inspect.getmembers(cmlw)]
                if function not in legal_functions:
                    msg = "@function #%i: couldn't find function '%s' in the module '%s' wrarpper" %(iblock,function,host)
                    raise NameError(msg)
                self.references(host,function) # check references
                self.Base.graph_info[iblock] = (host, function)
                cml_interface = [klass[1] for klass in inspect.getmembers(cmlw) if klass[0] == function][0]
                cmli = cml_interface(self.Base, parameters, iblock,task,function,host)
                cmli.run()
            elif host == 'pandas':
                # check methods
                legal_functions = [klass[0] for klass in inspect.getmembers(pdw)]
                if function not in legal_functions:
                    msg = "@function #%i: couldn't find function '%s' in the module '%s' wrarpper" %(iblock,function,host)
                    raise NameError(msg)
                self.references(host,function) # check references
                self.Base.graph_info[iblock] = (host, function)
                cml_interface = [klass[1] for klass in inspect.getmembers(pdw) if klass[0] == function][0]
                cmli = cml_interface(self.Base, parameters, iblock,task,function,host)
                cmli.run()
            elif host == 'tensorflow':
                # check methods
                legal_functions = [klass[0] for klass in inspect.getmembers(tfw)]
                if function not in legal_functions:
                    msg = "@function #%i: couldn't find function '%s' in the module '%s' wrarpper" %(iblock,function,host)
                    raise NameError(msg)
                self.references(host,function) # check references
                self.Base.graph_info[iblock] = (host, function)
                cml_interface = [klass[1] for klass in inspect.getmembers(tfw) if klass[0] == function][0]
                cmli = cml_interface(self.Base, parameters, iblock,task,function,host)
                cmli.run()

            end_time = tot_exec_time_str(start_time)
            tmp_str = "| ... done!"
            print tmp_str
            tmp_str = '| '+end_time
            print tmp_str
            tmp_str = "=======\n\n"
            print tmp_str
        self._save_references()
        tmp_str = "Total " + tot_exec_time_str(self.Base.start_time)
        print tmp_str
        tmp_str = std_datetime_str() + '\n'
        print tmp_str

class Settings(object):
    """
    makes the output directory.

    Parameters
    ----------
    output_directory: String, (default = "CheML.out")
        The directory path/name to store all the results and outputs

    input_copy: Boolean, (default = True)
        If True, keeps a copy of input script in the output_directory

    Returns
    -------
    output_directory
    """
    def __init__(self,output_directory="CMLWrapper.out", InputScript_copy = True):
        self.output_directory = output_directory
        self.InputScript_copy = InputScript_copy

    def fit(self,InputScript):
        initial_output_dir = copy.deepcopy(self.output_directory)
        i = 0
        while os.path.exists(self.output_directory):
            i+=1
            self.output_directory = initial_output_dir + '%i'%i
        os.makedirs(self.output_directory)
        # error_file = open(output_directory+'/'+errorfile,'a',0)
        if self.InputScript_copy:
            shutil.copyfile(InputScript, self.output_directory + '/InputScript.txt')
        logfile = open(self.output_directory + '/log.txt', 'a', 0)
        return self.output_directory, logfile

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = logfile

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

def run(INPUT_FILE, OUTPUT_DIRECTORY):
    """
    this is the only callable method in this file
    :param INPUT_FILE: path to the input file
    :return:
    """
    settings = Settings(OUTPUT_DIRECTORY)
    OUTPUT_DIRECTORY, logfile= settings.fit(INPUT_FILE)
    sys.stdout = Logger(logfile)
    script = open(INPUT_FILE, 'r')
    script = script.readlines()
    cmls, ImpOrder, CompGraph = Parser(script).fit()
    # print cmls
    # print ImpOrder
    # print CompGraph
    # sys.exit('this is how much you get till now!')
    wrapper = Wrapper(cmls, ImpOrder, CompGraph, INPUT_FILE, OUTPUT_DIRECTORY)
    wrapper.call()


#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
"""*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
 
 									  ChemML PySCRIPT

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#"""
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*

if __name__=="__main__":
    sys.exit()
