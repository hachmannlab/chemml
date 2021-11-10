#!/usr/bin/env python

import sys
import os
import time
import copy
import inspect

# from .. import __version__, __author__
from chemml.wrapper.pandas_pd import pdw
from chemml.wrapper.chemml_cml import cmlw
from chemml.wrapper.sklearn_skl import sklw
# from tensorflow_tf import tfw

from chemml.utils import isint, value, std_datetime_str, tot_exec_time_str
from chemml.wrapper.base import LIBRARY

def banner():
    PROGRAM_NAME = "ChemML"
    # PROGRAM_VERSION = __version__
    REVISION_DATE = "2018-03-20"
    # AUTHORS = __author__
    CONTRIBUTORS = " "
    DESCRIPTION = "ChemML is a machine learning and informatics program suite for the chemical and materials sciences."
    str = []
    str.append("=================================================")
    # str.append(PROGRAM_NAME + " " + PROGRAM_VERSION + " (" + REVISION_DATE + ")")
    # for AUTHOR in AUTHORS:
    #     str.append(AUTHOR)
    str.append("=================================================")
    str.append(time.ctime())
    str.append("")
    # str.append(DESCRIPTION)
    # str.append("")

    # print
    for line in str:
        print (line)

class Parser(object):
    """
    script: list of strings
        A list of lines in the chemml script file.
    """
    def __init__(self, script):
        self.script = script

    def fit(self):
        """
        The main funtion for parsing chemml script.
        It starts with finding blocks and then runs other functions.
        :return:
        cmls: chemml script
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
        for item in range(len(blocks)):
            block = blocks[item]
            function = self._functions(block[0])
            parameters, send, recv = self._parameters(block,item)
            cmls.append({"task": function,
                         "parameters": parameters,
                         "send": send,
                         "recv": recv})
        return cmls

    def _print_out(self, cmls):
        item = 0
        for block in cmls:
            item+=1
            line = '%s\n' %(block['task'])
            line = line.rstrip("\n")
            tmp_str =  '%i'%item+' '*(4-len(str(item)))+'Task: '+line
            print (tmp_str)
            line = '<<<<<<<'
            line = line.rstrip("\n")
            tmp_str = '        ' + line
            print (tmp_str)
            if len(block['parameters']) > 0 :
                for param in block['parameters']:
                    line = '%s = %s\n'%(param,block['parameters'][param])
                    line = line.rstrip("\n")
                    tmp_str =  '        '+line
                    print (tmp_str)
            else:
                line = ' :no parameter passed: set to default values if available'
                line = line.rstrip("\n")
                tmp_str =  '        ' + line
                print (tmp_str)
            line = '>>>>>>>'
            line = line.rstrip("\n")
            tmp_str =  '        ' + line
            print (tmp_str)
            if len(block['send']) > 0:
                for param in block['send']:
                    line = '%s -> send (id=%i)\n' %(param[0],param[1])
                    line = line.rstrip("\n")
                    tmp_str =  '        ' + line
                    print (tmp_str)
            else:
                line = ' :nothing to send:'
                line = line.rstrip("\n")
                tmp_str =  '        ' + line
                print (tmp_str)
            if len(block['recv']) > 0:
                for param in block['recv']:
                    line = '%s <- recv (id=%i)\n' %(param[0],param[1])
                    line = line.rstrip("\n")
                    tmp_str = '        ' + line
                    print (tmp_str)
            else:
                line = ' :nothing to receive:'
                line = line.rstrip("\n")
                tmp_str = '        ' + line
                print (tmp_str)
            line = ''
            line = line.rstrip("\n")
            tmp_str = '        ' + line
            print (tmp_str)

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
            msg = '@chemml script - number of sent tokens must be less or equal to number of received tokens'
            raise ValueError(msg)
        send_ids = [k[1] for k,v in send_all]
        recv_ids = [k[1] for k,v in recv_all]
        for id in send_ids:
            if send_ids.count(id)>1:
                msg = 'identified non unique send id (id#%i)'%id
                raise NameError(msg)
        if set(send_ids) != set(recv_ids):
            print (set(send_ids),set(recv_ids))
            msg = 'missing pairs of send and receive id:\n send IDs:%s\n recv IDs:%s\n'%(str(set(send_ids)),str(set(recv_ids)))
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
        self.send = {}      # {(iblock,token):output class}
        self.requirements = ['pandas']
        self.start_time = time.time()
        self.block_time = 0
        self.date = std_datetime_str('date')
        self.time = std_datetime_str('time')
        self.InputScript = ''
        self.output_directory = '.'
        self.log = []

class Wrapper(LIBRARY):
    """
    Todo: documentation
    """
    def __init__(self, cmls, ImpOrder, CompGraph, InputScript, output_directory):
        self.Base = BASE(CompGraph)     # initial and only instance of BASE during the entire wrapper run
        self.Base.InputScript = InputScript
        self.Base.output_directory = output_directory
        self.ImpOrder = ImpOrder
        self.cmls = cmls
        tmp_str = "=================================================\n"
        print (tmp_str)
        self._checker()

    def _checker(self):
        """
        check for key parameters like module and function and
        any possible typo in other params.
        """
        # get params
        # legal_tasks = ['DataRepresentation','Script','Input','Output','Preprocessor','FeatureSelection',
        #                         'FeatureTransformation','Divider','Regression','Classification','Postprocessor',
        #                         'Evaluation','Visualization','Optimizer']
        # legal_tasks = ['Enter','Prepare','Model','Search','Mix','Visualize','Store']
        # run over graph
        for iblock, block in enumerate(self.cmls):
            # check super function
            task = block['task']
            # if task not in legal_tasks:
            #     msg = '@Task #%i(%s): %s is not a valid task' %(iblock + 1, task,task)
            #     raise NameError(msg)
            # check parameters
            parameters = block['parameters']
            if 'host' not in parameters:
                msg = "@Task #%i(%s): no host name found" % (iblock + 1, task)
                raise NameError(msg)
            if 'function' not in parameters:
                msg = "@Task #%i(%s): no function name found" % (iblock + 1, task)
                raise NameError(msg)
            # check host and function
            # host_function = (block['parameters']['host'], block['parameters']['function'])
        return 'The input file is in a correct format.'

    def call(self):
        self.refs = {}
        for iblock in self.ImpOrder:
            task = self.cmls[iblock]['task']
            parameters = self.cmls[iblock]['parameters']
            host = parameters.pop('host')
            function = parameters.pop('function')
            start_time = time.time()
            tmp_str =  "======= block#%i: (%s, %s)" % (iblock + 1, host, function)
            print (tmp_str)
            tmp_str = "| run ...\n"
            print (tmp_str)
            if host == 'sklearn':
                # check methods
                legal_functions = [klass[0] for klass in inspect.getmembers(sklw)]
                if function in legal_functions:
                    self.references(host, function)  # check references
                    self.Base.graph_info[iblock] = (host, function)
                    cml_interface = [klass[1] for klass in inspect.getmembers(sklw) if klass[0] == function][0]
                    cmli = cml_interface(self.Base, parameters, iblock, task, function, host)
                    cmli.run()
                else:
                    self.references(host, function)  # check references
                    self.Base.graph_info[iblock] = (host, function)
                    cml_interface = [klass[1] for klass in inspect.getmembers(sklw) if klass[0] == 'automatic_run'][0]
                    cmli = cml_interface(self.Base, parameters, iblock, task, function, host)
                    cmli.run()
            elif host == 'chemml':
                # check methods
                legal_functions = [klass[0] for klass in inspect.getmembers(cmlw)]
                # print("legal_functions: ", legal_functions)
                if function not in legal_functions:
                    msg = "@function #%i: couldn't find function '%s' in the module '%s' wrapper" %(iblock,function,host)
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
            print (tmp_str)
            tmp_str = '| '+end_time
            print (tmp_str)
            tmp_str = "=======\n\n"
            print (tmp_str)
        self._save_references()
        tmp_str = "Total " + tot_exec_time_str(self.Base.start_time)
        print (tmp_str)
        tmp_str = std_datetime_str() + '\n'
        print (tmp_str)

class Settings(object):
    """
    makes the output directory.
    Parameters
    ----------
    output_directory: String, (default = "ChemML.out")
        The directory path/name to store all the results and outputs
    input_copy: Boolean, (default = True)
        If True, keeps a copy of input script in the output_directory
    Returns
    -------
    output_directory
    """
    def __init__(self,output_directory="CMLWrapper_out"):
        self.output_directory = output_directory

    def fit(self):
        initial_output_dir = copy.deepcopy(self.output_directory)
        i = 0
        while os.path.exists(self.output_directory):
            i+=1
            self.output_directory = initial_output_dir + '%i'%i
        os.makedirs(self.output_directory)
        logfile = open(self.output_directory + '/log.txt', 'a')
        errorfile = open(self.output_directory + '/error.txt', 'a')
        return self.output_directory, logfile, errorfile

    def write_InputScript(self,InputScript):
        with open(self.output_directory + '/InputScript.txt','w') as f:
            for line in InputScript:
                f.write("%s\n"%line)

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = logfile

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class Error(object):
    def __init__(self,errorfile):
        self.terminal = sys.stderr
        self.err = errorfile

    def write(self, message):
        self.terminal.write(message)
        self.err.write(message)

    def flush(self):
        pass


def run(INPUT_FILE, OUTPUT_DIRECTORY):
    """
    this is the only callable method in this file
    :param INPUT_FILE: path to the input file or the actual input script
    :return:
    """
    try:
        script = open(INPUT_FILE, 'r')
        script = script.readlines()
        tmp_str = "parsing the input file: %s ..."%INPUT_FILE
    except:
        if isinstance(INPUT_FILE, list):
            script = INPUT_FILE
            tmp_str = "parsing the input file: received as a list of lines ..."
        elif isinstance(INPUT_FILE, str):
            if '##' in INPUT_FILE and '>>' in INPUT_FILE:
                script = INPUT_FILE.split('\n')
                tmp_str = "parsing the input file: received in string format ..."
            else:
                banner()
                msg = "couldn't find the input file path or the input script is not valid"
                raise IOError(msg)
    settings = Settings(OUTPUT_DIRECTORY)
    OUTPUT_DIRECTORY, logfile, errorfile= settings.fit()
    sys.stdout = Logger(logfile)
    sys.stderr = Error(errorfile)
    banner()
    print (tmp_str + '\n')
    settings.write_InputScript(script)
    cmls, ImpOrder, CompGraph = Parser(script).fit()
    # print("CMLS: ",cmls)
    # print ImpOrder
    # print CompGraph
    # sys.exit('this is how much you get till now!')
    wrapper = Wrapper(cmls, ImpOrder, CompGraph, INPUT_FILE, OUTPUT_DIRECTORY)
    # print(wrapper)
    wrapper.call()


#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
"""*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
 
 									  ChemML PySCRIPT
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#"""
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*

if __name__=="__main__":
    sys.exit()