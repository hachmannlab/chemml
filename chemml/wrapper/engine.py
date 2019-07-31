#!/usr/bin/env python

# python 2 and 3 compatible
from __future__ import print_function
from builtins import range

import sys
import os
import time
import copy
import inspect
import json
import logging
import importlib

import numpy as np

import chemml
from .pandas_pd import pdw
from .chemml_cml import cmlw
from .sklearn_skl import sklw

from ..utils import isint, value, std_datetime_str, tot_exec_time_str
from .base import LIBRARY


def banner(logger):
    PROGRAM_NAME = "ChemMLWrapper"
    PROGRAM_VERSION = chemml.__version__
    AUTHORS = chemml.__author__
    release_date = chemml.__release__
    str = []
    str.append("=================================================")
    str.append(PROGRAM_NAME + " " + PROGRAM_VERSION + " (" + release_date +
               ")")
    for AUTHOR in AUTHORS:
        str.append(AUTHOR)
    str.append("=================================================")
    str.append(time.ctime())
    str.append("")
    # str.append(DESCRIPTION)
    # str.append("")

    print('\n')
    for line in str:
        print(line)
        logger.info(line)


def evaluate_param_value(param_val):
    """
    evaluate the string input value to python data structure

    Parameters
    ----------
    param_val: str
        an entry of type str

    Returns
    -------
    bool
        True if the input can become an integer, False otherwise

    Notes
    -----
    returns 'type' for the entry 'type', although type is a code object
    """
    try:
        val = eval(param_val)
        if isinstance(val, type):
            return param_val
        else:
            return val
    except:
        return param_val


def cycle_in_graph(graph):
    """
    Return True if the directed graph has a cycle.

    Parameters
    ----------
    graph: dict
        The graph must be represented as a dictionary mapping vertices to
        iterables of neighbouring vertices.

    Examples
    --------
    >>> cycle_in_graph({1: (2,3), 2: (3,)})
    False
    >>> cycle_in_graph({1: (2,), 2: (3,), 3: (1,)})
    True
    """
    visited = set()
    path = [object()]
    path_set = set(path)
    stack = [iter(graph)]
    while stack:
        for v in stack[-1]:
            if v in path_set:
                return True
            elif v not in visited:
                visited.add(v)
                path.append(v)
                path_set.add(v)
                stack.append(iter(graph.get(v, ())))
                break
        else:
            path_set.remove(path.pop())
            stack.pop()
    return False


class Parser(object):
    """
    make sense of the input json.

    Parameters
    ----------

    input_dict: dict
        A dictionary based on the input json file.

    logger: logging.Logger
        the logger

    """

    def __init__(self, input_dict, logger):
        self.input_dict = input_dict
        self.logger = logger

    def serialize(self):
        """
        The main funtion for parsing chemml input.
        It starts with finding blocks and then runs other functions.
       """
        # validate the main components
        if 'nodes' not in self.input_dict.keys():
            msg = "The input json is not a valid ChemMLWrapper input."
            self.logger.error(msg)
            raise ValueError(msg)

        # available keys in each block:
        #   [name, library, module, inputs, method, outputs, wrapper_io]
        # if method is available, it contains: [name, inputs, outputs]
        # evaluate all variables inside inputs, outputs, and wrapper_io to store
        # extract all send/recv
        send_recv = {}
        for block_id in self.input_dict['nodes'].keys():
            # shrink the var name
            block = self.input_dict['nodes'][block_id]

            # collect send and recive to create the graph
            send_recv[block_id] = {'send':[], 'recv':[]}

            # main inputs
            if 'inputs' in block:
                for var in block['inputs']:
                    val = block['inputs'][var]
                    if isinstance(val, str) and val[0] == '@' and val.count('@') == 2:
                        temp = val.strip().split('@')     # "@ID2@df" >> ['', 'ID2', 'df']
                        send_recv[block_id]['recv'].append((temp[1], temp[2]))

            # main outputs
            if 'outputs' in block:
                for var in block['outputs']:
                    val = block['outputs'][var]
                    if isinstance(val, bool) and val:
                        send_recv[block_id]['send'].append(var)

            # wrapper i/o
            if 'wrapper_io' in block:
                for var in block['wrapper_io']:
                    val = block['wrapper_io'][var]
                    if isinstance(val, str) and val[0] == '@' and val.count('@') == 2:
                        temp = val.strip().split('@')     # e.g. "@ID2@df" >> ['', 'ID2', 'df']
                        send_recv[block_id]['recv'].append((temp[1], temp[2])) #(ID, name)
                    elif isinstance(val, bool) and val:
                        send_recv[block_id]['send'].append(var)

            # method inputs / outputs
            if 'method' in block:
                method_block = block['method']
                if 'inputs' in method_block:
                    for var in method_block['inputs']:
                        val = method_block['inputs'][var]
                        if isinstance(val, str) and val[0] == '@' and val.count('@') == 2:
                            temp = val.strip().split('@')     # e.g. "@ID2@df" >> ['', 'ID2', 'df']
                            send_recv[block_id]['recv'].append((temp[1], temp[2])) #(ID, name)
                if 'outputs' in method_block:
                    for var in method_block['outputs']:
                        val = method_block['outputs'][var]
                        if isinstance(val, bool) and val:
                            send_recv[block_id]['send'].append(var)

        ## clean the redundant send tokens in the send_recv for the memory efficiency
        # collect all send and received items
        all_received = []
        all_sent = []
        for id in send_recv:
            all_received+=send_recv[id]['recv']
            for token in send_recv[id]['send']:
                all_sent.append((id, token))

        # find redundants sends
        redundant = []
        for item in all_sent:
            if item not in all_received:
                redundant.append(item)

        # turn redundant sends off in the send_recv and input_dict (overwrite)
        for item in redundant:
            id = item[0]
            var = item[1]
            ## remove from send_recv
            send_recv[id]['send'].remove(var)
            ## remove from input_dict
            # outputs
            outputs = self.input_dict['nodes'][id].get('outputs', {})
            if var in outputs: outputs[var] = False
            # method outputs
            method = self.input_dict['nodes'][id].get('method', {})
            outputs = method.get('outputs', {})
            if var in outputs: outputs[var] = False
            # other outputs
            wrapper_io = self.input_dict['nodes'][id].get('wrapper_io', {})
            if var in wrapper_io: wrapper_io[var] = False

        # graph representatoins
        graph, graph_send, graph_recv = self.make_graph(send_recv)

        # no cycle is allowed in the graph
        if cycle_in_graph(graph_send):
            msg = "The input graph is constructed of cycles that is not allowed due to the interdependency."
            self.logger.error(msg)
            raise ValueError(msg)

        # find layers of running nodes based on their dependencies
        layers = self.hierarchicy(graph_send, graph_recv)

        # pretty print input dict
        self.prettyprinter(layers, graph_send, graph_recv)

        return send_recv, all_received, graph, graph_send, graph_recv, layers

    def make_graph(self, send_recv):
        """
        create a directed graph as a dictionary mapping vertices to
        iterables of neighbouring vertices.

        Parameters
        ----------
        send_recv: dict
            send_recv[node_id] = {'send':[name], 'recv':[(node_id, name)]}

        Returns
        -------
        graph: dict
            a directed graph as a dictionary mapping node IDs to
            iterables of neighbouring vertices.
        """
        graph_send = {id:[] for id in send_recv}
        graph_recv = {id:[] for id in send_recv}
        graph = [] # [ (sender, reciver) ]
        for node_id in send_recv:
            for item in send_recv[node_id]['recv']:
                ref = send_recv.get(item[0], {})
                if 'send' not in ref or item[1] not in ref['send']:
                    msg = 'The input is not valid. Ther is a broken pipe in the graph construction.'
                    self.logger.error(msg)
                    raise ValueError(msg)
                else:
                    graph_send[item[0]].append(node_id) # item[0] is sender and node_id is receiver
                    graph_recv[node_id].append(item[0])  # item[0] is sender and node_id is receiver
                    graph.append((item[0], node_id)) # (send, recv)
        return graph, graph_send, graph_recv

    def prettyprinter(self, layers, graph_send, graph_recv):

        # the order of implementation
        ordered = []
        for layer in layers:
            ordered+=layer

        # print them by order
        item = 0
        for id in ordered:
            block = self.input_dict['nodes'][id]
            item += 1
            line = '%s (%s)\n' % (block['name'], block['library'])
            line = line.rstrip("\n")
            tmp_str = '%i' % item + ' ' * (
                4 - len(str(item))) + '%s: '%str(id) + line
            print(tmp_str)
            self.logger.info(tmp_str)

            # line = 'library = %s\n' % (block['library'])
            # line = line.rstrip("\n")
            # tmp_str = '        ' + line
            # print(tmp_str)
            # self.logger.info(tmp_str)

            if 'method' in block:
                line = 'method = %s\n' % (block['method']['name'])
                line = line.rstrip("\n")
                tmp_str = '        ' + line
                print(tmp_str)
                self.logger.info(tmp_str)

            line = '<<<<<<< receive from:'
            line = line.rstrip("\n")
            tmp_str = '        ' + line
            print(tmp_str)
            self.logger.info(tmp_str)

            if len(graph_recv[id])>0:
                for param in graph_recv[id]:
                    line = '%s\n' % (param)
                    line = line.rstrip("\n")
                    tmp_str = '        ' + line
                    print(tmp_str)
                    self.logger.info(tmp_str)
            else:
                line = '%s\n' % ("nothing to receive!")
                line = line.rstrip("\n")
                tmp_str = '        ' + line
                print(tmp_str)
                self.logger.info(tmp_str)

            line = '>>>>>>> send to:'
            line = line.rstrip("\n")
            tmp_str = '        ' + line
            print(tmp_str)
            self.logger.info(tmp_str)

            if len(graph_send[id])>0:
                for param in graph_send[id]:
                    line = '%s\n' % (param)
                    line = line.rstrip("\n")
                    tmp_str = '        ' + line
                    print(tmp_str)
                    self.logger.info(tmp_str)
            else:
                line = '%s\n' % ("nothing to send!")
                line = line.rstrip("\n")
                tmp_str = '        ' + line
                print(tmp_str)
                self.logger.info(tmp_str)

            line = ''
            line = line.rstrip("\n")
            tmp_str = '        ' + line
            print(tmp_str)
            self.logger.info(tmp_str)

    def hierarchicy(self, graph_send, graph_recv):
        """
        find the order of implementation of functions based on sends and receives

        """
        start_nodes = []
        end_nodes = []
        for id in graph_send:
            if len(graph_send[id]) == 0:
                end_nodes.append(id)
            elif len(graph_recv[id]) == 0:
                start_nodes.append(id)

        # find layers
        layers = [start_nodes]
        collected = [i for i in start_nodes]
        while len(collected)<len(graph_send):
            next_layer = []
            for id in graph_recv:
                if id not in collected and set(graph_recv[id]) <= set(collected):
                    next_layer.append(id)
            layers.append(next_layer)
            collected+=next_layer

        # validate layers
        if set(end_nodes) != set(layers[-1]):
            msg = "The input graph is not constructed properly."
            self.logger.error(msg)
            raise ValueError(msg)

        return layers


class BASE(object):
    def __init__(self, CompGraph):
        self.graph = CompGraph
        self.graph_info = {}
        self.send = {}  # {(iblock,token):output class}
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
    The main class to run the input script block by block in the order of the ImpOrder.

    """

    def __init__(self, cmls, ImpOrder, CompGraph, InputScript,
                 output_directory):
        self.Base = BASE(
            CompGraph
        )  # only instance of BASE during the entire workflow run
        self.Base.InputScript = InputScript
        self.Base.output_directory = output_directory
        self.ImpOrder = ImpOrder
        self.cmls = cmls
        tmp_str = "=================================================\n"
        print(tmp_str)
        print ('* Based on the dependencies, We run blocks in the order of: %s'%str(np.array(ImpOrder)+1))
        print ('* The outputs will be stored in the following directory: %s\n' %output_directory)
        tmp_str = "=================================================\n"
        print(tmp_str)

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
                msg = "@Task #%i(%s): no function name found" % (iblock + 1,
                                                                 task)
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
            tmp_str = "======= block#%i: (%s, %s)" % (iblock + 1, host,
                                                      function)
            print(tmp_str)
            tmp_str = "| run ...\n"
            print(tmp_str)
            if host == 'sklearn':
                # check methods
                legal_functions = [
                    klass[0] for klass in inspect.getmembers(sklw)
                ]
                if function in legal_functions:
                    self.references(host, function)  # check references
                    self.Base.graph_info[iblock] = (host, function)
                    cml_interface = [
                        klass[1] for klass in inspect.getmembers(sklw)
                        if klass[0] == function
                    ][0]
                    cmli = cml_interface(self.Base, parameters, iblock, task,
                                         function, host)
                    cmli.run()
                else:
                    self.references(host, function)  # check references
                    self.Base.graph_info[iblock] = (host, function)
                    cml_interface = [
                        klass[1] for klass in inspect.getmembers(sklw)
                        if klass[0] == 'automatic_run'
                    ][0]
                    cmli = cml_interface(self.Base, parameters, iblock, task,
                                         function, host)
                    cmli.run()
            elif host == 'chemml':
                # check methods
                legal_functions = [
                    klass[0] for klass in inspect.getmembers(cmlw)
                ]
                if function not in legal_functions:
                    msg = "@function #%i: couldn't find function '%s' in the module '%s' wrarpper" % (
                        iblock, function, host)
                    raise NameError(msg)
                self.references(host, function)  # check references
                self.Base.graph_info[iblock] = (host, function)
                cml_interface = [
                    klass[1] for klass in inspect.getmembers(cmlw)
                    if klass[0] == function
                ][0]
                cmli = cml_interface(self.Base, parameters, iblock, task,
                                     function, host)
                cmli.run()
            elif host == 'pandas':
                # check methods
                legal_functions = [
                    klass[0] for klass in inspect.getmembers(pdw)
                ]
                if function not in legal_functions:
                    msg = "@function #%i: couldn't find function '%s' in the module '%s' wrarpper" % (
                        iblock, function, host)
                    raise NameError(msg)
                self.references(host, function)  # check references
                self.Base.graph_info[iblock] = (host, function)
                cml_interface = [
                    klass[1] for klass in inspect.getmembers(pdw)
                    if klass[0] == function
                ][0]
                cmli = cml_interface(self.Base, parameters, iblock, task,
                                     function, host)
                cmli.run()

            end_time = tot_exec_time_str(start_time)
            tmp_str = "| ... done!"
            print(tmp_str)
            tmp_str = '| ' + end_time
            print(tmp_str)
            tmp_str = "=======\n\n"
            print(tmp_str)
        self._save_references()
        tmp_str = "Total " + tot_exec_time_str(self.Base.start_time)
        print(tmp_str)
        tmp_str = std_datetime_str() + '\n'
        print(tmp_str)


class Settings(object):
    """
    This class creates the output directory and the logger.

    Parameters
    ----------
    output_directory: String, (default = "ChemMLWrapper_output")
        The directory path/name to store all the results and outputs


    Returns
    -------
    output_directory
    """

    def __init__(self, output_directory="ChemMLWrapper_output"):
        self.output_directory = output_directory

    def create_output(self):
        initial_output_dir = copy.deepcopy(self.output_directory)
        i = 1
        while os.path.exists(self.output_directory):
            i += 1
            self.output_directory = initial_output_dir + '-%i' % i
        os.makedirs(self.output_directory)
        return self.output_directory

    def create_logger(self):
        """
        must be called after create_output to have access to the most updated output_directory
        """
        importlib.reload(logging)
        logfile = os.path.join(self.output_directory, 'log.txt')
        logging.basicConfig(filename=logfile,
                            filemode='a',
                            format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level=logging.DEBUG)
        logger = logging.getLogger('ChemML')
        return logger

    def copy_inputscript(self, input_dict):
        file_path = os.path.join(self.output_directory , 'input.json')
        with open(file_path, 'w') as f:
            json.dump(input_dict, f)


def run(input_json, output_dir):
    """
    This is the main function to run ChemMLWrapper for an input script.
    
    Parameters
    __________
    input_json: str
        This should be a path to the ChemMLWrapper input file or the actual input script in string format.
        The input must have a valid json format.
        
    output_dir: str
        This is the path to the output directory. If the directory already exist, we add an integer to the end
        of the folder name incrementally, until the name of the folder is unique.

    """
    # input must be string
    if not isinstance(input_json, str):
        msg = "First parameter must be the path to the input file with json format."
        raise IOError(msg)

    # try to convert json to dictionary
    try:
        file_json = open(input_json, 'rb')
        input_dict = json.load(file_json)
        tmp_str = "parsing input file: %s ..." % input_json
    except:
        try:
            input_dict = json.loads(input_json)
            tmp_str = "parsing the input string ..."
        except:
            msg = "The input is not a serializable json format."
            raise IOError(msg)

    # create output directory and logger
    settings = Settings(output_dir)
    output_dir = settings.create_output()
    logger = settings.create_logger()
    # copy input to the output directory for the record
    settings.copy_inputscript(input_dict)

    # print banner
    banner(logger)

    # confirm the input string is parsed
    print(tmp_str + '\n')
    logger.info(tmp_str + '\n')

    # parse the input dict
    parser = Parser(input_dict, logger)
    send_recv, all_received, graph, graph_send, graph_recv, layers = parser.serialize()
    input_dict = parser.input_dict # updated input_dict with switched off


    # wrapper = Wrapper(cmls, run_order, comp_graph, input_json, output_dir)
    # wrapper.call()


#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
"""*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
 
 									  ChemML PySCRIPT

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#"""
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*

if __name__ == "__main__":
    sys.exit()
