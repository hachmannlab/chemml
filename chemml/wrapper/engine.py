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
from collections import Counter

import numpy as np
import pandas as pd

import chemml
from ..utils import isint, value, std_datetime_str, tot_exec_time_str
from .interfaces import get_api, get_method, get_attributes, evaluate_param_value, evaluate_inputs


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
        The main funtion for parsing chemml's input json.
        It starts with finding blocks (nodes) of workflow and then runs other functions.
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

            # validate keys
            self.validate_keys(block_id, block)

            # collect send and recive to create the graph
            send_recv[block_id] = {'send':[], 'recv':[]}

            # main inputs
            if 'inputs' in block:
                for var in block['inputs']:
                    val = block['inputs'][var]
                    if isinstance(val, str) and val[0] == '@' and val.count('@')% 2 == 0:
                        temp = val.strip().split('@')[1:]     # "@ID2@df" >> ['', 'ID2', 'df']
                        for item in zip(temp[0::2], temp[1::2]):
                            send_recv[block_id]['recv'].append(item)

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
                    if isinstance(val, str) and val[0] == '@' and val.count('@')%2 == 0:
                        temp = val.strip().split('@')[1:]     # "@ID2@df" >> ['', 'ID2', 'df']
                        for item in zip(temp[0::2], temp[1::2]):
                            send_recv[block_id]['recv'].append(item)
                    elif isinstance(val, bool):
                        if val:
                            send_recv[block_id]['send'].append(var)

            # method inputs / outputs
            if 'method' in block:
                method_block = block['method']
                if 'inputs' in method_block:
                    for var in method_block['inputs']:
                        val = method_block['inputs'][var]
                        if isinstance(val, str) and val[0] == '@' and val.count('@')%2 == 0:
                            temp = val.strip().split('@')[1:]  # "@ID2@df" >> ['', 'ID2', 'df']
                            for item in zip(temp[0::2], temp[1::2]):
                                send_recv[block_id]['recv'].append(item)
                if 'outputs' in method_block:
                    for var in method_block['outputs']:
                        val = method_block['outputs'][var]
                        if isinstance(val, bool):
                            if val:
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
                    msg = 'The input is not valid. There is a broken pipe in the graph construction.'
                    self.logger.error(msg)
                    raise ValueError(msg)
                else:
                    graph_send[item[0]].append(node_id) # item[0] is sender and node_id is receiver
                    graph_recv[node_id].append(item[0])  # item[0] is sender and node_id is receiver
                    graph.append((item[0], node_id)) # (send, recv)
        return graph, graph_send, graph_recv

    def validate_keys(self, block_id, block):
        if 'name' not in block:
            msg = "The input json is not valid. @node ID#%s: name of class/function is missing." % (str(block_id))
            self.logger.error(msg)
            raise ValueError(msg)
        if 'library' not in block:
            msg = "The input json is not valid. @node ID#%s: name of library is missing." % (str(block_id))
            raise ValueError(msg)
        if 'module' not in block:
            msg = "The input json is not valid. @node ID#%s: The module name is missing." % (str(block_id))
            raise ValueError(msg)
        # check rest of keys
        available_keys = block.keys()
        all_keys = ['name', 'library', 'module', 'inputs', 'outputs',
                    'method', 'wrapper_io']
        if not set(available_keys) <= set(all_keys):
            msg = "The input json is not valid. @node ID#%s: redundant or irrelevant keys" % (str(block_id))
            self.logger.error(msg)
            raise ValueError(msg)

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

            if len(block['method']) > 0:
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
        print(set(end_nodes))
        print(set(layers[-1]))
        print(layers)
        if set(end_nodes) < set(layers[-1]):
            msg = "The input graph is not constructed properly."
            self.logger.error(msg)
            raise ValueError(msg)

        return layers


class Stream(object):
    """
    This is a container for the flowing data in the workflow.

    Parameters
    ----------
    token: tuple
        The token is a tuple of sender ID and the variable name.

    value: any data structure
        An arbitrary data structure that is flowing on the edges.


    """
    def __init__(self, token, value, count=1):
        self.token = token      # tuple
        self.value = value      # arbitrary type
        self.count = count      # int
        self.size = sys.getsizeof(value)    # int (bytes)


class Stack(object):
    def __init__(self, all_received, logger):

        # count number of times a sent token is used in the workflow
        self.initial_count = Counter(all_received)

        self.logger = logger

        # the stack of data on the edges
        self.stack = {}

        # start time should be available during running nodes
        self.start_time = time.time()

        # references can be updated by each node
        self.references = {}


        # self.graph = CompGraph
        # self.graph_info = {}
        # self.send = {}  # {(iblock,token):output class}
        # self.requirements = ['pandas']
        # self.block_time = 0
        # self.date = std_datetime_str('date')
        # self.time = std_datetime_str('time')
        # self.InputScript = ''
        # self.output_directory = '.'
        # self.log = []

    def push(self, token, value):
        """
        This function stores the sent data from each node.

        Parameters
        ----------
        token: tuple
            A tuple of two elements: first element presents the node ID of sender, and the second element is the variable name.

        value: any type
            The data that token is sending around.

        """
        if token in self.initial_count:
            if token not in self.stack:
                self.stack[token] = Stream(token, value, self.initial_count[token])

    def pull(self, token):
        """
        This function returns the data for a token.
        It also removes used data from memory.

        Parameters
        ----------
        token: tuple
            A tuple of two elements: first element presents the ID of sender node, and the second element is the variable name.

        Returns
        -------
        value: any type
            The stored date for the specified token
        """
        value = self.stack[token].value

        # clean memory
        if self.stack[token].count == 1:
            del self.stack[token]
        else:
            self.stack[token].count -= 1

        return value

    def getsizeof(self, token=None):
        """
        This function returns the size of a token's data in bytes.
        It also returns the total size of edges.

        Parameters
        ----------
        token: tuple, optional (default=None)

        Returns
        -------
        tuple
            A tuple of two elements: the first one presents the specified token's size (zero if None), and
            the second element is the total memory of flowing data.
        """
        token_size = 0
        if token is not None:
            token_size = self.stack[token].size

        total_size = 0
        for token in self.stack:
            total_size += self.stack[token].size

        return (token_size, total_size)


class Wrapper(object):
    """
    The main class to run the input json node by node.

    """

    def __init__(self,
                 input_dict,
                 logger,
                 output_dir,
                 send_recv=None,
                 all_received=None,
                 graph=None,
                 graph_send=None,
                 graph_recv=None,
                 layers=None,
                 ):

        # only instance of Stack to run workflow
        self.stack = Stack(all_received, logger)

        # other class attributes
        self.input_dict = input_dict
        self.logger = logger
        self.output_dir = output_dir
        self.layers = layers
        # self.Base.InputScript = InputScript
        # self.Base.output_directory = output_directory
        # self.ImpOrder = ImpOrder
        # self.cmls = cmls

        # print and log banner info
        self.prettyprint('banner')

        # run nodes one by one
        self.call()

    def call(self):
        self.refs = {}
        for group in self.layers:
            for block_id in group:
                block = self.input_dict['nodes'][block_id]

                # find the function/class
                name = block["name"]
                library = block["library"]

                # begin
                start_time = time.time()
                self.prettyprint('block_start', block_id, name, library)

                ### run wrappers
                self.interface(block, block_id)

                # end
                run_time = tot_exec_time_str(start_time)
                self.prettyprint('block_end', run_time)

                # self._save_references()

        # finish
        total_time = tot_exec_time_str(self.stack.start_time)
        self.prettyprint('finish', total_time)

    def prettyprint(self, level, *args):
        if level == 'banner':
            tmp_str = "================================================="
            print(tmp_str)
            self.logger.info(tmp_str)

            tmp_str = '* Based on the dependencies, we run nodes in the '
            print(tmp_str)
            self.logger.info(tmp_str)

            tmp_str = '  following order:'
            print(tmp_str)
            self.logger.info(tmp_str)

            for group in self.layers:
                tmp_str = "  " + str(group)
                print(tmp_str)
                self.logger.info(tmp_str)

            tmp_str = "\n"
            print(tmp_str)

            tmp_str = '* The outputs will be stored in the following '
            print(tmp_str)
            self.logger.info(tmp_str)

            tmp_str = 'directory: %s' % self.output_dir
            print(tmp_str)
            self.logger.info(tmp_str)

            tmp_str = "\n"
            print(tmp_str)
            self.logger.info(tmp_str)

        elif level=='block_start':
            tmp_str = "======= node ID#%s: (%s, %s)" % (args[0], args[1], args[2])
            print(tmp_str)
            self.logger.info(tmp_str)

            tmp_str = "| run ...\n"
            print(tmp_str)
            self.logger.info(tmp_str)

        elif level == 'block_end':
            tmp_str = "\n"
            print(tmp_str)
            self.logger.info(tmp_str)

            tmp_str = "| ... done!"
            print(tmp_str)
            self.logger.info(tmp_str)

            tmp_str = '| ' + args[0]
            print(tmp_str)
            self.logger.info(tmp_str)

            tmp_str = "=======\n\n"
            print(tmp_str)
            self.logger.info(tmp_str)

        elif level == 'finish':
            tmp_str = "Total " + args[0]
            print(tmp_str)
            self.logger.info(tmp_str)

            tmp_str = std_datetime_str() + '\n'
            print(tmp_str)
            self.logger.info(tmp_str)

            tmp_str = "================================================="
            print(tmp_str)
            self.logger.info(tmp_str)

        elif level == 'output':
            if not self.prettyprint_output:
                tmp_str = "... preparing outputs:"
                print(tmp_str)
                self.logger.info(tmp_str)
                self.prettyprint_output = True

            tmp_str = "      name: %s"%args[0]
            print(tmp_str)
            self.logger.info(tmp_str)

            tmp_str = "      size: %i bytes (total: %i bytes)"%(args[1][0],args[1][1])
            print(tmp_str)
            self.logger.info(tmp_str)

            tmp_str = "      type: %s"%str(type(args[2]))
            print(tmp_str)
            self.logger.info(tmp_str)

            if isinstance(args[2], np.ndarray) or isinstance(args[2], pd.DataFrame):
                tmp_str = "      shape: %s"%str(args[2].shape)
                print(tmp_str)
                self.logger.info(tmp_str)

            tmp_str = "      -----"
            print(tmp_str)
            self.logger.info(tmp_str)

    @staticmethod
    def get_func_output(function_output_,
                        output,
                        name, library, module, method_name=None):
        """
        maps the output name to the index of output for functions/methods that return more than one.
        Parameters
        ----------
        function_output_: tuple or anything else
            the returned outputs of a function

        output: str
            output name (token name)

        name: str
            function/class name

        library: str
            name of library

        module: str
            name of submodule

        method_name: str, optional (default=None)
            name of a method of a class

        Returns
        -------
        -1
            if it's a single-output

        int, 0 or positive
            if it's a multi-output function: the exact index of output

        """
        lame_metadata = {
            'chemml.wrapper.preprocessing':{
                'SplitColumns.fit': ['X1', 'X2']
            },
            'chemml.chem':{
                'tensorise_molecules': ['atoms','bonds','edges']
            },
            'chemml.datasets':{
                'load_cep_homo': ['smiles', 'homo'],
                'load_organic_density': ['smiles', 'density', 'features'],
                'load_xyz_polarizability': ['molecules', 'pol'],
                'load_comp_energy': ['entries', 'energy'],
                # 'load_crystal_structures': ['entries']

            }
        }

        # if it's a multiple output, must be a tuple
        # otherwise return the single output, I don't care about the output name (but GUI must care!!)
        if isinstance(function_output_, tuple):
            # parse metadata
            if method_name is None:
                output_names = lame_metadata['%s.%s'%(library,module)][name]
            else:
                output_names = lame_metadata['%s.%s'%(library,module)]['%s.%s'%(name,method_name)]

            # extract info from metadata
            n = len(output_names) # requires info from meta data if more than one
            # print(function_output_)
            if n > 1:
                index = output_names.index(output)
                return function_output_[index]
        else:
            return function_output_


    def interface(self, block, block_id):
        # flags
        self.prettyprint_output = False

        # classifiers
        name = block["name"]
        library = block["library"]
        module = block["module"]

        # get api
        try:
            api, api_type = get_api(name, library, module)
        except:
            msg = "Unable to import %s from %s.%s" % (name, library, module)
            self.logger.error(msg)
            raise ValueError(msg)

        # run api
        if name == 'train_test_split' and library == 'sklearn':
            from chemml.wrapper.sklearn_skl import train_test_split
            output_dict = train_test_split(block, self.stack)

            # function outputs
            if len(block['outputs']) > 0:
                outputs = block['outputs']
                for out_ in outputs:
                    if outputs[out_]:  # if it's True
                        val = output_dict[out_]
                        token = (block_id, out_)
                        self.stack.push(token, val)
                        self.prettyprint('output', out_, self.stack.getsizeof(token), val)

        elif api_type == 'class':
            # evaluate inputs
            inputs = evaluate_inputs(block['inputs'], self.stack, 'class')

            # instantiate class
            if inputs['obj'] is None:
                obj = api(**inputs['kwargs'])
            else:
                obj = inputs['obj']

            # get method
            if len(block['method']) > 0:

                # get method and its inputs
                method = get_method(obj, block['method']['name'])
                inputs = evaluate_inputs(block['method']['inputs'], self.stack)

                # run method
                function_output_ = self.run_function(name, library, module, block['method']['name'],
                                                     method, inputs)

                # method outputs
                # requires info from meta data if more than one
                if len(block['method']['outputs']) > 0:
                    outputs = block['method']['outputs']
                    for out_ in outputs:
                        if outputs[out_]: # if it's True
                            val = self.get_func_output(function_output_,
                                                         out_,
                                                         name, library, module,
                                                         block['method']['name'])
                            token = (block_id, out_)
                            self.stack.push(token, val)
                            self.prettyprint('output', out_, self.stack.getsizeof(token), val)

            # class outputs (attributes)
            if len(block['outputs'])>0 :
                attributes = list(block['outputs'].keys())
                for attr in attributes:
                    # all classes can send out an instance of that class, called obj
                    if attr == 'obj' and block['outputs'][attr]:
                        token = (block_id, 'obj')
                        self.stack.push(token, obj)
                        self.prettyprint('output', attr, self.stack.getsizeof(token), obj)
                    # the other outputs of a class are attributes of that class
                    else:
                        if block['outputs'][attr]: # if it's True
                            val = get_attributes(obj, attr)
                            token = (block_id, attr)
                            self.stack.push(token, val)
                            self.prettyprint('output', attr, self.stack.getsizeof(token), val)

        elif api_type == 'function':
            # evaluate function inputs
            inputs = evaluate_inputs(block['inputs'], self.stack)

            # run function
            function_output_ = self.run_function(name, library, module, '',
                                                 api, inputs)

            # function outputs
            if len(block['outputs']) > 0:
                outputs = block['outputs']
                for out_ in outputs:
                    if outputs[out_]:  # if it's True
                        val = self.get_func_output(function_output_,
                                                   out_,
                                                   name, library, module,
                                                   None)
                        token = (block_id, out_)
                        self.stack.push(token, val)
                        self.prettyprint('output', out_, self.stack.getsizeof(token), val)

    def run_function(self, name, library, module, method_name='', method=None, inputs=None):
        """
        run function with different input types
        """
        # exceptions
        # takes care of nodes that are more tricky to just automatically run them.
        if (name, library, module) == ("SaveCSV", "chemml", "wrapper.preprocessing"):
            if method_name == 'write':
                inputs['kwargs']['main_directory'] = self.output_dir
        elif (name, library, module) == ("SaveFile", "chemml", "wrapper.preprocessing"):
            if method_name == 'write':
                inputs['kwargs']['main_directory'] = self.output_dir
        elif (name, library, module) == ("SaveHDF5", "chemml", "wrapper.preprocessing"):
            if method_name == 'write':
                inputs['kwargs']['main_directory'] = self.output_dir

        if inputs['args'] is None:
            return method(**inputs['kwargs'])
        else:
            return method(*inputs['args'], **inputs['kwargs'])


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
            json.dump(input_dict, f, indent=2, sort_keys=True)


def variable_description(input_dict=None,
                         send_recv=None,
                         all_received=None,
                         graph=None,
                         graph_send=None,
                         graph_recv=None,
                         layers=None,):
    """
    This function prints out the output variables of the Parser with brief description to facilitate
    future development and contributions.
    """
    print ("\n\ninternal variables' description for developers:\n\n")

    # example
    if input_dict is not None:
        print("*** input_dict example:\n", input_dict, "\n")

    # description
    print("--- input_dict descriptoin:\n",
          "    type:    dictionary\n",
          "        keys:    'nodes', 'gui_format', 'template_id'\n",
          "        nodes:   dictionary\n",
          "            keys: 'name', 'library', 'module', 'inputs',\n "
          "                  'outputs', 'method', 'wrapper_io'\n",
          "            name:        str\n",
          "            library:     str\n",
          "            module:      str\n",
          "            inputs:      dictionary\n",
          "            outputs:     dictionary\n",
          "            wrapper_io:  dictionary\n",
          "            method:      dictionary\n",
          "                keys:    'name', 'inputs', 'outputs'\n"
          "### \n")

    # example
    if send_recv is not None:
        print("*** send_recv example:\n", send_recv, "\n")

    # description
    print("--- send_recv descriptoin:\n",
          "    type:    dictionary\n",
          "        keys:    node IDs\n",
          "        values:  dictionary\n",
          "            keys: 'send' and 'recv'\n",
          "            values: list\n",
          "                send elements: just variable name\n",
          "                recv elements: tuple of two elements\n",
          "                    first element:  ID of sender\n",
          "                    second element: variable name\n",
          "### \n")

    # example
    if all_received is not None:
        print("*** all_received example:\n", all_received, "\n")

    # description
    print("--- all_received descriptoin:\n",
          "    type:    list\n",
          "        elements:    tuple of two elements\n",
          "            first element:    ID of sender\n",
          "            second element:   variale name\n",
          "### \n")

    # example
    if graph is not None:
        print("*** graph example:\n", graph, "\n")

    # description
    print("--- graph descriptoin:\n",
          "    type:    list\n",
          "        elements:    tuple of two elements\n",
          "            first element:    ID of sender\n",
          "            second element:   ID of receiver\n",
          "### \n")

    # example
    if graph_send is not None:
        print("*** graph_send example:\n", graph_send, "\n")

    # description
    print("--- graph_send descriptoin:\n",
          "    type:    dictionary\n",
          "        keys:    node IDs\n",
          "        values:  list\n",
          "            elements: node IDs that each node is sending to\n",
          "### \n")

    # example
    if graph_recv is not None:
        print("*** graph_recv example:\n", graph_recv, "\n")

    # description
    print("--- graph_recv descriptoin:\n",
          "    type:    dictionary\n",
          "        keys:    node IDs\n",
          "        values:  list\n",
          "            elements: node IDs that each node is receiving from\n",
          "### \n")

    # example
    if layers is not None:
        print("*** layers example:\n", layers, "\n")

    # description
    print("--- layers descriptoin:\n",
          "    type:    list\n",
          "        elements:    list\n",
          "            elements:    node IDs\n",
          "            significance: order of runing based on their dependencies\n",
          "### \n")


def run(input_json, output_dir):
    """
    This is the main function to run ChemMLWrapper for an input script.
    
    Parameters
    __________
    input_json: str or dict
        This should be a path to the ChemMLWrapper input file or the actual input script in string or dictionary format.
        The input must have a valid json format.
        
    output_dir: str
        This is the path to the output directory. If the directory already exist, we add an integer to the end
        of the folder name incrementally, until the name of the folder is unique.

    """

    # try to convert json to dictionary or just get it as a dictionary
    if isinstance(input_json, dict):
        input_dict = input_json
        tmp_str = "parsing the input dictionary ..."
    elif isinstance(input_json, str):
        try:
            file_json = open(input_json, 'r')
            input_dict = json.load(file_json)
            tmp_str = "parsing input file: %s ..." % input_json
        except:
            try:
                input_dict = json.loads(input_json)
                tmp_str = "parsing the input string ..."
            except:
                msg = "The input is not a serializable json format."
                raise IOError(msg)
    else:
        msg = "First parameter must be the json object (a dictionary) or path to the input file with json format."
        raise IOError(msg)

    # create output directory and logger
    settings = Settings(output_dir)
    output_dir = settings.create_output()
    logger = settings.create_logger()
    # copy input to the output directory for the record
    settings.copy_inputscript(input_dict)

    # print banner
    banner(logger)

    # confirm if the input string is parsed
    print(tmp_str + '\n')
    logger.info(tmp_str + '\n')

    # parse the input dict
    parser = Parser(input_dict, logger)
    send_recv, all_received, graph, graph_send, graph_recv, layers = parser.serialize()
    input_dict = parser.input_dict # updated input_dict with switched off unused sent tokens

    # only for developers, comment out when you are done
    # variable_description(input_dict, send_recv, all_received, graph, graph_send, graph_recv, layers)

    # run wrappers for each node
    wrapper = Wrapper(input_dict, logger, output_dir,
                      send_recv, all_received, graph,
                      graph_send, graph_recv, layers)


#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
"""*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
 
 									  ChemML PySCRIPT

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#"""
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*

if __name__ == "__main__":
    sys.exit()
