import os
import sys
from ..database import sklearn_db, cheml_db
from ..database.TSHF import tshf
import copy

def txt(ntab=1, text1='', ntab12=1, adj12=False, text2=None, ntab23=1, adj23=False, text3=None,
        ntab34=1, adj34=False, text4=None, ntab45=1, adj45=False, text5=None):
    tab = '    '
    line = tab*ntab + text1
    if text2 is not None:
        if adj12:
            line += (ntab12*4 - len(text1))*' '
        else:
            line += tab*ntab12
        line += text2
        if text3 is not None:
            if adj23:
                line += (ntab23 * 4 - len(text2)) * ' '
            else:
                line += tab * ntab23
            line += text3
            if text4 is not None:
                if adj34:
                    line += (ntab34 * 4 - len(text3)) * ' '
                else:
                    line += tab * ntab34
                line += text4
                if text5 is not None:
                    if adj45:
                        line += (ntab45 * 4 - len(text4)) * ' '
                    else:
                        line += tab * ntab45
                    line += text5

    line = line.rstrip("\n")
    return line

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def transform(blocks):
    """
    goals:
        - collect all sends and receives
        - check send and receive format.
        - make the computational graph
        - find the order of implementation of functions based on sends and receives

    :param cmls:
    :return implementation order:
    """
    cmls = [blocks[key] for key in sorted(blocks.iterkeys())]
    send_all = []
    recv_all = []
    for block in cmls:
        send_all += block['send'].items()
        recv_all += block['recv'].items()

    # make graph
    reformat_send = {k[1]: [v, k[0]] for k, v in send_all}
    CompGraph = tuple([tuple(reformat_send[k[1]] + [v, k[0]]) for k, v in recv_all])

    # find orders
    ids_sent = []
    ImpOrder = []
    inf_checker = 0
    while len(ImpOrder) < len(cmls):
        inf_checker += 1
        for i in range(len(cmls)):
            if i not in ImpOrder:
                ids_recvd = [k[1] for k, v in cmls[i]['recv'].items()]
                if len(ids_recvd) == 0:
                    ids_sent += [k[1] for k, v in cmls[i]['send'].items()]
                    ImpOrder.append(i)
                elif len(set(ids_recvd) - set(ids_sent)) == 0:
                    ids_sent += [k[1] for k, v in cmls[i]['send'].items()]
                    ImpOrder.append(i)
        if inf_checker > len(cmls):
            msg = 'Your design of send and receive tokens makes a loop of interdependencies. You can avoid such loops by designing your workflow hierarchichally'
            raise IOError(msg)
    return tuple(ImpOrder), CompGraph


class ui(object):
    def __init__(self):
        self.tasks, self.combinations = tshf()
        self.blocks = {}
        self.block_id = -1
        self.comp_graph = []
        self.out_dir = "CMLWrapper.out"
        self.home_page()


    def home_page(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print '\n'
        print txt(1, "home           tasks           subtasks           hosts           functions               parameters           pipelines")
        print txt(1, "  @..............:.................:................:.................:........................:...................:")
        print '\n'
        print bcolors.BOLD + txt(11, "Welcome to the ChemML Wrapper Interface") + '\n' + bcolors.ENDC
        print txt(9, "*** Please enlarge the terminal window for a better view ***") + '\n'
        print txt(2, "enter n to start with a new script")
        print txt(2, "enter e to load an existing script")
        print txt(2, "enter t to start from a template")
        key = False
        while not key:
            k = raw_input(bcolors.BOLD + txt(2, "Choose how to start (n : new  ,  e : existing  ,  t : template): ") + bcolors.ENDC)
            k = k.strip()
            if k=='n':
                ko = raw_input(bcolors.BOLD + txt(2, "enter the name of output directory (or press enter for the current name = '%s'): "%self.out_dir) + bcolors.ENDC)
                ko = ko.strip()
                if ko == '':
                    key = True
                else:
                    self.out_dir = ko
                    key = True
            elif k=='e':
                key = True
            elif k=='t':
                key = True
        self.task_page()


    def task_page(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print '\n'
        print txt(1, "home           tasks           subtasks           hosts           functions               parameters           pipelines")
        print txt(1, "  :..............@.................:................:.................:........................:...................:")
        print '\n'
        print txt(2, "pick a task ...")+ '\n'
        print txt(3, 'id', 1, True, 'task')
        print txt(3, '--', 1, True, '----')
        for i,task in enumerate(self.tasks):
            print txt(3, str(i), 1, True, task)
        print '\n'
        options = [str(i) for i in range(len(self.tasks))] + ['s','q','b', 'p']
        key = False
        while not key:
            k = raw_input(bcolors.BOLD + txt(1,"enter one of ( id ,  p : pipeline page  ,  b : back  ,  s : save  ,  q : quit ): ") + bcolors.ENDC)
            k = k.strip()
            if k in options:
                key = True
        if k == 'q':
            sys.exit()
        elif k == 's':
            os.system('cls' if os.name == 'nt' else 'clear')
        elif k == 'b':
            self.home_page()
        elif k == 'p':
            self.graph_page()
        elif k in [str(i) for i in range(len(self.tasks))]:
            # os.system('cls' if os.name == 'nt' else 'clear')
            self.task = '%s'%self.tasks[int(k)]
            self.subtask_page()


    def subtask_page(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print '\n'
        print txt(1, "home           tasks           subtasks           hosts           functions               parameters           pipelines")
        print txt(1, "  :..............:.................@................:.................:........................:...................:")
        print '\n'
        subtasks = [i for i in self.combinations[self.task]]
        print txt(1, "task:", 3, True, self.task ) + '\n'
        print txt(2, "pick a subtask ...")+ '\n'
        print txt(3, 'id', 1, True, 'subtask')
        print txt(3, '--', 1, True, '----')
        for i, subtask in enumerate(subtasks):
            print txt(3, str(i), 1, True, subtask)
        print '\n'
        options = [str(i) for i in range(len(subtasks))] + ['b', 'q']
        key = False
        while not key:
            k = raw_input(bcolors.BOLD + txt(1, "enter one of (id, 'b':back, 'q':quit): ") + bcolors.ENDC)
            k = k.strip()
            if k in options:
                key = True
        if k == 'q':
            sys.exit()
        elif k == 'b':
            self.task_page()
        elif k in [str(i) for i in range(len(subtasks))]:
            # os.system('cls' if os.name == 'nt' else 'clear')
            self.subtask = '%s' % subtasks[int(k)]
            self.host_page()


    def host_page(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print '\n'
        print txt(1, "home           tasks           subtasks           hosts           functions               parameters           pipelines")
        print txt(1, "  :..............:.................:................@.................:........................:...................:")
        print '\n'
        hosts = [i for i in self.combinations[self.task][self.subtask]]
        print txt(1, "task:", 3, True, self.task )
        print txt(1, "subtask:", 3, True , self.subtask) + '\n'
        print txt(2, "pick a host ...")+ '\n'
        print txt(3, 'id', 1, True, 'host')
        print txt(3, '--', 1, True, '----')
        for i, host in enumerate(hosts):
            print txt(3, str(i), 1, True, host)
        print '\n'
        options = [str(i) for i in range(len(hosts))] + ['b', 'q']
        key = False
        while not key:
            k = raw_input(bcolors.BOLD + txt(1, "enter one of (id, 'b':back, 'q':quit): ") + bcolors.ENDC)
            k = k.strip()
            if k in options:
                key = True
        if k == 'q':
            sys.exit()
        elif k == 'b':
            self.subtask_page()
        elif k in [str(i) for i in range(len(hosts))]:
            # os.system('cls' if os.name == 'nt' else 'clear')
            self.host = '%s' % hosts[int(k)]
            self.function_page()


    def function_page(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print '\n'
        print txt(1, "home           tasks           subtasks           hosts           functions               parameters           pipelines")
        print txt(1, "  :..............:.................:................:.................@........................:...................:")
        print '\n'
        functions = [i for i in self.combinations[self.task][self.subtask][self.host]]
        print txt(1, "task:", 3, True, '%s' % self.task)
        print txt(1, "subtask:", 3, True , '%s' % self.subtask)
        print txt(1, "host:", 3, True , '%s' % self.host) + '\n'
        print txt(2, "pick a function ...")
        print txt(3, 'id', 1, True, 'function')
        print txt(3, '--', 1, True, '----')
        for i, func in enumerate(functions):
            print txt(3, str(i), 1, True, func)
        print '\n'
        options = [str(i) for i in range(len(functions))] + ['b', 'q']
        key = False
        while not key:
            k = raw_input(bcolors.BOLD + txt(1, "enter one of (id, 'b':back, 'q':quit): ") + bcolors.ENDC)
            k = k.strip()
            if k in options:
                key = True
        if k == 'q':
            sys.exit()
        elif k == 'b':
            self.host_page()
        elif k in [str(i) for i in range(len(functions))]:
            # os.system('cls' if os.name == 'nt' else 'clear')
            self.function = '%s' % functions[int(k)]
            self.db_extract_function()
            self.custom_function_page()


    ##################################


    def db_extract_function(self):
        if self.host == 'sklearn':
            self.metadata = getattr(sklearn_db, self.function)()
        elif self.host == 'cheml':
            self.metadata = getattr(cheml_db, self.function)()
        self.wparams = {i:copy.deepcopy(vars(self.metadata.WParameters)[i]) for i in vars(self.metadata.WParameters).keys() if
                     i not in ('__module__', '__doc__')}
        self.fparams = {i:copy.deepcopy(vars(self.metadata.FParameters)[i]) for i in vars(self.metadata.FParameters).keys() if
                     i not in ('__module__', '__doc__')}
        self.parameters = {}
        self.poptions = []
        for i,j in enumerate(self.wparams):
            self.parameters['%iw'%i] = self.wparams[j]
            self.poptions.append('%iw'%i)
        for i,j in enumerate(self.fparams):
            self.parameters['%if'%i] = self.fparams[j]
            self.poptions.append('%if' % i)


    def custom_function_help(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print '\n'
        print bcolors.BOLD + txt(1, "parameters page keys") + bcolors.ENDC
        print txt(2, bcolors.BOLD + 'a' + bcolors.ENDC, 1, True, "    add function to the workflow as a node.")
        print txt(2, bcolors.BOLD + 'id' + bcolors.ENDC, 1, True, "   edit a parameter's value. This is the parameter id. Next enter the python format of the new value (e.g. string with quotation mark).")
        print txt(2, bcolors.BOLD + 't' + bcolors.ENDC, 1, True, "    go to the task page and ignore this function.")
        print txt(2, bcolors.BOLD + 'b' + bcolors.ENDC, 1, True, "    back to the previous page (functions page).")
        print txt(2, bcolors.BOLD + 'q' + bcolors.ENDC, 1, True, "    quit the ChemML wrapper's interface without saving.")
        print txt(2, bcolors.BOLD + 'h' + bcolors.ENDC, 1, True, "    help page, this page!")
        print '\n'

        key = False
        while not key:
            k = raw_input(bcolors.BOLD + txt(1, "enter q to quit help page: ") + bcolors.ENDC)
            k = k.strip()
            if k in ['q']:
                self.custom_function_page()


    def custom_function_keys(self):
        key = False
        print txt(2, "** edit values by entering parameter's id and new value separately")
        print txt(2, "*** when you are done with editing enter 'a' to add this function to the workflow") + '\n'
        while not key:
            k = raw_input(bcolors.BOLD + txt(1, "enter one of ( id new_value  ,  a : add  ,  b : back  ,  t : tasks page  ,  h : help  ,  q : quit): ") + bcolors.ENDC)
            k = k.strip()
            if k in ['b', 'q', 'a', 't', 'h']:
                key = True
            elif ' ' in k:
                ks = k.split(' ')
                while '' in ks:
                    ks.remove('')
                if len(ks) == 2 and ks[0].strip() in self.parameters:
                    try:
                        param_value = eval(ks[1])
                        key = True
                    except Exception:
                        print bcolors.FAIL+ txt(1,"wrong format: %s - hint: use quotation mark for string"%ks[1]) +bcolors.ENDC

        if k == 'q':
            sys.exit()
        elif k == 'b':
            self.function_page()
        elif k == 'h':
            self.custom_function_help()
        elif k == 'a':
            self.block_id += 1
            self.blocks[self.block_id] = {'function':self.function,'host':self.host,'subtask':self.subtask,'task':self.task,
                                         'parameters': {self.parameters[z].name:self.parameters[z].default for z in self.parameters},
                                         'send': {}, 'recv': {}
                                          }
            self.graph_page()
        elif k == 't':
            self.task_page()
        elif ks[0] in self.parameters:
            self.parameters[ks[0]].default = param_value
            self.custom_function_page()


    def custom_function_page(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print '\n'
        print txt(1, "home           tasks           subtasks           hosts           functions               parameters           pipelines")
        print txt(1, "  :..............:.................:................:.................:........................@...................:")
        print '\n'
        print txt(1, "task:", 5, True, '%s' % self.task)
        print txt(1, "subtask:", 5, True , '%s' % self.subtask)
        print txt(1, "host:", 5, True , '%s' % self.host)
        print txt(1, "function:", 5, True , '%s' % self.function) + '\n'
        print txt(1, "requirements:", 5, True, '%s'%str(self.metadata.requirements))
        print txt(1, "documentation:", 5, True, '%s'%self.metadata.documentation) + '\n'
        print txt(2, "* parameters with their default values:") + '\n'
        print txt(3, 'id', 2, True, 'name',5,True,'value',5,True,'options')
        print txt(3, '--', 2, True, '----',5,True,'-----',5,True,'-------')
        for i in self.poptions:
            print txt(3,i,2,True,self.parameters[i].name,5,True,str(self.parameters[i].default),
                      5,True, str(self.parameters[i].options))
        print '\n'
        self.custom_function_keys()


    #################################


    def IO(self, ib):
        host = self.blocks[ib]['host']
        function = self.blocks[ib]['function']
        if host == 'sklearn':
            metadata = getattr(sklearn_db, function)()
        elif host == 'cheml':
            metadata = getattr(cheml_db, function)()
        self.inputs = {i:copy.deepcopy(vars(metadata.Inputs)[i]) for i in vars(metadata.Inputs).keys() if
                       i not in ('__module__','__doc__')}
        self.outputs = {i:copy.deepcopy(vars(metadata.Outputs)[i]) for i in vars(metadata.Outputs).keys() if
                        i not in ('__module__', '__doc__')}


    def graph_help(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print '\n'
        print bcolors.BOLD + txt(1, "general hints") + bcolors.ENDC
        print txt(2, "The list of nodes might have same functions (with different/same parameter values) for different parts of the workflow.")
        print txt(2, "The id of nodes are unique.")
        print txt(2, "After pipelining the nodes, you can see the tree design of the workflow.")
        print '\n'
        print bcolors.BOLD + txt(1, "pipelines page keys") + bcolors.ENDC
        print txt(2, bcolors.BOLD + 'd' + bcolors.ENDC, 1, True, "    delete a node. Next enter a node id with empty space in between (e.g. d 7)")
        print txt(2, bcolors.BOLD + 'id' + bcolors.ENDC, 1, True, "   initiate a pipeline. This is the sender node id. Next enter a receiver node id (e.g. 3 7)")
        print txt(2, '', 1, True, " Next we show you all the sender's output tokens and the receiver's input tokens. enter one token of each respectively.")
        print txt(2, bcolors.BOLD + 't' + bcolors.ENDC, 1, True, "    go to the task page.")
        print txt(2, bcolors.BOLD + 's' + bcolors.ENDC, 1, True, "    save the current workflow on the disk.")
        print txt(2, bcolors.BOLD + 'q' + bcolors.ENDC, 1, True, "    quit the ChemML wrapper's interface without saving.")
        print txt(2, bcolors.BOLD + 'h' + bcolors.ENDC, 1, True, "    help page, this page!")
        print '\n'

        key = False
        while not key:
            k = raw_input(bcolors.BOLD + txt(1, "enter q to quit page: ") + bcolors.ENDC)
            k = k.strip()
            if k in ['q']:
                self.graph_page()


    def No_short_loops(self):
        return True


    def graph_page_pipeline_keys(self,ks):
        print '\n'
        ibs = int(ks[0])
        ibr = int(ks[1])
        sender_function = "%s outputs" % self.blocks[ibs]['function']
        receiver_function = "%s inputs" % self.blocks[ibr]['function']
        print txt(2, sender_function + " " * 8 + ">>>>" + " " * 8 + receiver_function)
        print txt(2, "-" * len(sender_function) + " " * 20 + "-" * len(receiver_function))

        self.IO(ibs)
        sender_outputs = [self.outputs[i].name for i in self.outputs]
        self.IO(ibr)
        receiver_inputs = [self.inputs[i].name for i in self.inputs]
        outputs = ", ".join(sender_outputs)
        inputs = ", ".join(receiver_inputs)
        print txt(2, outputs + " " * (20 + len(sender_function) - len(outputs)) + inputs) + '\n'
        key = False
        while not key:
            k = raw_input(bcolors.BOLD + txt(2,"enter sender output and receiver input ('token token' or enter c to cancel): ") + bcolors.ENDC)
            k = k.strip()
            if k == 'c':
                self.graph_page()
            elif ' ' in k:
                tokens = k.split(' ')
                while '' in ks:
                    tokens.remove('')
                if tokens[0] in sender_outputs and tokens[1] in receiver_inputs:
                    new_edge = [ibs, tokens[0], ibr, tokens[1]]
                    if new_edge in self.comp_graph:
                        print bcolors.OKGREEN + txt(2, "this edge is already in the workflow tree") + bcolors.ENDC
                    else:
                        self.comp_graph.append(new_edge)
                        if self.No_short_loops():
                            key = True
                        else:
                            self.comp_graph.remove(new_edge)
                            print bcolors.FAIL + txt(2, "Exception (short loop): You arenot allowed to make interdependent nodes") + bcolors.ENDC

                    print self.comp_graph
        self.graph_page()


    def graph_page_keys(self):
        key = False
        print txt(1, "*** keys: ( id id : pipeline  ,  d id : delete  ,  t : task page  ,  s : save,  h : help  ,  q : quit )")
        while not key:
            k = raw_input(bcolors.BOLD + txt(1, "enter one of the above keys: ") + bcolors.ENDC)
            k = k.strip()
            if k in ['t', 'q', 's', 'h']:
                key = True
            elif ' ' in k:
                ks = k.split(' ')
                ibs = [str(ib) for ib in self.blocks]
                while '' in ks:
                    ks.remove('')
                if ks[0]=='d' and len(ks)==2:
                    if ks[1] in ibs:
                        key = True
                elif len(ks) == 2 and ks[0] in ibs and ks[1] in ibs:
                    if ks[0] == ks[1]:
                        print bcolors.FAIL + txt(1, "Exception (short loop): You arenot allowed to send info from one node to itself") + bcolors.ENDC
                        key = True # remove later
                    else:
                        key = True
        if k == 'q':
            sys.exit()
        elif k == 's':
            sys.exit()
        elif k == 't':
            self.task_page()
        elif k == 'h':
            self.graph_help()
        elif ks[0]=='d':
            del self.blocks[int(ks[1])]
            self.graph_page()
        else:
            self.graph_page_pipeline_keys(ks)


    def graph_page(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print '\n'
        print txt(1, "home           tasks           subtasks           hosts           functions               parameters           pipelines")
        print txt(1, "  :..............:.................:................:.................:........................:...................@")
        print '\n'
        print txt(1, "* list of added nodes:") + '\n'
        print txt(2, 'id', 2, True, 'host', 3, True, 'function', 5, True,'input tokens', 9, True, 'output tokens')
        print txt(2, '--', 2, True, '----', 3, True, '--------', 5, True,'------------', 9, True, '-------------')
        for ib in sorted(self.blocks.iterkeys()):
            self.IO(ib)
            host = self.blocks[ib]['host']
            function = self.blocks[ib]['function']
            inputs = [self.inputs[i].name for i in self.inputs]
            outputs = [self.outputs[i].name for i in self.outputs]
            print txt(2, str(ib), 2, True, host, 3, True, function, 5, True, ", ".join(inputs), 9, True, ", ".join(outputs))
        print '\n'
        print txt(1, "** workflow tree:") + '\n'

        self.graph_page_keys()


