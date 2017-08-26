def BANK():
    tasks = ['Enter','Prepare','Model','Search','Mix','Visualize','Store']
    info = {
            'Enter':{
                        'input_data':{
                                    'pandas':['read_excel', 'read_csv'],
                                     }
                    },
            'Prepare':{
                        'descriptor': {'cheml': ['RDKitFingerprint', 'Dragon', 'CoulombMatrix'],
                                       'sklearn': ['PolynomialFeatures', 'Binarizer','OneHotEncoder']
                                       },
                        'scaler': {
                                    'sklearn': ['StandardScaler','MinMaxScaler','MaxAbsScaler','RobustScaler','Normalizer']
                                  },
                        'feature selector': {
                                                'sklearn': ['PCA','KernelPCA']
                                            },
                        'feature transformer': {
                                                'cheml': ['TBFS']
                                                },
                        'basic operator': {
                                        'cheml':['PyScript','Merge','Split', 'Constant','MissingValues','Trimmer','Uniformer'],
                                        'sklearn': ['Imputer']
                                          },
                        'splitter': {
                                        'sklearn': ['Train_Test_Split','KFold']
                                    },
                      },
            'Model':{
                        'regression':{
                                        'cheml':['NN_PSGD','nn_dsgd'],
                                        'sklearn':[
                                                'OLS','Ridge','KernelRidge','Lasso','MultiTaskLasso','',
                                                'ElasticNet','MultiTaskElasticNet','Lars','LassoLars',
                                                'BayesianRidge', 'ARDRegression', 'LogisticRegression',
                                                'SGDRegressor','SVR','NuSVR','LinearSVR','MLPRegressor',
                                                ]
                                        },
                        'classification': {},
                        'clustering': {},
                    },
            'Search':{
                        'evolutionary': {
                                        'cheml': ['GeneticAlgorithm_binary'],
                                        'deep': []
                                        },
                        'swarm': {
                                    'pyswarm': ['pso']
                                 },
                        'grid':{
                                    'sklearn': ['GridSearchCV',]
                                },
                        'metrics':{
                                        'sklearn':['Evaluate_Regression']
                                   },
                     },
            'Mix':{
                    'A': {
                            'sklearn': ['cross_val_score',]
                          },
                    'B': {}
                  },
            'Visualize':{
                            'matplotlib': [],
                            'seaborn': []
                        },
            'Store':{
                        'output_data':{
                                        'cheml': ['SaveFile'],
                                      }
                    }
            }
    return info, tasks

import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output

global W
W = ['main_accordion_box']

class wrapperGUI(object):
    def __init__(self, script='new'):
        self.script = script
        self.graph = {}
        self.accordion_children = {}
        self.prev_accordion = None
        self.function_counter = 0
        self.token_counter = 0
        self.childeren = [widgets.Label(value='choose a method:')]
        self.add = add_box()
        self.add.widgets()
        self.accordion_children[0] = self.add

    def run(self):
        if self.script == 'new':
            self.display_accordion()

    def run_script(self,path):
        pass

    def display_accordion(self):
        children = [self.accordion_children[i].VBox for i in self.accordion_children]
        self.accordion = widgets.Accordion(children=children, \
                                           selected_index=0, layout=widgets.Layout(border='solid lightblue 2px'))
        self.accordion.set_title(0, 'Add a block ...')
        for i in range(1, len(self.accordion_children)):
            acc_i = self.accordion_children[i]
            self.accordion.set_title(i, '%s' % acc_i.name)
        display(self.accordion)
        W[0] = self.accordion

    def close_accordion(self):
        W[0].close()



class cheml_RDKitFingerprint(wrapperGUI):
    def display(self,graph_temp):
        self.graph_temp = graph_temp
        self._input_vbox()
        self._param_vbox()
        self._output_vbox()
        self._buttons()

        caption = widgets.Label(value='${RDKitFingerprint}$', \
                                layout=widgets.Layout(width='50%', margin='10px 0px 0px 440px'))
        accordion = widgets.Accordion(children=[self.input_box, self.param_box, self.output_box], \
                                      layout=widgets.Layout(width='80%', margin='10px 0px 10px 100px'))
        accordion.set_title(0, 'input/receivers')
        accordion.set_title(1, 'parameters')
        accordion.set_title(2, 'output/senders')

        layout = widgets.Layout(border='solid')
        self.func_block = widgets.VBox([caption, accordion, self.buttons_box], layout=layout)
        display(self.func_block)

    def close_block(self):
        self.func_block.close()

    def _param_vbox(self):
        self.parameters = {'removeHs': True, 'FPtype': 'Morgan', 'vector': 'bit', 'nBits': 1024, 'radius ': 2}
        # caption = widgets.Label(value='Parameters:',layout=widgets.Layout(width='50%'))
        self.removeHs = widgets.Checkbox(
            value=False,
            description='removeHs:',
            disabled=False)
        if self.removeHs.value:
            print "Hallloooo"
        self.FPtype = widgets.Dropdown(
            options=['HAP', 'AP', 'MACCS', 'Morgan', 'HTT', 'TT'],
            value='HAP',
            description='FPtype:',
            disabled=False,
            button_style=''  # 'success', 'info', 'warning', 'danger' or ''
        )
        self.param_box = widgets.VBox([self.removeHs, self.FPtype])

    def _input_vbox(self):
        self.in_1 = widgets.Checkbox(
            value=False,
            description='molfile',
            disabled=False)
        self.in_1_senders = widgets.Dropdown(
            options=['HAP', 'AP', 'MACCS', 'Morgan', 'HTT', 'TT'],
            value='HAP',
            description='senders:',
            button_style=''  # 'success', 'info', 'warning', 'danger' or ''
        )
        self.input_box = widgets.HBox([self.in_1, self.in_1_senders])

    def _output_vbox(self):
        self.out_df = widgets.Checkbox(
            value=False,
            description='df',
            disabled=False)
        self.out_rows = widgets.Checkbox(
            value=False,
            description='removed_rows',
            disabled=False)
        self.output_box = widgets.VBox([self.out_df, self.out_rows])

    def _buttons(self):
        self.add = widgets.Button(description="Add")
        self.add.on_click(self.on_add_clicked)

        self.cancel = widgets.Button(description="Cancel")
        # cancel.on_click(self.on_cancel_clicked)

        self.buttons_box = widgets.HBox([self.add, self.cancel], layout=widgets.Layout(margin='10px 0px 10px 350px'))

    def on_add_clicked(self,b):
        self.function_counter+=1
        self.graph[self.function_counter] = self.graph_temp
        self.graph[self.function_counter]['parameters'] = {'removeHs':self.removeHs.value,'FPtype':self.FPtype.value}
        self.graph[self.function_counter]['send'] = {'df':self.out_df.value,'removed_rows':self.out_rows.value}
        self.graph[self.function_counter]['recv'] = {'molfile': self.in_1.value}
        print self.graph
        item = add_accordion(f='RDKitFingerprint',h='cheml',n=self.function_counter)
        item.widgets()
        self.accordion_children[len(self.accordion_children)]=item
        self.close_block()
        self.close_accordion()
        self.display_accordion()


class add_accordion(object):
    def __init__(self,f,h,n):
        self.f = f
        self.h = h
        self.n = n
        self.name = '%i %s'%(self.n,self.f)
    def widgets(self):
        f = widgets.Text(
                        value=self.f,
                        # placeholder='Type something',
                        description='function:',
                        disabled=True
                    )

        h = widgets.Text(
            value=self.h,
            # placeholder='Type something',
            description='host:',
            disabled=True
        )


        self.edit = widgets.Button(description="Edit")
        # self.edit.on_click(self.on_edit_clicked)

        self.remove = widgets.Button(description="Remove")
        # cancel.on_click(self.on_cancel_clicked)

        buttons_box = widgets.HBox([self.edit, self.remove], layout=widgets.Layout(margin='10px 0px 10px 350px'))
        self.VBox = widgets.VBox([f,h,buttons_box])

class add_box(object):
    """
    attributes:
        widgets: the main function to make a VBox of all 5 elements
        task: the task widget
        subtask: the subtask widget
        host: the host widget
        func: the function widget
    """
    def widgets(self):
        # caption = widgets.Label(value='choose a method:',layout=widgets.Layout(width='50%'))
        self.bank, self.task_options = BANK()
        self._task()
        self._subtask()
        self._host()
        self._func()

        self.select = widgets.Button(description="Select", layout=widgets.Layout(margin='20px 0px 10px 115px'))
        self.select.on_click(self.on_select_clicked)

        self.VBox = widgets.VBox([self.task, self.subtask, self.host, self.func, self.select])

    def _task(self):
        self.task = widgets.Dropdown(
            options=self.task_options,
            value='Enter',
            description='Task:',
            disabled=False,
            button_style=''  # 'success', 'info', 'warning', 'danger' or ''
        )
        self.task.observe(self.handle_task_change, names='value')

    def _subtask(self):
        subtask_options = [i for i in self.bank[self.task.value]]
        self.subtask = widgets.Dropdown(
            options=subtask_options,
            value=subtask_options[0],
            description='Subtask:',
            disabled=False,
            button_style=''  # 'success', 'info', 'warning', 'danger' or ''
        )
        self.subtask.observe(self.handle_subtask_change, names='value')

    def _host(self):
        host_options = [i for i in self.bank[self.task.value][self.subtask.value]]
        self.host = widgets.Dropdown(
            options=host_options,
            value=host_options[0],
            description='Host:',
            disabled=False,
            button_style=''  # 'success', 'info', 'warning', 'danger' or ''
        )
        self.host.observe(self.handle_host_change, names='value')

    def _func(self):
        func_options = [i for i in self.bank[self.task.value][self.subtask.value][self.host.value]]
        self.func = widgets.Dropdown(
            options=func_options,
            value=func_options[0],
            description='Function:',
            disabled=False,
            button_style=''  # 'success', 'info', 'warning', 'danger' or ''
        )

    def _subtask_update(self):
        subtask_opts = [i for i in self.bank[self.task.value]]
        self.subtask.options = subtask_opts
        self.subtask.value = subtask_opts[0]

    def _host_update(self):
        host_opts = [i for i in self.bank[self.task.value][self.subtask.value]]
        self.host.options = host_opts
        self.host.value = host_opts[0]

    def _func_update(self):
        func_opts = [i for i in self.bank[self.task.value][self.subtask.value][self.host.value]]
        self.func.options = func_opts
        self.func.value = func_opts[0]

    def handle_task_change(self, change):
        self._subtask_update()
        self._host_update()
        self._func_update()

    def handle_subtask_change(self, change):
        self._host_update()
        self._func_update()

    def handle_host_change(self, change):
        self._func_update()

    def on_select_clicked(self, b):
        # display function
        # if function added ==> close all other displays and display a new accardion with new function
        # if function added ==> function_counter += 1
        # if function canceled ==> close function displays
        self.graph_temp = {'task': self.task.value, 'subtask': self.subtask.value, \
                           'host': self.host.value, 'function': self.func.value}
        print self.graph_temp
        # self.prev_accordion.close()
        self.select.disabled = True
        self.select.button_style = 'danger'
        block_done = True
        if self.graph_temp['function'] == 'RDKitFingerprint':
            block = cheml_RDKitFingerprint()
            block.display(self.graph_temp)


gui = wrapperGUI(script='new')
gui.run()