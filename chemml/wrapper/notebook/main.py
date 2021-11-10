from __future__ import print_function

import ipywidgets as widgets
from IPython.display import display
# from IPython.display import clear_output
import sys

import copy
import os
import pandas as pd     # newly added
import sklearn          # newly added


from chemml.wrapper.database import sklearn_db, chemml_db, pandas_db
from chemml.wrapper.notebook.TSHF import tshf
from chemml.utils.validation import isint

try:
    from graphviz import Digraph
except Exception as e:
    print("Graphviz not installed. Please install graphviz to view workflow. Use the following commands in the ChemmL environment: ")
    print("conda install -c anaconda python-graphviz")
    print("conda install -c anaconda pydot")
    sys.exit(e)


##########################################################
# Todo: bring back receivers - no need another recursive function for bidR; (currentbids - bidS = bidR)
# Todo: profile the cpu/clock time

class container_page(object):
    def __init__(self, title, index, widget, block_params={}):
        self.title = title
        self.index = index
        self.widget = widget
        self.block_params = block_params
        # IO_refresher will be added later for each function block

class ChemMLNotebook(object):
    def __init__(self):
        self.tasks, self.combinations = tshf()  # task, subtask, host, function connections; keep the tasks for the order
        self.blocks = {}  # similar to cmls in the parser, {'task':{}, 'subtask':{}, 'host':{}, 'function':{}, 'parameters':{}, 'send':{'NA':'here'}, 'recv':{'NA':'here'} }
        self.block_id = 1
        self.comp_graph = []    # list of (iblock_send,token,iblock_recv,token)
        self.out_dir = "CMLWrapper_out"
        self.pages = {}
        self.graph = widgets.Image()
        self.accordion = widgets.Accordion()
        self.home_page()

    ######################################################

    def display_accordion(self, active_id=0, layout=widgets.Layout(border='solid gray 2px')):
        def index_change(change):
            ind = self.accordion.selected_index
            if ind not in [0,1,None]:
                ib = [i for i in sorted(self.pages)][ind]
                self.pages[ib].IO_refresher()

        children = [self.pages[i].widget for i in sorted(self.pages)]
        self.accordion.children = children
        self.accordion.selected_index = active_id
        self.accordion.layout = layout
        new_titles = {}
        for i,ib in enumerate(sorted(self.pages)):
            if i in [0, 1]:
                self.accordion.set_title(i, '%s' % self.pages[i].title)
                # new_titles[str(i)] = '%s' % self.pages[i].title
            else:
                self.accordion.set_title(i, 'block# %i: %s' % (ib, self.pages[ib].title))
                # new_titles[str(i)] = 'block# %i: %s' % (ib, self.pages[ib].title)
        # self.accordion._titles = new_titles
        self.accordion.observe(index_change, names='selected_index')

    def close_accordion(self):
        pass
        # self.accordion.close()

    ##########################################################

    def check_filename(self,filename):
        if filename is None:
            filename = 'chemml_config.txt'
        initial_file_name = filename
        i = 0
        while os.path.exists(filename):
            i+=1
            base = os.path.splitext(initial_file_name)[0]
            format = os.path.splitext(initial_file_name)[1]
            if format == '':
                format = 'txt'
            filename = base + '%i'%i + format
        return filename

    def configurer(self):
        def value(string):
            try:
                val =  eval(string)
                if isinstance(val,str):
                    val = '"' + val + '"'
                return val
            except NameError:
                return '"' + string + '"'

        tab = '    '
        lines = []
        uids = {}   #comp_graph : unique ids
        for i in range(len(self.comp_graph)):
            uids[self.comp_graph[i]] = i
        for ib in self.pages:
            if ib not in [0,1]:
                edges = [e for e in self.comp_graph if ib in e]
                task = self.pages[ib].block_params['task']
                subtask = self.pages[ib].block_params['subtask']
                if len(edges) == 0:
                    lines.append('# (%s,%s)\n'%(task,subtask))
                else:
                    lines.append('## (%s,%s)\n'%(task,subtask))
                host = self.pages[ib].block_params['host']
                lines.append(tab+"<< host = %s\n"%host)
                function = self.pages[ib].block_params['function']
                lines.append(tab+"<< function = %s\n"%function)
                for param in self.pages[ib].block_params['wparams']:
                    container = self.pages[ib].block_params['wparams'][param]
                    if container.checker.value:
                        lines.append(tab+"<< %s = %s\n"%(container.name, str(container.widget.value)))
                for param in self.pages[ib].block_params['fparams']:
                    container = self.pages[ib].block_params['fparams'][param]
                    if container.checker.value:
                        lines.append(tab+"<< %s = %s\n"%(container.name, str(container.widget.value)))
                for e in edges:
                    if e[0] == ib:
                        lines.append(tab+">> %s %i\n"%(e[1],uids[e]))
                    if e[2] == ib:
                        lines.append(tab + ">> %i %s\n" % (uids[e], e[3]))
                lines.append('\n')
        return lines

    def templates_widgets(self):
        headerT = widgets.Label(value='Start with a template workflow', layout=widgets.Layout(width='50%'))
        # style = {'description_width': 'initial'}
        # outdir = widgets.Text(
        #     value='CMLWrapper_out',
        #     placeholder='Type something',
        #     description='Output directory:',
        #     disabled=False,
        #     style = style,
        #     layout = widgets.Layout(margin='30px 0px 30px 0px'))21


        TEMPLATES = []

        #######################***********************#######################***********************
        #######################:::::::::::::::::::::::#######################
        ### Datasets Overview  (DATAOVER)
        headerDATAOVER = widgets.HTML(value='<b> Datasets Overview: </b>', layout=widgets.Layout(width='50%',margin='10px 0px 10px 0px'))
        TEMPLATES.append(headerDATAOVER)
        
        ######################################################
        ## Template1
        def on_selectTe1_clicked(b):
            # template1.txt is a chemml wrapper config file
            from .templates import template1
            script = template1()
            old = [i for i in self.pages]

            try:
                self.parser(script)
                # update the current_bid
                self.block_id = max(self.pages)
                selectTe1.icon = 'check'
            except Exception as err:
                print( "Invalid configuration file ...")
                print( "    IOError: %s"%err.message)
                print( "... Not loaded!")
                selectTe1.icon = 'remove'
                rm = [i for i in self.pages if i not in old]
                for ib in rm:
                    if ib in self.pages:
                        del self.pages[ib]

            self.debut = False
            self.add_page()

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(), format='png')
            display(self.graph)

        te1 = widgets.Label(value="Template 1: load_cep_homo --> plot histogram of HOMO energies --> print 5 SMILES")#, layout=widgets.Layout(width='70%'))
        selectTe1 = widgets.Button(description="Select")
        selectTe1.style.button_color = 'lightblue'
        selectTe1.on_click(on_selectTe1_clicked)
        # viewT1 = widgets.Button(description="Overview")
        # viewT1.style.button_color = 'lightblue'
        # viewT1.(on_viewT1_clicked)
        hboxTe1 = widgets.HBox([te1, selectTe1],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
                                                               # height='40px', align_items='center',   justify_content = 'space-between',
                                                               # margin='0px 0px 0px 10px'))
        TEMPLATES.append(hboxTe1)

        ######################################################
        ## Template2
        def on_selectTe2_clicked(b):
            # template8.txt is a chemml wrapper config file
            from .templates import template2
            script = template2()
            old = [i for i in self.pages]

            try:
                self.parser(script)
                # update the current_bid
                self.block_id = max(self.pages)
                selectTe2.icon = 'check'
            except Exception as err:
                print( "Invalid configuration file ...")
                print( "    IOError: %s"%err.message)
                print( "... Not loaded!")
                selectTe2.icon = 'remove'
                rm = [i for i in self.pages if i not in old]
                for ib in rm:
                    if ib in self.pages:
                        del self.pages[ib]

            self.debut = False
            self.add_page()

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(), format='png')
            display(self.graph)

        te2 = widgets.Label(value="Template 2: load_organic_density --> plot feature AMW vs. density --> print 5 SMILES")#, layout=widgets.Layout(width='70%'))
        selectTe2 = widgets.Button(description="Select")
        selectTe2.style.button_color = 'lightblue'
        selectTe2.on_click(on_selectTe2_clicked)
        # viewT1 = widgets.Button(description="Overview")
        # viewT1.style.button_color = 'lightblue'
        # viewT1.on_click(on_viewT1_clicked)
        hboxTe2 = widgets.HBox([te2, selectTe2],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
                                                               # height='40px', align_items='center',   justify_content = 'space-between',
                                                               # margin='0px 0px 0px 10px'))
        TEMPLATES.append(hboxTe2)

        ######################################################
        ## Template3
        def on_selectTe3_clicked(b):
            # template9.txt is a chemml wrapper config file
            from .templates import template3
            script = template3()
            old = [i for i in self.pages]

            try:
                self.parser(script)
                # update the current_bid
                self.block_id = max(self.pages)
                selectTe3.icon = 'check'
            except Exception as err:
                print( "Invalid configuration file ...")
                print( "    IOError: %s"%err.message)
                print( "... Not loaded!")
                selectTe3.icon = 'remove'
                rm = [i for i in self.pages if i not in old]
                for ib in rm:
                    if ib in self.pages:
                        del self.pages[ib]

            self.debut = False
            self.add_page()

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(), format='png')
            display(self.graph)

        te3 = widgets.Label(value="Template 3: load_xyz_polarizability --> plot all 50 polarizabilities --> print first item of coordinates output")#, layout=widgets.Layout(width='70%'))
        selectTe3 = widgets.Button(description="Select")
        selectTe3.style.button_color = 'lightblue'
        selectTe3.on_click(on_selectTe3_clicked)
        # viewT1 = widgets.Button(description="Overview")
        # viewT1.style.button_color = 'lightblue'
        # viewT1.on_click(on_viewT1_clicked)
        hboxTe3 = widgets.HBox([te3, selectTe3],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
                                                               # height='40px', align_items='center',   justify_content = 'space-between',
                                                               # margin='0px 0px 0px 10px'))
        TEMPLATES.append(hboxTe3)



        #######################***********************#######################***********************
        #######################:::::::::::::::::::::::#######################:::::::::::::::::::::::
        ### Molecular Descriptors (MOLDES)
        headerMOLDES = widgets.HTML(value='<b> Molecular Descriptors: </b>', layout=widgets.Layout(width='50%',margin='10px 0px 10px 0px'))
        TEMPLATES.append(headerMOLDES)
        
        ######################################################
        ## Template4
        def on_selectTe4_clicked(b):
            # template4.txt is a chemml wrapper config file
            from .templates import template4
            script = template4()
            old = [i for i in self.pages]

            try:
                self.parser(script)
                # update the current_bid
                self.block_id = max(self.pages)
                selectTe7.icon = 'check'
            except Exception as err:
                print( "Invalid configuration file ...")
                print( "    IOError: %s"%err.message)
                print( "... Not loaded!")
                selectTe4.icon = 'remove'
                rm = [i for i in self.pages if i not in old]
                for ib in rm:
                    if ib in self.pages:
                        del self.pages[ib]

            self.debut = False
            self.add_page()

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(), format='png')
            display(self.graph)

        te4 = widgets.Label(value="Template 1: load_xyz_polarizability --> generate CoulombMatrix features --> save features")#, layout=widgets.Layout(width='70%'))
        selectTe4 = widgets.Button(description="Select")
        selectTe4.style.button_color = 'lightblue'
        selectTe4.on_click(on_selectTe4_clicked)
        # viewT1 = widgets.Button(description="Overview")
        # viewT1.style.button_color = 'lightblue'
        # viewT1.on_click(on_viewT1_clicked)
        hboxTe4 = widgets.HBox([te4, selectTe4],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
                                                               # height='40px', align_items='center',   justify_content = 'space-between',
                                                               # margin='0px 0px 0px 10px'))
        TEMPLATES.append(hboxTe4)

        ######################################################
        ## Template5
        def on_selectTe5_clicked(b):
            # template5.txt is a chemml wrapper config file
            from .templates import template5
            script = template5()
            old = [i for i in self.pages]

            try:
                self.parser(script)
                # update the current_bid
                self.block_id = max(self.pages)
                selectTe5.icon = 'check'
            except Exception as err:
                print( "Invalid configuration file ...")
                print( "    IOError: %s"%err.message)
                print( "... Not loaded!")
                selectTe5.icon = 'remove'
                rm = [i for i in self.pages if i not in old]
                for ib in rm:
                    if ib in self.pages:
                        del self.pages[ib]

            self.debut = False
            self.add_page()

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(), format='png')
            display(self.graph)

        te5 = widgets.Label(value="Template 2: load_xyz_polarizability --> generate BagofBonds features --> save features")#, layout=widgets.Layout(width='70%'))
        selectTe5 = widgets.Button(description="Select")
        selectTe5.style.button_color = 'lightblue'
        selectTe5.on_click(on_selectTe5_clicked)
        # viewT1 = widgets.Button(description="Overview")
        # viewT1.style.button_color = 'lightblue'
        # viewT1.on_click(on_viewT1_clicked)
        hboxTe5 = widgets.HBox([te5, selectTe5],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
                                                               # height='40px', align_items='center',   justify_content = 'space-between',
                                                               # margin='0px 0px 0px 10px'))
        TEMPLATES.append(hboxTe5)

        ######################################################
        ## Template6
        def on_selectTe6_clicked(b):
            # template3.txt is a chemml wrapper config file
            from .templates import template6
            script = template6()
            old = [i for i in self.pages]

            try:
                self.parser(script)
                # update the current_bid
                self.block_id = max(self.pages)
                selectTe6.icon = 'check'
            except Exception as err:
                print( "Invalid configuration file ...")
                print( "    IOError: %s"%err.message)
                print( "... Not loaded!")
                selectTe6.icon = 'remove'
                rm = [i for i in self.pages if i not in old]
                for ib in rm:
                    if ib in self.pages:
                        del self.pages[ib]

            self.debut = False
            self.add_page()

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(), format='png')
            display(self.graph)

        te6 = widgets.Label(value="Template 3: get SMILES reperesentaion of molecules --> save them --> generate Morgan Fingerprints  --> save features")#, layout=widgets.Layout(width='70%'))
        selectTe6 = widgets.Button(description="Select")
        selectTe6.style.button_color = 'lightblue'
        selectTe6.on_click(on_selectTe6_clicked)
        # viewT1 = widgets.Button(description="Overview")
        # viewT1.style.button_color = 'lightblue'
        # viewT1.on_click(on_viewT1_clicked)
        hboxTe6 = widgets.HBox([te6, selectTe6],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
                                                               # height='40px', align_items='center',   justify_content = 'space-between',
                                                               # margin='0px 0px 0px 10px'))
        TEMPLATES.append(hboxTe6)

        ######################################################
        ## Template7
        def on_selectTe7_clicked(b):
            # template7.txt is a chemml wrapper config file
            from .templates import template7
            script = template7()
            old = [i for i in self.pages]

            try:
                self.parser(script)
                # update the current_bid
                self.block_id = max(self.pages)
                selectTe7.icon = 'check'
            except Exception as err:
                print( "Invalid configuration file ...")
                print( "    IOError: %s"%err.message)
                print( "... Not loaded!")
                selectTe7.icon = 'remove'
                rm = [i for i in self.pages if i not in old]
                for ib in rm:
                    if ib in self.pages:
                        del self.pages[ib]

            self.debut = False
            self.add_page()

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(), format='png')
            display(self.graph)

        te7 = widgets.Label(value="Template 4: get SMILES reperesentaion of molecules --> save them --> generate Dragon descriptors  --> save features")#, layout=widgets.Layout(width='70%'))
        selectTe7 = widgets.Button(description="Select")
        selectTe7.style.button_color = 'lightblue'
        selectTe7.on_click(on_selectTe7_clicked)
        # viewT1 = widgets.Button(description="Overview")
        # viewT1.style.button_color = 'lightblue'
        # viewT1.on_click(on_viewT1_clicked)
        hboxTe7 = widgets.HBox([te7, selectTe7],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
                                                               # height='40px', align_items='center',   justify_content = 'space-between',
                                                               # margin='0px 0px 0px 10px'))
        TEMPLATES.append(hboxTe7)


        #######################***********************#######################***********************
        #######################:::::::::::::::::::::::#######################:::::::::::::::::::::::
        ### Inorganic Descriptors (InorgDes)
        headerInorgDes = widgets.HTML(value='<b> Inorganic Descriptors: </b>', layout=widgets.Layout(width='50%',margin='10px 0px 10px 0px'))
        TEMPLATES.append(headerInorgDes)

        ######################################################
        ## Template8
        def on_selectTe8_clicked(b):
            # template8.txt is a chemml wrapper config file
            from .templates import template8
            script = template8()
            old = [i for i in self.pages]

            try:
                self.parser(script)
                # update the current_bid
                self.block_id = max(self.pages)
                selectTe8.icon = 'check'
            except Exception as err:
                print( "Invalid configuration file ...")
                print( "    IOError: %s"%err.message)
                print( "... Not loaded!")
                selectTe8.icon = 'remove'
                rm = [i for i in self.pages if i not in old]
                for ib in rm:
                    if ib in self.pages:
                        del self.pages[ib]

            self.debut = False
            self.add_page()

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(), format='png')
            display(self.graph)

        te8 = widgets.Label(value="Template 1: load_comp_energy --> inorganic descriptors for composition entries --> concatenate and print shape")#, layout=widgets.Layout(width='70%'))
        selectTe8 = widgets.Button(description="Select")
        selectTe8.style.button_color = 'lightblue'
        selectTe8.on_click(on_selectTe8_clicked)
        # viewT1 = widgets.Button(description="Overview")
        # viewT1.style.button_color = 'lightblue'
        # viewT1.on_click(on_viewT1_clicked)
        hboxTe8 = widgets.HBox([te8, selectTe8],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
                                                               # height='40px', align_items='center',   justify_content = 'space-between',
                                                               # margin='0px 0px 0px 10px'))
        TEMPLATES.append(hboxTe8)

        ######################################################
        ## Template9
        def on_selectTe9_clicked(b):
            # template9.txt is a chemml wrapper config file
            from .templates import template9
            script = template9()
            old = [i for i in self.pages]

            try:
                self.parser(script)
                # update the current_bid
                self.block_id = max(self.pages)
                selectTe9.icon = 'check'
            except Exception as err:
                print( "Invalid configuration file ...")
                print( "    IOError: %s"%err.message)
                print( "... Not loaded!")
                selectTe9.icon = 'remove'
                rm = [i for i in self.pages if i not in old]
                for ib in rm:
                    if ib in self.pages:
                        del self.pages[ib]

            self.debut = False
            self.add_page()

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(), format='png')
            display(self.graph)

        te9 = widgets.Label(value="Template 2: load_crystal_structures --> inorganic descriptors for crystal structures --> concatenate and print shape")#, layout=widgets.Layout(width='70%'))
        selectTe9 = widgets.Button(description="Select")
        selectTe9.style.button_color = 'lightblue'
        selectTe9.on_click(on_selectTe9_clicked)
        # viewT1 = widgets.Button(description="Overview")
        # viewT1.style.button_color = 'lightblue'
        # viewT1.on_click(on_viewT1_clicked)
        hboxTe9 = widgets.HBox([te9, selectTe9],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
                                                               # height='40px', align_items='center',   justify_content = 'space-between',
                                                               # margin='0px 0px 0px 10px'))
        TEMPLATES.append(hboxTe9)


        #######################***********************#######################***********************
        #######################:::::::::::::::::::::::#######################:::::::::::::::::::::::
        ### Custom Datasets
        headerMOLDES = widgets.HTML(value='<b> Generate Morgan fingerprints from SMILES codes: </b>', layout=widgets.Layout(width='50%',margin='10px 0px 10px 0px'))
        TEMPLATES.append(headerMOLDES)
        
        ######################################################
        #Template13
        def on_selectTe13_clicked(b):
            # template13.txt is a chemml wrapper config file
            print("""
                    Please ensure that you are supplying an excel file from your PC.\n 
                    Template includes a random file which is not a part of the ChemML library.\n
                    This template will not work if a custom file is not supplied.
                    """)
            from .templates import template13
            script = template13()
            old = [i for i in self.pages]

            try:
                self.parser(script)
                # update the current_bid
                self.block_id = max(self.pages)
                selectTe13.icon = 'check'
            except Exception as err:
                print( "Invalid configuration file ...")
                print( "    IOError: %s"%err.message)
                print( "... Not loaded!")
                selectTe13.icon = 'remove'
                rm = [i for i in self.pages if i not in old]
                for ib in rm:
                    if ib in self.pages:
                        del self.pages[ib]

            self.debut = False
            self.add_page()

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(), format='png')
            display(self.graph)

        te13 = widgets.Label(value="Template 1: load excel sheet with SMILES codes in first column --> save them --> generate Morgan Fingerprints \n --> save features")#, layout=widgets.Layout(width='70%'))
        selectTe13 = widgets.Button(description="Select")
        selectTe13.style.button_color = 'lightblue'
        selectTe13.on_click(on_selectTe13_clicked)
        # viewT1 = widgets.Button(description="Overview")
        # viewT1.style.button_color = 'lightblue'
        # viewT1.on_click(on_viewT1_clicked)
        hboxTe13 = widgets.HBox([te13, selectTe13],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
                                                               # height='40px', align_items='center',   justify_content = 'space-between',
                                                               # margin='0px 0px 0px 10px'))
        TEMPLATES.append(hboxTe13)

        #######################***********************#######################***********************
        #######################:::::::::::::::::::::::#######################:::::::::::::::::::::::
        ### Data Mining (DMine)
        headerDMine = widgets.HTML(value='<b> Data Mining: </b>', layout=widgets.Layout(width='50%',margin='10px 0px 10px 0px'))
        TEMPLATES.append(headerDMine)

        ######################################################
        #Template14
        def on_selectTe14_clicked(b):
            # template14.txt is a chemml wrapper config file
            from .templates import template14
            script = template14()
            old = [i for i in self.pages]

            try:
                self.parser(script)
                # update the current_bid
                self.block_id = max(self.pages)
                selectTe14.icon = 'check'
            except Exception as err:
                print( "Invalid configuration file ...")
                print( "    IOError: %s"%err.message)
                print( "... Not loaded!")
                selectTe14.icon = 'remove'
                rm = [i for i in self.pages if i not in old]
                for ib in rm:
                    if ib in self.pages:
                        del self.pages[ib]

            self.debut = False
            self.add_page()

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(), format='png')
            display(self.graph)

        te14 = widgets.Label(value="Template 1: A simple machine learning model workflow")#, layout=widgets.Layout(width='70%'))
        selectTe14 = widgets.Button(description="Select")
        selectTe14.style.button_color = 'lightblue'
        selectTe14.on_click(on_selectTe14_clicked)
        # viewT1 = widgets.Button(description="Overview")
        # viewT1.style.button_color = 'lightblue'
        # viewT1.on_click(on_viewT1_clicked)
        hboxTe14 = widgets.HBox([te14, selectTe14],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
                                                               # height='40px', align_items='center',   justify_content = 'space-between',
                                                               # margin='0px 0px 0px 10px'))
        TEMPLATES.append(hboxTe14)


        # ######################################################
        ## Template11
        def on_selectTe11_clicked(b):
            # template11.txt is a chemml wrapper config file
            from .templates import template11
            script = template11()
            old = [i for i in self.pages]

            try:
                self.parser(script)
                # update the current_bid
                self.block_id = max(self.pages)
                selectTe11.icon = 'check'
            except Exception as err:
                print( "Invalid configuration file ...")
                print( "    IOError: %s"%err.message)
                print( "... Not loaded!")
                selectTe11.icon = 'remove'
                rm = [i for i in self.pages if i not in old]
                for ib in rm:
                    if ib in self.pages:
                        del self.pages[ib]

            self.debut = False
            self.add_page()

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(), format='png')
            display(self.graph)

        te11 = widgets.Label(value="Template 2: model selection with grid search and cross validation")#, layout=widgets.Layout(width='70%'))
        selectTe11 = widgets.Button(description="Select")
        selectTe11.style.button_color = 'lightblue'
        selectTe11.on_click(on_selectTe11_clicked)
        # viewT1 = widgets.Button(description="Overview")
        # viewT1.style.button_color = 'lightblue'
        # viewT1.on_click(on_viewT1_clicked)
        hboxTe11 = widgets.HBox([te11, selectTe11],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
                                                               # height='40px', align_items='center',   justify_content = 'space-between',
                                                               # margin='0px 0px 0px 10px'))
        TEMPLATES.append(hboxTe11)

        ######################################################
        ## Template12
        def on_selectTe12_clicked(b):
            # template12.txt is a chemml wrapper config file
            from .templates import template12
            script = template12()
            old = [i for i in self.pages]

            try:
                self.parser(script)
                # update the current_bid
                self.block_id = max(self.pages)
                selectTe12.icon = 'check'
            except Exception as err:
                print( "Invalid configuration file ...")
                print( "    IOError: %s"%err.message)
                print( "... Not loaded!")
                selectTe12.icon = 'remove'
                rm = [i for i in self.pages if i not in old]
                for ib in rm:
                    if ib in self.pages:
                        del self.pages[ib]

            self.debut = False
            self.add_page()

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(), format='png')
            display(self.graph)

        te12 = widgets.Label(value="Template 3: a complete machine learning workflow")#, layout=widgets.Layout(width='70%'))
        selectTe12 = widgets.Button(description="Select")
        selectTe12.style.button_color = 'lightblue'
        selectTe12.on_click(on_selectTe12_clicked)
        # viewT1 = widgets.Button(description="Overview")
        # viewT1.style.button_color = 'lightblue'
        # viewT1.on_click(on_viewT1_clicked)
        hboxTe12 = widgets.HBox([te12, selectTe12],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
                                                               # height='40px', align_items='center',   justify_content = 'space-between',
                                                               # margin='0px 0px 0px 10px'))
        TEMPLATES.append(hboxTe12)

        ######################################################

        ### Hyperparamter Optimization (GA_optimize)
        headerGA_optimize = widgets.HTML(value='<b> Hyperparamter Optimization: </b>', layout=widgets.Layout(width='50%',margin='10px 0px 10px 0px'))
        TEMPLATES.append(headerGA_optimize)


        #Template15
        def on_selectTe15_clicked(b):
            # template15.txt is a chemml wrapper config file
            from .templates import template15
            script = template15()
            old = [i for i in self.pages]

            try:
                self.parser(script)
                # update the current_bid
                self.block_id = max(self.pages)
                selectTe15.icon = 'check'
            except Exception as err:
                print( "Invalid configuration file ...")
                print( "    IOError: %s"%err.message)
                print( "... Not loaded!")
                selectTe15.icon = 'remove'
                rm = [i for i in self.pages if i not in old]
                for ib in rm:
                    if ib in self.pages:
                        del self.pages[ib]

            self.debut = False
            self.add_page()

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(), format='png')
            display(self.graph)

        te15 = widgets.Label(value="Template 1: Genetic Algorithm for MLPRegressor")#, layout=widgets.Layout(width='70%'))
        selectTe15 = widgets.Button(description="Select")
        selectTe15.style.button_color = 'lightblue'
        selectTe15.on_click(on_selectTe15_clicked)
        # viewT1 = widgets.Button(description="Overview")
        # viewT1.style.button_color = 'lightblue'
        # viewT1.on_click(on_viewT1_clicked)
        hboxTe15 = widgets.HBox([te15, selectTe15],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
                                                               # height='40px', align_items='center',   justify_content = 'space-between',
                                                               # margin='0px 0px 0px 10px'))
        TEMPLATES.append(hboxTe15)

        ######################################################
        # ## TemplateMHL
        # def on_selectTeMHL_clicked(b):
        #     # templateMHL.txt is a chemml wrapper config file
        #     from .templates import templateMHL
        #     script = templateMHL()
        #     old = [i for i in self.pages]
        #
        #     try:
        #         self.parser(script)
        #         # update the current_bid
        #         self.block_id = max(self.pages)
        #         selectTeMHL.icon = 'check'
        #     except Exception as err:
        #         print( "Invalid configuration file ...")
        #         print( "    IOError: %s"%err.message)
        #         print( "... Not loaded!")
        #         selectTeMHL.icon = 'remove'
        #         rm = [i for i in self.pages if i not in old]
        #         for ib in rm:
        #             if ib in self.pages:
        #                 del self.pages[ib]
        #
        #     self.debut = False
        #     self.add_page()
        #
        #     ## clear ouput and update the graph viz
        #     self.graph.close()
        #     dot = Digraph(format='png')
        #     for edge in self.comp_graph:
        #         dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
        #         dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
        #         dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
        #     self.graph = widgets.Image(value=dot.pipe(), format='png')
        #     display(self.graph)
        #
        # teMHL = widgets.Label(value="Template MHL: read XYZ files --> generate BagofBonds features --> save features")#, layout=widgets.Layout(width='70%'))
        # selectTeMHL = widgets.Button(description="Select")
        # selectTeMHL.style.button_color = 'lightblue'
        # selectTeMHL.on_click(on_selectTeMHL_clicked)
        # # viewT1 = widgets.Button(description="Overview")
        # # viewT1.style.button_color = 'lightblue'
        # # viewT1.on_click(on_viewT1_clicked)
        # hboxTeMHL = widgets.HBox([teMHL, selectTeMHL],layout=widgets.Layout( border='dotted black 1px',justify_content = 'space-between'))
        #                                                        # height='40px', align_items='center',   justify_content = 'space-between',
        #                                                        # margin='0px 0px 0px 10px'))
        # TEMPLATES.append(hboxTeMHL)
        #

        vb = widgets.VBox([headerT]+TEMPLATES)#+TUTORIALS)
        return vb

    def home_page_widgets(self):
        def on_selectN_clicked(b):
            self.block_id = 1   #restart the block ids
            self.output_directory = outdir.value
            rm = [i for i in self.pages if i not in [0,1]]
            for j in rm:
                del self.pages[j]
            self.comp_graph = []

            ## clear ouput and update the graph viz
            self.graph.close()
            # dot = Digraph(format='png')
            # for edge in self.comp_graph:
            #     dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
            #     dot.node('%i' % edge[2], label='%i %s' % (edge[2],self.pages[edge[2]].title))
            #     dot.edge('%i' % edge[0], '%i'%edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            # self.graph = widgets.Image(value=dot.pipe(),format='png')
            # display(self.graph)

            self.add_page()

        def on_selectE_clicked(b):
            if txtarea.value != '':
                old = [i for i in self.pages]
                try:
                    self.parser(txtarea.value.split('\n'))
                    # update the current_bid
                    self.block_id = max(self.pages)
                    selectE.icon = 'check'
                except Exception as err:
                    print( "Invalid configuration file ...")
                    print( "    IOError: %s"%err.message)
                    print( "... Not loaded!")
                    selectE.icon = 'remove'
                    rm = [i for i in self.pages if i not in old]
                    for ib in rm:
                        if ib in self.pages:
                            del self.pages[ib]

                self.debut = False
                self.add_page()

                ## clear ouput and update the graph viz
                self.graph.close()
                dot = Digraph(format='png')
                for edge in self.comp_graph:
                    dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                    dot.node('%i' % edge[2], label='%i %s' % (edge[2], self.pages[edge[2]].title))
                    dot.edge('%i' % edge[0], '%i' % edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
                self.graph = widgets.Image(value=dot.pipe(), format='png')
                display(self.graph)

        def on_show_clicked(b):
            lines = self.configurer()
            txt.value = ''.join(lines)

        def on_save_clicked(b):
            path = self.check_filename(filename.value)
            lines = self.configurer()
            try:
                if len(lines) >0:
                    with open(path, 'w') as config:
                        for line in lines:
                            config.write(line)
                    print( "\nThe ChemML Wrapper's config file has been successfully saved ...")
                    print( "    config file name: %s" % path)
                    print( "    current directory: %s" % os.getcwd())
                    print( "    what's next? run the ChemML Wrapper using the config file with the following codes:")
                    print( "        >>> from chemml.wrapper.engine import run")
                    print( "        >>> run(INPUT_FILE = 'path_to_the_config_file', OUTPUT_DIRECTORY = '%s')" % outdir.value)
                    print( "... you can also create a python script of the above codes and run it on any cluster that ChemML is installed.")
                    save.icon = 'check'
                else:
                    print( "The config file is empty ...")
                    print( "... Not saved!")
                    save.icon = 'remove'
            except:
                print( "The config file path or name is not supported ...")
                print( "... Not saved!")
                save.icon = 'remove'

        header = widgets.Label(value='Choose how to start:', layout=widgets.Layout(width='50%'))
        # Tab: new script
        style = {'description_width': 'initial'}
        outdir = widgets.Text(
            value='CMLWrapper_out',
            placeholder='Type something',
            description='Output directory:',
            disabled=False,
            style = style,
            layout = widgets.Layout(margin='30px 0px 10px 0px'))
        headerN = widgets.Label(value='Start with a new script', layout=widgets.Layout(width='50%'))
        selectN = widgets.Button(description="Start", layout=widgets.Layout(margin='20px 0px 10px 115px'))
        selectN.style.button_color = 'lightblue'
        selectN.on_click(on_selectN_clicked)

        line = widgets.HBox([],layout=widgets.Layout(height='0px', border='dotted black 1px',margin='20px 0px 0px 0px'))
        note = widgets.HTML(value="<b>Note:</b> Don't forget to print/save the ChemML Wrapper's input file (configuration script) when you are done with editing:",
                            layout=widgets.Layout(margin='10px 0px 0px 0px'))
        show = widgets.Button(description="print Script", layout=widgets.Layout(width= "120px",margin='10px 0px 0px 0px'))
        show.style.button_color = 'lightblue'
        show.on_click(on_show_clicked)
        txt = widgets.Textarea(
            placeholder="Press the 'print Script' button to print the ChemML Wrapper's configuration script here. Copy and save it for the future use.",
            disabled=False,
            layout=widgets.Layout(width='100%',margin='0px 0px 0px 10px'))
        hbox1 = widgets.HBox([show,txt],layout=widgets.Layout(margin='10px 0px 0px 0px',align_items='center', justify_content = 'center'))

        save = widgets.Button(description="Save Script", layout=widgets.Layout(width= "120px",margin='20px 0px 0px 0px'))
        save.style.button_color = 'lightblue'
        save.on_click(on_save_clicked)
        # f_label = widgets.Label('config file name:',layout=widgets.Layout(width='100%',margin='20px 0px 0px 20px'))
        style = {'description_width': 'initial'}
        filename = widgets.Text(
            value='chemML_config.txt',
            description = 'config file name:',
            placeholder='chemML_config.txt',
            disabled=False,
            style= style,
            layout=widgets.Layout(margin='20px 0px 0px 20px'))
        hbox2 = widgets.HBox([save, filename],layout=widgets.Layout(margin='10px 0px 0px 0px',align_items='center', justify_content = 'center'))

        vboxN = widgets.VBox([headerN,outdir,selectN])
        # Tab: existing script
        headerE = widgets.Label(value='Load an existing script', layout=widgets.Layout(width='50%'))
        txtarea = widgets.Textarea(
            placeholder='copy a ChemML script (config file) here',
            disabled=False,
            layout = widgets.Layout(width='50%'))
        selectE = widgets.Button(description="Load", layout=widgets.Layout(margin='20px 0px 10px 115px'))
        selectE.style.button_color = 'lightblue'
        selectE.on_click(on_selectE_clicked)
        vboxE = widgets.VBox([headerE, txtarea, selectE])
        # Tab: template
        vboxT = self.templates_widgets()

        tabs = widgets.Tab()
        tabs.children = [vboxN, vboxE, vboxT]
        tabs.set_title(0, 'New script')
        tabs.set_title(1, 'Existing script')
        tabs.set_title(2, 'Template workflow')

        self.home_page_VBox = widgets.VBox([header, tabs,line,note,hbox1, hbox2])

    def home_page(self):
        id = 0
        # clear_output()
        self.home_page_widgets()
        self.pages[id] = container_page('Home page', id, self.home_page_VBox)
        self.display_accordion(id)
        display(self.accordion)
        print( 'The computation graph will be displayed here:')
        display(self.graph)

    ################################

    def add_page_widgets(self):
        def on_select_clicked(b):
            self.block_id += 1
            block_params = {'task': task_w.value, 'subtask': subtask_w.value, \
                                          'host': host_w.value, 'function': func_w.value, \
                                          'wparams':{}, 'fparams':{}, 'inputs':{}, 'outputs':{}}
            self.pages[self.block_id] = container_page('%s'%func_w.value, self.block_id, None, block_params)
            self.current_bid = self.block_id
            self.debut = True
            self.custom_function_page()

        def _subtask_update():
            subtask_opts = [i for i in self.combinations[task_w.value]]
            subtask_w.options = subtask_opts
            subtask_w.value = subtask_opts[0]

        def _host_update():
            host_opts = [i for i in self.combinations[task_w.value][subtask_w.value]]
            host_w.options = host_opts
            host_w.value = host_opts[0]

        def _func_update():
            func_opts = [i for i in self.combinations[task_w.value][subtask_w.value][host_w.value]]
            func_w.options = func_opts
            func_w.value = func_opts[0]

        def handle_task_change(t):
            _subtask_update()
            _host_update()
            _func_update()

        def handle_subtask_change(s):
            _host_update()
            _func_update()

        def handle_host_change(h):
            _func_update()

        header = widgets.Label(value='Choose a method:', layout=widgets.Layout(width='50%'))
        task_options = self.tasks[0:8]
        task_w = widgets.Dropdown(
            options=task_options,
            value=task_options[0],
            description='Task:')
        subtask_options = [i for i in self.combinations[task_w.value]]
        subtask_w = widgets.Dropdown(
            options=subtask_options,
            value=subtask_options[0],
            description='Subtask:')
        host_options = [i for i in self.combinations[task_w.value][subtask_w.value]]
        host_w = widgets.Dropdown(
            options=host_options,
            value=host_options[0],
            description='Host:')
        func_options = [i for i in self.combinations[task_w.value][subtask_w.value][host_w.value]]
        func_w = widgets.Dropdown(
            options=func_options,
            value=func_options[0],
            description='Function:')
        select = widgets.Button(description="Select", layout=widgets.Layout(margin='20px 0px 10px 115px'))
        select.style.button_color = 'lightblue'
        select.on_click(on_select_clicked)

        task_w.observe(handle_task_change, names='value')
        subtask_w.observe(handle_subtask_change, names='value')
        host_w.observe(handle_host_change, names='value')

        add_page_VBox = widgets.VBox([header, task_w, subtask_w, host_w, func_w, select])

        return add_page_VBox

    def add_page(self):
        id = 1
        add_page_VBox = self.add_page_widgets()
        self.pages[id] = container_page('Add a block', id, add_page_VBox)
        self.display_accordion(id)

    ################################

    def db_extract_function(self, host, function):
        # print("host:", host)
        if host == 'sklearn':
            metadata = getattr(sklearn_db, function)()
        elif host == 'chemml':
            metadata = getattr(chemml_db, function)()
        elif host == 'pandas':
            metadata = getattr(pandas_db, function)()
        wparams = {i:copy.deepcopy(vars(metadata.WParameters)[i]) for i in vars(metadata.WParameters).keys() if '__' not in i}
        fparams = {i:copy.deepcopy(vars(metadata.FParameters)[i]) for i in vars(metadata.FParameters).keys() if '__' not in i}
        inputs = {i: copy.deepcopy(vars(metadata.Inputs)[i]) for i in vars(metadata.Inputs).keys() if '__' not in i}
        outputs = {i: copy.deepcopy(vars(metadata.Outputs)[i]) for i in vars(metadata.Outputs).keys() if '__' not in i}
        return wparams, fparams, inputs, outputs, metadata

    def custom_function_IO_w(self):
        if not self.debut:
            self.current_bid = sorted(self.pages)[self.accordion.selected_index]

        def remove_nodes_toSEND(n = self.current_bid, remove_bids = set()):
            # find all the nodes that the current node can not send info to them
            remove_bids.add(n)  # including itself
            for e in self.comp_graph:
                if e[2]==n:
                    remove_bids.add(e[0])
                    remove_nodes_toSEND(n=e[0], remove_bids=remove_bids)
            return remove_bids

        def remove_nodes_toRECV(n = self.current_bid, remove_bids = set()):
            # find all the nodes that the current node can not receive info from them
            remove_bids.add(n)  # including itself
            for e in self.comp_graph:
                if e[0]==n:
                    remove_bids.add(e[2])
                    remove_nodes_toRECV(n=e[2], remove_bids=remove_bids)
            return remove_bids

        def refresh_tasks():
            if not self.debut:
                self.current_bid = sorted(self.pages)[self.accordion.selected_index]

            ### Senders
            ## block ids that can receive info from the current block
            bidS_options = [i for i in self.pages if i not in [0, 1]]
            rm_n_2S = remove_nodes_toSEND(self.current_bid,set())
            for n in rm_n_2S:
                bidS_options.remove(n)
            bidS.options = sorted(bidS_options)
            if len(bidS_options)>0:
                bidS.value = sorted(bidS_options)[0]
                external_inputs = [token for token in self.pages[bidS.value].block_params['inputs']]
                # remove input tokens that are already taken
                rm = []
                for t in external_inputs:
                    for e in self.comp_graph:
                        if e[2:] == (bidS.value, t):
                            rm.append(t)
                for token in rm:
                    external_inputs.remove(token)
                external_receivers.options = sorted(external_inputs)
                if len(external_inputs)>0:
                    external_receivers.value = sorted(external_inputs)[0]
            else:
                external_receivers.options = []

            ### Receivers
            ## update input tokens to receive only once
            input_tokens = [token for token in self.pages[self.current_bid].block_params['inputs']]
            # remove input tokens that are already taken
            rm = []
            for t in input_tokens:
                for e in self.comp_graph:
                    if e[2:] == (self.current_bid, t):
                        rm.append(t)
            for token in rm:
                input_tokens.remove(token)
            receiver.options = sorted(input_tokens)
            if len(input_tokens)>0:
                receiver.value = sorted(input_tokens)[0]
                # block ids that can send info to the current block
                bidR_options = [i for i in self.pages if i not in [0, 1]]
                rm = remove_nodes_toRECV(self.current_bid,set())
                for n in rm:
                    bidR_options.remove(n)
                bidR.options = sorted(bidR_options)
                if len(bidR_options)>0:
                    bidR.value = sorted(bidR_options)[0]
                    external_outputs = [token for token in self.pages[bidR.value].block_params['outputs']]
                    external_senders.options = external_outputs
                else:
                    external_senders.options = []
            else:
                bidR.options = []
                external_senders.options = []

            ## update pipes options from comp_graph
            ps = ["%i, %s      >>>      %i, %s" % (e[0], e[1], e[2], e[3]) for e in self.comp_graph if self.current_bid in e]
            pipes.options = sorted(ps)

        def display_graph():
            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2],self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i'%edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(),format='png')
            display(self.graph)

        def on_refresh_clicked(b):
            if not self.debut:
                self.current_bid = sorted(self.pages)[self.accordion.selected_index]
            refresh_tasks()

        def on_addS_clicked(b):
            if not self.debut:
                self.current_bid = sorted(self.pages)[self.accordion.selected_index]
            if sender.value is not None and bidS.value is not None and external_receivers.value is not None:
                input_type = set(self.pages[bidS.value].block_params['inputs'][external_receivers.value].types)
                output_type = set(self.pages[self.current_bid].block_params['outputs'][sender.value].types)
                if input_type.issubset(output_type) or output_type.issubset(input_type):
                    all_receivers = [e[2:] for e in self.comp_graph]
                    if (bidS.value, external_receivers.value) not in all_receivers:
                        edge = (self.current_bid, sender.value, bidS.value, external_receivers.value)
                        if edge not in self.comp_graph:
                            self.comp_graph.append(edge)
            refresh_tasks()
            display_graph()

        def on_addR_clicked(b):
            if not self.debut:
                self.current_bid = sorted(self.pages)[self.accordion.selected_index]
            if receiver.value is not None and bidR.value is not None and external_senders.value is not None:
                output_type = set(self.pages[bidR.value].block_params['outputs'][external_senders.value].types)
                input_type = set(self.pages[self.current_bid].block_params['inputs'][receiver.value].types)
                if input_type.issubset(output_type) or output_type.issubset(input_type):
                    all_receivers = [e[2:] for e in self.comp_graph]
                    if (self.current_bid, receiver.value) not in all_receivers:
                        edge = (bidR.value, external_senders.value, self.current_bid, receiver.value)
                        if edge not in self.comp_graph:
                            self.comp_graph.append(edge)
            ## all refresh tasks:
            refresh_tasks()
            display_graph()

        def bidS_value_change(change):
            if not self.debut:
                self.current_bid = sorted(self.pages)[self.accordion.selected_index]
            if bidS.value is not None:
                external_inputs = [token for token in self.pages[bidS.value].block_params['inputs']]
                rm = []
                for t in external_inputs:
                    for e in self.comp_graph:
                        if e[2:] == (bidS.value, t):
                            rm.append(t)
                for token in rm:
                    external_inputs.remove(token)
                external_receivers.options = sorted(external_inputs)
                if len(external_inputs)>0:
                    external_receivers.value = sorted(external_inputs)[0]
            else:
                external_receivers.options = []

        def bidR_value_change(change):
            if not self.debut:
                self.current_bid = sorted(self.pages)[self.accordion.selected_index]
            if bidR.value is not None:
                external_outputs = [token for token in self.pages[bidR.value].block_params['outputs']]
                external_senders.options = sorted(external_outputs)
                if len(external_outputs) > 0:
                    external_senders.value = sorted(external_outputs)[0]
            else:
                external_senders.options = []

        def on_remove_clicked(b):
            if not self.debut:
                self.current_bid = sorted(self.pages)[self.accordion.selected_index]
            rm = pipes.value
            for p in rm:
                p = p.strip().split(',')
                m = p[1].strip().split('      >>>      ')
                edge = (int(p[0]), m[0].strip(), int(m[1]), p[2].strip())
                if edge in self.comp_graph:
                    self.comp_graph.remove(edge)
            refresh_tasks()
            display_graph()


        self.pages[self.current_bid].IO_refresher = refresh_tasks

        ## Sender
        output_tokens = [token for token in self.pages[self.current_bid].block_params['outputs']]
        headerS = widgets.HTML(value='<b> Send >>> </b>', layout=widgets.Layout(width='50%',margin='10px 0px 0px 10px'))
        # listS = widgets.HTML(value='output tokens: %s'%', '.join(sorted(output_tokens)), \
        #                      layout=widgets.Layout(margin='3px 0px 0px 20px'))
        sender = widgets.Dropdown(
            options=sorted(output_tokens),
            # value=output_tokens[0],
            description='output token:')
        hbox1S = widgets.HBox([sender],layout=widgets.Layout(height='40px', border='dotted black 1px',
                                                               align_items='center',  # justify_content = 'center',
                                                               margin='0px 0px 0px 10px'))
        toS = widgets.HTML(value='<b> >>> </b>')
        bidS_options = [i for i in self.pages if i not in [0, 1]]
        rm_n_2S = remove_nodes_toSEND(self.current_bid,set())
        for n in rm_n_2S:
            bidS_options.remove(n)
        bidS = widgets.Dropdown(
            options=sorted(bidS_options),
            description='block#:',
            layout=widgets.Layout(width='140px'))
        bidS.observe(bidS_value_change,names='value')
        refreshS = widgets.Button(icon='refresh', layout=widgets.Layout(width='40px'))
        # refreshS.style.button_color = 'darkseagreen'
        refreshS.on_click(on_refresh_clicked)
        if bidS.value is not None:
            external_inputs = [token for token in self.pages[bidS.value].block_params['inputs']]
            rm = []
            for t in external_inputs:
                for e in self.comp_graph:
                    if e[2:] == (bidS.value, t):
                        rm.append(t)
            for token in rm:
                external_inputs.remove(token)
        else:
            external_inputs = []
        external_receivers = widgets.Dropdown(
            options = sorted(external_inputs),
            description='input token:')
        hbox2S = widgets.HBox([bidS,refreshS, external_receivers],
                             layout=widgets.Layout(height='40px', border='dotted black 1px',
                                                   align_items='center', justify_content='center'))
        addS = widgets.Button(description="Add", layout=widgets.Layout(width='60px', margin='0px 10px 0px 0px'))
        addS.style.button_color = 'lightblue'
        addS.on_click(on_addS_clicked)
        hboxS = widgets.HBox([hbox1S, toS, hbox2S, addS],
                             layout=widgets.Layout(justify_content='space-between',
                                                   align_items='center',
                                                   margin='10px 0px 20px 0px'))

        ## Receiver
        note = widgets.HTML(value='<b> Note: </b> This page automatically avoid: (1) loops, (2) type inconsistency, and (3) more than one input per token.', layout=widgets.Layout(margin='20px 0px 0px 10px'))
        headerR = widgets.HTML(value='<b> Receive <<< </b>', layout=widgets.Layout(width='50%',margin='10px 0px 0px 10px'))
        input_tokens = [token for token in self.pages[self.current_bid].block_params['inputs']]
        listR = widgets.HTML(value='all input tokens: %s'%', '.join(sorted(input_tokens)),
                             layout=widgets.Layout(margin='0px 0px 0px 20px'))
        rm = []
        for t in input_tokens:
            for e in self.comp_graph:
                if e[2:] == (self.current_bid, t):
                    rm.append(t)
        for token in rm:
            input_tokens.remove(token)
        receiver = widgets.Dropdown(
            options=input_tokens,
            # value=input_tokens[0],
            disabled = False,
            description = 'input token:')
        hbox1R = widgets.HBox([receiver],layout=widgets.Layout(height='40px', border='dotted black 1px',
                                                               align_items='center',  # justify_content = 'center',
                                                               margin='0px 0px 0px 10px'))
        fro = widgets.HTML(value='<b> <<< </b>')
        bidR_options = [i for i in self.pages if i not in [0, 1]]
        rm = remove_nodes_toRECV(self.current_bid, set())
        for n in rm:
            bidR_options.remove(n)
        bidR = widgets.Dropdown(
            options = sorted(bidR_options),
            description='block#:',
            disabled=False,
            layout=widgets.Layout(width='140px'))
        bidR.observe(bidR_value_change,names='value')
        refresh = widgets.Button(icon='refresh',disabled=False, layout=widgets.Layout(width='40px'))
        # refresh.style.button_color = 'darkseagreen'
        refresh.on_click(on_refresh_clicked)
        if bidR.value is not None:
            external_outputs = [token for token in self.pages[bidR.value].block_params['outputs']]
        else:
            external_outputs = []
        external_senders = widgets.Dropdown(
            options = sorted(external_outputs),
            disabled=False,
            description='output token:')
        hbox2R = widgets.HBox([bidR, refresh,external_senders],
                             layout=widgets.Layout(height='40px', border='dotted black 1px',
                                                   align_items='center', justify_content='center'))
        addR = widgets.Button(description="Add", disabled=False, layout=widgets.Layout(width='60px', margin='0px 10px 0px 0px'))
        addR.style.button_color = 'lightblue'
        addR.on_click(on_addR_clicked)
        hboxR = widgets.HBox([hbox1R, fro, hbox2R, addR],
                             layout=widgets.Layout(justify_content='space-between',
                                                   align_items='center',
                                                   margin='10px 0px 20px 0px'))

        ## vbox
        pipes_options = [edge for edge in self.comp_graph if self.current_bid in edge]
        pipes = widgets.SelectMultiple(
            options= sorted(pipes_options),
            # description='list of pipes:',
            layout=widgets.Layout(margin='20px 0px 20px 250px'))
        remove = widgets.Button(description="Remove pipe", icon = 'remove',
                                layout=widgets.Layout(width='120px', margin='40px 0px 0px 10px'))
        remove.style.button_color = 'lightblue'
        remove.on_click(on_remove_clicked)
        hbox4 = widgets.HBox([pipes, remove], margin='0px 0px 0px 100px')

        IO_vbox = widgets.VBox([note, headerS, hboxS, headerR, listR, hboxR, hbox4])#, layout=widgets.Layout(border='solid darkslategray 1px'))
        return IO_vbox

    def custom_function_params_w(self):
        # def handle_param_change(b):
        #     for item in self.pages[current_bid].block_params['wparams']:
        #         self.pages[current_bid].block_params['wparams'][item].value = self.pages[current_bid].block_params['wparams'][item].widget.value
        #     for item in self.pages[current_bid].block_params['fparams']:
        #         self.pages[current_bid].block_params['fparams'][item].value = self.pages[current_bid].block_params['fparams'][item].widget.value
        if not self.debut:
            self.current_bid = sorted(self.pages)[self.accordion.selected_index]

        def on_default_clicked(b):
            if not self.debut:
                self.current_bid = sorted(self.pages)[self.accordion.selected_index]
            for item in self.pages[self.current_bid].block_params['wparams']:
                self.pages[self.current_bid].block_params['wparams'][item].widget.value = \
                str(self.pages[self.current_bid].block_params['wparams'][item].default)
            for item in self.pages[self.current_bid].block_params['fparams']:
                self.pages[self.current_bid].block_params['fparams'][item].widget.value = \
                str(self.pages[self.current_bid].block_params['fparams'][item].default)

        # wrapper parameters widgets
        header = widgets.HTML(value='<b>Wrapper parameters:</b>', layout=widgets.Layout(width='50%'))
        wparams = self.pages[self.current_bid].block_params['wparams']
        wparams_boxes = []
        for item in sorted(wparams):
            style = {'description_width': 'initial'}
            wp = widgets.Text(
                value=str(wparams[item].default),
                placeholder=str(wparams[item].options),
                description=wparams[item].name,
                disabled=False,
                style = style,
                )
            wparams[item].widget = wp
            # wp.observe(handle_param_change, names='value')
            wp_checker = widgets.Checkbox(value=False,
                                          description='Check to set',
                                          disabled=False,
                                          indent=False,
                                          layout = widgets.Layout(margin='0px 0px 0px 10px')
            )
            if self.pages[self.current_bid].block_params['wparams'][item].required:
                wp_checker.value = True
            wparams[item].checker = wp_checker

            wformat = widgets.Text(
                value=str(wparams[item].format),
                description= 'Type:',
                disabled=True)

            hwp = widgets.Box([wp])
            if self.pages[self.current_bid].block_params['wparams'][item].required:
                hwp.layout = widgets.Layout(border='dotted red 1px')

            hbox = widgets.HBox([hwp, wp_checker, wformat],)
                                # layout=widgets.Layout(display='flex',
                                #                       flex_flow='row',
                                #                       align_items='stretch'))
            wparams_boxes.append(hbox)
        wparams_vbox = widgets.VBox([header]+ wparams_boxes)
        self.pages[self.current_bid].block_params['wparams'] = wparams

        # function parameters widgets
        header = widgets.HTML(value='<b>Function parameters:</b>',
                              layout=widgets.Layout(width='50%',margin='20px 0px 0px 0px'))
        fparams = self.pages[self.current_bid].block_params['fparams']
        fparams_boxes = []
        for item in sorted(fparams):
            style = {'description_width': 'initial'}
            wp = widgets.Text(
                value=str(fparams[item].default),
                placeholder=str(fparams[item].options),
                description=fparams[item].name,
                disabled=False,
                style = style,
                )
            fparams[item].widget = wp
            # wp.observe(handle_param_change, names='value')
            wp_checker = widgets.Checkbox(value=False,
                                          description='Check to set',
                                          disabled=False,
                                          indent = False,
                                          layout = widgets.Layout(margin='0px 0px 0px 10px')
                                         )

            if self.pages[self.current_bid].block_params['fparams'][item].required:
                wp_checker.value = True
            fparams[item].checker = wp_checker

            wformat = widgets.Text(
                value=str(fparams[item].format),
                description='Type:',
                disabled=True,
                layout = widgets.Layout(width='40%'))

            hwp = widgets.Box([wp])
            if self.pages[self.current_bid].block_params['fparams'][item].required:
                hwp.layout = widgets.Layout(border='dotted red 1px')


            hbox = widgets.HBox([ hwp, wp_checker, wformat],)
                                # layout=widgets.Layout(display='flex',
                                #                       flex_flow='row',))
                                                      # align_items='stretch'))
            fparams_boxes.append(hbox)
        fparams_vbox = widgets.VBox([header]+ fparams_boxes)
        self.pages[self.current_bid].block_params['fparams'] = fparams

        defaultB = widgets.Button(description="Default values",
                                  layout=widgets.Layout(width='110px',margin='10px 0px 10px 180px'))
        defaultB.style.button_color = 'lightblue'
        defaultB.on_click(on_default_clicked)

        params_vbox = widgets.VBox([wparams_vbox,fparams_vbox,defaultB],
                                   layout=widgets.Layout(justify_content='center', align_content='center'))

        return params_vbox

    def set_default_params_IO(self):
        if not self.debut:
            self.current_bid = sorted(self.pages)[self.accordion.selected_index]
        host = self.pages[self.current_bid].block_params['host']
        # print('host_1',host)
        function = self.pages[self.current_bid].block_params['function']
        # print("funct_1",function)
        wparams, fparams, inputs, outputs, metadata = self.db_extract_function(host,function)
        self.pages[self.current_bid].block_params['wparams'] = wparams
        self.pages[self.current_bid].block_params['fparams'] = fparams
        self.pages[self.current_bid].block_params['inputs'] = inputs
        self.pages[self.current_bid].block_params['outputs'] = outputs
        self.pages[self.current_bid].block_params['requirements'] = metadata.requirements
        self.pages[self.current_bid].block_params['documentation'] = metadata.documentation
        self.pages[self.current_bid].block_params['task'] = metadata.task
        self.pages[self.current_bid].block_params['subtask'] = metadata.subtask

    def custom_function_widgets(self):
        if not self.debut:
            self.current_bid = sorted(self.pages)[self.accordion.selected_index]

        def on_remove_clicked(b):
            if not self.debut:
                self.current_bid = sorted(self.pages)[self.accordion.selected_index]
            del self.pages[self.current_bid]
            rm = [edge for edge in self.comp_graph if self.current_bid in edge]
            for e in rm:
                self.comp_graph.remove(e)

            ## clear ouput and update the graph viz
            self.graph.close()
            dot = Digraph(format='png')
            for edge in self.comp_graph:
                dot.node('%i' % edge[0], label='%i %s' % (edge[0], self.pages[edge[0]].title))
                dot.node('%i' % edge[2], label='%i %s' % (edge[2],self.pages[edge[2]].title))
                dot.edge('%i' % edge[0], '%i'%edge[2], label='%s > %s' % (edge[1], edge[3]), labelfontcolor='green')
            self.graph = widgets.Image(value=dot.pipe(),format='png')
            display(self.graph)


            self.add_page()

        # initialize, only if fparams, inputs and ouputs are empty
        if len(self.pages[self.current_bid].block_params['fparams']) == len(self.pages[self.current_bid].block_params['inputs']) ==\
                len(self.pages[self.current_bid].block_params['outputs']) == 0:
            self.set_default_params_IO()
        # info
        t = self.pages[self.current_bid].block_params['task']
        s = self.pages[self.current_bid].block_params['subtask']
        h = self.pages[self.current_bid].block_params['host']
        f = self.pages[self.current_bid].block_params['function']
        tshfTB = widgets.ToggleButtons(
            options=[t, s, h, f],
            disabled=False,
            button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
            layout=widgets.Layout(margin='0px 0px 0px 180px'))
        reqL = widgets.HTML(value="<b>Requirements:  </b>"+str(self.pages[self.current_bid].block_params['requirements']),
                            layout=widgets.Layout(width='100%'))
        docL = widgets.HTML(value="<b>Documentation: </b>"+self.pages[self.current_bid].block_params['documentation'],
                            layout=widgets.Layout(width='100%'))

        # parameters widgets
        params_vbox = self.custom_function_params_w()

        # inputs and outputs
        IO_vbox = self.custom_function_IO_w()

        removeB = widgets.Button(description="Remove block",layout=widgets.Layout(margin='10px 0px 10px 400px'))
        removeB.style.button_color = 'lightblue'
        removeB.on_click(on_remove_clicked)

        custom_f_accordion = widgets.Tab(layout=widgets.Layout(margin='10px 0px 10px 0px'))
        custom_f_accordion.children = [params_vbox, IO_vbox]
        custom_f_accordion.set_title(0, 'Parameters')
        custom_f_accordion.set_title(1, 'Input/Output')
        # custom_f_accordion.selected_index = 0

        custom_f_vbox = widgets.VBox([tshfTB, reqL, docL, custom_f_accordion, removeB],
                                     layout=widgets.Layout(justify_content='center', align_content='center'))

        return custom_f_vbox

    def custom_function_page(self):
        custom_function_VBox = self.custom_function_widgets()
        self.pages[self.current_bid].widget = custom_function_VBox
        # display
        active_id = [i for i in sorted(self.pages)].index(self.current_bid)
        active_id = 1 # keep it on the add page (if comment out this line the newly added block will become active)
        self.display_accordion(active_id)
        self.debut = False

    ################################

    def parser(self, script):
        """
        The main funtion for parsing chemml script.
        It starts with finding blocks and then runs other functions.

        :return:
        cmls: pages
        """
        blocks={}
        it = max(self.pages)+1  # must be +2 but there is another +1 in the loop

        check_block = False
        for line in script:
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
        self.loading_bids = range(min(blocks),max(blocks)+1)
        self._options(blocks)
        # to make self.comp_graph with cmls (only requires send and receive)
        self._transform()

    def _db_extract_function(self, host, function):
        print("host_db_extract",host)
        if host == 'sklearn':
            metadata = getattr(sklearn_db, function)()
        elif host == 'chemml':
            metadata = getattr(chemml_db, function)()
        wparams = {i:copy.deepcopy(vars(metadata.WParameters)[i]) for i in vars(metadata.WParameters).keys() if
                     i not in ('__module__', '__doc__')}
        fparams = {i:copy.deepcopy(vars(metadata.FParameters)[i]) for i in vars(metadata.FParameters).keys() if
                     i not in ('__module__', '__doc__')}
        inputs = {i: copy.deepcopy(vars(metadata.Inputs)[i]) for i in vars(metadata.Inputs).keys() if
                  i not in ('__module__', '__doc__')}
        outputs = {i: copy.deepcopy(vars(metadata.Outputs)[i]) for i in vars(metadata.Outputs).keys() if
                   i not in ('__module__', '__doc__')}
        return wparams, fparams, inputs, outputs, metadata

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
                parameters[param] = val #value(val) #"%s"%val
            while '>>' in line:
                line = line[line.index('>>') + 2:].strip()
                if '>' in line:
                    args = line[:line.index('>')].strip()
                else:
                    args = line.strip()
                arg = args.split()
                # print(arg)
                if len(arg) == 2:
                    a, b = arg
                    # print("Hello 1",a,b)
                    if isint(b) and not isint(a):
                        # print("Hello 2",a,b)
                        send[(a, int(b))] = item
                    elif isint(a) and not isint(b):
                        recv[(b, int(a))] = item
                    else:
                        msg = 'wrong format of send and receive in block #%i at %s (send: >> var id; recv: >> id var)' % (item+1, args)
                        # raise IOError(msg)
                else:
                    msg = 'wrong format of send and receive in block #%i at %s (send: >> var id; recv: >> id var)'%(item+1,args)
                    # raise IOError(msg)
        return parameters, send, recv

    def _options(self, blocks):
        for bid in self.loading_bids:
            block = blocks[bid]
            # task = self._functions(block[0])
            parameters, send, recv = self._parameters(block,bid)
            host = parameters['host']
            function = parameters['function']
            # here we also add send and receive to be handled in the self.transform for making the comp_graph
            block_params = {'task': '', 'subtask': '', \
                            'host': host, 'function': function, \
                            'wparams': {}, 'fparams': {}, 'inputs': {}, 'outputs': {}, 'send':send, 'recv':recv}
            self.pages[bid] = container_page('%s'%function, bid, None, block_params)
            self.current_bid = bid
            self.debut = True
            # to make all the widgets (store widget in the page.widget and parameters and IOs db containers
            custom_function_VBox = self.custom_function_widgets()
            self.pages[bid].widget = custom_function_VBox

            # change the widget values to the values from config file
            # print("parameters: ",parameters)
            # print(self.pages[bid].block_params['fparams'])
            for param in parameters:
                if param not in ['host', 'function']:
                    if param in self.pages[bid].block_params['wparams']:
                        self.pages[bid].block_params['wparams'][param].widget.value = parameters[param]
                        # checkbox if value is different from default
                        if self.pages[bid].block_params['wparams'][param].default != parameters[param]:
                            self.pages[bid].block_params['wparams'][param].checker.value = True
                    elif param in self.pages[bid].block_params['fparams']:
                        self.pages[bid].block_params['fparams'][param].widget.value = parameters[param]
                        # checkbox if value is different from default
                        if self.pages[bid].block_params['fparams'][param].default != parameters[param]:
                            self.pages[bid].block_params['fparams'][param].checker.value = True
                    else:
                        msg = "(host: %s, function: %s): Not a valid parameter '%s'"%(host, function, param)
                        raise IOError(msg)

    def _transform(self):
        """
        goals:
            - collect all sends and receives
            - check send and receive format.
            - make the computational graph

        """
        send_all = []
        recv_all = []
        for bid in self.loading_bids:
            send_all += self.pages[bid].block_params['send'].items()
            recv_all += self.pages[bid].block_params['recv'].items()
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
            print( set(send_ids),set(recv_ids))
            msg = 'missing pairs of send and receive id:\n send IDs:%s\n recv IDs:%s\n'%(str(set(send_ids)),str(set(recv_ids)))
            raise ValueError(msg)

        # make graph
        reformat_send = {k[1]:[v,k[0]] for k,v in send_all}
        self.comp_graph += [tuple(reformat_send[k[1]]+[v,k[0]]) for k,v in recv_all]



# examples for dictionaries' fomrat:

# comp_graph = [[0, 'df', 2, 'df'], [0, 'api', 1, 'iv1'], [1, 'ov1', 3, 'df2'], [2, 'df', 3, 'df1'], [1, 'ov2', 4, 'df']]

# self.pages[ib].block_params = { 'function': 'mlp_hogwild',
#                                 'inputs': {'api': <chemml.wrappers.database.containers.Input object at 0x12351c510>,
#                                             'dfy': <chemml.wrappers.database.containers.Input object at 0x12421a610>,
#                                             'dfx': <chemml.wrappers.database.containers.Input object at 0x12421a690>},
#                                 'subtask': 'regression',
#                                 'task': 'Model',
#                                 'requirements': (('ChemML', '0.1.0'), ('scikit-learn', '0.19.0'), ('pandas', '0.20.3')),
#                                 'wparams': {'func_method': <chemml.wrappers.database.containers.Parameter object at 0x123bf1d50>},
#                                 'fparams': {'rms_decay': <chemml.wrappers.database.containers.Parameter object at 0x1235cc750>,
#                                             'learn_rate': <chemml.wrappers.database.containers.Parameter object at 0x1235cc250>},
#                                 'outputs': {'api': <chemml.wrappers.database.containers.Output object at 0x12421a6d0>,
#                                             'dfy_predict': <chemml.wrappers.database.containers.Output object at 0x12421a710>},
#                                 'documentation': '',
#                                 'host': 'chemml'
#                               }