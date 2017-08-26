import ipywidgets as widgets
from IPython.display import display
import traitlets

from .base import BASE
from ..wrappers.base import BANK


# from cheml.wrappers.base import BANK
class wrapperGUI(BASE):
    def _main_accordion(self):
        main_accordion = widgets.Accordion(children=self.accordion_children.values(), selected_index=0,
                                           layout=widgets.Layout(border='solid'))
        main_accordion.set_title(0, 'Add a block ...')
        if self.prev_accordion is not None:
            self.prev_accordion.close()
        display(main_accordion)
        return main_accordion

    def _add_box(self):
        # caption = widgets.Label(value='choose a method:',layout=widgets.Layout(width='50%'))
        self.bank, _ = BANK()
        self._task()
        self._subtask()
        self._host()
        self._func()
        self.graph_temp = {'task': self.task.value, 'subtask': self.subtask.value, \
                           'host': self.host.value, 'function': self.func.value}

        self.select = widgets.Button(description="Select", layout=widgets.Layout(margin='20px 0px 10px 115px'))
        self.select.on_click(self.on_select_clicked)

        add_box = widgets.VBox([self.task, self.subtask, self.host, self.func, self.select])
        self.accordion_children['add'] = add_box
        self.prev_accordion = self._main_accordion()

    def _task(self):
        _, task_options = BANK()
        self.task = widgets.Dropdown(
            options=task_options,
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
        self.func.observe(self.handle_func_change, names='value')

    def _close_add_box(self):
        self._main_accordion.close()

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
        self.graph_temp = {'task': self.task.value, 'subtask': self.subtask.value, \
                           'host': self.host.value, 'function': self.func.value}

    def handle_subtask_change(self, change):
        self._host_update()
        self._func_update()
        self.graph_temp = {'task': self.task.value, 'subtask': self.subtask.value, \
                           'host': self.host.value, 'function': self.func.value}

    def handle_host_change(self, change):
        self._func_update()
        self.graph_temp = {'task': self.task.value, 'subtask': self.subtask.value, \
                           'host': self.host.value, 'function': self.func.value}

    def handle_func_change(self, change):
        self.graph['function'] = self.func.value

    def on_select_clicked(self, b):
        # display function
        # if function added ==> close all other displays and display a new accardion with new function
        # if function added ==> function_counter += 1
        # if function canceled ==> close function displays

        # self.prev_accordion.close()
        self.select.disabled = True
        self.select.button_style = 'danger'
        if self.graph_temp['function'] == 'RDKitFingerprint':
            AHA = cheml_RDKitFingerprint()
            AHA.display()


gui = wrapperGUI(script='new')
gui.run()