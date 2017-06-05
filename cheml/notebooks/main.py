import ipywidgets as widgets
from IPython.display import display
import traitlets

from ..wrappers.base import BIG_BANK

class rungui(BIG_BANK):

    def gui(self,script = 'new'):
        # caption = widgets.Label(value='Enter Data')
        task_options = [i for i in self.info]
        self.task = widgets.Dropdown(
            options = task_options,
            value='IO',
            description='Task:',
            disabled=False,
            button_style='' # 'success', 'info', 'warning', 'danger' or ''
            )

        host_options = [i for i in self.info[self.task.value]]
        self.host = widgets.Dropdown(
            options = host_options,
            value = host_options[0],
            description = 'Host:',
            disabled = False,
            button_style = '' # 'success', 'info', 'warning', 'danger' or ''
            )

        func_options = [i for i in self.info[self.task.value][self.host.value]]
        self.func = widgets.Dropdown(
            options = func_options,
            value = func_options[0],
            description = 'Function:',
            disabled = False,
            button_style = '' # 'success', 'info', 'warning', 'danger' or ''
            )
        display(caption, self.task, self.host, self.func)

        self.task.observe(self.handle_task_change,names='value')

        set = widgets.Button(description="Set")
        display(set)
        self.graph = {'task':self.task.value,'host':self.host.value,'function':self.func.value}
        set.on_click(self.on_download_clicked)

        if script == 'new':
            pass

    def handle_task_change(self,change):
        host_opts = [i for i in self.info[self.task.value]]
        self.host.options = host_opts
        self.host.value = host_opts[0]
        func_opts = [i for i in self.info[self.task.value][self.host.value]]
        self.func.options = func_opts
        self.func.value = func_opts[0]

        self.graph['task'] = self.task.value
        self.graph['host'] = self.host.value
        self.graph['function'] = self.func.value

        self.host.observe(self.handle_host_change, names='value')

    def handle_host_change(self,change):
        func_opts = [i for i in self.info[self.task.value][self.host.value]]
        self.func.options = func_opts
        self.func.value = func_opts[0]

        self.graph['task'] = self.task.value
        self.graph['host'] = self.host.value
        self.graph['function'] = self.func.value

        self.func.observe(self.handle_func_change, names='value')

    def handle_func_change(self,change):
        self.graph['function'] = self.func.value

    def on_download_clicked(self,b):
        print(self.graph)
        if self.graph['function']=='RDKitFingerprint':
            w3 = widgets.Checkbox(value=False,description='Hi there!',disabled=False)
            display(w3)
