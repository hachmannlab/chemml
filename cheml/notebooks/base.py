import ipywidgets as widgets
from IPython.display import display

class BASE(object):
    def __init__(self, script='new'):
        self.script = script
        self.childeren = [widgets.Label(value='choose a method:')]

        if script == 'new':
            self._main_box()
        else:
            self.run_graph()

    def run_graph(self):
        print 'workflow graph'


