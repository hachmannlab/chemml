from celery import Celery
from cheml.wrappers.celery.utils import parse_graph
from cheml.wrappers.engine import Wrapper


app = Celery('tasks', backend='redis://localhost', broker='pyamqp://guest@localhost//')


@app.task
def add(x, y):
    return x + y


@app.task
def run_cheml(data):
    print(data)

    cmls, dep_lists, comp_graph = parse_graph(data)
    wrapper = Wrapper(cmls, dep_lists, comp_graph, "", "/home/tinto/Workspace/tmp")
    wrapper.call()

    return cmls.__str__()