import sklearn_db
import cheml_db
import pandas_db
import inspect
from collections import OrderedDict
import json


def createJson(v):
    return {'display_name': v, 'name': v, 'type': 'string', 'desc': v, 'value': ''}


def tshf():
    """
    tshf stands for the combination of task, subtask, host, and function
    :return: combination, dictionary of the aforementioned combinations
    """
    # 7 tasks
    tasks = ['Enter', 'Represent', 'Prepare', 'Model', 'Search', 'Mix', 'Visualize', 'Store']
    extras = ['np', '__builtins__', '__doc__', '__file__', '__name__', '__package__', 'mask', 'Input', 'Output',
              'Parameter', 'req', 'regression_types', 'cv_classes']

    combination = OrderedDict()
    for task in tasks:
        combination.update({task: {}})

    all_classes = [k[1] for k in inspect.getmembers(sklearn_db) if k[0][0:2]!='__']
    all_classes += [k[1] for k in inspect.getmembers(cheml_db) if k[0][0:2]!='__' ]
    all_classes += [k[1] for k in inspect.getmembers(pandas_db) if k[0][0:2]!='__' ]
    for k in all_classes:
        vk = vars(k)
        if 'task' in vk and 'subtask' in vk:
            task, subtask, host, function = [vk['task'], vk['subtask'], vk['host'], vk['function']]
            if subtask not in combination[task]:
                combination[task][subtask] = {host: [function]}
            else:
                if host not in combination[task][subtask]:
                    combination[task][subtask][host] = [function]
                else:
                    combination[task][subtask][host].append(function)
    return json.dumps(tasks), json.dumps(combination)


def get_complete_meta():
    # 7 tasks
    tasks = ['Enter', 'Represent', 'Prepare', 'Model', 'Search', 'Mix', 'Visualize', 'Store']
    extras = ['np', '__builtins__', '__doc__', '__file__', '__name__', '__package__', 'mask', 'Input', 'Output',
              'Parameter', 'req', 'regression_types', 'cv_classes']

    combination = {task: {} for task in tasks}
    all_classes = [k[1] for k in inspect.getmembers(sklearn_db) if k[0][0:2] != '__']
    all_classes += [k[1] for k in inspect.getmembers(cheml_db) if k[0][0:2] != '__']
    all_classes += [k[1] for k in inspect.getmembers(pandas_db) if k[0][0:2] != '__']
    functions = {}
    for k in all_classes:
        vk = vars(k)
        if 'task' in vk and 'subtask' in vk:
            task, subtask, host, function = [vk['task'], vk['subtask'], vk['host'], vk['function']]
            if subtask not in combination[task]:
                combination[task][subtask] = {host: [function]}
            else:
                if host not in combination[task][subtask]:
                    combination[task][subtask][host] = [function]
                else:
                    combination[task][subtask][host].append(function)
            functions[function] = {}
            if function == 'read_excel':
                pass
            if hasattr(k, 'FParameters'):
                functions[function]['FParameters'] = [createJson(v[0]) for v in inspect.getmembers(k.FParameters) if v[0][0:2] != '__']
            if hasattr(k, 'WParameters'):
                functions[function]['WParameters'] = [createJson(v[0]) for v in inspect.getmembers(k.WParameters) if v[0][0:2] != '__']
            if hasattr(k, 'Inputs'):
                functions[function]['input'] = [createJson(v[0]) for v in inspect.getmembers(k.Inputs) if v[0][0:2] != '__']
            if hasattr(k, 'Outputs'):
                functions[function]['output'] = [createJson(v[0]) for v in inspect.getmembers(k.Outputs) if v[0][0:2] != '__']
    return json.dumps(functions)
