from chemml.wrapper.database import sklearn_db, chemml_db, pandas_db
import inspect

def tshf():
    """
    tshf stands for the combination of task, subtask, host, and function
    :return: combination, dictionary of the aforementioned combinations
    """
    # 7 tasks
    tasks = ['Input', 'Represent', 'Prepare', 'Model', 'Optimize', 'Visualize', 'Output']
    extras = ['np', '__builtins__', '__doc__', '__file__', '__name__', '__package__', 'mask', 'Input', 'Output',
              'Parameter', 'req', 'regression_types', 'cv_classes']

    # for task in tasks:
    #     print(task)
    combination = {task: {} for task in tasks}
    all_classes = [k[1] for k in inspect.getmembers(sklearn_db) if k[0][0:2]!='__']
    all_classes += [k[1] for k in inspect.getmembers(chemml_db) if k[0][0:2]!='__' ]
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
    return tasks, combination