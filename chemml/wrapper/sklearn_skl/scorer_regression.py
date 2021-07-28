"""
A wrapper for the sklearn.metrics.make_scorer
"""
import sys
from chemml.wrapper.interfaces import evaluate_inputs

def scorer_regression(block, stack):
    # evaluate function inputs
    inputs = evaluate_inputs(block['inputs'], stack)

    output_dict ={}
    # run function
    from sklearn.metrics import make_scorer
    for i in inputs['args']:
        if i == 'mean_squared_error' or i=='mean_absolute_error' or i=='accuracy_score':
            score_func_1=i
            output_dict ['score_'+str(i)] = make_scorer(score_func = score_func_1, *inputs['args'], **inputs['kwargs'])
        else: 
            sys.exit("Function not incoporated as yet")
       
    return output_dict
