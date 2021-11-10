"""
A wrapper for the sklearn.model_selection.KFold
"""

from chemml.wrapper.interfaces import evaluate_inputs

def KFold(block, stack):
    # evaluate function inputs
    inputs = evaluate_inputs(block['inputs'], stack)

    # run function
    from sklearn.model_selection import KFold    
    kf = KFold(**inputs['kwargs'])
    output_dict = {}
    
    fold_gen = kf.split(*inputs['args'])
    output_dict["api"] = kf
    output_dict["fold_gen"] = fold_gen
        
    return output_dict




   

