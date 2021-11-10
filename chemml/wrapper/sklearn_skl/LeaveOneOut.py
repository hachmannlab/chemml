"""
A wrapper for the sklearn.model_selection.LeaveOneOut
"""

from chemml.wrapper.interfaces import evaluate_inputs

def LeaveOneOut(block, stack):

    # evaluate function inputs
    inputs = evaluate_inputs(block['inputs'], stack)

    # run function
    from sklearn.model_selection import LeaveOneOut

    loo = LeaveOneOut(**inputs['kwargs'])
    output_dict = {}
    fold_gen = loo.split(*inputs['args'])
    output_dict["api"] = loo
    output_dict["fold_gen"] = fold_gen