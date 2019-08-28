"""
A wrapper for the sklearn.model_selection.train_test_split
"""

from chemml.wrapper.interfaces import evaluate_inputs

def train_test_split(block, stack):

    # evaluate function inputs
    inputs = evaluate_inputs(block['inputs'], stack)

    # run function
    from sklearn.model_selection import train_test_split
    function_output_ = train_test_split(*inputs['args'], **inputs['kwargs'])

    n_out = len(function_output_)
    assert n_out == 2*len(inputs['args'])

    # create outputs
    # names are in this order: train1, test1, train2, test2, train3, test3
    output_dict = {}
    for i in range(n_out):
        if i % 2 == 0:
            output_dict["train%i" % (int(i/2) + 1)] = function_output_[i]
        else:
            output_dict["test%i" % (int(i/2) + 1)] = function_output_[i]

    return output_dict

