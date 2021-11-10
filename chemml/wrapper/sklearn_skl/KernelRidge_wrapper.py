"""
A wrapper for the sklearn.kernel_ridge.KernelRidge
"""

from chemml.wrapper.interfaces import evaluate_inputs

class KernelRidge:
    def __init__(self,block,stack):

        # evaluate function inputs
        inputs = evaluate_inputs(block['inputs'], stack)

        # run function
        from sklearn.kernel_ridge import KernelRidge
        self.clf = KernelRidge(**inputs['kwargs'])
        self.clf.fit(*inputs['args'])
    
    

    def predict_y(self, X_test):
        y_predict= self.clf.predict(X_test)

        return y_predict


# model = KernelRidge(block, stack)
# model.predict_y(x)