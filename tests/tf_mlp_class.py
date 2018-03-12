import unittest
from cheml.nn.tensorflow import mlp_classification

class TestTFMLP(unittest.TestCase):
    """
    The basic test class
    """
    def test_test(self):
        """
        The actual training test.
        """
        model = mlp_classification(nneurons=[256,256],act_funcs='relu',cost='scel',
                                   optimizer='AdamOptimizer', training_epochs=15, display_step = 1)
        accuracy = model.test()
        print accuracy

if __name__== '__main__':
    unittest.main()