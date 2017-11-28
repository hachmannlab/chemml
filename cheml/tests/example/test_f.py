import unittest
from f import fact, div

class TestFactorial(unittest.TestCase):
    """
    Our basic test class
    """

    def test_fact(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """
        res = fact(5)
        self.assertEqual(res, 120)

    def test_error(self):
        """
        To test exception raise due to run time error
        """
        self.assertRaises(ZeroDivisionError, div, 0)


if __name__ == '__main__':
    unittest.main()
