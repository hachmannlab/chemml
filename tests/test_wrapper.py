import unittest
from cheml import wrapperRUN


class Testwrapper(unittest.TestCase):
    def test_template1(self):
        from cheml.notebooks.templates import template1
        script =  template1()
        wrapperRUN(script,'trash/test.out')



if __name__== '__main__':
    unittest.main()



