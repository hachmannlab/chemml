import unittest
from cheml import wrapperRUN


class Testwrapper(unittest.TestCase):
    def test_template1(self):
        from cheml.notebooks.templates import template1
        script =  template1()
        wrapperRUN(script,'trash/test.out')

    def test_template2(self):
        from cheml.notebooks.templates import template2
        script = template2()
        wrapperRUN(script, 'trash/test.out')

    def test_template3(self):
        from cheml.notebooks.templates import template3
        script = template3()
        wrapperRUN(script, 'trash/test.out')
    
    def test_template4(self):
        from cheml.notebooks.templates import template4
        script = template4()
        wrapperRUN(script, 'trash/test.out')
    
    def test_template5(self):
        from cheml.notebooks.templates import template5
        script = template5()
        wrapperRUN(script, 'trash/test.out')
    
    def test_template6(self):
        from cheml.notebooks.templates import template6
        script = template6()
        wrapperRUN(script, 'trash/test.out')
    
    def test_template7(self):
        from cheml.notebooks.templates import template7
        script = template7()
        wrapperRUN(script, 'trash/test.out')
    
    def test_template8(self):
        from cheml.notebooks.templates import template8
        script = template8()
        wrapperRUN(script, 'trash/test.out')
    
    def test_template9(self):
        from cheml.notebooks.templates import template9
        script = template9()
        wrapperRUN(script, 'trash/test.out')
    

    def test_template11(self):
        from cheml.notebooks.templates import template11
        script = template11()
        wrapperRUN(script, 'trash/test.out')

    def test_template12(self):
        from cheml.notebooks.templates import template12
        script = template12()
        wrapperRUN(script, 'trash/test.out')


if __name__== '__main__':
    unittest.main()



