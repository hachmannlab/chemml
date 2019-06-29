from __future__ import print_function

import inspect

def value(string):
    if string in (None ,False, True):
        return True
    else:
        return False


# example
import pandas as pd
a = inspect.getargspec(pd.read_excel).args
d = inspect.getargspec(pd.read_excel).defaults
d = ['* required'] + list(d)
assert len(a)==len(d)
for i in range(len(a)):
    v = d[i]
    if value(v):
        print( "%s = Parameter('%s', %s)" % (a[i], a[i], str(d[i])))
    else:
        print( "%s = Parameter('%s', '%s')" % (a[i], a[i], str(d[i])))

