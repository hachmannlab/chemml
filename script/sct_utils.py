import numpy as np

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False		

def islist(value):
    text = str(value)
    if text[0]=='[' and text[len(text)-1]==']':
        return True
    else:
        return False
        
def istuple(value):
    text = str(value)
    if text[0]=='(' and text[len(text)-1]==')':
        return True
    else:
        return False

def isnpdot(value):
    text = str(value)
    if text[0:3]=='np.':
        return True
    else:
        return False
