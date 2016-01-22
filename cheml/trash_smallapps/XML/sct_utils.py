import numpy as np
import datetime
import sys

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

def std_datetime_str(mode='datetime'):
    """(std_time_str):
        This function gives out the formatted time as a standard string, i.e., YYYY-MM-DD hh:mm:ss.
    """
    if mode == 'datetime':
        return str(datetime.datetime.now())[:19]
    elif mode == 'date':
        return str(datetime.datetime.now())[:10]
    elif mode == 'time':
        return str(datetime.datetime.now())[11:19]
    elif mode == 'datetime_ms':
        return str(datetime.datetime.now())
    elif mode == 'time_ms':
        return str(datetime.datetime.now())[11:]
    else:
        sys.exit("Invalid mode!")
