from .sct_utils import isfloat, islist, istuple, isnpdot, std_datetime_str

class Sklearn_Base(object):
    def __init__(self, function, parameters, send):
        self.function = function
        self.parameters = parameters
        self.send = send

class preprocessing(Base):
    pass

class regression(Base):
    pass

class classification(Base):
    pass

