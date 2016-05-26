class BASE(object):
    def __init__(self,a,b):
        self.graph = (10,100)
        self.a = a
        self.b = b

    def fit(self):
        self.graph += (1000,)

    def sum(self):
        self.Int()
        return self.a + self.b + self.x

class sub(BASE):
    # def __init__(self,a,b):
    #     super(sub,self).__init__(a,b)
    #     self.x = 7

    def add(self):
        self.graph += (self.sum(),)
        print self.graph
        print self.a , self.b, self.x

    def Int(self):
        self.x = 7

clf = sub(1,6)
# clf.Int()
clf.add()