class page1(object):
    def __init__(self, information):
        self.G = information
        self.G.graph = (edge for edge in self.G.graph if edge[0] != 0)
        print 'in page 1, graph:', self.G.graph
