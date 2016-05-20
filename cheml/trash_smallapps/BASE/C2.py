class page2(object):
    def __init__(self, information):
        self.G = information
        self.G.graph = (edge for edge in self.G.graph if edge[0] != 2)
        print 'in page 2, graph:', self.G.graph

