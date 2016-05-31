from PyQt4.QtGui import *
from PyQt4.QtCore import *

rad = 5

class Node(QGraphicsEllipseItem):
    def __init__(self, path, index):
        super(Node, self).__init__(-rad, -rad, 2*rad, 2*rad)

        self.rad = rad
        self.path = path
        self.index = index

        self.setZValue(1)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setBrush(Qt.green)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            self.path.updateElement(self.index, value.toPointF())
        return QGraphicsEllipseItem.itemChange(self, change, value)


class Path(QGraphicsPathItem):
    def __init__(self, path, scene):
        super(Path, self).__init__(path)
        for i in xrange(path.elementCount()):
            node = Node(self, i)
            node.setPos(QPointF(path.elementAt(i)))
            scene.addItem(node)
        self.setPen(QPen(Qt.red, 1.75))        

    def updateElement(self, index, pos):
        path.setElementPositionAt(index, pos.x(), pos.y())
        self.setPath(path)


if __name__ == "__main__":

    app = QApplication([])

    path = QPainterPath()
    path.moveTo(0,0)
    path.cubicTo(-30, 70, 35, 115, 100, 100);
    path.lineTo(200, 100);
    path.cubicTo(200, 30, 150, -35, 60, -30);

    scene = QGraphicsScene()
    scene.addItem(Path(path, scene))

    view = QGraphicsView(scene)
    view.setRenderHint(QPainter.Antialiasing)
    view.resize(600, 400)
    view.show()
    app.exec_()
