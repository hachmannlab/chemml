#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt4 import QtGui, QtCore


class Button(QtGui.QPushButton):
    def mouseMoveEvent(self, e):
        if e.buttons() != QtCore.Qt.RightButton:
            return

        # write the relative cursor position to mime data
        mimeData = QtCore.QMimeData()
        # simple string with 'x,y'
        mimeData.setText('%d,%d' % (e.x(), e.y()))

        # let's make it fancy. we'll show a "ghost" of the button as we drag
        # grab the button to a pixmap
        pixmap = QtGui.QPixmap.grabWidget(self)

        # below makes the pixmap half transparent
        painter = QtGui.QPainter(pixmap)
        painter.setCompositionMode(painter.CompositionMode_DestinationIn)
        painter.fillRect(pixmap.rect(), QtGui.QColor(0, 0, 0, 127))
        painter.end()

        # make a QDrag
        drag = QtGui.QDrag(self)
        # put our MimeData
        drag.setMimeData(mimeData)
        # set its Pixmap
        drag.setPixmap(pixmap)
        # shift the Pixmap so that it coincides with the cursor position
        drag.setHotSpot(e.pos())

        # start the drag operation
        # exec_ will return the accepted action from dropEvent
        if drag.exec_(QtCore.Qt.CopyAction | QtCore.Qt.MoveAction) == QtCore.Qt.MoveAction:
            print 'moved'
        else:
            print 'copied'


    def mousePressEvent(self, e):
        QtGui.QPushButton.mousePressEvent(self, e)
        if e.button() == QtCore.Qt.LeftButton:
            print 'press'



class Example(QtGui.QWidget):
    def __init__(self):
        super(Example, self).__init__()
        self.initUI()


    def initUI(self):
        self.setAcceptDrops(True)

        button = Button('Button', self)
        button.move(100, 65)

        self.buttons = [button]

        self.setWindowTitle('Copy or Move')
        self.setGeometry(300, 300, 280, 150)


    def dragEnterEvent(self, e):
        e.accept()


    def dropEvent(self, e):
        # get the relative position from the mime data
        mime = e.mimeData().text()
        x, y = map(int, mime.split(','))

        if e.keyboardModifiers() & QtCore.Qt.ShiftModifier:
            # copy
            # so create a new button
            button = Button('Button', self)
            # move it to the position adjusted with the cursor position at drag
            button.move(e.pos()-QtCore.QPoint(x, y))
            # show it
            button.show()
            # store it
            self.buttons.append(button)
            # set the drop action as Copy
            e.setDropAction(QtCore.Qt.CopyAction)
        else:
            # move
            # so move the dragged button (i.e. event.source())
            e.source().move(e.pos()-QtCore.QPoint(x, y))
            # set the drop action as Move
            e.setDropAction(QtCore.Qt.MoveAction)
        # tell the QDrag we accepted it
        e.accept()



if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    ex.show()
    app.exec_()