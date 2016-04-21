#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
 
# Create an PyQT4 application object.
a = QAppli:wcation(sys.argv)       
 
# The QWidget widget is the base class of all user interface objects in PyQt4.
# w = QWidget()
w = QMainWindow()

# Set window size. 
w.resize(320, 440)

# Set window title  
w.setWindowTitle("CheML") 


### Menu
# Create main menu
mainMenu = w.menuBar()
mainMenu.setNativeMenuBar(False)
fileMenu = mainMenu.addMenu('&File')

# Add exit button
exitButton = QAction(QIcon('exit24.png'), 'Exit', w)
exitButton.setShortcut('Ctrl+Q')
exitButton.setStatusTip('Exit application')
exitButton.triggered.connect(w.close)
fileMenu.addAction(exitButton)
###


### button
# Add a button
btn = QPushButton('Click me', w)
btn.setToolTip('triple action!')
#btn.clicked.connect(exit)
btn.resize(btn.sizeHint())
btn.move(100, 80)       

btn_about = QPushButton('About', w)
btn_about.setToolTip('About')
#btn.clicked.connect(exit)
btn_about.resize(btn_about.sizeHint())
btn_about.move(100, 120)       

# Create the actions 
@pyqtSlot()
def on_click_about():
    print('About clicked.')
    QMessageBox.about(w, "About", "CheML written in Python!")

@pyqtSlot()
def on_click():
    print('clicked')
    textbox1.setText("Button clicked.")
 
@pyqtSlot()
def on_press():
    print('pressed')
 
@pyqtSlot()
def on_release():
    print('released')

# connect the signals to the slots
btn_about.clicked.connect(on_click_about)
btn.clicked.connect(on_click)
btn.pressed.connect(on_press)
btn.released.connect(on_release)
###


### message box
# Show a message box
result = QMessageBox.question(w, 'Message', "Do you like Python?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

if result == QMessageBox.Yes:
    print 'Yes.'
    QMessageBox.information(w, "Message", "Python also likes you :*")
else:
    print 'No.'        
    QMessageBox.warning(w, "Message", "Ooops! You don't like Python :@")
# QMessageBox.critical(w, "Message", "Sorry! :~")
###


### textbox
# Create textbox
textbox1 = QLineEdit(w)
textbox1.move(20, 20)
textbox1.resize(280,40)

# Create textbox
textbox = QLineEdit(w)
textbox.move(20, 180)
textbox.resize(280,40)
###


### Table
table = QTableWidget(w)
tableItem = QTableWidgetItem()
# initiate table
table.setWindowTitle("QTableWidget")
table.resize(280, 180)
table.move(20, 240)
table.setRowCount(4)
table.setColumnCount(2)
# set label
table.setHorizontalHeaderLabels(QString("H1;H2;").split(";"))
table.setVerticalHeaderLabels(QString("V1;V2;V3;V4").split(";"))

# set data
table.setItem(0,0, QTableWidgetItem("(1,1)"))
table.setItem(0,1, QTableWidgetItem("(1,2)"))
table.setItem(1,0, QTableWidgetItem("(2,1)"))
table.setItem(1,1, QTableWidgetItem("(2,2)"))
table.setItem(2,0, QTableWidgetItem("(3,1)"))
table.setItem(2,1, QTableWidgetItem("(3,2)"))
table.setItem(3,0, QTableWidgetItem("(4,1)"))
table.setItem(3,1, QTableWidgetItem("(4,2)"))

# tooltip text
table.horizontalHeaderItem(0).setToolTip("Column 1 ")
table.horizontalHeaderItem(1).setToolTip("Column 2 ")

# on click function
def cellClick(row,col):
    print "Click on " + str(row) + " " + str(col)
table.cellClicked.connect(cellClick)
###

### Tabs
w1 = QMainWindow()
w1.resize(320, 440)
w1.move(400,20)
tabs = QTabWidget(w1)
    
# Create tabs
tab1 = QWidget() 
tab2 = QWidget()
tab3 = QWidget()
tab4 = QWidget()
 
# Resize width and height
tabs.resize(250, 350)
    
# Set layout of first tab
vBoxlayout = QVBoxLayout()
pushButton1 = QPushButton("Start")
pushButton2 = QPushButton("Settings")
pushButton3 = QPushButton("Stop")
vBoxlayout.addWidget(pushButton1)
vBoxlayout.addWidget(pushButton2)
vBoxlayout.addWidget(pushButton3)
tab2.setLayout(vBoxlayout)   
     
# Add tabs
tabs.addTab(tab1,"Tab 1")
tabs.addTab(tab2,"Tab 2")
tabs.addTab(tab3,"Tab 3")
tabs.addTab(tab4,"Tab 4") 
    
# Set title and show
tabs.setWindowTitle('PyQt QTabWidget @ pythonspot.com')
###


### Progress bar
class QProgBar(QProgressBar):
 
    value = 0
 
    @pyqtSlot()
    def increaseValue(progressBar):
        progressBar.setValue(progressBar.value)
        progressBar.value = progressBar.value+1 
 
# Create progressBar. 
bar = QProgBar(w1)
bar.resize(300,20)    
bar.setValue(0)
bar.move(10,400)

# create timer for progressBar
timer = QTimer()
bar.connect(timer,SIGNAL("timeout()"),bar,SLOT("increaseValue()"))
timer.start(200) 
###


### Pixmap

# Create widget
label = QLabel(w1)
pixmap = QPixmap(os.getcwd() + '/logo.png')
label.setPixmap(pixmap)
label.move(300,80)
label.resize(pixmap.width(),pixmap.height())
# w1.resize(pixmap.width(),pixmap.height())
###

### File Dialog
w2 = QWidget()
w2.resize(320, 240)
w2.setWindowTitle("File Dialog")
 
# Get filename using QFileDialog
filename = QFileDialog.getOpenFileName(w2, 'Open File', '/')
print filename
 
# print file contents
with open(filename, 'r') as f:
    print(f.read())


# Show window
w.show()
w1.show()
w2.show()

sys.exit(a.exec_())
