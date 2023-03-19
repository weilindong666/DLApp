# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'imageViewerpGJkwX.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from ui.ui.showGraph import showGraph


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet(u"QMainWindow{\n"
"	\n"
"	background-color: qlineargradient(spread:repeat, x1:0, y1:1, x2:0, y2:0, stop:0.125 rgb(239, 238, 236), stop:0.840909 rgb(67, 61, 59));\n"
"	border-radius: 5px;\n"
"}")
        self.action_openfile = QAction(MainWindow)
        self.action_openfile.setObjectName(u"action_openfile")
        self.action_openfolder = QAction(MainWindow)
        self.action_openfolder.setObjectName(u"action_openfolder")
        self.action_exit = QAction(MainWindow)
        self.action_exit.setObjectName(u"action_exit")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.image = showGraph(self.centralwidget)
        self.image.setObjectName(u"image")
        self.image.setStyleSheet(u"QWidget#image{\n"
"	background-color:rgba(255, 255, 255, 0)\n"
"}")
        self.horizontalLayout_2 = QHBoxLayout(self.image)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label = QLabel(self.image)
        self.label.setObjectName(u"label")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_2.addWidget(self.label)


        self.verticalLayout.addWidget(self.image)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSlider = QSlider(self.centralwidget)
        self.horizontalSlider.setObjectName(u"horizontalSlider")
        self.horizontalSlider.setOrientation(Qt.Horizontal)

        self.horizontalLayout.addWidget(self.horizontalSlider)

        self.spinBox = QSpinBox(self.centralwidget)
        self.spinBox.setObjectName(u"spinBox")

        self.horizontalLayout.addWidget(self.spinBox)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.verticalLayout.setStretch(0, 9)
        self.verticalLayout.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 23))
        self.menubar.setStyleSheet(u"QMenuBar#menubar{\n"
"	background-color: rgb(67, 61, 59);\n"
"	color: rgb(255, 255, 255);\n"
"}")
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        self.menu.setStyleSheet(u"")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menu.menuAction())
        self.menu.addAction(self.action_openfile)
        self.menu.addAction(self.action_openfolder)
        self.menu.addAction(self.action_exit)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.action_openfile.setText(QCoreApplication.translate("MainWindow", u"\u6253\u5f00\u6587\u4ef6", None))
        self.action_openfolder.setText(QCoreApplication.translate("MainWindow", u"\u6253\u5f00\u6587\u4ef6\u5939", None))
        self.action_exit.setText(QCoreApplication.translate("MainWindow", u"\u9000\u51fa", None))
        self.label.setText("")
        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"\u5f00\u59cb", None))
    # retranslateUi

