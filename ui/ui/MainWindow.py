# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindowuuBGxL.ui'
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
        MainWindow.resize(1284, 883)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setStyleSheet(u"")
        self.action_information = QAction(MainWindow)
        self.action_information.setObjectName(u"action_information")
        self.action_cut = QAction(MainWindow)
        self.action_cut.setObjectName(u"action_cut")
        self.action_setup = QAction(MainWindow)
        self.action_setup.setObjectName(u"action_setup")
        self.action_imageViewer = QAction(MainWindow)
        self.action_imageViewer.setObjectName(u"action_imageViewer")
        self.action_dcm = QAction(MainWindow)
        self.action_dcm.setObjectName(u"action_dcm")
        self.action_help = QAction(MainWindow)
        self.action_help.setObjectName(u"action_help")
        self.action_import_image = QAction(MainWindow)
        self.action_import_image.setObjectName(u"action_import_image")
        self.action_import_data = QAction(MainWindow)
        self.action_import_data.setObjectName(u"action_import_data")
        self.action_import_model = QAction(MainWindow)
        self.action_import_model.setObjectName(u"action_import_model")
        self.action_result = QAction(MainWindow)
        self.action_result.setObjectName(u"action_result")
        self.action_export_labels = QAction(MainWindow)
        self.action_export_labels.setObjectName(u"action_export_labels")
        self.action_export_graph = QAction(MainWindow)
        self.action_export_graph.setObjectName(u"action_export_graph")
        self.action_import_label = QAction(MainWindow)
        self.action_import_label.setObjectName(u"action_import_label")
        self.action_import_feature = QAction(MainWindow)
        self.action_import_feature.setObjectName(u"action_import_feature")
        self.action_showFeature = QAction(MainWindow)
        self.action_showFeature.setObjectName(u"action_showFeature")
        self.action_import_dataset = QAction(MainWindow)
        self.action_import_dataset.setObjectName(u"action_import_dataset")
        self.action_extractFeature = QAction(MainWindow)
        self.action_extractFeature.setObjectName(u"action_extractFeature")
        self.action_normalize = QAction(MainWindow)
        self.action_normalize.setObjectName(u"action_normalize")
        self.action_sift_statistic = QAction(MainWindow)
        self.action_sift_statistic.setObjectName(u"action_sift_statistic")
        self.action_sift_correlation_coefficient = QAction(MainWindow)
        self.action_sift_correlation_coefficient.setObjectName(u"action_sift_correlation_coefficient")
        self.action_sift_lasso = QAction(MainWindow)
        self.action_sift_lasso.setObjectName(u"action_sift_lasso")
        self.action_predict = QAction(MainWindow)
        self.action_predict.setObjectName(u"action_predict")
        self.actionsdf = QAction(MainWindow)
        self.actionsdf.setObjectName(u"actionsdf")
        self.action_export_features = QAction(MainWindow)
        self.action_export_features.setObjectName(u"action_export_features")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.verticalLayout_3 = QVBoxLayout(self.tab)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.treeWidget = QTreeWidget(self.tab)
        __qtreewidgetitem = QTreeWidgetItem()
        __qtreewidgetitem.setText(0, u"1");
        self.treeWidget.setHeaderItem(__qtreewidgetitem)
        self.treeWidget.setObjectName(u"treeWidget")
        self.treeWidget.setStyleSheet(u"QTreeWidget#treeWidget{\n"
"	font: 16pt \"\u65b9\u6b63\u8212\u4f53\";\n"
"	color: rgb(6, 64, 255);\n"
"	\n"
"	selection-color: rgb(255, 0, 4);\n"
"}\n"
"\n"
"")

        self.verticalLayout_3.addWidget(self.treeWidget)

        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.verticalLayout_4 = QVBoxLayout(self.tab_2)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.tableWidget = QTableWidget(self.tab_2)
        self.tableWidget.setObjectName(u"tableWidget")

        self.verticalLayout_4.addWidget(self.tableWidget)

        self.tabWidget.addTab(self.tab_2, "")

        self.verticalLayout_2.addWidget(self.tabWidget)

        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_2.addWidget(self.line)

        self.tabWidget_2 = QTabWidget(self.centralwidget)
        self.tabWidget_2.setObjectName(u"tabWidget_2")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.verticalLayout_5 = QVBoxLayout(self.tab_3)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.image = showGraph(self.tab_3)
        self.image.setObjectName(u"image")
        self.image.setStyleSheet(u"QWidget#image{\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.horizontalLayout_4 = QHBoxLayout(self.image)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")

        self.verticalLayout_5.addWidget(self.image)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalSlider = QSlider(self.tab_3)
        self.horizontalSlider.setObjectName(u"horizontalSlider")
        self.horizontalSlider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_3.addWidget(self.horizontalSlider)

        self.spinBox = QSpinBox(self.tab_3)
        self.spinBox.setObjectName(u"spinBox")

        self.horizontalLayout_3.addWidget(self.spinBox)

        self.horizontalLayout_3.setStretch(0, 9)
        self.horizontalLayout_3.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_3)

        self.verticalLayout_5.setStretch(0, 9)
        self.verticalLayout_5.setStretch(1, 1)
        self.tabWidget_2.addTab(self.tab_3, "")
        self.tab_4 = QWidget()
        self.tab_4.setObjectName(u"tab_4")
        self.verticalLayout_6 = QVBoxLayout(self.tab_4)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.treeWidget_2 = QTreeWidget(self.tab_4)
        __qtreewidgetitem1 = QTreeWidgetItem()
        __qtreewidgetitem1.setText(0, u"1");
        self.treeWidget_2.setHeaderItem(__qtreewidgetitem1)
        self.treeWidget_2.setObjectName(u"treeWidget_2")

        self.horizontalLayout_5.addWidget(self.treeWidget_2)

        self.textEdit = QTextEdit(self.tab_4)
        self.textEdit.setObjectName(u"textEdit")

        self.horizontalLayout_5.addWidget(self.textEdit)


        self.verticalLayout_6.addLayout(self.horizontalLayout_5)

        self.tabWidget_2.addTab(self.tab_4, "")

        self.verticalLayout_2.addWidget(self.tabWidget_2)

        self.verticalLayout_2.setStretch(0, 5)
        self.verticalLayout_2.setStretch(2, 5)

        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout.addWidget(self.line_2)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.line_3 = QFrame(self.centralwidget)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_3)

        self.graph = showGraph(self.centralwidget)
        self.graph.setObjectName(u"graph")
        self.graph.setStyleSheet(u"showGraph#graph{\n"
"	background-color:rgba(255, 255, 255, 0)\n"
"}")
        self.horizontalLayout_2 = QHBoxLayout(self.graph)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.previous = QPushButton(self.graph)
        self.previous.setObjectName(u"previous")
        sizePolicy.setHeightForWidth(self.previous.sizePolicy().hasHeightForWidth())
        self.previous.setSizePolicy(sizePolicy)
        self.previous.setStyleSheet(u"")

        self.horizontalLayout_2.addWidget(self.previous)

        self.next = QPushButton(self.graph)
        self.next.setObjectName(u"next")
        sizePolicy.setHeightForWidth(self.next.sizePolicy().hasHeightForWidth())
        self.next.setSizePolicy(sizePolicy)
        self.next.setStyleSheet(u"")

        self.horizontalLayout_2.addWidget(self.next)


        self.verticalLayout.addWidget(self.graph)

        self.textBrowser_2 = QTextBrowser(self.centralwidget)
        self.textBrowser_2.setObjectName(u"textBrowser_2")
        self.textBrowser_2.setStyleSheet(u"QTextBrowser#textBrowser_2{\n"
"	font: 20pt \"\u65b9\u6b63\u8212\u4f53\";\n"
"	color: rgb(255, 0, 0);\n"
"}")

        self.verticalLayout.addWidget(self.textBrowser_2)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 6)
        self.verticalLayout.setStretch(2, 3)

        self.horizontalLayout.addLayout(self.verticalLayout)

        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.tableWidget_2 = QTableWidget(self.centralwidget)
        self.tableWidget_2.setObjectName(u"tableWidget_2")
        self.tableWidget_2.setStyleSheet(u"QTableWidget#tableWidget_2{\n"
"	color: rgb(0, 26, 255);\n"
"	selection-color: rgb(255, 8, 0);\n"
"	font: 75 11pt \"Agency FB\";\n"
"}")

        self.verticalLayout_8.addWidget(self.tableWidget_2)


        self.horizontalLayout.addLayout(self.verticalLayout_8)

        self.horizontalLayout.setStretch(0, 3)
        self.horizontalLayout.setStretch(2, 4)
        self.horizontalLayout.setStretch(3, 3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1284, 23))
        self.menubar.setStyleSheet(u"QMenuBar#menubar {\n"
"    color: rgb(239, 238, 236);\n"
"    background-color: rgb(67, 61, 59);\n"
"}\n"
"\n"
"QMenuBar#menubar::item {\n"
"    background-color: transparent;\n"
"}\n"
"\n"
"QMenuBar#menubar::item:hover {\n"
"    color: rgb(255, 0, 0);\n"
"    background-color: transparent;\n"
"}")
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        self.menu_3 = QMenu(self.menu)
        self.menu_3.setObjectName(u"menu_3")
        self.menu_4 = QMenu(self.menu)
        self.menu_4.setObjectName(u"menu_4")
        self.menu_2 = QMenu(self.menubar)
        self.menu_2.setObjectName(u"menu_2")
        self.menu_5 = QMenu(self.menubar)
        self.menu_5.setObjectName(u"menu_5")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        self.toolBar.setStyleSheet(u"")
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())
        self.menu.addAction(self.menu_3.menuAction())
        self.menu.addAction(self.menu_4.menuAction())
        self.menu.addAction(self.action_imageViewer)
        self.menu.addAction(self.action_showFeature)
        self.menu.addAction(self.action_dcm)
        self.menu.addAction(self.action_cut)
        self.menu.addAction(self.action_setup)
        self.menu_3.addAction(self.action_import_image)
        self.menu_3.addAction(self.action_import_data)
        self.menu_3.addAction(self.action_import_model)
        self.menu_3.addAction(self.action_import_label)
        self.menu_3.addAction(self.action_import_feature)
        self.menu_4.addAction(self.action_result)
        self.menu_4.addAction(self.action_export_labels)
        self.menu_4.addAction(self.action_export_graph)
        self.menu_4.addAction(self.action_export_features)
        self.menu_2.addAction(self.action_information)
        self.menu_2.addAction(self.action_help)
        self.menu_5.addAction(self.action_import_dataset)
        self.menu_5.addAction(self.action_extractFeature)
        self.menu_5.addSeparator()
        self.menu_5.addAction(self.action_normalize)
        self.menu_5.addAction(self.action_sift_statistic)
        self.menu_5.addAction(self.action_sift_correlation_coefficient)
        self.menu_5.addAction(self.action_sift_lasso)
        self.menu_5.addSeparator()
        self.menu_5.addAction(self.action_predict)
        self.toolBar.addAction(self.action_imageViewer)
        self.toolBar.addAction(self.action_dcm)
        self.toolBar.addAction(self.action_extractFeature)
        self.toolBar.addAction(self.action_showFeature)
        self.toolBar.addAction(self.action_cut)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.action_setup)
        self.toolBar.addAction(self.action_information)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_2.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u6df1\u5ea6\u5b66\u4e60\u8f6f\u4ef6", None))
        self.action_information.setText(QCoreApplication.translate("MainWindow", u"\u7248\u672c\u53f7\uff1a2.0", None))
        self.action_cut.setText(QCoreApplication.translate("MainWindow", u"\u9884\u6d4b", None))
        self.action_setup.setText(QCoreApplication.translate("MainWindow", u"\u8bbe\u7f6e", None))
        self.action_imageViewer.setText(QCoreApplication.translate("MainWindow", u"\u67e5\u770b\u56fe\u50cf", None))
        self.action_dcm.setText(QCoreApplication.translate("MainWindow", u"\u8bfb\u53d6dcm\u6587\u4ef6", None))
        self.action_help.setText(QCoreApplication.translate("MainWindow", u"\u5e2e\u52a9", None))
        self.action_import_image.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u50cf", None))
#if QT_CONFIG(tooltip)
        self.action_import_image.setToolTip(QCoreApplication.translate("MainWindow", u"\u5bfc\u5165\u56fe\u50cf", None))
#endif // QT_CONFIG(tooltip)
        self.action_import_data.setText(QCoreApplication.translate("MainWindow", u"\u6570\u636e\u96c6", None))
#if QT_CONFIG(tooltip)
        self.action_import_data.setToolTip(QCoreApplication.translate("MainWindow", u"\u5bfc\u5165\u6570\u636e\u96c6", None))
#endif // QT_CONFIG(tooltip)
        self.action_import_model.setText(QCoreApplication.translate("MainWindow", u"\u6a21\u578b", None))
#if QT_CONFIG(tooltip)
        self.action_import_model.setToolTip(QCoreApplication.translate("MainWindow", u"\u5bfc\u5165\u6a21\u578b", None))
#endif // QT_CONFIG(tooltip)
        self.action_result.setText(QCoreApplication.translate("MainWindow", u"\u7ed3\u679c", None))
#if QT_CONFIG(tooltip)
        self.action_result.setToolTip(QCoreApplication.translate("MainWindow", u"\u5bfc\u51fa\u7ed3\u679c", None))
#endif // QT_CONFIG(tooltip)
        self.action_export_labels.setText(QCoreApplication.translate("MainWindow", u"\u6807\u7b7e", None))
#if QT_CONFIG(tooltip)
        self.action_export_labels.setToolTip(QCoreApplication.translate("MainWindow", u"\u5bfc\u51fa\u6807\u7b7e", None))
#endif // QT_CONFIG(tooltip)
        self.action_export_graph.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u8868", None))
#if QT_CONFIG(tooltip)
        self.action_export_graph.setToolTip(QCoreApplication.translate("MainWindow", u"\u56fe\u8868", None))
#endif // QT_CONFIG(tooltip)
        self.action_import_label.setText(QCoreApplication.translate("MainWindow", u"\u6807\u7b7e", None))
#if QT_CONFIG(tooltip)
        self.action_import_label.setToolTip(QCoreApplication.translate("MainWindow", u"\u5bfc\u5165\u6807\u7b7e", None))
#endif // QT_CONFIG(tooltip)
        self.action_import_feature.setText(QCoreApplication.translate("MainWindow", u"\u7279\u5f81", None))
#if QT_CONFIG(tooltip)
        self.action_import_feature.setToolTip(QCoreApplication.translate("MainWindow", u"\u5bfc\u5165\u7279\u5f81", None))
#endif // QT_CONFIG(tooltip)
        self.action_showFeature.setText(QCoreApplication.translate("MainWindow", u"\u67e5\u770b\u7279\u5f81", None))
        self.action_import_dataset.setText(QCoreApplication.translate("MainWindow", u"\u5bfc\u5165\u6570\u636e\u96c6", None))
        self.action_extractFeature.setText(QCoreApplication.translate("MainWindow", u"\u7279\u5f81\u63d0\u53d6", None))
        self.action_normalize.setText(QCoreApplication.translate("MainWindow", u"\u6b63\u5219\u5316", None))
        self.action_sift_statistic.setText(QCoreApplication.translate("MainWindow", u"\u7edf\u8ba1\u68c0\u9a8c", None))
        self.action_sift_correlation_coefficient.setText(QCoreApplication.translate("MainWindow", u"\u76f8\u5173\u6027\u7cfb\u6570-\u7279\u5f81\u7b5b\u9009", None))
        self.action_sift_lasso.setText(QCoreApplication.translate("MainWindow", u"lasso\u56de\u5f52-\u7279\u5f81\u7b5b\u9009", None))
        self.action_predict.setText(QCoreApplication.translate("MainWindow", u"\u9884\u6d4b\u7ed3\u679c", None))
        self.actionsdf.setText(QCoreApplication.translate("MainWindow", u"sdf", None))
        self.action_export_features.setText(QCoreApplication.translate("MainWindow", u"\u7279\u5f81", None))
#if QT_CONFIG(tooltip)
        self.action_export_features.setToolTip(QCoreApplication.translate("MainWindow", u"\u5bfc\u51fa\u7279\u5f81", None))
#endif // QT_CONFIG(tooltip)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"\u6587\u4ef6", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"\u6570\u636e\u96c6", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), QCoreApplication.translate("MainWindow", u"\u56fe\u50cf", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), QCoreApplication.translate("MainWindow", u"\u6587\u4ef6\u4fe1\u606f", None))
        self.previous.setText("")
        self.next.setText("")
        self.textBrowser_2.setPlaceholderText("")
        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"\u5f00\u59cb", None))
        self.menu_3.setTitle(QCoreApplication.translate("MainWindow", u"\u5bfc\u5165", None))
        self.menu_4.setTitle(QCoreApplication.translate("MainWindow", u"\u5bfc\u51fa", None))
        self.menu_2.setTitle(QCoreApplication.translate("MainWindow", u"\u5173\u4e8e", None))
        self.menu_5.setTitle(QCoreApplication.translate("MainWindow", u"\u5f71\u50cf\u7ec4\u5b66\u6d41\u7a0b", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

