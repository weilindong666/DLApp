# -*- coding: UTF-8 -*-
'''
@Time    : 2023/2/20 13:06
@Author  : 魏林栋
@Site    : 
@File    : showFeature.py
@Software: PyCharm
'''
from PySide2.QtWidgets import QWidget, QTableWidgetItem, QMainWindow, QAbstractItemView
from PySide2.QtGui import *
from ui.UIC import UIC
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from ui.ui.showFeature import Ui_MainWindow


class showFeature(QMainWindow, QWidget, UIC):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.ui.tableWidget.setRowCount(100)
        self.ui.tableWidget.setColumnCount(100)

        self.ui.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.features = None

    def showFeatures(self, features):
        self.features = features
        self.updateTableItems()

    def updateTableItems(self):
        self.ui.tableWidget.clear()
        self.ui.tableWidget.setRowCount(self.features.shape[0])
        self.ui.tableWidget.setColumnCount(self.features.shape[1])
        for indexC, columnName in enumerate(self.features.columns.tolist()):
            for indexR, indexName in enumerate(self.features.index.tolist()):
                data = self.features[columnName][indexName]
                if pd.isna(data):
                    continue
                try:
                    self.ui.tableWidget.setItem(indexR, indexC, QTableWidgetItem(str(data)))
                except Exception:
                    print(indexR, indexC, data)
        self.ui.tableWidget.setHorizontalHeaderLabels(self.features.columns.tolist())

if __name__ == '__main__':
    from PySide2.QtWidgets import QApplication

    app = QApplication()

    window = showFeature()
    window.show()

    app.exec_()