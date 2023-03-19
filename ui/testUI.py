# -*- coding: UTF-8 -*-
'''
@Time    : 2023/3/17 13:35
@Author  : 魏林栋
@Site    : 
@File    : testUI.py
@Software: PyCharm
'''
from PySide2.QtWidgets import QWidget, QApplication, QMainWindow
from ui.ui.ui_test import Ui_MainWindow

class TestUI(QMainWindow):
    def __init__(self):
        super(TestUI, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

    def showGraph(self):
        pass


if __name__ == '__main__':
    app = QApplication()
    obj = TestUI()
    obj.show()
    app.exec_()