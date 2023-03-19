# -*- coding: UTF-8 -*-
'''
@Time    : 2023/2/20 14:28
@Author  : 魏林栋
@Site    : 
@File    : Widget_2.py
@Software: PyCharm
'''
from PySide2.QtWidgets import QWidget
from PySide2.QtCore import Signal, Qt
from ui.UIC import UIC
from ui.ui.widget_2 import Ui_Form

class Widget_2(QWidget, UIC):
    def __init__(self):
        QWidget.__init__(self, parent=None)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        # self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)  # 使窗口打开时处于最上面
        self.ui.pushButton.clicked.connect(self.cancel)

    def updateProcessBar(self, percent):
        self.ui.progressBar.setValue(percent)

    def cancel(self):
        pass


if __name__ == '__main__':
    from PySide2.QtWidgets import QApplication

    app = QApplication()

    window = Widget_2()
    window.show()

    app.exec_()