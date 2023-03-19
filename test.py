# -*- coding: UTF-8 -*-
'''
@Time    : 2022/10/28 17:32
@Author  : 魏林栋
@Site    : 
@File    : test.py
@Software: PyCharm
'''
import sys
from PySide2.QtWidgets import QWidget, QApplication
from PySide2 import QtUiTools, QtCore


class mouseMoveEvent(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = UiLoader().loadUi('./ui/ui/test.ui', self)
        self.ui.setMouseTracking(True)
        self.show()

    def mouseMoveEvent(self, e):
        print('666666')

class UiLoader(QtUiTools.QUiLoader):
    _baseinstance = None

    def createWidget(self, classname, parent=None, name=''):
        if parent is None and self._baseinstance is not None:
            widget = self._baseinstance
        else:
            widget = super(UiLoader, self).createWidget(classname, parent, name)
            if self._baseinstance is not None:
                setattr(self._baseinstance, name, widget)
        return widget

    def loadUi(self, uifile, baseinstance=None):
        self._baseinstance = baseinstance
        widget = self.load(uifile)
        QtCore.QMetaObject.connectSlotsByName(widget)
        return widget

class test:
    count = 0
    def __init__(self):
        test.count += 1


from PySide2.QtWidgets import QApplication, QWidget, QStyle, QToolButton
from PySide2.QtGui import QIcon
from PySide2.QtCore import Qt

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Custom Close Button')
        self.setGeometry(300, 300, 400, 300)

        # 创建一个 QToolButton，它将充当关闭按钮
        self.closeButton = QToolButton(self)
        self.closeButton.setGeometry(0, 0, 20, 20)

        # 设置关闭按钮的图标
        self.closeButton.setIcon(QIcon('close.png'))

        # 使用样式表更改关闭按钮的图像
        self.closeButton.setStyleSheet('QToolButton:hover {'
                                        'image: url(./ui/texture/none.png);'
                                        '}'
                                        'QToolButton:pressed {'
                                        'image: url(./ui/texture/none.png);'
                                        '}'
                                        'QToolButton::menu-indicator {'
                                        'image: none;'
                                        '}')

        # 连接关闭按钮的 clicked 信号到 quit() 槽函数
        self.closeButton.clicked.connect(self.quit)

    def quit(self):
        # 退出应用程序
        QApplication.quit()

if __name__ == '__main__':
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()

