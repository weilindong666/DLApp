# -*- coding: UTF-8 -*-
'''
@Time    : 2022/9/20 15:00
@Author  : 魏林栋
@Site    : 
@File    : UIC.py
@Software: PyCharm
'''
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import os
from lib.Tools import Tools


class UIC(object):
    textBrowsers = {}
    image_engines = {}
    nowpath = os.path.dirname(__file__).replace('\\', '/')
    temPath = os.path.dirname(nowpath) + '/temp'
    image_none = nowpath + '/texture/none.png'
    image_next = nowpath + '/texture/next.png'
    image_previous = nowpath + '/texture/previous.png'
    image_background = nowpath + '/texture/background.png'
    icon_icon = nowpath + '/texture/1.png'
    tools = Tools()

    def __init__(self):
        pass

    def getCSS(self, ui, key, address=None):
        CSS = {'MainWindow': {
            'graph': 'QWidget#graph{' + f'background-image: url("{self.image_none if address is None else address}");' + 'background-size:10*10;background-position:center;}',
            'previous': "QPushButton#previous{background-color:rgba(255, 255, 255, 0);" + f'background-image:url("{self.image_previous}");' + "background-position:center;}",
            'image': "QWidget#image{" + f'background-image: url("{self.image_none if address is None else address}");' + "background-repeat:no-repeat;" + "background-position:center;}",
            'MainWindow': "#MainWindow{" + f'background-image:url("{self.image_background}")' + "}",
            'next': "QPushButton#next{background-color:rgba(255, 255, 255, 0);" + f'background-image:url("{self.image_next}");' + "background-position:center;}"},
        'reaDCMUI': {
            'tab': "QWidget#tab{" + f'background-image:url("{self.image_none if address is None else address}");' + "background-repeat:no-repeat;background-position:center;}"},
        'ImageViewerUI': {
            'image': "QWidget#image{" + f'background-image:url("{self.image_none if address is None else address}");' + "background-repeat:no-repeat;background-position:center;}"
        }}

        return CSS[ui][key]

    def dialogGetText(self, father):
        label, okPressed = QInputDialog.getText(
            father,
            "输入label",
            "label:",
            QLineEdit.Normal,
            "")
        if not okPressed:
            return None
        return label

    def warningWidget(self, father, info):
        QMessageBox.critical(
            father,
            '错误',
            info)