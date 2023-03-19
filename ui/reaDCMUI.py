# -*- coding: UTF-8 -*-
'''
@Time    : 2023/1/16 16:31
@Author  : 魏林栋
@Site    : 
@File    : reaDCMUI.py
@Software: PyCharm
'''
from PySide2.QtWidgets import QFileDialog, QWidget, QTreeWidgetItem, QStyle, QMainWindow, QAbstractItemView, QMessageBox
from PySide2.QtGui import *
from PySide2.QtCore import Qt
import pydicom
import os
from ui.ui.reaDCM2 import Ui_MainWindow
from ui.UIC import UIC
import pandas as pd


class reaDCMUI(QMainWindow, QWidget, UIC):
    data_type = (pydicom.valuerep.IS, int, str, pydicom.valuerep.DSfloat, pydicom.valuerep.PersonName, bytes, type(None))
    def __init__(self):
        super(reaDCMUI, self).__init__()
        # QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowFlags(self.windowFlags())  # 使窗口打开时处于最上面
        self.ui.tab.setStyleSheet(self.getCSS('reaDCMUI', 'tab'))
        self.ui.action_open.triggered.connect(self.openFile)
        self.ui.action_export.triggered.connect(self.export)
        self.ui.treeWidget.clicked.connect(self.mytreeClicked)
        self.ui.treeWidget.header().setVisible(False)
        self.exportTreeItem = set()
        # self.ui.treeWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)  # 设置树控件可以多选
        self.openResizeEvent = 0
        self.size_label = [self.ui.label.width(), self.ui.label.height()]


    def openFile(self):
        filePath,_ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "选择你想查看的dcm文件",  # 标题
            "./",        # 起始目录
            "文件类型 (*.dcm)"  # 选择类型过滤项，过滤内容在括号中
        )
        if os.path.isfile(filePath):
            self.ui.treeWidget.clear()
            data = pydicom.read_file(filePath)
            self.updateTreeItem(data, self.ui.treeWidget)
            # 显示图像
            try:
                self.tools.dcmToPng(filePath, f'{self.temPath}/image.png')
                self.ui.tab.setStyleSheet(self.getCSS('reaDCMUI', 'tab', f'{self.temPath}/image.png'))
                # self.ui.label.setPixmap(QPixmap())
            except Exception:
                pass


    def updateTreeItem(self, data, parent):
        if isinstance(data, pydicom.sequence.Sequence):
            num = 1
            if len(data) <= 1:
                for item in data:
                    for tag in item.dir():
                        info = item.data_element(tag).value
                        if isinstance(info, self.data_type):
                            self._generate_item(parent, tag, info, 0)
                        else:
                            dir_item = self._generate_item(parent, tag, info, 1)
                            result = self.updateTreeItem(info, dir_item)
                            if result:
                                dir_item.setData(0, Qt.UserRole, result)
                                dir_item.setIcon(0, self.style().standardIcon(QStyle.SP_FileIcon))
                    num += 1
            else:
                for item in data:
                    for tag in item.dir():
                        info = item.data_element(tag).value
                        tag = tag + str(num)
                        if isinstance(info, self.data_type):
                            self._generate_item(parent, tag, info, 0)
                        else:
                            dir_item = self._generate_item(parent, tag, info, 1)
                            result = self.updateTreeItem(info, dir_item)
                            if result:
                                dir_item.setData(0, Qt.UserRole, result)
                                dir_item.setIcon(0, self.style().standardIcon(QStyle.SP_FileIcon))
                    num += 1
        elif isinstance(data, pydicom.multival.MultiValue):
            return data
        elif isinstance(data, pydicom.dataset.FileDataset):
            for tag in data.dir():  # i是tag
                info = data.data_element(tag).value
                # if tag == 'AcquisitionNumber':
                #     print(type(info))
                if isinstance(info, self.data_type):
                    self._generate_item(parent, tag, info, 0)
                else:
                    dir_item = self._generate_item(parent, tag, info, 1)
                    result = self.updateTreeItem(info, dir_item)
                    if result:
                        dir_item.setData(0, Qt.UserRole, result)
                        dir_item.setIcon(0, self.style().standardIcon(QStyle.SP_FileIcon))
        return None

    def _generate_item(self, parent, name, data, node_type):
        item = QTreeWidgetItem(parent, node_type)
        item.setText(0, name)
        # item.setCheckState(0, Qt.Unchecked)
        if len(str(data)) <= 200:
            item.setToolTip(0, str(data))
        else:
            item.setToolTip(0, str(data)[:200])
        item.setData(0, Qt.UserRole, data)
        item.setIcon(0, self.style().standardIcon(QStyle.SP_DirIcon if node_type == 1 else QStyle.SP_FileIcon))
        return item

    def mytreeClicked(self):
        item = self.ui.treeWidget.currentItem()
        if item is None:
            return
        line = '- '*30 + '\n\n'
        info = str(item.text(0)) + ':\n' + line + str(item.data(0, Qt.UserRole))
        self.ui.textEdit.setText(info)
        if item in self.exportTreeItem:
            item.setTextColor(0, 'black')
            self.exportTreeItem.remove(item)
        else:
            item.setTextColor(0, 'red')
            self.exportTreeItem.add(item)

    def export(self):
        filePath, _ = QFileDialog.getSaveFileName(
            self,  # 父窗口对象
            "保存文件",  # 标题
            self.nowpath,  # 起始目录
            "(*.csv)"  # 选择类型过滤项，过滤内容在括号中
        )
        if filePath and self.exportTreeItem:
            indexs = []
            values = []
            for item in self.exportTreeItem:
                item.setTextColor(0, 'black')
                indexs.append(str(item.text(0)))
                values.append(str(item.data(0, Qt.UserRole)))
            df = pd.DataFrame(values, index=indexs)
            try:
                df.to_csv(filePath)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    '错误',
                    str(e))
                return
            self.exportTreeItem = set()
            # 提示
            QMessageBox.information(
                self,
                '导出成功',
                '请继续下一步操作')

    # def resizeEvent(self, event):
    #     if self.openResizeEvent <= 1:
    #         self.openResizeEvent += 1
    #         self.size_label = [self.ui.label.width(), self.ui.label.height()]
    #     else:
    #         if (self.ui.label.height() - self.size_label[1]) <= 2:
    #             return
    #         self.ui.label.setPixmap(QPixmap(self.image_none).scaled(self.ui.label.width(), self.ui.label.height()))
    #         self.size_label = [self.ui.label.width(), self.ui.label.height()]


if __name__ == '__main__':
    from PySide2.QtWidgets import QApplication
    app = QApplication()

    window = reaDCMUI()
    window.show()

    app.exec_()