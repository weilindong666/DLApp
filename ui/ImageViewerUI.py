# -*- coding: UTF-8 -*-
'''
@Time    : 2022/11/7 19:57
@Author  : 魏林栋
@Site    : 
@File    : ImageViewerUI.py
@Software: PyCharm
'''
import cv2
from PySide2.QtWidgets import QFileDialog, QWidget, QMessageBox, QMainWindow
from PySide2.QtGui import *
from ui.UIC import UIC
import os
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor
from ui.ui.imageViewer import Ui_MainWindow


class ImageViewerUI(QMainWindow, QWidget, UIC):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setMouseTracking(True)
        self.ui.label.setMouseTracking(True)
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        # self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)  # 使窗口打开时处于最上面
        # self.ui.label.setPixmap(QPixmap(self.image_none))
        # self.ui.image.setStyleSheet(self.getCSS('ImageViewerUI', 'image'))
        self.ui.image.updateA(self.image_none)
        self.ui.action_openfolder.triggered.connect(self.selectDir)
        self.ui.action_openfile.triggered.connect(self.selectFile)
        self.ui.action_exit.triggered.connect(self.exit)
        self.ui.horizontalSlider.valueChanged.connect(self.sliderMove)
        self.ui.spinBox.valueChanged.connect(self.spinBoxChange)
        self.nowDir = os.path.dirname(self.nowpath)
        self.dir = os.path.join(self.nowDir)
        self.images = []
        self.openResizeEvent = False

    def selectDir(self):
        filePath = QFileDialog.getExistingDirectory(
            self,  # 父窗口对象
            "选择文件夹"  # 标题
        )
        if not filePath:
            return
        if os.path.isdir(filePath):  # 判断是否是文件夹
            self.dir = filePath
            for item in os.listdir(self.dir):  # 判断是否有图像
                if item.endswith('.png') or item.endswith('.jpg'):
                    self.images = []
                    self.findAllImages('.png')
                    self.findAllImages('.jpg')
                    self.initShow()
                    return
                elif item.endswith('.dcm'):
                    self.clear()
                    self.findAllImages('.dcm')
                    images = []
                    with ThreadPoolExecutor(max_workers=5) as pool:
                        for address in self.images:
                            name = os.path.splitext(os.path.basename(address))[0] + '.png'
                            save_path = os.path.join(self.temPath, name)
                            images.append(save_path)
                            pool.submit(lambda cxp: self.tools.dcmToPng(*cxp), (address, save_path))
                    self.images = images
                    self.dir = self.temPath
                    self.initShow()
                    return
                elif item.endswith('.nii'):
                    pass
        QMessageBox.critical(
            self,
            '错误',
            '请选择包含图像的文件夹')

    def findAllImages(self, type_):
        for item in os.listdir(self.dir):
            if item.endswith(type_):
                self.images.append(os.path.join(self.dir, item))

    def initShow(self):
        # 显示第一张图片
        # self.ui.image.setStyleSheet(self.getCSS('ImageViewerUI', 'image', self.images[0].replace('\\', '/')))
        self.ui.image.updateA(self.images[0].replace('\\', '/'), aspect='equal', if_RGB=True)
        # 将滑块和数字显示恢复到0
        self.ui.horizontalSlider.setValue(0)
        self.ui.spinBox.setValue(0)
        # 将滑块和数字显示最大值设置成图片数
        self.ui.horizontalSlider.setMaximum(len(self.images) - 1)
        self.ui.spinBox.setMaximum(len(self.images) - 1)

    def sliderMove(self):
        '''
        滑块滑动时数字和图片同时改变
        '''
        # 更改数字框（spinBox）数字
        self.ui.spinBox.setValue(self.ui.horizontalSlider.value())
        # 更换原始图像
        if(os.listdir(self.dir)):
            # self.ui.image.setStyleSheet(self.getCSS('ImageViewerUI', 'image', self.images[self.ui.horizontalSlider.value()].replace('\\', '/')))
            self.ui.image.updateA(self.images[self.ui.horizontalSlider.value()].replace('\\', '/'), aspect='equal', if_RGB=True)

    def spinBoxChange(self):
        self.ui.horizontalSlider.setValue(self.ui.spinBox.value())
        # 更换原始图像
        if (os.listdir(self.dir)):
            # self.ui.image.setStyleSheet(self.getCSS('ImageViewerUI', 'image', self.images[self.ui.spinBox.value()].replace('\\', '/')))
            self.ui.image.updateA(self.images[self.ui.spinBox.value()].replace('\\', '/'), aspect='equal', if_RGB=True)

    def selectFile(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "选择你要解析的文件",  # 标题
            self.nowDir,  # 起始目录
            "图片类型 (*.nii *.nii.gz *.dcm)"  # 选择类型过滤项，过滤内容在括号中
        )
        if not filePath:
            return
        self.dir = self.temPath
        self.nowDir, _ = os.path.split(filePath)
        if filePath.endswith('.nii'):
            self.clear()
            print('666666666该文件是nii文件6666666666')
        elif filePath.endswith('.nii.gz'):
            self.clear()
            self.images = self.tools.niigzToPng(filePath, self.temPath)
        elif filePath.endswith('.dcm'):
            self.clear()
            self.images = [f'{self.temPath}/0.png']
            self.tools.dcmToPng(filePath, f'{self.temPath}/0.png')
        self.initShow()

    def clear(self):
        self.images = []
        self.tools.deleteTemp(self.temPath)

    def exit(self):
        self.close()

    # def resizeEvent(self, event):
    #     if not self.openResizeEvent:
    #         self.openResizeEvent = True
    #     else:
    #         self.ui.label.setPixmap(QPixmap(self.image_none).scaled(self.ui.label.width(), self.ui.label.height()))

if __name__ == '__main__':
    from PySide2.QtWidgets import QApplication

    app = QApplication()

    window = ImageViewerUI()
    window.show()

    app.exec_()