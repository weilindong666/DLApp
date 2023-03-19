# -*- coding: UTF-8 -*-
'''
@Time    : 2023/3/12 13:47
@Author  : 魏林栋
@Site    : 
@File    : showGraph.py
@Software: PyCharm
'''
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.image as img

class showGraph(FigureCanvas):
    def __init__(self, parent=None):
        fig = plt.figure(dpi=100, tight_layout=True, facecolor='black')
        fig.patch.set_alpha(0.0)
        FigureCanvas.__init__(self, fig)
        self.axes = fig.add_subplot(111)
        # self.axes.axis('off')
        self.axes.spines['top'].set_visible(False)  # 去掉绘图时上面的横线
        self.axes.spines['right'].set_visible(False)  # 去掉绘图时右面的横线
        self.axes.spines['left'].set_visible(False)  # 去掉绘图时左面的横线
        self.axes.spines['bottom'].set_visible(False)  # 去掉绘图时下面的横线
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        self.now_graph = None
        # self.updateA('./result/feature_stats.png')

    def updateA(self, address, aspect='auto', if_RGB=False):
        self.now_graph = address
        self.axes.clear()
        image = img.imread(address)
        if if_RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.axes.imshow(image, extent=(0, 1000, 0, 1000), aspect=aspect)
        self.draw()