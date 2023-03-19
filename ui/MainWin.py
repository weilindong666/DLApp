# -*- coding: UTF-8 -*-
'''
@Time    : 2022/9/17 10:56
@Author  : 魏林栋
@Site    : 
@File    : MainWin.py
@Software: PyCharm
'''
import numpy as np
from PySide2.QtWidgets import QFileDialog, QApplication, QWidget, QMainWindow, QTreeWidgetItem, QStyle, QInputDialog, \
    QMenu, QAction, QLineEdit, QTableWidgetItem, QHeaderView, QMessageBox, QProgressBar
from PySide2.QtGui import *
from PySide2.QtCore import Signal, QObject
from PySide2.QtUiTools import QUiLoader
from ui.UIC import UIC
from ui.Widget import Widget
from ui.ImageViewerUI import ImageViewerUI
from ui.reaDCMUI import reaDCMUI
from ui.showFeature import showFeature
from ui.Widget_2 import Widget_2
from Tool import A
import os
from threading import Thread
from ui.ui.MainWindow import Ui_MainWindow
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns
import matplotlib.pyplot as plt
from ui.ui.showGraph import showGraph


class MySignals(QObject):
    status_processbar_signal = Signal(QProgressBar, int)
    show_feature_signal = Signal()

class MainWin(QMainWindow, QWidget, UIC):

    def __init__(self):
        super(MainWin, self).__init__()
        # QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # self.ui.verticalLayout.addWidget(Ui_Form().setupUi(self))
        # 自定义信号
        self.mysignals = MySignals()
        self.mysignals.show_feature_signal.connect(self.showFeature)
        
        self.initImage()
        # self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)  # 使窗口打开时处于最上面
        self.ui.treeWidget.header().setVisible(False)
        self.ui.treeWidget_2.header().setVisible(False)
        # self.ui.treeWidget.clicked.connect(self.clickTreeWidget)
        self.ui.treeWidget.setContextMenuPolicy(Qt.CustomContextMenu)  # 打开右键菜单的策略
        self.ui.treeWidget.customContextMenuRequested.connect(self.treeWidgetItem_fun)  # 绑定事件

        self.widgetUI_1 = Widget()
        self.widgetUI_2 = Widget_2()
        self.ImageViewerUI = ImageViewerUI()
        self.reaDCMUI = reaDCMUI()
        self.showFeatureUI = showFeature()
        self.setWindowIcon(QIcon(self.icon_icon))
        # 导入
        self.ui.action_cut.triggered.connect(self.segmentation)
        self.ui.action_imageViewer.triggered.connect(self.openImageViewerUI)
        self.ui.action_dcm.triggered.connect(self.openreaDCMUI)
        self.ui.action_import_image.triggered.connect(self.importImage)
        self.ui.action_import_data.triggered.connect(self.importData)
        self.ui.action_import_model.triggered.connect(self.importModel)
        # 导出
        self.ui.action_export_features.triggered.connect(self.exportFeatures)
        self.ui.action_export_features.setIcon(QIcon(self.image_background))

        self.ui.horizontalSlider.valueChanged.connect(self.sliderMove)
        self.ui.spinBox.valueChanged.connect(self.spinBoxChange)
        self.widgetUI_1.signal_select_patient.connect(self.selectPatient)

        self.engines = {'textBrowser': [], 'image_engine': []}
        self.enginesInit()
        # 导入数据集
        self.ui.action_import_dataset.triggered.connect(self.import_dataset)

        self.tool = A()
        self.tools.deleteTemp(self.temPath)
        # 制作标签
        self.ui.tableWidget_2.setRowCount(1000)
        self.ui.tableWidget_2.setColumnCount(1000)
        self.ui.tableWidget_2.setHorizontalHeaderLabels(['ID', 'dataPath', 'maskPath', 'label'])
        self.ui.action_export_labels.triggered.connect(self.exportLabels)
        self.ui.action_import_label.triggered.connect(self.importLabels)
        self.ui.tableWidget_2.itemChanged.connect(self.tableItemChange)
        self.ui.tableWidget_2.setShowGrid(True)
        self.tableitemchangeswitch = True
        self.df = pd.DataFrame(columns=['ID', 'dataPath', 'maskPath', 'label'])
        self.images = []
        # 提取特征
        self.features = None
        self.progressBar = None
        self.mysignals.status_processbar_signal.connect(self.showProcessBar)
        self.ui.action_extractFeature.triggered.connect(self.extractFeature)
        self.ui.action_import_feature.triggered.connect(self.importFeature)
        self.ui.action_showFeature.triggered.connect(self.showFeature)
        # 正则化
        self.ui.action_normalize.triggered.connect(self.normalize)
        # 统计检验
        self.sift_statistic_pvalue = 0.05
        self.ui.action_sift_statistic.triggered.connect(self.sift_statistic)
        # 相关性筛选
        self.ui.action_sift_correlation_coefficient.triggered.connect(self.sift_correlation_coefficient)
        # lasso回归
        self.ui.action_sift_lasso.triggered.connect(self.sift_lasso)
        # 预测模型
        self.model = self.tools.model
        self.ui.action_predict.triggered.connect(self.predict)
        # 个性化参数设置
        self.corr = 'spearman'
        self.ui.action_setup.triggered.connect(self.setUp)
        # 图表
        self.graphs = []
        self.ui.previous.clicked.connect(self.graphPrevious)
        self.ui.next.clicked.connect(self.graphNext)

    def graphPrevious(self):
        if self.graphs:
            nowGraph = self.ui.graph.now_graph
            index = self.graphs.index(nowGraph)
            index -= 1
            if index < 0:
                index = len(self.graphs) - 1
            self.ui.graph.updateA(self.graphs[index])
            # self.ui.graph.paintEvent()

    def graphNext(self):
        if self.graphs:
            nowGraph = self.ui.graph.now_graph
            index = self.graphs.index(nowGraph)
            index += 1
            if index >= len(self.graphs):
                index = 0
            self.ui.graph.updateA(self.graphs[index])
            # self.ui.graph.paintEvent()

    def import_dataset(self):
        '''影像组学流程里的导入数据集'''
        filePath = QFileDialog.getExistingDirectory(
            self,  # 父窗口对象
            "选择文件夹"  # 标题
        )
        if filePath:
            files = os.listdir(filePath)
            if 'images' not in files:
                self.warningWidget(self, '没有images文件夹')
                return
            if 'masks' not in files:
                self.warningWidget(self, '没有masks文件夹')
                return
            # 导入images
            imagePath = filePath + '/images'
            maskPath = filePath + '/masks'
            dir_item = self._generate_item(self.ui.treeWidget, 'images', imagePath, 1)
            self.updateTreeItem(imagePath, dir_item)
            # 导入masks
            dir_item = self._generate_item(self.ui.treeWidget, 'masks', maskPath, 1)
            self.updateTreeItem(maskPath, dir_item)

    def importFeature(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "选择你要上传的特征列表",  # 标题
            self.nowpath,  # 起始目录
            "(*.csv)"  # 选择类型过滤项，过滤内容在括号中
        )
        if filePath:
            self.features = pd.read_csv(filePath)
            '''此处需要加一行判断，如果特征结果格式不符，进行报错！'''
            self.ui.statusbar.showMessage('特征导入完毕！')

    def showFeature(self):
        if self.features is not None:
            self.showFeatureUI.show()
            self.showFeatureUI.showFeatures(self.features)

    def exportFeatures(self):
        filePath, _ = QFileDialog.getSaveFileName(
            self,  # 父窗口对象
            "保存文件",  # 标题
            self.nowpath,  # 起始目录
            "(*.csv)"  # 选择类型过滤项，过滤内容在括号中
        )
        if filePath and self.features is not None:
            self.features.to_csv(filePath, index=False)

    def extractFeature(self):
        if self.df.empty:
            self.warningWidget(self, '还没有导入数据集')
            return
        self.features = pd.DataFrame()
        self.tools.createExtractor()
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.resize(100, 20)
        self.ui.statusbar.addWidget(self.progressBar)
        thread = Thread(target=self.extractFeatureThread)
        thread.start()

    def extractFeatureThread(self):
        lenth = len(self.df.index.tolist())
        self.ui.textBrowser_2.append(f'开始提取特征,一共有{lenth}个数据\n')
        for index in self.df.index.tolist():
            precrnt = int(((index + 1) / lenth) * 100)
            if index == 0:
                line, column = self.tools.featurExtract(self.df, index, True)
                self.features = self.features.reindex(columns=column, fill_value='')
                self.features.loc[index] = line
                self.mysignals.status_processbar_signal.emit(self.progressBar, precrnt)
                continue
            line = self.tools.featurExtract(self.df, index)
            self.features.loc[index] = line
            self.mysignals.status_processbar_signal.emit(self.progressBar, precrnt)
        self.progressBar.deleteLater()
        self.mysignals.show_feature_signal.emit()
        self.ui.textBrowser_2.append(f'特征提取结束! 一共提取到{len(self.features.columns)}个特征\n')
        # return
        # self.ui.statusbar.removeWidget(self.progressBar)  # 该方法会导致程序闪退

    def showProcessBar(self, processBar, num):
        processBar.setValue(num)
        return

    def normalize(self):
        self.ui.textBrowser_2.append(f'开始对特征数据进行正则化\n')
        self.features = self.tools.normalize_df(self.features, not_norm=['label', 'ID'])
        self.mysignals.show_feature_signal.emit()
        self.ui.textBrowser_2.append(f'正则化结束\n')

    def sift_statistic(self):
        self.ui.textBrowser_2.append(f'开始对特征数据进行统计检验\n')
        columns = [c for c in self.features.columns.tolist() if c not in ['ID', 'label']]
        features = self.tools.clinic_statistic(self.features, columns)

        features[['pvalue']] = features[['pvalue']].applymap(lambda x: float(str(x)[1:]))
        features[['group']] = features[['feature_name']].applymap(lambda x: x.split('_')[1])
        features = features[['feature_name', 'pvalue', 'group']]

        # 显示图像
        g = sns.catplot(x="group", y="pvalue", data=features, kind="violin")
        g.fig.set_size_inches(15, 10)
        sns.stripplot(x="group", y="pvalue", data=features, ax=g.ax, color='black')
        plt.savefig('./result/feature_stats.png', bbox_inches='tight')
        self.ui.graph.updateA('./result/feature_stats.png')
        self.graphs.append('./result/feature_stats.png')

        sel_feature = list(features[features['pvalue'] < self.sift_statistic_pvalue]['feature_name']) + ['label']
        self.features = self.features[sel_feature]

        self.mysignals.show_feature_signal.emit()  # 弹出特征列表
        self.ui.textBrowser_2.append(f'统计检验结束！\n')

    def sift_correlation_coefficient(self):
        '''利用相关性系数进行筛选！'''
        self.ui.textBrowser_2.append(f'开始对特征数据进行相关性筛选！\n')
        # pearson_corr = self.features.corr('pearson')  # 皮尔逊相关系数
        # kendall_corr = self.features.corr('kendall')  # 肯德尔相关性系数
        corr = self.features.corr(self.corr)  # 斯皮尔曼相关性系数

        sns.heatmap(corr, annot=False, cmap='YlGnBu', cbar=False, fmt='.3f')
        plt.title('spearman correlation coefficient')
        plt.savefig('./result/feature_corr.png', bbox_inches='tight')
        self.graphs.append('./result/feature_corr.png')
        pp = sns.clustermap(corr, linewidths=.5, figsize=(50.0, 40.0), cmap='YlGnBu')
        plt.setp(pp.ax_heatmap.get_yticklabels(), rotation=0)
        plt.title('feature cluster')
        plt.savefig('./result/feature_cluster.png', bbox_inches='tight')
        self.graphs.append('./result/feature_cluster.png')
        self.ui.graph.updateA('./result/feature_cluster.png')
        # 筛选
        sel_features = self.tools.select_feature(corr, threshold=0.9, topn=10, verbose=False)
        self.features = self.features[sel_features]

        self.mysignals.show_feature_signal.emit()  # 弹出特征列表
        self.ui.textBrowser_2.append(f'筛选完毕\n')

    def sift_lasso(self):
        self.ui.textBrowser_2.append(f'开始通过lasso回归对特征数据进行筛选！\n')
        y_data = self.features[['label']]
        X_data = self.features.drop(['label'], axis=1)
        # lasso回归
        alpha, images = self.tools.lasso_cv_coefs(X_data, y_data)  # 得出的是lasso回归(交叉验证)的惩罚量
        self.graphs += images

        # 特征筛选
        self.features, image = self.tools.lassoSiftFeature(self.features, alpha)
        self.graphs.append(image)
        self.ui.graph.updateA(image)
        self.ui.textBrowser_2.append(f'筛选完毕！\n')

    def predict(self):
        self.ui.textBrowser_2.append(f'开始预测！\n')
        y_data = self.features[['label']]
        X_data = self.features.drop(['label'], axis=1)
        images, result = self.tools.predict(X_data, y_data)
        self.graphs += images
        self.ui.graph.updateA(images[0])
        for key1, value1 in result.items():
            self.ui.textBrowser_2.append(f'{key1}数据集的结果\n')
            for key2, value2 in value1.items():
                self.ui.textBrowser_2.append(f'{key2}:{value2}\n')

    def tableItemChange(self, item):
        # 首先确定这个单元格的（row和column）有没有在表格里
        row = item.row()
        column = item.column()
        data = item.text()
        if self.tableitemchangeswitch:
            if row not in self.df.index.tolist():
                self.df = self.df.reindex(index=range(row+1))
            if column >= 4 and column not in self.df.columns.tolist():
                self.df = self.df.reindex(columns=['ID', 'dataPath', 'maskPath', 'label'] + list(range(4,column+1)), fill_value='')
            self.df.at[row, self.df.columns[column]] = data
        else:
            self.tableitemchangeswitch = True

    def setUp(self):
        '''设置，通过设置可以更改影像组学的setting，也可以更改相关性系数里的方法等等'''
        pass

    def treeWidgetItem_fun(self, pos):
        item = self.ui.treeWidget.currentItem()
        item1 = self.ui.treeWidget.itemAt(pos)

        if item != None and item1 != None:
            popMenu = QMenu()
            popMenu.addAction(QAction(u'生成数据标签', self))
            popMenu.addAction(QAction(u'生成mask标签', self))
            popMenu.addAction(QAction(u'从标签页中移除', self))
            popMenu.addAction(QAction(u'查看图像', self))
            popMenu.triggered[QAction].connect(self.processtrigger)
            popMenu.exec_(QCursor.pos())

    def processtrigger(self, q):
        menu = {'生成数据标签': 'dataPath', '生成mask标签': 'maskPath'}
        item = self.ui.treeWidget.currentItem()
        if q.text() == '查看图像':
            self.images = []
            self.tools.deleteTemp(self.temPath)
            self.images = self.tools.niigzToPng(item.data(0, Qt.UserRole), self.temPath)
            self.initShow()
        elif q.text() in menu.keys():
            # 如果是文件，点击后名称其绝对地址和名称加入表格，再次点击从表格中删除
            # 如果是文件夹，把其内的文件绝对地址和名称加入表格，其内的文件夹忽略，再次点击从表格中删除
            if item.data(1, Qt.UserRole) == 1:
                label = self.dialogGetText(self)
                if label is not None:
                    for num in range(item.childCount()):
                        tableItem = item.child(num)
                        if tableItem.text(0) not in self.df.ID.tolist():
                            self.addData(tableItem.text(0), menu[q.text()], tableItem.data(0, Qt.UserRole), label)
                            continue
                        self.addData(tableItem.text(0), menu[q.text()], tableItem.data(0, Qt.UserRole))
            else:
                if item.text(0) not in self.df.ID.tolist():
                    label = self.dialogGetText(self)
                    if label is not None:
                        self.addData(item.text(0), menu[q.text()], item.data(0, Qt.UserRole), label)
                    return
                self.addData(item.text(0), menu[q.text()], item.data(0, Qt.UserRole))
        elif q.text() == '从标签页中移除':
            if item.data(1, Qt.UserRole) == 1:
                for num in range(item.childCount()):
                    tableItem = item.child(num)
                    if tableItem.text(0) not in self.df.ID.tolist():
                        # 警告
                        self.warningWidget(self, '标签页中没有此项')
                        return
                    index = self.df.ID.tolist().index(tableItem.text(0))
                    self.ui.tableWidget_2.removeRow(index)
                    self.df.drop(index, axis=0, inplace=True)
                    self.df.reset_index(drop=True, inplace=True)
            else:
                if item.text(0) not in self.df.ID.tolist():
                    # 警告
                    self.warningWidget(self, '标签页中没有此项')
                    return
                index = self.df.ID.tolist().index(item.text(0))
                self.ui.tableWidget_2.removeRow(index)
                self.df.drop(index, axis=0, inplace=True)
                self.df.reset_index(drop=True, inplace=True)

    def addData(self, ID, column, data, label=''):
        info = []
        if ID not in self.df.ID.tolist():
            if column == 'dataPath':
                info = {'ID':ID, 'dataPath':data, 'maskPath':'', 'label':label}
            elif column == 'maskPath':
                info = {'ID':ID, 'dataPath':'', 'maskPath':data, 'label':label}
            new_df = pd.DataFrame(info, index=[0])
            self.df = self.df.append(new_df, ignore_index=True)
            self.addTableItem(ID, column, label, addID=1)
        else:
            index = self.df[self.df.ID == ID].index
            if column == 'dataPath':
                info = np.array([ID, data, self.df.maskPath[index].values[0], self.df.label[index].values[0]], dtype=object)
            elif column == 'maskPath':
                info = np.array([ID, self.df.dataPath[index].values[0], data, self.df.label[index].values[0]], dtype=object)
            self.df.loc[index] = info
            self.addTableItem(ID, column)

    def addTableItem(self, ID, column, label=None, addID=None):
        x = list(self.df.ID).index(ID)
        y = list(self.df.columns).index(column)
        itemName = str(self.df[column][self.df[self.df.ID == ID].index].values[0])
        if addID:
            y_ = list(self.df.columns).index('ID')
            self.tableitemchangeswitch = False
            self.ui.tableWidget_2.setItem(x, y_, QTableWidgetItem(ID))
        self.tableitemchangeswitch = False
        self.ui.tableWidget_2.setItem(x, y, QTableWidgetItem(itemName))
        if label:
            y_ = list(self.df.columns).index('label')
            self.tableitemchangeswitch = False
            self.ui.tableWidget_2.setItem(x, y_, QTableWidgetItem(label))

    def exportLabels(self):
        filePath, _ = QFileDialog.getSaveFileName(
            self,  # 父窗口对象
            "保存文件",  # 标题
            self.nowpath,  # 起始目录
            "(*.csv)"  # 选择类型过滤项，过滤内容在括号中
        )
        if filePath:
            self.df.to_csv(filePath, index=False)
            QMessageBox.information(
                self,
                '保存成功',
                '标签保存成功！')

    def importLabels(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "选择你要上传的文件",  # 标题
            self.nowpath,  # 起始目录
            "(*.csv)"  # 选择类型过滤项，过滤内容在括号中
        )
        if filePath:
            df = pd.read_csv(filePath)
            self.df = self.df.append(df, ignore_index=True)
            self.df = self.df.drop_duplicates(subset=['ID'])
            self.updateTableItems()

    def updateTableItems(self):
        for indexC, columnName in enumerate(self.df.columns.tolist()):
            for indexR, indexName in enumerate(self.df.index.tolist()):
                data = self.df[columnName][indexName]
                if pd.isna(data):
                    continue
                try:
                    self.ui.tableWidget_2.setItem(indexR, indexC, QTableWidgetItem(str(data)))
                    self.tableitemchangeswitch = False
                except Exception:
                    print(indexR, indexC, data)

    def initShow(self):
        '''这里的显示图像方法可以优化，参考 showGraph.py 文件里的方法'''
        # 显示第一张图片
        # self.ui.image.setStyleSheet(self.getCSS('MainWindow', 'image', self.images[0]))
        self.ui.image.updateA(self.images[0], aspect='equal', if_RGB=True)
        # 将滑块和数字显示恢复到0
        self.ui.horizontalSlider.setValue(0)
        self.ui.spinBox.setValue(0)
        # 将滑块和数字显示最大值设置成图片数
        self.ui.horizontalSlider.setMaximum(len(self.images) - 1)
        self.ui.spinBox.setMaximum(len(self.images) - 1)

    def initImage(self):
        self.ui.next.setStyleSheet(self.getCSS('MainWindow', 'next'))
        self.setStyleSheet(self.getCSS('MainWindow', 'MainWindow'))
        # self.ui.image.setStyleSheet(self.getCSS('MainWindow', 'image'))
        self.ui.image.updateA(self.image_none)
        self.ui.previous.setStyleSheet(self.getCSS('MainWindow', 'previous'))
        # self.ui.graph.setStyleSheet(self.getCSS('MainWindow', 'graph'))
        self.ui.graph.updateA(self.image_none)

    def enginesInit(self):
        '''
        初始化引擎，方便简化代码
        :return:
        '''
        # self.engines['textBrowser'].append(self.ui.textBrowser_1)
        self.engines['textBrowser'].append(self.ui.textBrowser_2)
        # self.engines['image_engine'].append(self.ui.original_image)
        # self.engines['image_engine'].append(self.ui.processed_image)

    def sliderMove(self):
        # 更改数字框（spinBox）数字
        self.ui.spinBox.setValue(self.ui.horizontalSlider.value())
        # 更换原始图像
        if(os.listdir(self.temPath)):
            # self.ui.image.setStyleSheet(self.getCSS('MainWindow', 'image', self.images[self.ui.horizontalSlider.value()]))
            self.ui.image.updateA(self.images[self.ui.horizontalSlider.value()], aspect='equal', if_RGB=True)

    def spinBoxChange(self):
        self.ui.horizontalSlider.setValue(self.ui.spinBox.value())
        # 更换原始图像
        if (os.listdir(self.temPath)):
            # self.ui.image.setStyleSheet(self.getCSS('MainWindow', 'image', self.images[self.ui.spinBox.value()]))
            self.ui.image.updateA(self.images[self.ui.spinBox.value()], aspect='equal', if_RGB=True)

    def selectPatientThread(self, patient):
        print(f'子线程开始,病人是{patient}')
        self.tool.preserveImage()
        # 初始化原图像显示区域
        self.showImage(0, './temp/0.png')
        self.ui.horizontalSlider.setValue(0)
        self.ui.spinBox.setValue(0)
        # 设置滑块的最大值
        self.ui.horizontalSlider.setMaximum(len(os.listdir('./temp')) - 1)
        self.ui.spinBox.setMaximum(len(os.listdir('./temp')) - 1)
        self.ui.statusbar.showMessage(f'')
        self.tool = self.tool
        print('子线程结束')

    def selectPatient(self, patient):
        '''
        :param patient: 回调函数，返回的使病人的编号
        '''
        self.ui.statusbar.showMessage(f'正在导入病人{patient}的CT图像')
        self.tool.selectPatient(patient)
        thread = Thread(target=self.selectPatientThread, args=(patient,), name='Thread-1')
        thread.start()
        print('主线程结束')

    def onOpen(self):
        '''
        打开文件夹
        :return:
        '''
        pass

    def segmentationThread(self):
        fig, image_original, y_true, ArrayDicom_mask_ = self.tool.read_dataset()
        # 图像矩阵归一化
        image = self.tool.normalize_hu(image_original)
        # 切割并储存图像到文件夹,返回矩阵中心点坐标
        save_dir = r"C:\Users\Administrator\Desktop\DLApp\data"

        x_, y_, z_, z_z, ii = self.tool.save_data2img(image, save_dir, ArrayDicom_mask_)
        # 文件储存的位置
        read_dir = r'C:\Users\Administrator\Desktop\DLApp\data'
        # 得到预测的结果和图像
        fig, y = self.tool.predict_image(read_dir, x_, y_, z_, z_z, ii)
        # 输出指标
        AUC, PPV, TPR, ACC, fig, ASD, DICE, HSD = self.tool.evaluate(image, y, y_true, x_, y_)

        Result = []
        Evaluate = []
        Result.append('AUC： ' + str(AUC) + '\n')
        Result.append('PPV： ' + str(PPV) + '\n')
        Result.append('TPR： ' + str(TPR) + '\n')
        Result.append('ACC： ' + str(ACC) + '\n')
        Evaluate.append('ASD： ' + str(ACC) + '\n')
        Evaluate.append('DICE： ' + str(ACC) + '\n')
        Evaluate.append('HSD： ' + str(ACC) + '\n')
        self.showResult(1, Evaluate)
        self.showResult(0, Result)
        self.showImage(1, './result.png')
        print(AUC, PPV, TPR, ACC, fig, ASD, DICE, HSD)

    def segmentation(self):
        '''
        自动分割按钮函数
        :return:
        '''
        thread = Thread(target=self.segmentationThread, name='Thread-2')
        thread.start()
        self.ui.statusbar.showMessage(f'模型正在预测！')

    def openImageViewerUI(self):
        '''
        显示图像查看界面
        '''
        self.ImageViewerUI.show()

    def openreaDCMUI(self):
        self.reaDCMUI.show()

    def importImage(self):
        '''
        输入按钮函数
        :return:
        '''
        self.widgetUI_1.show()
        pass

    def importModel(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "选择你要上传的模型",  # 标题
            "./",  # 起始目录
            "模型类型 (*.h5)"  # 选择类型过滤项，过滤内容在括号中
        )
        if filePath:
            self.tool.loadModel(filePath)

    def showResult(self, slot, linedit):
        '''
        :param slot: 有两个槽位，第一个是展示结果，第二个是评价指标
        :param results: 要显示的数据，这里的文本框是逐行展示的，所以数据也要逐行进行添加
        :return:
        '''
        for line in linedit:
            self.engines['textBrowser'][slot].append(line)

    def showImage(self, slot, image_address):
        '''
        :param slot: 图像显示位置，0是左边原图像，1是右边处理后的图像
        :param image_address: 图像地址
        :return:
        '''
        self.engines['image_engine'][slot].setPixmap(QPixmap(image_address))

    def importData(self):
        filePath = QFileDialog.getExistingDirectory(
            self,  # 父窗口对象
            "选择文件夹"  # 标题
        )
        if filePath:
            file = os.path.basename(filePath)
            dir_item = self._generate_item(self.ui.treeWidget, file, filePath, 1)
            self.updateTreeItem(filePath, dir_item)


    def updateTreeItem(self, address, parent):
        for file in os.listdir(address):
            path = address + '/' + file
            if os.path.isdir(path):
                # dir_item = self._generate_item(parent, file, path, 1)
                self.updateTreeItem(path, parent)
            else:
                if file.endswith('.nii.gz'):
                    self._generate_item(parent, file, path, 0)

    def _generate_item(self, parent, name, data, node_type):
        item = QTreeWidgetItem(parent, node_type)
        item.setText(0, name)
        item.setData(0, Qt.UserRole, data)
        item.setData(1, Qt.UserRole, node_type)
        item.setIcon(0, self.style().standardIcon(QStyle.SP_DirIcon if node_type == 1 else QStyle.SP_DirHomeIcon))
        # item.set
        return item

    def clickTreeWidget(self):
        # 如果是文件，点击后名称其绝对地址和名称加入表格，再次点击从表格中删除
        # 如果是文件夹，把其内的文件绝对地址和名称加入表格，其内的文件夹忽略，再次点击从表格中删除

        item = self.ui.treeWidget.currentItem()
        if item in self.exportTreeItem:
            item.setTextColor(0, 'black')
            self.exportTreeItem.remove(item)
        else:
            item.setTextColor(0, 'red')
            self.exportTreeItem.add(item)


if __name__ == '__main__':
    app = QApplication()
    obj = MainWin()
    obj.show()
    app.exec_()
