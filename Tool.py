# -*- coding: UTF-8 -*-
'''
@Time    : 2022/10/29 18:17
@Author  : 魏林栋
@Site    : 
@File    : Tool.py
@Software: PyCharm
'''
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
import cv2
import pylidc as pl
from pylidc.utils import consensus
from keras.models import load_model
import os
from keras.preprocessing.image import img_to_array
import SimpleITK as sitk
from pycm import ConfusionMatrix
from itertools import chain
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from concurrent.futures import ThreadPoolExecutor

class A:

    def __init__(self):
        self.example = 'LIDC-IDRI-0001'  # 默认第一个病人
        self.scan = None
        self.nods = None
        self.vol = None
        self.model = None
        self.clearTemp('./temp')

    def loadModel(self, address):
        print('正在加载模型')
        self.model = load_model(address)
        print('加载完毕！')

    def selectPatient(self, patient):
        print(f'正在加载{patient}病人信息！')
        self.example = patient
        self.clearTemp('./temp')
        self.scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == self.example).first()
        self.vol = self.scan.to_volume()  # 图像信息矩阵
        self.nods = self.scan.cluster_annotations()  # 患者的肺结节特征列表
        print('加载完毕！')

    def preserveImage(self):
        # 用进程池批量保存图像
        images = self.scan.load_all_dicom_images()
        with ThreadPoolExecutor(max_workers=5) as pool:
            for i in range(self.scan.to_volume().shape[2]):
                pool.submit(lambda cxp: self.preserve(*cxp), (images, i))


    def preserve(self, images, num):
        '''
        :param images: 图像数据ndarray矩阵
        :param num: 图片的片数
        '''
        name = './temp/' + str(num) + '.png'
        normalization = self.normaliza(images[num].pixel_array)
        cv2.imwrite(name, normalization*255.0)

    def normaliza(self, x):
        '''
        :param x: 输入ndarray矩阵
        :return: 归一化矩阵数据
        '''
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def showVisualize(self):
        self.scan.visualize(annotation_groups=self.nods)

    def read_dataset(self, ):
        try:
            anns = self.nods[0]  # 第一个肺结节的所有特征
            contour_slice_indices = anns[0]  # 第一个肺结节的第一个特征
        except:
            print("[INFO]文件名字不正确...")
            return

        cmask, cbbox, masks = consensus(anns, clevel=0.5,
                                        pad=[(4, 4), (4, 4), (10, 10)])

        image = self.vol[cbbox]
        # image = self.normaliza(image)

        ArrayDicom_mask_ = cmask
        ArrayDicom_mask_ = ArrayDicom_mask_.astype(np.uint8)

        # 读取含有z范围内的切片位置
        contour_slice_indices = contour_slice_indices.contour_slice_indices
        zz_ = len(contour_slice_indices) / 2
        zz_ = int(zz_)
        k = int(0.5 * (cbbox[2].stop - cbbox[2].start))

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))  # 创建了一个（5*5）大小的子图
        ax.imshow(self.vol[cbbox][:, :, k], cmap=plt.cm.gray, alpha=1)

        colors = ['r', 'g', 'b', 'y']
        for j in range(len(masks)):
            for c in find_contours(masks[j][:, :, k].astype(float), 0.5):  # 返回二值图像轮廓（二值图像：由黑白两色组成的图像像素值是0或者255）
                label = "Annotation %d" % (j + 1)
                plt.plot(c[:, 1], c[:, 0], colors[j], label=label)

        # Plot the 50% consensus contour for the kth slice.
        for c in find_contours(cmask[:, :, k].astype(float), 0.5):
            plt.plot(c[:, 1], c[:, 0], '--k', label='50% Consensus')

        ax.axis('off')
        ax.legend()
        plt.tight_layout()
        # plt.savefig("../images/consensus.png", bbox_inches="tight")
        # plt.show()
        y_true = cmask[:, :, k]
        image_original = image
        # fig.show()
        return fig, image_original, y_true, ArrayDicom_mask_

    # 归一化
    def normalize_hu(self, image):
        print("[INFO]数据 预处理...")
        # 将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image

    def isPointinPolygon(self, point, rangelist):  # [[0,0],[1,1],[0,1],[0,0]] [1,0.8]

        lnglist = []
        latlist = []
        for i in range(len(rangelist) - 1):
            lnglist.append(rangelist[i][0])
            latlist.append(rangelist[i][1])
        # print(lnglist, latlist)
        maxlng = max(lnglist)
        minlng = min(lnglist)
        maxlat = max(latlist)
        minlat = min(latlist)
        # print(maxlng, minlng, maxlat, minlat)
        if (point[0] > maxlng or point[0] < minlng or
                point[1] > maxlat or point[1] < minlat):
            return 0
        count = 0
        point1 = rangelist[0]
        for i in range(1, len(rangelist)):
            point2 = rangelist[i]
            # 点与多边形顶点重合
            if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
                # print("在顶点上")
                return 1
            # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
            if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
                # 求线段与射线交点 再和lat比较
                point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1])
                # print(point12lng)
                # 点在多边形边上
                if (point12lng == point[0]):
                    # print("点在多边形边上")
                    return 1
                if (point12lng < point[0]):
                    count += 1
            point1 = point2
        # print(count)
        if count % 2 == 0:
            return 0
        else:
            return 1

    def save_data2img(self, image, save_dir, ArrayDicom_mask_):
        print("[INFO]图像矩阵储存为图片格式...")
        # save_dir = r"I:\dataset_zhang\tmp\data"
        self.clearTemp('./data')
        # 矩阵增广  和传参
        x_, y_, z_ = np.shape(image)  # 中心点定位
        x_ = int(x_)
        y_ = int(y_)
        z_ = int(z_)
        z_z = int(z_ / 2)

        ArrayDicom = np.zeros((x_ + 31, y_ + 31, z_ + 31), dtype=float)
        ArrayDicom[0:(x_), 0:(y_), 0:(z_)] = image

        ArrayDicom_mask = np.zeros((x_ + 31, y_ + 31, z_ + 31), dtype=float)
        ArrayDicom_mask[0:(x_), 0:(y_), 0:(z_)] = ArrayDicom_mask_

        # z_ = int(z_) / 2
        # z_ = int(z_)
        #####################################

        # print(z_z)
        # z = 15
        z = z_z
        ii = 0
        for x in range(0, x_):
            for y in range(0, y_):
                # 图像进行切片处理

                x_silc_35 = ArrayDicom[x, (y):(y + 30), (z):(z + 30)] * 255
                # 数据类型转换
                x_silc_35 = x_silc_35.astype(np.uint8)

                cv2.imwrite(save_dir + '\\x\\' + str(ii) + '.jpg', x_silc_35)

                #################################################################################
                x_silc_30_mask = ArrayDicom_mask[x, (y):(y + 30), (z):(z + 30)]
                x_silc_30_mask = x_silc_30_mask.astype(np.uint8)  # 数据类型转换
                x_silc_30_mask = cv2.resize(x_silc_30_mask, (30, 30), interpolation=cv2.INTER_LINEAR) * 255
                cv2.imwrite(save_dir + '\\x1\\' + str(ii) + '.jpg ', x_silc_30_mask)

                y_silc_35 = ArrayDicom[(x):(x + 30), y, (z):(z + 30)] * 255

                y_silc_35 = y_silc_35.astype(np.uint8)
                cv2.imwrite(save_dir + '\\y\\' + str(ii) + '.jpg ', y_silc_35)

                ################
                y_silc_30_mask = ArrayDicom_mask[(x):(x + 30), y, (z): (z + 30)]
                y_silc_30_mask = y_silc_30_mask.astype(np.uint8)  # 数据类型转换
                y_silc_30_mask = cv2.resize(y_silc_30_mask, (30, 30), interpolation=cv2.INTER_LINEAR) * 255
                cv2.imwrite(save_dir + '\\y1\\' + str(ii) + '.jpg ', y_silc_30_mask)

                z_silc_35 = ArrayDicom[(x):(x + 30), (y):(y + 30), z] * 255

                z_silc_35 = z_silc_35.astype(np.uint8)
                cv2.imwrite(save_dir + '\\z\\' + str(ii) + '.jpg ', z_silc_35)

                z_silc_30_mask = ArrayDicom_mask[(x):(x + 30), (y):(y + 30), z]
                z_silc_30_mask = z_silc_30_mask.astype(np.uint8)  # 数据类型转换
                z_silc_30_mask = cv2.resize(z_silc_30_mask, (30, 30), interpolation=cv2.INTER_LINEAR) * 255
                cv2.imwrite(save_dir + '\\z1\\' + str(ii) + '.jpg', z_silc_30_mask)
                ii += 1
        print(ii)

        return x_, y_, z_, z_z, ii

    def predict_image(self, read_dir, x_, y_, z_, z_z, ii):
        model = self.model
        # 数据预处理
        data_x = []
        print(ii)
        for i in range(ii):
            i = str(i) + '.jpg'
            Path = os.path.join(read_dir, 'x', i)
            try:
                image = cv2.imread(Path)
                image = img_to_array(image)
            except Exception:
                # print(Path)
                continue
            data_x.append(image)

        data_x = np.array(data_x, dtype='float') / 255.0
        # labels = np.array(labels)
        # 转化标签为张量
        # labels = to_categorical(labels)

        data_y = []
        # labels = []
        # f = open('D:\\lung\\segmiton\\u-net\\test\\trainlabel.txt')
        # label = f.readlines()
        # for Pathimg in os.listdir('D:\\lung\\segmiton\\u-net\\test\\train\\x\\65\\'):
        for i in range(ii):
            i = str(i) + '.jpg'
            Path = os.path.join(read_dir, 'y', i)

            image = cv2.imread(Path)
            image = img_to_array(image)
            data_y.append(image)
        data_y = np.array(data_y, dtype='float') / 255.0

        data_z = []
        # labels = []
        # f = open('D:\\lung\\segmiton\\u-net\\test\\trainlabel.txt')
        # label = f.readlines()
        # for Pathimg in os.listdir('D:\\lung\\segmiton\\u-net\\test\\train\\x\\65\\'):
        for i in range(ii):
            i = str(i) + '.jpg'
            Path = os.path.join(read_dir, 'z', i)
            image = cv2.imread(Path)
            image = img_to_array(image)
            data_z.append(image)
        data_z = np.array(data_z, dtype='float') / 255.0

        # 数据预处理
        data_x1 = []
        for i in range(ii):
            i = str(i) + '.jpg'
            Path = os.path.join(read_dir, 'x1', i)
            # print(Path)
            image = cv2.imread(Path)
            image = img_to_array(image)
            data_x1.append(image)

        data_x1 = np.array(data_x1, dtype='float') / 255.0
        # labels = np.array(labels)
        # 转化标签为张量
        # labels = to_categorical(labels)

        data_y1 = []
        # labels = []
        # f = open('D:\\lung\\segmiton\\u-net\\test\\trainlabel.txt')
        # label = f.readlines()
        # for Pathimg in os.listdir('D:\\lung\\segmiton\\u-net\\test\\train\\x\\65\\'):
        for i in range(ii):
            i = str(i) + '.jpg'
            Path = os.path.join(read_dir, 'y1', i)

            image = cv2.imread(Path)
            image = img_to_array(image)
            data_y1.append(image)
        data_y1 = np.array(data_y1, dtype='float') / 255.0

        data_z1 = []
        # labels = []
        # f = open('D:\\lung\\segmiton\\u-net\\test\\trainlabel.txt')
        # label = f.readlines()
        # for Pathimg in os.listdir('D:\\lung\\segmiton\\u-net\\test\\train\\x\\65\\'):
        for i in range(ii):
            i = str(i) + '.jpg'
            Path = os.path.join(read_dir, 'z1', i)
            image = cv2.imread(Path)
            image = img_to_array(image)
            data_z1.append(image)
        data_z1 = np.array(data_z1, dtype='float') / 255.0

        x = [data_x, data_y, data_z, data_x1, data_y1, data_z1]
        # classify the input image
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True    #当使用GPU时候，Tensorflow运行自动慢慢达到最大GPU的内存

        y = model.predict(x, batch_size=64)
        # print (result.shape)
        y_pred = y
        y_pred = y_pred.reshape(x_, y_)
        ret, thresh1 = cv2.threshold(y_pred, 0.9, 1, cv2.THRESH_BINARY)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.imshow(thresh1, cmap=plt.cm.gray, alpha=1)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        ax.axis('off')
        plt.savefig('./1.jpg')
        # plt.savefig('D:\\524\\lung\\segmiton\\u-net\\result_fenxi\\cancha\\data\\tupian\\' + PathDicom  + str(z) + '.jpg', dpi=1400)
        return fig, y

    def evaluate(self, image_original, y, y_true1, x_, y_):
        # image_original,y,y_true1  = image,y,y_true
        image = image_original

        ################################ppv  he  sen   acc  auc
        y_pred = y
        y_true = y_true1
        y_true = y_true.astype(np.uint8)

        iii, y_pred = cv2.threshold(y, 0.5, 1, cv2.THRESH_BINARY)
        y_actu = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred = y_pred.astype(np.uint8)
        y_pred = list(chain.from_iterable(y_pred))
        y_actu = list(chain.from_iterable(y_actu))

        # 混淆矩阵
        cm = ConfusionMatrix(actual_vector=y_actu, predict_vector=y_pred)  # Create CM From Data
        cm.classes
        cm.table
        AUC = ((cm.AUC[0] + cm.AUC[1]) / 2)
        PPV = ((cm.PPV[0] + cm.PPV[1]) / 2)
        TPR = ((cm.TPR[0] + cm.TPR[1]) / 2)
        ACC = ((cm.ACC[0] + cm.ACC[1]) / 2)
        print('AUC', (cm.AUC[0] + cm.AUC[1]) / 2)
        print('PPV', (cm.PPV[0] + cm.PPV[1]) / 2)
        print('TPR', (cm.TPR[0] + cm.TPR[1]) / 2)
        print('ACC', (cm.ACC[0] + cm.ACC[1]) / 2)

        #########################dice

        y_true = y_true1
        y_pred = y
        ret, y_pred = cv2.threshold(y_pred, 0.5, 1, cv2.THRESH_BINARY)
        y_pred = y_pred.reshape(x_, y_)
        y_true = y_true.astype(np.uint8)
        y_pred = y_pred.astype(np.uint8)

        def computeQualityMeasures(lP, lT):
            quality = dict()
            labelPred = sitk.GetImageFromArray(lP, isVector=False)
            labelTrue = sitk.GetImageFromArray(lT, isVector=False)
            hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
            hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
            quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
            quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()

            dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
            dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
            quality["dice"] = dicecomputer.GetDiceCoefficient()

            return quality

        quality = computeQualityMeasures(y_pred, y_true)
        ASD = quality['avgHausdorff']
        DICE = quality["dice"]
        HSD = quality["Hausdorff"]

        print(quality)

        ############################################roc
        # from sklearn import cross_validation
        y_pred = y
        y_actu = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred = list(chain.from_iterable(y_pred))
        y_actu = list(chain.from_iterable(y_actu))

        # Compute ROC curve and ROC area for each class
        fpr, tpr, threshold = roc_curve(y_actu, y_pred)  ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)  ###计算auc的值

        fig = plt.figure()
        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('./result.png')
        return AUC, PPV, TPR, ACC, fig, ASD, DICE, HSD

    def clearTemp(self, path_file):
        '''
        清空一个文件夹里的所有文件和文件夹
        '''
        ls = os.listdir(path_file)
        for i in ls:
            f_path = os.path.join(path_file, i)
            # 判断是否是一个目录,若是,则递归删除
            if os.path.isdir(f_path):
                self.clearTemp(f_path)
            else:
                os.remove(f_path)