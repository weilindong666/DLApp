# -*- coding: utf-8 -*-
"""
Created on Wed May 22 18:56:41 2019

@author: Administrator
"""
import os
import h5py
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from skimage.measure import find_contours
import cv2
import pylidc as pl
from pylidc.utils import consensus
import xml.dom.minidom
from random import choice
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import keras.layers as K
from pycm import ConfusionMatrix
from itertools import chain
import argparse
from PIL import Image
import SimpleITK as sitk
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from itertools import chain
# from sklearn import cross_validation


class A:
    
    def __init__(self,ID):
        self.example = ID
        print("[INFO] 加载模型中，可能需要几分钟...")
        #读取模型
        self.model = load_model(r'C:\Users\Administrator\Desktop\DLApp\model_2_test.h5')
        
    #读取所有结节


    def read_dataset(self):
        example = self.example  # 病人编号， example = 'LIDC-IDRI-0057'

        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == example).first()  # 这个库是用来把病人信息当作数据库来操作，（pl.Scan.patient_id 就是提取病人id的操作）

        vol = scan.to_volume()  # 将扫描图像值转换为 NumPy 数组，vol.shape=(512, 512, 87)
        # Cluster the annotations for the scan, and grab one.
        nods = scan.cluster_annotations()
        try:
            anns = nods[0]  # 装的是第一个肺结节的特征信息
            contour_slice_indices = anns[0]  # 第一个肺结节的第一个特征
        except :
            print("[INFO]文件名字不正确...")
            return
            # pass
        # Perform a consensus consolidation and 50% agreement level.
        # We pad the slices to add context for viewing.

        cmask,cbbox,masks = consensus(anns, clevel=0.5, pad=[(4,4), (4,4), (10,10)])
        image = vol[cbbox]


        #读取含有z范围内的切片位置
        contour_slice_indices = contour_slice_indices.contour_slice_indices
        zz_ = int(len(contour_slice_indices) / 2)
        k = int(0.5*(cbbox[2].stop - cbbox[2].start))

        fig,ax = plt.subplots(1,1,figsize=(5,5))
        ax.imshow(vol[cbbox][:,:,k], cmap=plt.cm.gray, alpha=1)

        colors = ['r', 'g', 'b', 'y']
        for j in range(len(masks)):
            for c in find_contours(masks[j][:,:,k].astype(float), 0.5):
                label = "Annotation %d" % (j+1)
                plt.plot(c[:,1], c[:,0], colors[j], label=label)

        # Plot the 50% consensus contour for the kth slice.
        for c in find_contours(cmask[:,:,k].astype(float), 0.5):
            plt.plot(c[:,1], c[:,0], '--k', label='50% Consensus')

        ax.axis('off')
        ax.legend()
        plt.tight_layout()
        #plt.savefig("../images/consensus.png", bbox_inches="tight")
        plt.show()
        y_true = cmask[:,:,k]
        image_original = image
        return fig,image_original,y_true


      #归一化
    def normalize_hu(self,image):
        print("[INFO]数据 预处理...") #将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image





    def isPointinPolygon(self,point, rangelist):  #[[0,0],[1,1],[0,1],[0,0]] [1,0.8]

        lnglist = []
        latlist = []
        for i in range(len(rangelist)-1):
            lnglist.append(rangelist[i][0])
            latlist.append(rangelist[i][1])
        #print(lnglist, latlist)
        maxlng = max(lnglist)
        minlng = min(lnglist)
        maxlat = max(latlist)
        minlat = min(latlist)
        #print(maxlng, minlng, maxlat, minlat)
        if (point[0] > maxlng or point[0] < minlng or
            point[1] > maxlat or point[1] < minlat):
            return 0
        count = 0
        point1 = rangelist[0]
        for i in range(1, len(rangelist)):
            point2 = rangelist[i]
            # 点与多边形顶点重合
            if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
                #print("在顶点上")
                return 1
            # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
            if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
                # 求线段与射线交点 再和lat比较
                point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0])/(point2[1] - point1[1])
                #print(point12lng)
                # 点在多边形边上
                if (point12lng == point[0]):
                    #print("点在多边形边上")
                    return 1
                if (point12lng < point[0]):
                    count +=1
            point1 = point2
        #print(count)
        if count%2 == 0:
            return 0
        else:
            return 1


    def save_data2img(self,image,save_dir):
        print("[INFO]图像矩阵储存为图片格式...")
        #矩阵增广  和传参
        x_,y_,z_ = np.shape(image)  #中心点定位
        x_ = int(x_)
        y_ = int(y_)
        z_ = int(z_)
        z_z = int(z_/2)

        ArrayDicom = np.zeros((x_ + 31, y_ + 31, z_ + 31), dtype=float)
        ArrayDicom[0:(x_ ), 0:(y_ ), 0:(z_ )] = image

        ArrayDicom_mask = np.zeros((x_ + 31, y_ + 31, z_ + 31), dtype=float)
        ArrayDicom_mask[0:(x_ ), 0:(y_ ), 0:(z_ )] = ArrayDicom_mask
        print(z_z)
    #z = 15
        z = z_z
        ii = 0
        for x in range(0,x_):
            for y in range(0,y_):
            #定位标记点坐标并且判断
                print(ii)
                # label=1
                # txtfile = open(save_dir+'label.txt', mode='a')
                # txtfile.writelines('%s %d \n' % (str(ii)+'.jpg', label))
                # txtfile.close()
                #图像进行切片处理

                x_silc_35 = ArrayDicom[x,(y ):(y + 30), (z ):(z + 30)] * 255
                #数据类型转换
                x_silc_35 = x_silc_35.astype(np.uint8)

                                # z_image[i,:,:,1]=z_silc_65
                                # z_image[i,:,:,0] = z_silc_35
                                # cv2.imwrite(save_dir+'z/35/' + str(ii) + '.jpg',z_silc_35*255)
                                # cv2.imwrite(save_dir+'z/65/' + str(ii) + '.jpg',z_silc_65*25

                cv2.imwrite(save_dir + 'x\\' + str(ii) + '.jpg', x_silc_35)

                #################################################################################
                x_silc_30_mask = ArrayDicom_mask[x ,(y) :(y + 30), (z):(z + 30)]
                x_silc_30_mask = x_silc_30_mask.astype(np.uint8)   #数据类型转换
                x_silc_30_mask = cv2.resize(x_silc_30_mask, (30, 30), interpolation=cv2.INTER_LINEAR) * 255
                cv2.imwrite(save_dir + 'x1\\'  + str(ii) +  '.jpg ', x_silc_30_mask)






                y_silc_35 = ArrayDicom[(x ):(x + 30),y, (z ):(z + 30) ] * 255

                y_silc_35 = y_silc_35.astype(np.uint8)

            # x_image[i, :, :, 1] = x_silc_65
            # x_image[i, :, :, 0] = x_silc_35
          #  cv2.imwrite(save_dir+'x/35/' + str(ii) + '.jpg', x_silc_35*255)
           # cv2.imwrite(save_dir+'x/65/' + str(ii) + '.jpg', x_silc_65*25)
                cv2.imwrite(save_dir + 'y\\'  + str(ii) +  '.jpg ', y_silc_35)

                ################
                y_silc_30_mask = ArrayDicom_mask[(x ):(x + 30) ,y , (z): (z+30) ]
                y_silc_30_mask = y_silc_30_mask.astype(np.uint8)   #数据类型转换
                y_silc_30_mask = cv2.resize(y_silc_30_mask, (30, 30), interpolation=cv2.INTER_LINEAR) * 255
                cv2.imwrite(save_dir + 'y1\\'  + str(ii) +  '.jpg ', y_silc_30_mask)


                z_silc_35 = ArrayDicom[(x ):(x + 30),(y ):(y + 30), z ] * 255


                z_silc_35 = z_silc_35.astype(np.uint8)
            # y_image[i, :, :, 1] = y_silc_65
            # y_image[i, :, :, 0] = y_silc_35
           # cv2.imwrite(save_dir+'y/35/' + str(ii) + '.jpg', y_silc_35*255)
           # cv2.imwrite(save_dir+'y/65/' + str(ii) + '.jpg', y_silc_65*255)
                cv2.imwrite(save_dir + 'z\\' + str(ii) +  '.jpg ', z_silc_35)


                z_silc_30_mask = ArrayDicom_mask[(x ):(x + 30),(y ):(y + 30), z ]
                z_silc_30_mask = z_silc_30_mask.astype(np.uint8)   #数据类型转换
                z_silc_30_mask = cv2.resize(z_silc_30_mask, (30, 30), interpolation=cv2.INTER_LINEAR) * 255
                cv2.imwrite(save_dir + 'z1\\'  + str(ii) +  '.jpg', z_silc_30_mask)
                ii += 1

        return x_,y_,z_,z_z,ii


    def predict_image(self,read_dir,x_,y_,z_,z_z,ii):
        model = self.model
        ii = ii

        # load the trained convolutional neural network
        #print("[INFO] loading network...")
        #model = load_model('D:\\524\\lung\\segmiton\\u-net\\result_fenxi\\cancha\\model_2_test.h5')
        #数据预处理
        data_x = []
        for i in range(0,ii):
            i = str(i) + '.jpg'
            Path    = os.path.join(read_dir + 'x\\',i)
            print(Path)
            image = cv2.imread(Path)
            image = img_to_array(image)
            data_x.append(image)

        data_x = np.array(data_x,dtype='float') / 255.0
        #labels = np.array(labels)
        #转化标签为张量
        #labels = to_categorical(labels)


        data_y = []
        #labels = []
        #f = open('D:\\lung\\segmiton\\u-net\\test\\trainlabel.txt')
        #label = f.readlines()
        #for Pathimg in os.listdir('D:\\lung\\segmiton\\u-net\\test\\train\\x\\65\\'):
        for i in range(0,ii):
            i = str(i) + '.jpg'
            Path    = os.path.join(read_dir + 'y\\',i)

            image = cv2.imread(Path)
            image = img_to_array(image)
            data_y.append(image)
        data_y = np.array(data_y,dtype='float') / 255.0



        data_z = []
        #labels = []
        #f = open('D:\\lung\\segmiton\\u-net\\test\\trainlabel.txt')
        #label = f.readlines()
        #for Pathimg in os.listdir('D:\\lung\\segmiton\\u-net\\test\\train\\x\\65\\'):
        for i in range(0,ii):
            i = str(i) + '.jpg'
            Path    = os.path.join(read_dir + 'z\\',i)
            image = cv2.imread(Path)
            image = img_to_array(image)
            data_z.append(image)
        data_z = np.array(data_z,dtype='float') / 255.0


        #数据预处理
        data_x1 = []
        for i in range(0,ii):
            i = str(i) + '.jpg'
            Path    = os.path.join(read_dir + 'x1\\',i)
            print(Path)
            image = cv2.imread(Path)
            image = img_to_array(image)
            data_x1.append(image)

        data_x1 = np.array(data_x1,dtype='float') / 255.0
        #labels = np.array(labels)
        #转化标签为张量
        #labels = to_categorical(labels)


        data_y1 = []
        #labels = []
        #f = open('D:\\lung\\segmiton\\u-net\\test\\trainlabel.txt')
        #label = f.readlines()
        #for Pathimg in os.listdir('D:\\lung\\segmiton\\u-net\\test\\train\\x\\65\\'):
        for i in range(0,ii):
            i = str(i) + '.jpg'
            Path    = os.path.join(read_dir + 'y1\\',i)

            image = cv2.imread(Path)
            image = img_to_array(image)
            data_y1.append(image)
        data_y1 = np.array(data_y1,dtype='float') / 255.0


        data_z1 = []
        #labels = []
        #f = open('D:\\lung\\segmiton\\u-net\\test\\trainlabel.txt')
        #label = f.readlines()
        #for Pathimg in os.listdir('D:\\lung\\segmiton\\u-net\\test\\train\\x\\65\\'):
        for i in range(0,ii):
            i = str(i) + '.jpg'
            Path    = os.path.join(read_dir + 'z1\\',i)
            image = cv2.imread(Path)
            image = img_to_array(image)
            data_z1.append(image)
        data_z1 = np.array(data_z1,dtype='float') / 255.0


        x = [data_x,data_y,data_z,data_x1,data_y1,data_z1]
            # classify the input image
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True    #预测结果于输出

        y = model.predict(x,batch_size=64)
            #print (result.shape)
        y_pred = y
        y_pred = y_pred.reshape(x_,y_)
        ret, thresh1 = cv2.threshold(y_pred, 0.9, 1, cv2.THRESH_BINARY)
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        ax.imshow(thresh1, cmap=plt.cm.gray,alpha=1)
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
        plt.margins(0,0)
        ax.axis('off')
        #plt.savefig('D:\\524\\lung\\segmiton\\u-net\\result_fenxi\\cancha\\data\\tupian\\' + PathDicom  + str(z) + '.jpg', dpi=1400)
        return fig,y




    def evaluate(self,image_original,y,y_true1):
        image = image_original

        # ppv  he  sen   acc  auc
        y_pred = y
        y_true = y_true1
        y_true = y_true.astype(np.uint8)
        iii,y_pred = cv2.threshold(y, 0.5, 1, cv2.THRESH_BINARY)
        y_actu = np.array(y_true)
        y_pred =  np.array(y_pred)
        y_pred = y_pred.astype(np.uint8)
        y_pred = list(chain.from_iterable(y_pred))
        y_actu = list(chain.from_iterable(y_actu))
        cm = ConfusionMatrix(actual_vector=y_actu, predict_vector=y_pred) # Create CM From Data
        cm.classes
        cm.table
        AUC = ((cm.AUC[0] + cm.AUC[1])/2)
        PPV = ((cm.PPV[0]+cm.PPV[1])/2)
        TPR = ((cm.TPR[0]+cm.TPR[1])/2)
        ACC = ((cm.ACC[0]+cm.ACC[1])/2)
        print((cm.AUC[0] + cm.AUC[1])/2)
        print((cm.PPV[0]+cm.PPV[1])/2)
        print((cm.TPR[0]+cm.TPR[1])/2)
        print((cm.ACC[0]+cm.ACC[1])/2)
        # dice
        y_true =   y_true1
        y_pred = y
        ret, y_pred = cv2.threshold(y_pred, 0.5, 1, cv2.THRESH_BINARY)
        y_pred = y_pred.reshape(x_,y_)
        y_true = y_true.astype(np.uint8)
        y_pred = y_pred.astype(np.uint8)



        def computeQualityMeasures(lP,lT):
            quality=dict()
            labelPred=sitk.GetImageFromArray(lP, isVector=False)
            labelTrue=sitk.GetImageFromArray(lT, isVector=False)
            hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
            hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
            quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance()
            quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()

            dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
            dicecomputer.Execute(labelTrue>0.5,labelPred>0.5)
            quality["dice"]=dicecomputer.GetDiceCoefficient()

            return quality

        quality = computeQualityMeasures(y_pred, y_true)
        ASD = quality['avgHausdorff']
        DICE = quality["dice"]
        HSD = quality["Hausdorff"]

        print(quality)

        ############################################roc
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

        return fig, ASD, DICE, HSD



    
  
    
                

            
                        
                    
                


                  