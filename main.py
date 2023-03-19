# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 22:38:42 2022

@author: dell
"""
from PySide2.QtWidgets import QApplication
from ui.MainWin import MainWin

# from daima.create_data_test import A
#
# A = A(ID = 'LIDC-IDRI-0057')
#
# #在数据集中读取文件
# fig,image_original,y_true,ArrayDicom_mask_ = A.read_dataset()
# #图像矩阵归一化
# image = A.normalize_hu(image_original)
# #切割并储存图像到文件夹,返回矩阵中心点坐标
# save_dir = r"I:/dataset_zhang/tmp/data"
#
#
# x_,y_,z_,z_z,ii = A.save_data2img(image,save_dir,ArrayDicom_mask_)
# #读取图像并且使用模型预测图像
# ii = 3538
#
# #文件储存的位置
# read_dir = 'C:/Users/Administrator/Desktop/tmp/data'
# #得到预测的结果和图像
# fig,y = A.predict_image(read_dir,x_,y_,z_,z_z,ii)
# #输出指标
# AUC,PPV,TPR,ACC,fig,ASD,DICE,HSD= A.evaluate(image,y,y_true, x_, y_)


if __name__ == '__main__':
    app = QApplication()
    obj = MainWin()
    obj.show()
    app.exec_()
















