from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import argparse

import os
import cv2
import dlib
import numpy as np
from mtcnn.mtcnn import MTCNN
from includes.Face.matlab_cp2tform import get_similarity_transself_for_cv2
import includes.Face.net_sphere  as net_sphere
import sys
from PyQt5.QtCore import *
import cv2
import numpy as np
import warnings
from mtcnn.mtcnn import MTCNN
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
from includes.pymysql.PyMySQL import *
#识别算法的线程
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class DetectionThread(QThread):
    #传出的信号为图片中人脸的位置矩形以及识别出的人名
    Bound_Name = pyqtSignal(int,int,int,int,str)
    Dynamic_Bound_Name = pyqtSignal(int,int,int,int,str)
    Dynamic_Show_Time = pyqtSignal(int)
    Face_Count = pyqtSignal(int)
    def __init__(self,detector,net):
        super(DetectionThread, self).__init__()
        #导入识别和检测模型
        self.net = net
        self.detector = detector
    def SetImg(self,img):
        self.img = img
        self.method = method

        self.start()
    def SetThresHold(self,thres):
        self.thres = thres
    def run(self):

        result = self.detector.detect_faces(self.img)
        print('results', result)

        self.Face_Count.emit(len(result))
        #如果没有检测出人脸，发出一个信号并且提前停止线程
        if len(result) == 0 :
            return

        # 检测，标定landmark
        for face in result:
            temp_landmarks = []
            bouding_boxes = face['box']
            for axis in bouding_boxes:
                if axis<=0 or axis>=self.img.shape[0]-1 or axis>=self.img.shape[1]-1:
                    return

            keypoints = face['keypoints']

            faces = self.img[bouding_boxes[1]:bouding_boxes[1] + bouding_boxes[3],
                    bouding_boxes[0]:bouding_boxes[0] + bouding_boxes[2]]
            originfaces.append(faces)
            lefteye = keypoints['left_eye']
            righteye = keypoints['right_eye']
            nose = keypoints['nose']
            mouthleft = keypoints['mouth_left']
            mouthright = keypoints['mouth_right']
            temp_landmarks.append(lefteye[0])
            temp_landmarks.append(lefteye[1])
            temp_landmarks.append(righteye[0])
            temp_landmarks.append(righteye[1])
            temp_landmarks.append(nose[0])
            temp_landmarks.append(nose[1])
            temp_landmarks.append(mouthleft[0])
            temp_landmarks.append(mouthleft[1])
            temp_landmarks.append(mouthright[0])
            temp_landmarks.append(mouthright[1])
            for i, num in enumerate(temp_landmarks):
                if i % 2:
                    temp_landmarks[i] = num - bouding_boxes[1]
                else:
                    temp_landmarks[i] = num - bouding_boxes[0]

            faces = self.alignment(faces, temp_landmarks)
            faces = np.transpose(faces, (2, 0, 1)).reshape(1, 3, 112, 96)
            faces = (faces - 127.5) / 128.0
            aligment_imgs.append(faces)

        # print('face ok')
        length = len(aligment_imgs)
        aligment_imgs = np.array(aligment_imgs)
        aligment_imgs = np.reshape(aligment_imgs, (length, 3, 112, 96))
        output_imgs_features = self.get_imgs_features(aligment_imgs)
        cos_distances_list = []
        #和数据库内的每一个向量进行计算对比
        imgs_features = self.db.get_all_vector()
        NameIndb = self.db.get_all_name()
        NameList = []
        for img_feature in output_imgs_features:
            cos_distance_list = [self.cal_cosdistance(img_feature, test_img_feature) for test_img_feature in
                                 imgs_features]
            cos_distances_list.append(cos_distance_list)

        # print('\n',cos_distances_list)

        for sub_cos_distances_list in cos_distances_list:

            if max(sub_cos_distances_list) < self.thres:
                NameList.append('Unknown')
            else:
                NameList.append(NameIndb[sub_cos_distances_list.index(max(sub_cos_distances_list))])

        #method = 0: 签到
        #method = 1: 动态识别（画人脸）
        if self.method ==0:
            for i, name in enumerate(NameList):
                bound = result[i]['box']
                #发送信号
                #    print('Signal emit:',bound,name)
                self.Bound_Name.emit(bound[0],bound[1],bound[2],bound[3],name)
        elif self.method ==1:
            for i, name in enumerate(NameList):
                bound = result[i]['box']
                #发送信号
                self.Dynamic_Bound_Name.emit(bound[0],bound[1],bound[2],bound[3],name)
            self.Dynamic_Show_Time.emit(self.show_time)

        else:
            pass


    def cal_cosdistance(self, vec1, vec2):
        vec1 = np.reshape(vec1, (1, -1))
        vec2 = np.reshape(vec2, (-1, 1))
        length1 = np.sqrt(np.square(vec1).sum())
        length2 = np.sqrt(np.square(vec2).sum())
        cosdistance = vec1.dot(vec2) / (length1 * length2)
        cosdistance = cosdistance[0][0]
        return cosdistance

    def get_imgs_features(self, imgs_alignment):
        input_images = Variable(torch.from_numpy(imgs_alignment).float(), volatile=True)
        output_features = self.net(input_images)
        output_features = output_features.data.numpy()
        return output_features

    def alignment(self,src_img,src_pts):
        ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
            [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
        crop_size = (96, 112)
        src_pts = np.array(src_pts).reshape(5,2)

        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)
        tfm = get_similarity_transself_for_cv2(s, r)
        face_img = cv2.warpAffine(src_img, tfm, crop_size)
        return face_img