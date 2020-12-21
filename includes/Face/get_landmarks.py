from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os 
import cv2
import dlib
import argparse
import numpy as np
from matlab_cp2tself import get_similarity_transself_for_cv2
import face_recognition

from PIL import Image

def get_five_points_landmarks(path):
    landmark = []
    
    img = face_recognition.load_image_file(path)
    face_landmarks_list = face_recognition.face_landmarks(img)
    if len(face_landmarks_list) == 0:
        print(path)

    face_landmarks = face_landmarks_list[0]

    first_point = face_landmarks['left_eye'][1]
    landmark.append(first_point[0])
    landmark.append(first_point[1])

    second_point = face_landmarks['right_eye'][1]
    landmark.append(second_point[0])
    landmark.append(second_point[1])

    third_point = face_landmarks['nose_bridge'][3]    
    landmark.append(third_point[0])
    landmark.append(third_point[1])

    forth_point = face_landmarks['top_lip'][0]
    landmark.append(forth_point[0])
    landmark.append(forth_point[1])

    fifth_point = face_landmarks['top_lip'][6]
    landmark.append(fifth_point[0])
    landmark.append(fifth_point[1])

    landmark = [float(item) for item in landmark]

    return landmark






















    
