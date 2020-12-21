'''
面部显示窗口，继承自Widget的一个子类
'''

from PyQt5.QtWidgets import  *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class FaceWidget(QStackedWidget):
    def __init__(self):
        super(FaceWidget,self).__init__()
        self.setObjectName('FaceWidget')
        self.pos_x = 0
        self.pos_y = 0
        self.resize(400,1000)
        self.faceWidth = 240
        self.faceHeight = 320
        self.facenum = 0
        self.pagelist = []
        self.FaceLabelList = []
        self.TextLabelList = []

    def SetPage(self):
        self.page1  = QWidget()
        self.page2  = QWidget()
        self.page3  = QWidget()
        self.page4  = QWidget()

        self.pagelist.append(self.page1)
        self.pagelist.append(self.page2)
        self.pagelist.append(self.page3)
        self.pagelist.append(self.page4)
    def SetLabel(self):


    #添加人脸到窗口里
    def PushFace(self):
        self.alabel = QLabel(self.page1)
        self.alabel.setGeometry(QRect(self.pos_x,self.pos_y,self.faceWidth,self.faceHeight))
        self.pos_y = self.pos_y+ self.faceHeight


    #自动布局
    def AutoReset(self):
        s