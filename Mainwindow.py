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
import numpy as np
from mtcnn.mtcnn import MTCNN
from includes.Face.matlab_cp2tform import get_similarity_transself_for_cv2
from includes.thread.AddFaceThread import AddFaceThread
from includes.thread.DetectionThread import DetectionThread
import includes.Face.net_sphere  as net_sphere
import qdarkstyle

#import network model
parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='sphere20a', type=str, choices=['sphere20a', 'DCR'])
parser.add_argument('--model','-m', default='./model/sphere20a_20171020.pth', type=str)
args = parser.parse_args()


net = getattr(net_sphere,args.net)()
net.load_state_dict(torch.load(args.model))
# device = torch.device('cuda',0) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
net.to(device)
net.eval()
net.feature = True

import sys
from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import os

sys.setrecursionlimit(1000000)
myFolder = os.path.split(os.path.realpath(__file__))[0]
sys.path = [os.path.join(myFolder, 'thread')
           ,os.path.join(myFolder,'resources')
] + sys.path

os.chdir(myFolder)
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import time
import warnings
from includes.pymysql.PyMySQL import *
from Widgets.DBWidge import DBWidge
warnings.filterwarnings('ignore')

def mask_image(show, size=200):
    """Return a ``QPixmap`` from *imgdata* masked with a smooth circle.

    *imgdata* are the raw image bytes, *imgtype* denotes the image type.

    The returned image will have a size of *size* × *size* pixels.

    """
    # Load image and convert to 32-bit ARGB (adds an alpha channel):
    image = QImage(show.data, show.shape[1], show.shape[0],QImage.Format_RGB888)
    # image.convertToFormat(QImage.Format_RGB888)

    # Crop image to a square:
    imgsize = min(image.width(), image.height())
    rect = QRect(
        (image.width() - imgsize) / 2,
        (image.height() - imgsize) / 2,
        imgsize,
        imgsize,
    )
    image = image.copy(rect)

    # Create the output image with the same dimensions and an alpha channel
    # and make it completely transparent:
    out_img = QImage(imgsize, imgsize, QImage.Format_ARGB32)
    out_img.fill(Qt.transparent)

    # Create a texture brush and paint a circle with the original image onto
    # the output image:
    brush = QBrush(image)        # Create texture brush
    painter = QPainter(out_img)  # Paint the output image
    painter.setBrush(brush)      # Use the image texture brush
    painter.setPen(Qt.NoPen)     # Don't draw an outline
    painter.setRenderHint(QPainter.Antialiasing, True)  # Use AA
    painter.drawEllipse(0, 0, imgsize, imgsize)  # Actually draw the circle
    painter.end()                # We are done (segfault if you forget this)

    # Convert the image to a pixmap and rescale it.  Take pixel ratio into
    # account to get a sharp image on retina displays:
    pr = QWindow().devicePixelRatio()
    pm = QPixmap.fromImage(out_img)
    pm.setDevicePixelRatio(pr)
    size *= pr
    pm = pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    return pm

class Label(QLabel):
    def __init__(self, *args, antialiasing=True, **kwargs):
        super(Label, self).__init__(*args, **kwargs)
        self.Antialiasing = antialiasing
        self.setMaximumSize(50, 50)
        self.setMinimumSize(50, 50)
        self.radius = 25

        self.target = QPixmap(self.size())
        self.target.fill(Qt.transparent)

        p = QPixmap.fromImage().scaled(
            50, 50, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

        painter = QPainter(self.target)
        if self.Antialiasing:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.HighQualityAntialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        path = QPainterPath()
        path.addRoundedRect(
            0, 0, self.width(), self.height(), self.radius, self.radius)

        painter.setClipPath(path)
        painter.drawPixmap(0, 0, p)
        self.setPixmap(self.target)

class Ui_MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)


        #数据库调用
        self.dbWidge = DBWidge()
        self.dbWidge.setHidden(True)
        self.db = PyMySQL('localhost','root','CockTail','TESTDATABASE')
        #相机区域

        #人脸识别与记录线程
        self.detector = MTCNN()
        self.FaceThread = DetectionThread(self.detector,net)
        #添加新人脸的线程
        self.AddFaceThread = AddFaceThread(self.detector,net)
        #定时器
        self.timer_camera =   QTimer()
        self.timer_camera_counter = 0
        self.timer_clear_label = QTimer()
        self.timer_dynamic_recog = QTimer()
        self.timer_long_name = QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0    #Camera used
        self.dynamic_draw_flag = False


        #初始化
        # self.setBackGround()

        self.facelabel_list = []
        self.textlabel_list = []
        self.name_list = []
        self.long_name_list = []
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.initMenu()
        self.initAnimation()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.timer_clear_label.start(5000)
        self.timer_long_name.start(60000)

    def set_ui(self):
        self.resize(1600,900)

        self.textBrowser = QtWidgets.QTextBrowser(self)
        self.textBrowser.setGeometry(QtCore.QRect(10, 650, 661, 151))
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setFont(QFont("Timers",14))
        self.tabWidget = QtWidgets.QScrollArea(self)
        self.tabWidget.setGeometry(QtCore.QRect(670, 40, 500, 800))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setMinimumSize(400,2000)
        self.tab.setObjectName("tab")
        self.tabWidget.setWidget(self.tab)
        self.gridLayoutWidget = QtWidgets.QWidget(self.tab)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 400, 1200))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        self.append_label()
        self.append_label()
        self.append_label()
        self.append_label()

        self.lcdNumber = QtWidgets.QLCDNumber(self)
        self.lcdNumber.setGeometry(QtCore.QRect(470, 40, 201, 41))
        self.lcdNumber.setObjectName("lcdNumber")
        self.lcdNumber.setDigitCount(2)
        self.camera_label = QtWidgets.QLabel(self)
        self.camera_label.setGeometry(QtCore.QRect(10, 90, 661, 551))
        self.camera_label.setObjectName("camera_label")


        self.horizontalLayoutWidget = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 395, 81))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.pushButton_4 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.setText("人脸显示")
        self.horizontalLayout.addWidget(self.pushButton_4)
        self.pushButton_3 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText("手动签到")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("清除列表")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("开启相机")
        self.horizontalLayout.addWidget(self.pushButton)




    def append_label(self):
        label_num = self.facelabel_list.__len__()
        temp_text = QLabel(self.gridLayoutWidget)
        temp_text.setFont(QFont("Timers",12))
        self.facelabel_list.append(QLabel(self.gridLayoutWidget))
        self.textlabel_list.append(temp_text)
        self.gridLayout.addWidget(self.facelabel_list[-1],label_num,0,1,1)
        self.gridLayout.addWidget(self.textlabel_list[-1],label_num,1,1,1)


    def contextMenuEvent(self, event):
        pos = event.globalPos()
        size = self._contextMenu.sizeHint()
        x, y, w, h = pos.x(), pos.y(), size.width(), size.height()
        self._animation.stop()
        self._animation.setStartValue(QRect(x, y, 0, 0))
        self._animation.setEndValue(QRect(x, y, w, h))
        self._animation.start()
        self._contextMenu.exec_(event.globalPos())

    def initMenu(self):
        self._contextMenu = QMenu(self)
        self.ac_open_cama = self._contextMenu.addAction('打开相机', self.CameraOperation)
        self.ac_detection = self._contextMenu.addAction('一键签到', self.Checkin)
        self.ac_Addface = self._contextMenu.addAction('添加新人脸',self.AddFace)
        self.ac_DynamicRecog = self._contextMenu.addAction('关闭动态识别',self.DynamicRecogOn)
        self.ac_dbManager = self._contextMenu.addAction('数据库操作',self.openDBmanager)
        self.ac_delete_text = self._contextMenu.addAction('删除信息显示',self.clear_all_text)
    def initAnimation(self):
        # 按钮动画
        self._animation = QPropertyAnimation(
            self._contextMenu, b'geometry', self,
            easingCurve=QEasingCurve.Linear, duration=300)
        # easingCurve 修改该变量可以实现不同的效果

    #定义信号槽
    def slot_init(self):

        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_camera.timeout.connect(self.frame_count)
        self.timer_clear_label.timeout.connect(self.del_instant_label)
        self.timer_dynamic_recog.timeout.connect(self.Checkin)
        self.timer_long_name.timeout.connect(self.del_long_name)
        #人脸识别算法完成后在右边的tab widget 中显示
        self.AddFaceThread.No_face.connect(self.TextShowNoFace)
        self.FaceThread.Bound_Name.connect(self.ShowInTab)
        self.FaceThread.Face_Count.connect(self.ShowInLCD)
        self.pushButton.clicked.connect(self.CameraOperation)
        self.pushButton_2.clicked.connect(self.clear_all_label)
        self.pushButton_3.clicked.connect(self.Checkin)
        self.pushButton_4.clicked.connect(self.OpenDraw)

    def OpenDraw(self):
        self.dynamic_draw_flag = 1 - self.dynamic_draw_flag

    def frame_count(self):
        if self.timer_camera_counter is None:
            self.timer_camera_counter = 0

        else:
            self.timer_camera_counter = self.timer_camera_counter + 1

        if self.timer_camera_counter>=5:
            self.timer_camera_counter = 0

    def ShowInLCD(self,number):
        self.lcdNumber.display(number)

    def TextShowNoFace(self):
        self.textBrowser.insertPlainText("未检测到人脸，请重试")
    def openDBmanager(self):
        if self.dbWidge.isHidden():
            self.dbWidge.setHidden(False)


    def del_instant_label(self):
        #删除第一个Label,剩余label后移动
        if not self.textlabel_list[0].text():
            return

        self.facelabel_list[0].clear()
        name = self.textlabel_list[0].text().split('#')[1]
        self.textlabel_list[0].clear()
        self.name_list.remove(name)
        for i in range(self.textlabel_list.__len__()-1):
            print(self.textlabel_list[i].text())
            if self.textlabel_list[i+1].text():
                print('p2')
                self.facelabel_list[i].setPixmap(self.facelabel_list[i+1].pixmap())
                self.textlabel_list[i].setText(self.textlabel_list[i+1].text())
                self.textlabel_list[i+1].clear()
                self.facelabel_list[i+1].clear()





    def AddFace(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please check you have connected your camera", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.Addface_img = self.image.copy()
            try:
                self.AddFaceThread.SetImg(self.Addface_img)
            except:
                pass
    def DynamicRecogOn(self):
        if self.timer_camera.isActive()==False:
            msg = QtWidgets.QMessageBox.warning(self, u"warning", u"没有检测到摄像头", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            if self.timer_dynamic_recog.isActive() == False:
                self.timer_dynamic_recog.start(400)
                self.ac_DynamicRecog.setText('关闭动态识别')
            else:
                self.timer_dynamic_recog.stop()
                self.ac_DynamicRecog.setText('开启动态识别')

    def CameraOperation(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please check you have connected your camera", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        
            else:
                self.timer_camera.start(75)
                self.timer_dynamic_recog.start(400)
                self.ac_DynamicRecog.setText('关闭动态识别')
                self.ac_open_cama.setText('关闭摄像头')
                self.pushButton.setText('关闭摄像头')
        else:
            if self.timer_dynamic_recog.isActive():
                self.timer_dynamic_recog.stop()
                self.ac_DynamicRecog.setText('开启动态识别')
            self.timer_camera.stop()
            self.cap.release()
            self.camera_label.clear()
            self.ac_open_cama.setText('打开摄像头')
            self.pushButton.setText('打开摄像头')
    #相机显示
    def show_camera(self):
        flag, self.image= self.cap.read()
        if self.dynamic_draw_flag:
            if self.timer_camera_counter==0:
                self.draw_face_rec()


        show = cv2.resize(self.image, (800, 600))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],QImage.Format_RGB888 )
        self.camera_label.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def draw_face_rec(self):
        result = self.detector.detect_faces(self.image)
        if len(result) == 0 :
            return

        for face in result:
            bouding_boxes = face['box']
            for axis in bouding_boxes:
                if axis<=0 or axis>=self.image.shape[0]-1 or axis>=self.image.shape[1]-1:
                    return

            cv2.rectangle(self.image,(bouding_boxes[0],bouding_boxes[1]),(bouding_boxes[0]+bouding_boxes[2],bouding_boxes[1]+bouding_boxes[3]),(255,0,0),2)


    def Checkin(self):

        if self.timer_camera.isActive()==False:
            msg = QtWidgets.QMessageBox.warning(self, u"warning", u"没有检测到摄像头", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)

        else:
            #启动识别算法线程
            self.RecogImage = self.image.copy()
            try:
                self.FaceThread.SetImg(self.image)
            except:
                pass
    # def button_wrtieface_click(self):
    #     if self.timer_camera.isActive() == False:
    #         msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please open your camara ", buttons=QtWidgets.QMessageBox.Ok,
    #                                             defaultButton=QtWidgets.QMessageBox.Ok)
    #     else:
    #         name,ok = QInputDialog.getText(self, "Your name ", "Your name",
    #                                         QLineEdit.Normal, self.nameLable.text())
    #         if(ok and (len(name)!=0)):
    #             add_new_face(self.image,name)
    def ShowInTab(self,bound0,bound1,bound2,bound3,name):
        try:
            face = self.RecogImage[bound1:bound1 + bound3,
                        bound0:bound0 + bound2]
            show = cv2.resize(face, (200,200))
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            pix = mask_image(show)
            if self.textlabel_list.__len__()==0:
                self.append_label()
        except:
            return

        try:
            if self.check_name(name)==True:
                for i,text_label in enumerate(self.textlabel_list):
                    if not text_label.text():
                        print('doing')
                        self.facelabel_list[i].setPixmap(pix)
                        tx = time.strftime('%Y-%m-%d\n%H:%M:%S')
                        all_str = '姓名:#' + name + '#\n' + '时间:' + tx
                        text_label.setText(all_str)
                        break

                    if i==self.textlabel_list.__len__()-1 and text_label.text():
                        self.append_label()
                        print('1')
                        self.facelabel_list[-1].setPixmap(pix)
                        tx = time.strftime('%Y-%m-%d\n%H:%M:%S')
                        all_str = '姓名:#' + name + '#\n' + '时间:' + tx
                        self.textlabel_list[-1].setText(all_str)
        except:
            return

    def check_name(self,name):
        if name not in self.long_name_list:
            if name == 'Unknown':
                tx1 = time.strftime('%Y-%m-%d %H:%M:%S')
                str_1 = '检测到未知人员于' + tx1 + '出现\n'
                # self.textBrowser.insertPlainText(str_1)
                # self.textBrowser.verticalScrollBar().setValue(self.textBrowser.verticalScrollBar().maximum())
            else:
                self.long_name_list.append(name)
                tx1 = time.strftime('%Y-%m-%d %H:%M:%S')
                str_1 = '检测到#' + name + '#于' + tx1 + '出现\n'
                self.textBrowser.insertPlainText(str_1)
                self.textBrowser.verticalScrollBar().setValue(self.textBrowser.verticalScrollBar().maximum())
        if name not in self.name_list:
            self.name_list.append(name)
            return True

        else:
            return False

    def del_long_name(self):
        self.long_name_list.clear()

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"关闭?")

        msg.addButton(ok,QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'是')
        cacel.setText(u'否')

        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()

    def clear_all_label(self):
        self.facelabel_list = []
        self.textlabel_list = []

    def clear_all_text(self):
        self.textBrowser.clear()

    def draw_face(self,src,radius):
        if src is None:
            return

        si = QSize(2*radius,2*radius)
        mask = QBitmap(si)
        painter = QPainter(mask)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(0,0,si.width(),si.height(),Qt.white)
        painter.setBrush(QColor(0,0,0))
        painter.drawRoundedRect(0,0,si.width(),si.height(),99,99)
        image = QPixmap(src.scaled(si))
        image.setMask(mask)

        return image

    def label_draw_face(self,label,image):
        label.setMaximumSize(200, 200)
        label.setMinimumSize(200, 200)
        label.radius = 100
        target = QPixmap(label.size())
        target.fill(Qt.transparent)

        p = QPixmap.fromImage(image).scaled(
            200, 200, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

        painter = QPainter(target)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.HighQualityAntialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        path = QPainterPath()
        path.addRoundedRect(0, 0, label.width(), label.height(), label.radius, label.radius)

        painter.setClipPath(path)
        painter.drawPixmap(0, 0, p)
        label.setPixmap(target)
        print('ck1')
        return label



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
