from __future__ import print_function
import sys
from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtWidgets import QApplication, QLineEdit, QInputDialog, QGridLayout, QLabel, QPushButton, QFrame, QWidget,QMenu
import os
sys.setrecursionlimit(1000000)


import cv2
import numpy as np



class Showface(QtWidgets.QScrollArea):
    def __init__(self, parent=None):
        super(Showface, self).__init__(parent)

        self.CAM_NUM = 0
        self.resize(1022, 670)

        self.set_ui()
        self.slot_init()


        #初始化右键下拉菜单
        self.initMenu()
        self.initAnimation()
    def set_ui(self):
        self.nameLable = QLabel(" ")
        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        #Scroll area reset
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setGeometry(QtCore.QRect(830, 0, 291, 751))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        #为scroll area 添加一个布局
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 289, 749))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")


        self.label_show_camera = QtWidgets.QLabel()

        
        self.label_show_camera.setFixedSize(600, 600)
        self.label_show_camera.setAutoFillBackground(False)


        self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)

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
        self.ac_open_cama = self._contextMenu.addAction('打开相机', self.button_open_camera_click)
        self.ac_detection = self._contextMenu.addAction('识别', self.button_detection_click)
        self.ac_record = self._contextMenu.addAction('记录', self.button_record_click)
    def initAnimation(self):
        # 按钮动画
        self._animation = QPropertyAnimation(
            self._contextMenu, b'geometry', self,
            easingCurve=QEasingCurve.Linear, duration=300)
        # easingCurve 修改该变量可以实现不同的效果


    def slot_init(self):

        self.timer_camera.timeout.connect(self.show_camera)

    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please check you have connected your camera", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        
            else:
                self.timer_camera.start(50)
                self.ac_open_cama.setText('关闭相机')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.ac_open_cama.setText('打开相机')
    def show_camera(self):
        flag, self.image= self.cap.read()
        if self.recognition_flag==True:
            self.detect_recognition()
        show = cv2.resize(self.image, (800, 600))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.selfat_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def button_detection_click(self):
        if self.timer_camera.isActive()==False:
            msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"pleas open your camara", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)

        else:
            if self.recognition_flag==False:
                self.recognition_flag=True

            else:
                self.recognition_flag=False


    def button_record_click(self):
        if self.timer_camera.isActive()==False:
            msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"please open your camara", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            if self.recognition_flag==False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"you are not using recognition", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                file=open('record.txt','a')
                file.write('name: ')
                file.write(str(self.name_list))
                tx = time.strftime('%Y-%m-%d %H:%M:%S')
                file.write('\n')
                file.write(tx)
                file.close()

    def button_wrtieface_click(self):
        if self.timer_camera.isActive() == False:
            msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please open your camara ", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            name,ok = QInputDialog.getText(self, "Your name ", "Your name",
                                            QLineEdit.Normal, self.nameLable.text())
            if(ok and (len(name)!=0)):
                add_new_face(self.image,name)
    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"close", u"close?")

        msg.addButton(ok,QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'Yes')
        cacel.setText(u'Cancel')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()
    def detect_recognition(self):
        result = datas.detector.detect_faces(self.image)
        aligment_imgs = []
        originfaces = []
        # 检测，标定landmark
        for face in result:
            temp_landmarks = []
            bouding_boxes = face['box']
            keypoints = face['keypoints']

            cv2.rectangle(self.image, (bouding_boxes[0], bouding_boxes[1]),
                          (bouding_boxes[0] + bouding_boxes[2], bouding_boxes[1] + bouding_boxes[3]), (255, 0, 0), 2)

            faces = self.image[bouding_boxes[1]:bouding_boxes[1] + bouding_boxes[3],
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

            faces = DataPrepare.alignment(faces, temp_landmarks)
            faces = np.transpose(faces, (2, 0, 1)).reshape(1, 3, 112, 96)
            faces = (faces - 127.5) / 128.0
            aligment_imgs.append(faces)
        length = len(aligment_imgs)
        aligment_imgs = np.array(aligment_imgs)
        aligment_imgs = np.reshape(aligment_imgs, (length, 3, 112, 96))
        output_imgs_features = datas.get_imgs_features(aligment_imgs)
        cos_distances_list = []
        result_index = []
        for img_feature in output_imgs_features:
            cos_distance_list = [datas.cal_cosdistance(img_feature, test_img_feature) for test_img_feature in
                                 imgs_features]
            cos_distances_list.append(cos_distance_list)
        for imgfeature in cos_distances_list:
            if max(imgfeature) < thres:
                result_index.append(-1)
            else:
                result_index.append(imgfeature.index(max(imgfeature)))
        for i, index in enumerate(result_index):
            name = imgs_name_list[i]
            tx = time.strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(self.image, name, (result[i]['box'][0], result[i]['box'][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                        1)
            cv2.putText(self.image, str('time:') + str(tx), (result[i]['box'][0] + 10, result[i]['box'][1] + 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

app = QtWidgets.QApplication(sys.argv)
ui = Ui_MainWindow()
ui.show()
ui.scrollAreaWidgetContents.show()
sys.exit(app.exec_())