from __future__ import print_function
import sys
from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtWidgets import QApplication, QLineEdit, QInputDialog, QGridLayout, QLabel, QPushButton, QFrame, QWidget,QMenu



# 显示相机的窗口(继承label的一个类

class CameraWidget(QtWidgets.QLabel):
    def __init__(self):
        super(CameraWidget, self).__init__(parent)

        #相机区域

        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.resize(800,600)

    def On_Capture(self):


app = QtWidgets.QApplication(sys.argv)
ui = Ui_MainWindow()
ui.show()
ui.scrollAreaWidgetContents.show()
sys.exit(app.exec_())