'''
This is label that used to dynamically show face in tab widge
'''
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class faceLabel(QLabel):
    def __init__(self):
        super(Qlabel,self).__init__()
        self.dynamicTimer = QTimer()