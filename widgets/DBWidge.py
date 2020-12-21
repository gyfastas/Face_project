from __future__ import print_function
import sys
from includes.pymysql.PyMySQL import PyMySQL
from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtSql import *
import qdarkstyle

#数据库操作窗口


class TableWidge(QWidget):
    def __init__(self):
        super(TableWidge, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.resize(500,300)
        self.db = PyMySQL('localhost', 'root', 'CockTail', 'TESTDATABASE')
        self.people_size = self.db.get_all_info().__len__()
        self.model = QStandardItemModel(self.people_size,2)
        self.model.setHorizontalHeaderLabels(['姓名','数据记录时间'])
        self.initMenu()
        self.initAnimation()
        for col in range(self.people_size):
            item = QStandardItem(self.db.get_all_name()[col])
            self.model.setItem(col,0,item)
            item = QStandardItem(str(self.db.get_all_time()[col]))
            self.model.setItem(col,1,item)

        self.tableView = QTableView()
        self.tableView.setModel(self.model)
        self.tableView.resizeColumnsToContents()
        layout = QVBoxLayout()
        layout.addWidget(self.tableView)
        self.setLayout(layout)

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
        self.ac_delete_all  = self._contextMenu.addAction('删除所有数据',self.delete_all)
    def initAnimation(self):
        # 按钮动画
        self._animation = QPropertyAnimation(
            self._contextMenu, b'geometry', self,
            easingCurve=QEasingCurve.Linear, duration=300)
        # easingCurve 修改该变量可以实现不同的效果



    def delete_all(self):
        self.db.delete_all()
        self.model.clear()


class DBWidge(QWidget):
    def __init__(self,parent=None):
        super(DBWidge,self).__init__(parent)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        self.db = PyMySQL('localhost', 'root', 'CockTail', 'TESTDATABASE')
        self.initButton()
        self.initSlot()
        self.table = TableWidge()
        self.show()

    def initButton(self):
        self.connectButton = QPushButton('连接数据库',self)
        self.connectButton.setGeometry(100,100,100,50)
        self.checkButton = QPushButton('查看数据',self)
        self.checkButton.setGeometry(100,150,100,50)

    def initSlot(self):
        self.connectButton.clicked.connect(self.connectDB)
        self.checkButton.clicked.connect(self.openView)

    def connectDB(self):
        try:
            self.db.connect()
            QMessageBox.about(self,'Connection','成功连接数据库')

        except:
            QMessageBox.about(self,'Connection','连接数据库失败')

    def openView(self):
        if self.table.isHidden():
            self.table.show()

        else:
            self.table.hide()


if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = DBWidge()
    ui.show()
    sys.exit(app.exec_())
