# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/new_layers.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(318, 202)
        MainWindow.setMinimumSize(QtCore.QSize(318, 0))
        MainWindow.setMaximumSize(QtCore.QSize(318, 16777215))
        MainWindow.setStyleSheet("background-color: rgb(242, 242, 242);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(16, 16, 286, 140))
        self.widget.setStyleSheet("background-color: rgb(196, 196, 196);")
        self.widget.setObjectName("widget")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(20, 10, 90, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.comboBox_loss = QtWidgets.QComboBox(self.widget)
        self.comboBox_loss.setGeometry(QtCore.QRect(120, 10, 122, 20))
        self.comboBox_loss.setStyleSheet("background-color: rgb(242, 242, 242);")
        self.comboBox_loss.setObjectName("comboBox_loss")
        self.btn_save = QtWidgets.QPushButton(self.widget)
        self.btn_save.setGeometry(QtCore.QRect(242, 10, 30, 20))
        self.btn_save.setStyleSheet("background-color: rgb(48, 71, 140);\n"
"color: rgb(242, 242, 242);")
        self.btn_save.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icons/add.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_save.setIcon(icon)
        self.btn_save.setObjectName("btn_save")
        self.listWidget = QtWidgets.QListWidget(self.widget)
        self.listWidget.setGeometry(QtCore.QRect(20, 40, 252, 90))
        self.listWidget.setObjectName("listWidget")
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        self.btn_apply = QtWidgets.QPushButton(self.centralwidget)
        self.btn_apply.setGeometry(QtCore.QRect(240, 166, 60, 20))
        self.btn_apply.setStyleSheet("background-color: rgb(48, 71, 140);\n"
"color: rgb(242, 242, 242);")
        self.btn_apply.setObjectName("btn_apply")
        self.btn_save_2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save_2.setGeometry(QtCore.QRect(180, 166, 50, 20))
        self.btn_save_2.setStyleSheet("background-color: rgb(48, 71, 140);\n"
"color: rgb(242, 242, 242);")
        self.btn_save_2.setObjectName("btn_save_2")
        self.btn_load = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load.setGeometry(QtCore.QRect(120, 166, 50, 20))
        self.btn_load.setStyleSheet("background-color: rgb(48, 71, 140);\n"
"color: rgb(242, 242, 242);")
        self.btn_load.setObjectName("btn_load")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "New layers"))
        self.label_2.setText(_translate("MainWindow", "Output layer"))
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        item = self.listWidget.item(0)
        item.setText(_translate("MainWindow", "ex"))
        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.btn_apply.setText(_translate("MainWindow", "Apply"))
        self.btn_save_2.setText(_translate("MainWindow", "Save"))
        self.btn_load.setText(_translate("MainWindow", "Load"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
