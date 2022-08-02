# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/pre_trained.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(318, 152)
        MainWindow.setStyleSheet("background-color: rgb(242, 242, 242);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(16, 16, 286, 90))
        self.widget.setStyleSheet("background-color: rgb(196, 196, 196);")
        self.widget.setObjectName("widget")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(10, 10, 60, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(20, 40, 92, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.comboBox_loss = QtWidgets.QComboBox(self.widget)
        self.comboBox_loss.setGeometry(QtCore.QRect(120, 40, 132, 20))
        self.comboBox_loss.setStyleSheet("background-color: rgb(242, 242, 242);")
        self.comboBox_loss.setObjectName("comboBox_loss")
        self.btn_load = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load.setGeometry(QtCore.QRect(122, 116, 50, 20))
        self.btn_load.setStyleSheet("background-color: rgb(48, 71, 140);\n"
"color: rgb(242, 242, 242);")
        self.btn_load.setObjectName("btn_load")
        self.btn_save = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save.setGeometry(QtCore.QRect(182, 116, 50, 20))
        self.btn_save.setStyleSheet("background-color: rgb(48, 71, 140);\n"
"color: rgb(242, 242, 242);")
        self.btn_save.setObjectName("btn_save")
        self.btn_apply = QtWidgets.QPushButton(self.centralwidget)
        self.btn_apply.setGeometry(QtCore.QRect(242, 116, 60, 20))
        self.btn_apply.setStyleSheet("background-color: rgb(48, 71, 140);\n"
"color: rgb(242, 242, 242);")
        self.btn_apply.setObjectName("btn_apply")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pre trained"))
        self.label.setText(_translate("MainWindow", "Setup"))
        self.label_2.setText(_translate("MainWindow", "Output layer"))
        self.btn_load.setText(_translate("MainWindow", "Load"))
        self.btn_save.setText(_translate("MainWindow", "Save"))
        self.btn_apply.setText(_translate("MainWindow", "Apply"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
