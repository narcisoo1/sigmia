# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/fine_tuning.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(318, 472)
        MainWindow.setMinimumSize(QtCore.QSize(318, 472))
        MainWindow.setMaximumSize(QtCore.QSize(318, 472))
        MainWindow.setStyleSheet("background-color: rgb(242, 242, 242);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(16, 16, 286, 110))
        self.widget.setStyleSheet("background-color: rgb(196, 196, 196);")
        self.widget.setObjectName("widget")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(10, 10, 60, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(20, 40, 91, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(20, 70, 91, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.comboBox_output_layer = QtWidgets.QComboBox(self.widget)
        self.comboBox_output_layer.setGeometry(QtCore.QRect(120, 40, 131, 20))
        self.comboBox_output_layer.setStyleSheet("background-color: rgb(242, 242, 242);")
        self.comboBox_output_layer.setObjectName("comboBox_output_layer")
        self.btn_apply_2 = QtWidgets.QPushButton(self.widget)
        self.btn_apply_2.setGeometry(QtCore.QRect(120, 70, 131, 20))
        self.btn_apply_2.setStyleSheet("background-color: rgb(48, 71, 140);\n"
"color: rgb(242, 242, 242);")
        self.btn_apply_2.setObjectName("btn_apply_2")
        self.widget_2 = QtWidgets.QWidget(self.centralwidget)
        self.widget_2.setGeometry(QtCore.QRect(16, 136, 286, 140))
        self.widget_2.setStyleSheet("background-color: rgb(196, 196, 196);")
        self.widget_2.setObjectName("widget_2")
        self.label_5 = QtWidgets.QLabel(self.widget_2)
        self.label_5.setGeometry(QtCore.QRect(10, 10, 60, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.widget_2)
        self.label_6.setGeometry(QtCore.QRect(20, 40, 91, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.widget_2)
        self.label_7.setGeometry(QtCore.QRect(20, 70, 91, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.widget_2)
        self.label_8.setGeometry(QtCore.QRect(20, 100, 91, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.comboBox_loss = QtWidgets.QComboBox(self.widget_2)
        self.comboBox_loss.setGeometry(QtCore.QRect(120, 40, 131, 20))
        self.comboBox_loss.setStyleSheet("background-color: rgb(242, 242, 242);")
        self.comboBox_loss.setObjectName("comboBox_loss")
        self.comboBox_opt = QtWidgets.QComboBox(self.widget_2)
        self.comboBox_opt.setGeometry(QtCore.QRect(120, 70, 131, 20))
        self.comboBox_opt.setStyleSheet("background-color: rgb(242, 242, 242);")
        self.comboBox_opt.setObjectName("comboBox_opt")
        self.comboBox_metrics = QtWidgets.QComboBox(self.widget_2)
        self.comboBox_metrics.setGeometry(QtCore.QRect(120, 100, 131, 20))
        self.comboBox_metrics.setStyleSheet("background-color: rgb(242, 242, 242);")
        self.comboBox_metrics.setObjectName("comboBox_metrics")
        self.widget_3 = QtWidgets.QWidget(self.centralwidget)
        self.widget_3.setGeometry(QtCore.QRect(16, 286, 286, 140))
        self.widget_3.setStyleSheet("background-color: rgb(196, 196, 196);")
        self.widget_3.setObjectName("widget_3")
        self.label_9 = QtWidgets.QLabel(self.widget_3)
        self.label_9.setGeometry(QtCore.QRect(10, 10, 60, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.widget_3)
        self.label_10.setGeometry(QtCore.QRect(20, 40, 91, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.widget_3)
        self.label_11.setGeometry(QtCore.QRect(20, 70, 91, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.comboBox_epochs = QtWidgets.QComboBox(self.widget_3)
        self.comboBox_epochs.setGeometry(QtCore.QRect(120, 40, 131, 20))
        self.comboBox_epochs.setStyleSheet("background-color: rgb(242, 242, 242);")
        self.comboBox_epochs.setObjectName("comboBox_epochs")
        self.comboBox_batch = QtWidgets.QComboBox(self.widget_3)
        self.comboBox_batch.setGeometry(QtCore.QRect(120, 70, 131, 20))
        self.comboBox_batch.setStyleSheet("background-color: rgb(242, 242, 242);")
        self.comboBox_batch.setObjectName("comboBox_batch")
        self.btn_load = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load.setGeometry(QtCore.QRect(122, 436, 50, 20))
        self.btn_load.setStyleSheet("background-color: rgb(48, 71, 140);\n"
"color: rgb(242, 242, 242);")
        self.btn_load.setObjectName("btn_load")
        self.btn_save = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save.setGeometry(QtCore.QRect(182, 436, 50, 20))
        self.btn_save.setStyleSheet("background-color: rgb(48, 71, 140);\n"
"color: rgb(242, 242, 242);")
        self.btn_save.setObjectName("btn_save")
        self.btn_apply = QtWidgets.QPushButton(self.centralwidget)
        self.btn_apply.setGeometry(QtCore.QRect(242, 436, 60, 20))
        self.btn_apply.setStyleSheet("background-color: rgb(48, 71, 140);\n"
"color: rgb(242, 242, 242);")
        self.btn_apply.setObjectName("btn_apply")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Fine tuning"))
        self.label.setText(_translate("MainWindow", "Setup"))
        self.label_2.setText(_translate("MainWindow", "Outout layer"))
        self.label_3.setText(_translate("MainWindow", "New layers"))
        self.btn_apply_2.setText(_translate("MainWindow", "Add layers"))
        self.label_5.setText(_translate("MainWindow", "Compile"))
        self.label_6.setText(_translate("MainWindow", "Loss functions"))
        self.label_7.setText(_translate("MainWindow", "Optimizer"))
        self.label_8.setText(_translate("MainWindow", "Metrics"))
        self.label_9.setText(_translate("MainWindow", "Training"))
        self.label_10.setText(_translate("MainWindow", "Epochs"))
        self.label_11.setText(_translate("MainWindow", "Batch  size"))
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