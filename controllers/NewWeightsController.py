from PyQt5 import QtWidgets, QtCore
from interfaces.NewWeightsWindow import Ui_MainWindow
import json

class ControlNewWeights(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(ControlNewWeights, self).__init__(parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.translate_ui()


    def translate_ui(self):
        self.core = json.load(open('controllers/configs.json'))

        _translate = QtCore.QCoreApplication.translate
        for item in self.core['losses']:
            self.ui.comboBox_loss.addItem(item['name'])

        for item in self.core['optimizers']:
            self.ui.comboBox_optimizer.addItem(item)

        for item in self.core['metrics-cnn']:
            self.ui.comboBox_metric.addItem(item['name'])