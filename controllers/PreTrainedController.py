from PyQt5 import QtWidgets
from interfaces.PreTrainedWindow import Ui_MainWindow


class ControlPreTrainedWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(ControlPreTrainedWindow, self).__init__(parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
