from PyQt5 import QtWidgets
from interfaces.NewLayersWindow import Ui_MainWindow


class ControlNewLayers(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(ControlNewLayers, self).__init__(parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
