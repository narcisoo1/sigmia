from PyQt5 import QtWidgets
from interfaces.OutputWindow import Ui_MainWindow


class OutputController(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(OutputController, self).__init__(parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
