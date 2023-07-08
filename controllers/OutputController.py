from PyQt5 import QtWidgets
from interfaces.OutputWindow import Ui_MainWindow

class OutputController(QtWidgets.QMainWindow):
    """
    Classe de controle para a janela de saída.
    Herda de QtWidgets.QMainWindow para criar uma janela principal.
    """

    def __init__(self, parent=None):
        """
        Construtor da classe OutputController.

        Args:
            parent: O widget pai, se houver.
        """
        super(OutputController, self).__init__(parent)

        self.ui = Ui_MainWindow()  # Instancia a classe de interface do usuário
        self.ui.setupUi(self)  # Configura a interface do usuário na janela principal
