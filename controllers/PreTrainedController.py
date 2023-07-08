from PyQt5 import QtWidgets
from interfaces.PreTrainedWindow import Ui_MainWindow

class ControlPreTrainedWindow(QtWidgets.QMainWindow):
    """
    Classe de controle para a janela de modelos pré-treinados.
    Herda de QtWidgets.QMainWindow para criar uma janela principal.
    """

    def __init__(self, parent=None):
        """
        Construtor da classe ControlPreTrainedWindow.

        Args:
            parent: O widget pai, se houver.
        """
        super(ControlPreTrainedWindow, self).__init__(parent)

        self.ui = Ui_MainWindow()  # Instancia a classe de interface do usuário
        self.ui.setupUi(self)  # Configura a interface do usuário na janela principal
