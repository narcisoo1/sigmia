from PyQt5 import QtWidgets
from interfaces.NewLayersWindow import Ui_MainWindow

class ControlNewLayers(QtWidgets.QMainWindow):
    """
    Classe de controle para a janela de novas camadas.
    Herda de QtWidgets.QMainWindow para criar uma janela principal.
    """

    def __init__(self, parent=None):
        """
        Construtor da classe ControlNewLayers.

        Args:
            parent: O widget pai, se houver.
        """
        super(ControlNewLayers, self).__init__(parent)

        self.ui = Ui_MainWindow()  # Instancia a classe de interface do usuário
        self.ui.setupUi(self)  # Configura a interface do usuário na janela principal
