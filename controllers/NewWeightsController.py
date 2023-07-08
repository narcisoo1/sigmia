from PyQt5 import QtWidgets, QtCore
from interfaces.NewWeightsWindow import Ui_MainWindow
import json

class ControlNewWeights(QtWidgets.QMainWindow):
    """
    Classe de controle para a janela de novos pesos.
    Herda de QtWidgets.QMainWindow para criar uma janela principal.
    """

    def __init__(self, parent=None):
        """
        Construtor da classe ControlNewWeights.

        Args:
            parent: O widget pai, se houver.
        """
        super(ControlNewWeights, self).__init__(parent)

        self.ui = Ui_MainWindow()  # Instancia a classe de interface do usuário
        self.ui.setupUi(self)  # Configura a interface do usuário na janela principal

        self.translate_ui()  # Traduz os elementos da interface do usuário

    def translate_ui(self):
        """
        Traduz os elementos da interface do usuário com base nos dados do arquivo 'configs.json'.
        """
        self.core = json.load(open('controllers/configs.json'))  # Carrega os dados do arquivo 'configs.json'

        _translate = QtCore.QCoreApplication.translate

        # Adiciona os itens da lista 'losses' como opções no comboBox_loss
        for item in self.core['losses']:
            self.ui.comboBox_loss.addItem(item['name'])

        # Adiciona os itens da lista 'optimizers' como opções no comboBox_optimizer
        for item in self.core['optimizers']:
            self.ui.comboBox_optimizer.addItem(item)

        # Adiciona os itens da lista 'metrics-cnn' como opções no comboBox_metric
        for item in self.core['metrics-cnn']:
            self.ui.comboBox_metric.addItem(item['name'])
