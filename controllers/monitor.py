from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
import pyqtgraph as pg

class LossMonitor(QWidget):
    """
    Uma classe QWidget personalizada para monitorar curvas de perda usando PyQt5 e pyqtgraph.
    """

    def __init__(self):
        super().__init__()
        self.model_names = []   # Lista para armazenar os nomes dos modelos
        self.loss_curves = {}   # Dicionário para armazenar as curvas de perda de cada modelo

        layout = QVBoxLayout()  # Layout vertical para organizar os widgets
        self.setLayout(layout)  # Define o layout para o widget

        self.plot_widget = pg.PlotWidget()  # Cria um widget de plotagem usando pyqtgraph
        layout.addWidget(self.plot_widget)  # Adiciona o widget de plotagem ao layout

        self.plot_widget.setLabel('left', 'Perda')  # Define o rótulo para o eixo Y do gráfico
        self.plot_widget.setLabel('bottom', 'Época/Dobragem')  # Define o rótulo para o eixo X do gráfico

        self.plot = self.plot_widget.plot()  # Cria um item de plotagem para o widget de plotagem

    def set_model_names(self, model_names):
        """
        Define os nomes dos modelos para os quais as curvas de perda serão monitoradas.

        Args:
            model_names (list): Lista de nomes dos modelos.
        """
        self.model_names = model_names

        # Inicializa uma curva de perda vazia para cada modelo
        for model_name in model_names:
            self.loss_curves[model_name] = []

    def update_loss(self, model_name, loss):
        """
        Atualiza a curva de perda para um modelo específico.

        Args:
            model_name (str): Nome do modelo.
            loss (float): Valor de perda a ser adicionado à curva.
        """
        self.loss_curves[model_name].append(loss)

        # Atualiza os dados do gráfico com a nova curva de perda
        x = list(range(len(self.loss_curves[model_name])))
        y = self.loss_curves[model_name]
        self.plot.setData(x, y)


class MyWidget(QWidget):
    """
    Uma classe QWidget personalizada para exibir um rótulo.
    """

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        Inicializa a interface do usuário criando e organizando os widgets.
        """
        self.label = QLabel("Olá, mundo!")  # Cria um widget QLabel com o texto inicial
        layout = QVBoxLayout()  # Layout vertical para organizar os widgets
        layout.addWidget(self.label)  # Adiciona o widget de rótulo ao layout
        self.setLayout(layout)  # Define o layout para o widget

    def update_label_text(self, new_text):
        """
        Atualiza o texto do rótulo.

        Args:
            new_text (str): Novo texto para o rótulo.
        """
        self.label.setText(new_text)
