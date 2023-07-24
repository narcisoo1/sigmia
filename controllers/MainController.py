from importlib.util import LazyLoader
import sys
import os
from os.path import expanduser
from pathlib import Path
import json
import pickle
import time
import copy
import matplotlib
import numpy as num
from datetime import datetime
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from matplotlib import image, pyplot as plt
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

from csv import writer

from controllers.CoreCNN import CoreCNN
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QTimer
from interfaces.MainWindow import Ui_MainWindow
from controllers.NewWeightsController import ControlNewWeights
from controllers.FineTuningController import ControlFineTuning
from controllers.PreTrainedController import ControlPreTrainedWindow
from controllers.OutputController import OutputController
import threading
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import time
import traceback, sys
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, precision_recall_curve, roc_curve, auc, f1_score, confusion_matrix

#Import svm model
from sklearn import svm
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import gc
import os
import glob
import csv
from .FeatureExtractor import FeatureExtractor
from .monitor import LossMonitor,MyWidget

os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
pin_memory=False




class BarGraphItem(pg.BarGraphItem):
    """
    Classe que herda de pg.BarGraphItem e adiciona um evento de clique duplo do mouse para definir a escala do gráfico.

    Métodos:
        __init__: Construtor da classe.
        mouseDoubleClickEvent: Manipulador do evento de clique duplo do mouse.
    """

    def __init__(self, *args, **kwargs):
        """
        Construtor da classe.

        Parâmetros:
            *args: Argumentos posicionais para o construtor da classe base.
            **kwargs: Argumentos nomeados para o construtor da classe base.
        """
        pg.BarGraphItem.__init__(self, *args, **kwargs)

    def mouseDoubleClickEvent(self, e):
        """
        Manipulador do evento de clique duplo do mouse.

        Parâmetros:
            e: Evento de clique do mouse.
        """
        # Definir a escala do gráfico para 0.2
        self.setScale(0.2)


class WorkerSignals(QObject):
    """
    Classe que define os sinais utilizados pela classe Worker.

    Sinais:
        finished: Sinal emitido quando a tarefa é concluída.
        error: Sinal emitido em caso de erro durante a execução da tarefa.
               O sinal contém uma tupla com informações sobre a exceção ocorrida.
        result: Sinal emitido com o resultado da execução da tarefa.
        progress: Sinal emitido para informar o progresso da tarefa.
                  O sinal contém um valor inteiro representando o progresso em percentagem.
    """

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)



class Worker(QRunnable):
    """
    Classe que representa uma tarefa executada em segundo plano.

    Esta classe utiliza a funcionalidade de sinal e slot do PyQt para executar uma função em segundo plano
    e enviar sinais de progresso, resultado e finalização.

    Parâmetros:
        fn (callable): Função a ser executada em segundo plano.
        *args: Argumentos posicionais da função.
        **kwargs: Argumentos nomeados da função.
    """

    finished = pyqtSignal()
    
    def __init__(self, fn, *args, **kwargs):
        """
        Construtor da classe Worker.

        Armazena os argumentos do construtor (reutilizados para processamento) e configura os sinais utilizados.

        Parâmetros:
            fn (callable): Função a ser executada em segundo plano.
            *args: Argumentos posicionais da função.
            **kwargs: Argumentos nomeados da função.
        """
        super(Worker, self).__init__()

        # Armazena os argumentos do construtor
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Adiciona o callback ao dicionário de argumentos nomeados
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Inicializa a função em segundo plano com os argumentos passados.
        '''

        # Obtém os argumentos/kwargs aqui e inicia o processamento usando-os
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Retorna o resultado do processamento
        finally:
            self.signals.finished.emit()  # Concluído





class ApplicationWindow(QtWidgets.QMainWindow,QObject):
    def __init__(self):
        """
        Construtor da classe ApplicationWindow.
        """

        super(ApplicationWindow, self).__init__()
        self.loss_monitor = LossMonitor()
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.coreCNN = CoreCNN()
        self.ui.widget_monitor.setLayout(QVBoxLayout())
        self.ui.widget_monitor.layout().addWidget(self.loss_monitor)
        self.use_gpu = torch.cuda.is_available()
        self.dataset_sizes = None
        self.image_datasets = None
        self.dataloaders = None
        self.trainDataloader = None
        self.testDataloader = None
        self.model = []
        self.data_dir = None
        self.data_iter = None
        self.dataset = None
        self.labels = None
        self.output = None
        self.p_train = None
        self.INPUT = 'input'
        self.TRAIN = 'train'
        self.VAL = 'val'
        self.TEST = 'test'
        self.data_transforms = self.dt_transforms()
        self.class_names = None
        self.optimizer = None
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.finished = None
        self.data = {}
        self.translate_ui()
        self.init_signals()
        self.gpu_disponivel()
        self.threadpool = QThreadPool()


    def init_signals(self):
        """
        Configura os sinais e slots na interface do usuário.
        """

        self.ui.stackedWidget.setCurrentIndex(0)

        self.tabs = [self.ui.btn_tab_CNN, self.ui.btn_tab_Classf, self.ui.btn_tab_MC]
        self.logs = [self.ui.plainTextEdit_log1, self.ui.plainTextEdit_log2]

        # Instanciação dos controladores
        self.controlNewWeights = ControlNewWeights()
        self.controlFineTuning = ControlFineTuning()
        self.controlPreTrainedWindow = ControlPreTrainedWindow()
        self.controlOutput = OutputController()

        # Configuração dos botões de navegação de pastas
        self.ui.btn_browser_input_1.clicked.connect(lambda: self.open_browse(self.ui.line_input_1, 'cnn', 'input_folder'))
        self.ui.btn_browser_input_2.clicked.connect(lambda: self.open_browse(self.ui.line_input_2, 'classif', 'input_folder'))
        self.ui.btn_browser_input_3.clicked.connect(lambda: self.open_browse(self.ui.line_input_3, 'exib', 'input_folder'))
        self.ui.btn_browser_output_1.clicked.connect(lambda: self.open_browse(self.ui.line_output_1, 'cnn', 'output_folder'))
        self.ui.btn_browser_output_2.clicked.connect(lambda: self.open_browse(self.ui.line_output_2, 'classif', 'output_folder'))
        self.ui.btn_browser_output_3.clicked.connect(lambda: self.open_browse(self.ui.line_output_3, 'exib', 'output_folder'))

        # Configuração dos botões de troca de abas
        self.ui.btn_tab_CNN.clicked.connect(lambda: self.charge_tab(0))
        self.ui.btn_tab_Classf.clicked.connect(lambda: self.charge_tab(1))
        self.ui.btn_tab_MC.clicked.connect(lambda: self.charge_tab(2))

        # Configuração dos botões de configuração
        self.ui.conf_btn_nw.clicked.connect(lambda: self.controlNewWeights.show())
        self.ui.conf_btn_ft.clicked.connect(lambda: self.controlFineTuning.show())
        self.ui.conf_btn_pt.clicked.connect(lambda: self.controlPreTrainedWindow.show())

        # Configuração dos botões de execução
        self.ui.btn_cnn_run.clicked.connect(lambda: self.run())
        self.ui.btn_classifi_run.clicked.connect(lambda: self.run_classificacao())
        self.ui.btn_exib_run.clicked.connect(lambda: self.run_exib())

        # Configuração dos controles deslizantes e caixas de seleção
        self.charged_init = False
        self.ui.horizontalSlider_cnn.valueChanged.connect(self.onChangeSliderCNN)
        self.ui.spinBox_cnn_train.valueChanged.connect(self.onChangeSpinBoxCNNTrain)
        self.ui.spinBox_cnn_test.valueChanged.connect(self.onChangeSpinBoxCNNTest)

        # Configuração dos botões de seleção de configuração e tipo de treinamento
        self.radio_setup_cnn_selected = 0
        self.btns_setup = [self.ui.conf_btn_nw, self.ui.conf_btn_ft, self.ui.conf_btn_pt]
        self.ui.radio_btn_nw.clicked.connect(lambda: self.onClickedRadioSetup(0))
        self.ui.radio_btn_ft.clicked.connect(lambda: self.onClickedRadioSetup(1))
        self.ui.radio_btn_pt.clicked.connect(lambda: self.onClickedRadioSetup(2))

        self.radio_tt_cnn_selected = 0
        self.btsns_tt_cnn = [self.ui.radio_btn_tt_cnn, self.ui.radio_btn_tt_classifi]
        self.ui.radio_btn_tt_cnn.clicked.connect(lambda: self.onClickedRadioTT(0))
        self.ui.radio_btn_kf_cnn.clicked.connect(lambda: self.onClickedRadioTT(1))

        # Configuração do duplo clique em item da lista de datasets
        self.ui.listWidget_datasets.itemDoubleClicked.connect(self.onItemDoubleClicked)
        
        #Configuração botão reset tabela de exibição
        self.ui.btn_exib_reset.clicked.connect(lambda: self.reset_tablewidget())
        
        #oculta botões sem uso
        self.ui.btn_cnn_pause.hide()
        self.ui.btn_cnn_stop.hide()
        self.ui.conf_btn_pt.hide()
        self.ui.btn_output_layer.hide()
        self.ui.btn_load_setup.hide()
        self.ui.btn_save_setup.hide()
        self.ui.btn_save_tt_cnn.hide()
        self.ui.btn_load_tt_cnn.hide()
        self.ui.btn_gen_tt_cnn.hide()
        self.ui.btn_classifi_pause.hide()
        self.ui.btn_classifi_stop.hide()
        self.ui.btn_save_tt_classif.hide()
        self.ui.btn_load_tt_classif.hide()
        self.ui.btn_gen_tt_classif.hide()


        #Reseta tabela dos resultados
        self.reset_tablewidget()



    def translate_ui(self):
        """
        Carrega as configurações de aplicativos, classificações e métricas a partir de um arquivo JSON
        e traduz os textos para exibição na interface do usuário.
        """

        # Carrega as configurações do arquivo JSON
        self.core = json.load(open('controllers/configs.json'))

        _translate = QtCore.QCoreApplication.translate

        # Configurações dos aplicativos
        for name, _ in self.core['applications'].items():
            item = QtWidgets.QListWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setText(_translate("MainWindow", name))
            self.ui.listWidget_arch.addItem(item)

        # Limpa a lista de modelos
        self.ui.listWidget_models.clear()

        # Configurações das classificações
        for name, _ in self.core['classifications'].items():
            item = QtWidgets.QListWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setText(_translate("MainWindow", name))
            self.ui.listWidget_models.addItem(item)

        # Configurações das métricas
        for name, _ in self.core['metrics'].items():
            self._append_items(self.ui.list_metrics, name)


    def onClickedRadioSetup(self, radio_n):
        """
        Evento acionado quando um dos botões de rádio relacionados à configuração é clicado.
        """

        # Define a opção selecionada como radio_n
        self.radio_setup_cnn_selected = radio_n

        # Atualiza as configurações dos botões com base na opção selecionada
        for i in range(3):
            if i == radio_n:
                # Ativa o botão selecionado
                self.btns_setup[i].setEnabled(True)
            else:
                # Desativa os outros botões
                self.btns_setup[i].setEnabled(False)


    def onClickedRadioTT(self, radio_n):
        """
        Evento acionado quando um dos botões de rádio relacionados à divisão de treinamento/teste é clicado.
        """

        # Define a opção selecionada como 0 (nenhuma)
        self.radio_tt_cnn_selected = 0

        # Verifica a opção selecionada e atualiza as configurações dos elementos de interface do usuário
        if radio_n == 0:
            # Opção: Divisão manual
            self.ui.spinBox_cnn_train.setEnabled(True)
            self.ui.spinBox_cnn_test.setEnabled(True)
            self.ui.horizontalSlider_cnn.setEnabled(True)
            self.ui.spinBox_cnn_kf.setEnabled(False)
        elif radio_n == 1:
            # Opção: Divisão k-fold
            self.ui.spinBox_cnn_train.setEnabled(False)
            self.ui.spinBox_cnn_test.setEnabled(False)
            self.ui.horizontalSlider_cnn.setEnabled(False)
            self.ui.spinBox_cnn_kf.setEnabled(True)


    def onChangeSliderCNN(self):
        """
        Evento acionado quando o valor do QSlider horizontalSlider_cnn é alterado.
        """

        # Verifica se a inicialização já foi carregada
        if not self.charged_init:
            # Marca a inicialização como carregada
            self.charged_init = True

            # Atualiza o valor do QSpinBox spinBox_cnn_train com o valor atual do horizontalSlider_cnn
            self.ui.spinBox_cnn_train.setValue(self.ui.horizontalSlider_cnn.value())

            # Atualiza o valor do QSpinBox spinBox_cnn_test com a diferença entre 100 e o valor do horizontalSlider_cnn
            self.ui.spinBox_cnn_test.setValue(abs(self.ui.horizontalSlider_cnn.value() - 100))

            # Marca a inicialização como não carregada
            self.charged_init = False


    def onChangeSpinBoxCNNTrain(self):
        """
        Evento acionado quando o valor do QSpinBox spinBox_cnn_train é alterado.
        """

        # Verifica se a inicialização já foi carregada
        if not self.charged_init:
            # Marca a inicialização como carregada
            self.charged_init = True

            # Atualiza o valor do QSlider horizontalSlider_cnn com o valor atual do spinBox_cnn_train
            self.ui.horizontalSlider_cnn.setValue(self.ui.spinBox_cnn_train.value())

            # Atualiza o valor do QSpinBox spinBox_cnn_test com a diferença entre 100 e o valor do spinBox_cnn_train
            self.ui.spinBox_cnn_test.setValue(abs(self.ui.spinBox_cnn_train.value() - 100))

            # Marca a inicialização como não carregada
            self.charged_init = False


    def onChangeSpinBoxCNNTest(self):
        """
        Evento acionado quando o valor do QSpinBox spinBox_cnn_test é alterado.
        """

        # Verifica se a inicialização já foi carregada
        if not self.charged_init:
            # Marca a inicialização como carregada
            self.charged_init = True

            # Atualiza o valor do QSlider horizontalSlider_cnn com a diferença entre 100 e o valor do spinBox_cnn_test
            self.ui.horizontalSlider_cnn.setValue(abs(self.ui.spinBox_cnn_test.value() - 100))

            # Atualiza o valor do QSpinBox spinBox_cnn_train com a diferença entre 100 e o valor do spinBox_cnn_test
            self.ui.spinBox_cnn_train.setValue(abs(self.ui.spinBox_cnn_test.value() - 100))

            # Marca a inicialização como não carregada
            self.charged_init = False

    def add_items(self, list_widget: QtWidgets.QListWidget, items: list):
        """
        Adiciona uma lista de itens a um QListWidget.

        Parâmetros:
            - list_widget: O QListWidget ao qual os itens serão adicionados.
            - items: Uma lista de itens a serem adicionados.
        """

        # Obtém a tradução localizada do texto dos itens
        _translate = QtCore.QCoreApplication.translate

        # Limpa o QListWidget e adiciona o item 'All'
        self.reset_list(list_widget)

        # Adiciona cada item individualmente ao QListWidget
        for item in items:
            self._append_items(list_widget, item)


    def _append_items(self, list_widget: QtWidgets.QListWidget, item: str):
        """
        Adiciona itens a um QListWidget.

        Parâmetros:
            - list_widget: O QListWidget ao qual os itens serão adicionados.
            - item: O texto do item a ser adicionado.
        """

        # Obtém a tradução localizada do texto do item
        _translate = QtCore.QCoreApplication.translate

        # Cria um novo QListWidgetItem
        _item = QtWidgets.QListWidgetItem()

        # Define as flags do item para permitir que o usuário marque ou desmarque o item
        _item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)

        # Define o estado de marcação do item como não marcado
        _item.setCheckState(QtCore.Qt.Unchecked)

        # Define o texto do item usando a tradução localizada
        _item.setText(_translate("MainWindow", item))

        # Adiciona o item ao QListWidget
        list_widget.addItem(_item)


    def reset_list(self, list_widget: QtWidgets.QListWidget):
        """
        Limpa um QListWidget e adiciona um item especial chamado 'All'.

        Parâmetros:
            - list_widget: O QListWidget a ser resetado.
        """

        # Limpa o QListWidget
        list_widget.clear()

        # Adiciona o item 'All' ao QListWidget
        self._append_items(list_widget, 'All')


    def onItemDoubleClicked(self, item):
        """
        Evento acionado quando um item é clicado duas vezes em uma lista (ou widget semelhante).

        Parâmetros:
            - item: O item que foi clicado duas vezes.
        """

        # Chama a função add_log para adicionar uma mensagem de log indicando que o item foi clicado duas vezes
        self.add_log(f'Debug - {item.text()} double\n')

    def charge_tab(self, index):
        """
        Alterna entre as guias (tabs) na interface do usuário e aplica estilos diferentes para a guia ativa e as guias inativas.

        Parâmetros:
            - index (int): O índice da guia que será ativada.
        """

        # Itera sobre as guias
        for i in range(len(self.tabs)):
            # Define o índice da guia ativa na interface do usuário
            self.ui.stackedWidget.setCurrentIndex(index)

            # Aplica estilos diferentes para a guia ativa e as guias inativas
            if i == index:
                self.tabs[i].setStyleSheet(
                    "background-color: rgb(9, 16, 38);\n"
                    "color: rgb(242, 242, 242);"
                )
            else:
                self.tabs[i].setStyleSheet(
                    "background-color: rgb(67, 83, 115);\n"
                    "color: rgb(242, 242, 242);"
                )


    def add_log(self, text):
        """
        Adiciona texto a uma área de log na interface do usuário.

        Parâmetros:
            - text (str): O texto a ser adicionado ao log.
        """

        # Itera sobre os objetos plainText na lista de logs
        for plainText in self.logs:
            # Insere o texto fornecido na área de log
            plainText.insertPlainText(text)


    def gpu_disponivel(self):
        for x in range(torch.cuda.device_count()):
            self.ui.comboBox_gpu.addItem(torch.cuda.get_device_name(x))
    
    def model_layers(self, model):
        """
        Preenche o combobox na interface do usuário com os nomes dos nós (layers) presentes em um modelo.

        Parâmetros:
            - model: O modelo do qual se deseja obter os nomes dos nós.
        """

        # Obtém os nomes dos nós do modelo
        nodes, _ = get_graph_node_names(model)

        # Itera sobre os nomes dos nós e adiciona-os ao combobox
        for x in nodes:
            self.ui.comboBox_gpu_2.addItem(torch.cuda.get_device_name(x))

    
    def device(self):
        """
        Obtém o dispositivo de processamento selecionado pelo usuário.

        Retorna:
            - device (str): O dispositivo de processamento selecionado ("cpu" para CPU ou "cuda:x" para GPU, onde x é o índice da GPU).
        """

        # Verifica se a opção "Do Not Use" está selecionada no combobox
        if self.ui.comboBox_gpu.currentText() == "Do Not Use":
            return "cpu"
        else:
            # Itera sobre os dispositivos CUDA disponíveis
            for x in range(torch.cuda.device_count()):
                # Verifica se o dispositivo selecionado é igual ao dispositivo atual
                if self.ui.comboBox_gpu.currentText() == torch.cuda.get_device_name(x):
                    # Retorna o dispositivo no formato "cuda:x" (onde x é o índice da GPU)
                    return "cuda:" + str(x)

    def open_browse(self, line, tab_name, type_folder):
        """
        Abre a caixa de diálogo de seleção de diretório e atualiza os campos de linha apropriados na interface do usuário.

        Parâmetros:
            - line (str): A linha correspondente na interface do usuário que será atualizada com o diretório selecionado.
            - tab_name (str): O nome da aba na qual o diretório está sendo selecionado.
            - type_folder (str): O tipo de diretório sendo selecionado (input_folder ou output_folder).

        """

        # Abre a caixa de diálogo de seleção de diretório
        path = QtWidgets.QFileDialog.getExistingDirectory(self)

        # Verifica se um diretório foi selecionado
        if path != '':
            if tab_name == 'cnn' and type_folder == 'input_folder':
                # Atualiza o campo de linha de entrada correspondente na interface do usuário
                self.ui.line_input_1.setText(path)
                self.input = path
            if tab_name == 'cnn' and type_folder == 'output_folder':
                # Atualiza o campo de linha de saída correspondente na interface do usuário
                self.ui.line_output_1.setText(path)
                self.output = path

            if tab_name == 'classif' and type_folder == 'input_folder':
                # Atualiza o campo de linha de entrada correspondente na interface do usuário
                self.ui.line_input_2.setText(path)
                self.ui.line_output_2.setText(path)
                # Popula o widget de lista com os arquivos do diretório selecionado
                self.populate_list_widget(path)
            if tab_name == 'classif' and type_folder == 'output_folder':
                # Atualiza o campo de linha de saída correspondente na interface do usuário
                self.ui.line_output_2.setText(path)

            if tab_name == 'exib' and type_folder == 'input_folder':
                # Atualiza o campo de linha de entrada correspondente na interface do usuário
                self.ui.line_input_3.setText(path)
                # Popula o widget de lista com os arquivos do diretório selecionado
                self.populate_list_widget_exibition(path)
            if tab_name == 'exib' and type_folder == 'output_folder':
                # Atualiza o campo de linha de saída correspondente na interface do usuário
                self.ui.line_output_3.setText(path)



    def open_browse2(self, line, tab_name, type_folder):
        #path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a source directory:', expanduser('~'))
        path = QtWidgets.QFileDialog.getExistingDirectory(self)
        # getOpenFileName(self, 'Open a file', '', 'All Files (*.*)')
        if path != '':
            if tab_name == 'cnn' and type_folder=='input_folder':
                self.ui.line_input_1.setText(path)
                image_datasets = {
                    x: datasets.ImageFolder(
                        os.path.join(path, x), 
                        transform=self.data_transforms[x]
                    )
                    for x in [self.INPUT]
                }
                self.p_train=self.ui.horizontalSlider_cnn.value()/100
                print('\n\n',self.p_train,'\n\n')
                train_size = int(self.p_train * len(image_datasets['input']))
                test_size = int(len(image_datasets['input'])-train_size)
                self.TRAIN, self.TEST = torch.utils.data.random_split(image_datasets['input'], [train_size, test_size])
                #print(len(image_datasets['input']))
                self.image_datasets=image_datasets
                self.labels=image_datasets['input'].classes
                self.dataset_sizes = train_size+test_size
                self.class_names = self.labels
            if tab_name == 'cnn' and type_folder=='output_folder':
                self.ui.line_output_1.setText(path)
                self.output=path
    def load_dataloader(self, train, test, batch_size):
        """
        Carrega os conjuntos de treinamento e teste em objetos DataLoader.

        Parâmetros:
            - train (torch.utils.data.Dataset ou lista): O conjunto de dados de treinamento.
            - test (torch.utils.data.Dataset ou lista): O conjunto de dados de teste.
            - batch_size (int): O tamanho do lote para os objetos DataLoader.

        Retorna:
            - trainDataloader (torch.utils.data.DataLoader ou lista): O DataLoader do conjunto de treinamento.
            - testDataloader (torch.utils.data.DataLoader ou lista): O DataLoader do conjunto de teste.
        """

        # Verifica se a opção de validação cruzada k-fold está selecionada
        if self.ui.radio_btn_kf_cnn.isChecked():
            traindt = []
            testdt = []

            # Itera sobre os splits de treinamento e teste
            for x in train:
                # Cria um DataLoader para cada split de treinamento
                traindt.append(torch.utils.data.DataLoader(x, batch_size))
            for x in test:
                # Cria um DataLoader para cada split de teste
                testdt.append(torch.utils.data.DataLoader(x, batch_size))

            # Armazena os DataLoaders dos splits de treinamento e teste
            trainDataloader = traindt
            testDataloader = testdt
        else:
            # Cria um DataLoader para o conjunto de treinamento
            trainDataloader = torch.utils.data.DataLoader(train, batch_size)
            # Cria um DataLoader para o conjunto de teste
            testDataloader = torch.utils.data.DataLoader(test, batch_size)

        # Retorna os DataLoaders do conjunto de treinamento e teste
        return trainDataloader, testDataloader


    def kfold(self,n_splits):
        path=self.ui.line_input_1.text()
        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(path, x), 
                transform=self.data_transforms[x]
            )
            for x in [self.INPUT]
        }
        kf = KFold(n_splits, shuffle=True)
        train_index, test_index = next(kf.split(image_datasets['input']), None)
        print('\n\n',train_index[1],'\n\n')
        train=[]
        for x in train_index:
            train.append(image_datasets['input'][x])
        test=[]
        for x in test_index:
            test.append(image_datasets['input'][x])
        self.TRAIN, self.TEST = train,test
        self.image_datasets=image_datasets
        self.labels=image_datasets['input'].classes
        self.class_names = self.labels
        print(self.labels)

    def kfold_split(self, n_splits):
        """
        Divide um conjunto de dados em conjuntos de treinamento e teste usando a validação cruzada k-fold.

        Parâmetros:
            - n_splits (int): O número de splits desejado para a validação cruzada k-fold.

        Retorna:
            - train_splits (list): Uma lista contendo os conjuntos de treinamento para cada split.
            - test_splits (list): Uma lista contendo os conjuntos de teste para cada split.
        """

        # Obtém o caminho do diretório de imagens a partir da interface do usuário
        path = self.ui.line_input_1.text()

        # Carrega os dados das imagens a partir do diretório especificado
        image_datasets = datasets.ImageFolder(os.path.join(path), transform=self.data_transforms)

        # Obtém os rótulos das amostras
        labels = image_datasets.targets

        # Cria um objeto KFold para realizar a validação cruzada k-fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Realiza a divisão dos dados em conjuntos de treinamento e teste
        splits = kf.split(image_datasets, labels)

        # Inicializa listas para armazenar os conjuntos de treinamento e teste
        train_splits = []
        test_splits = []

        # Calcula o tamanho aproximado do conjunto de treinamento e o resto da divisão
        num_samples = len(image_datasets)
        train_size = num_samples // n_splits
        remainder = num_samples % n_splits

        # Itera sobre os splits gerados pela validação cruzada k-fold
        for i, (train_index, test_index) in enumerate(splits):
            # Verifica se o split atual está antes do resto
            if i < remainder:
                # Cria um subconjunto de treinamento com tamanho aumentado
                train_dataset = torch.utils.data.Subset(image_datasets, train_index)
                train_size_current = train_size + 1
            else:
                # Cria um subconjunto de treinamento com tamanho normal
                train_dataset = torch.utils.data.Subset(image_datasets, train_index)
                train_size_current = train_size

            # Cria um subconjunto de teste
            test_dataset = torch.utils.data.Subset(image_datasets, test_index)

            # Adiciona os subconjuntos de treinamento e teste às listas correspondentes
            train_splits.append(train_dataset)
            test_splits.append(test_dataset)

            # Imprime o tamanho do conjunto de treinamento atual
            print(len(train_dataset))

        # Retorna as listas contendo os conjuntos de treinamento e teste para cada split
        return train_splits, test_splits




    def train_test(self):
        """
        Divide um conjunto de dados em conjuntos de treinamento e teste usando uma proporção especificada.

        Retorna:
            - train (torch.utils.data.Dataset): O conjunto de dados de treinamento.
            - test (torch.utils.data.Dataset): O conjunto de dados de teste.
        """

        # Obtém o caminho do diretório de imagens a partir da interface do usuário
        path = self.ui.line_input_1.text()

        # Carrega os dados das imagens a partir do diretório especificado
        image_datasets = datasets.ImageFolder(os.path.join(path), transform=self.data_transforms)

        # Obtém a proporção de treinamento especificada pelo usuário
        self.p_train = self.ui.horizontalSlider_cnn.value() / 100

        # Calcula o tamanho do conjunto de treinamento e teste
        train_size = int(self.p_train * len(image_datasets))
        test_size = len(image_datasets) - train_size

        # Realiza a divisão aleatória dos dados em conjuntos de treinamento e teste
        train, test = torch.utils.data.random_split(image_datasets, [train_size, test_size])

        # Armazena os dados do conjunto de dados completo, classes e tamanhos
        self.image_datasets = image_datasets
        self.labels = image_datasets.classes
        self.dataset_sizes = len(image_datasets)
        self.class_names = self.labels

        # Retorna os conjuntos de treinamento e teste
        return train, test




    def accuracy(self, predictions, labels):
        """
        Calcula a precisão do modelo de classificação comparando as predições geradas pelo modelo com as etiquetas verdadeiras.

        Parâmetros:
            predictions: As predições geradas pelo modelo.
            labels: As etiquetas verdadeiras.

        Retorna:
            A precisão do modelo como um tensor.

        Exemplo:
            predictions = tensor([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
            labels = tensor([1, 0, 0])
            accuracy(predictions, labels)  # Retorna tensor(0.6667)
        """
        # Obter as classes previstas com base nas predições
        classes = torch.argmax(predictions, dim=1)
        
        # Calcular a média das correspondências entre as classes previstas e as etiquetas verdadeiras
        accuracy = torch.mean((classes == labels).float())
        
        return accuracy


    def newweights(self,epoch):
        minvalid_loss = num.inf
        for i in range(epoch):
            num_correct=0.0
            num_samples=0.0
            trainloss = 0.0
            self.model.train()     
            for data, label in self.trainDataloader:
                if torch.cuda.is_available():
                    data, label = data.cuda(), label.cuda()
                    self.model.to('cuda')
                
                self.optimizer.zero_grad()
                targets = self.model(data)
                #print("\n\n\n",len(targets),"\n",len(label),"\n\n\n")
                #label = torch.nn.functional.one_hot(label)
                loss = self.loss_fn(targets,label)
                loss.backward()
                self.optimizer.step()
                trainloss += loss.item()
            
            testloss = 0.0
            self.model.eval()
            running_accuracy = 0.00    
            for data, label in self.testDataloader:
                if torch.cuda.is_available():
                    data, label = data.cuda(), label.cuda()
                    self.model.to('cuda')
                
                targets = self.model(data)
                running_accuracy += self.accuracy(targets,label)
                _, predictions = targets.max(1)
                num_correct += (predictions == label).sum()
                num_samples += predictions.size(0)
                loss = self.loss_fn(targets,label)
                testloss = loss.item() * data.size(0)
            running_accuracy /= len(self.testDataloader)

            print(f'Epoch {i+1} \t\t Training data: {trainloss / len(self.trainDataloader)} \t\t Test data: {testloss / len(self.testDataloader)} \t\t Acurácia: {running_accuracy}')
            print(f'Epoch {i+1} \t\t Calculate {num_correct} / {num_samples} \t\t\t\t Acurácia: {float(num_correct/num_samples)}')
            
            if minvalid_loss > testloss:
                print(f'Test data Decreased({minvalid_loss:.6f}--->{testloss:.6f}) \t Saving The Model')
                self.save(self.model)
                minvalid_loss = testloss

    def pretrained(self,epoch):
        for i in range(epoch):
            num_correct=0.0
            num_samples=0.0
            self.model.eval()
            running_accuracy = 0.00    
            for data, label in self.testDataloader:
                #print("\nRODOU\n")
                if torch.cuda.is_available():
                    data, label = data.cuda(), label.cuda()
                    self.model.to('cuda')
                
                targets = self.model(data)
                _, predictions = targets.max(1)
                num_correct += (predictions == label).sum()
                num_samples += predictions.size(0)
                loss = self.loss_fn(targets,label)
                testloss = loss.item() * data.size(0)
            running_accuracy /= len(self.testDataloader)

            print(f'Epoch {i+1} \t\t Calculate {num_correct} / {num_samples} \t\t Acurácia: {float(num_correct/num_samples)}')
            #if minvalid_loss > testloss:
                #print(f'Test data Decreased({minvalid_loss:.6f}--->{testloss:.6f}) \t Saving The Model')
                #self.save(self.model)
                #minvalid_loss = testloss
                #pass
    
    def fineTuning(self, epoch, device, train_dataloader, test_dataloader, models):
        """
        Realiza o processo de ajuste fino (fine-tuning) de um modelo de classificação.

        Parâmetros:
            epoch: O número de épocas de treinamento.
            device: O dispositivo de execução (por exemplo, 'cuda' para GPU ou 'cpu' para CPU).
            train_dataloader: O dataloader de treinamento que fornece os dados de treinamento.
            test_dataloader: O dataloader de teste que fornece os dados de teste.
            models: Uma lista de modelos a serem finetunados.

        Retorna:
            Nenhum valor de retorno.

        Exemplo:
            epoch = 10
            device = 'cuda'
            train_dataloader = [...]
            test_dataloader = [...]
            models = ['resnet50', 'vgg16']
            fineTuning(epoch, device, train_dataloader, test_dataloader, models)
        """
        trainTestORKFold = None
        if self.ui.radio_btn_tt_cnn.isChecked():
            trainTestORKFold = "tt" + self.ui.spinBox_cnn_train.text()
        elif self.ui.radio_btn_kf_cnn.isChecked():
            trainTestORKFold = "kf" + self.ui.radio_btn_kf_cnn.text()

        accurate = []
        epochs = []
        self.loss_monitor.set_model_names(models)

        for j in range(len(models)):
            contador_treino = 0
            x = 0
            minvalid_loss = None
            nome = (
                models[x]
                + "_"
                + "Fine-tuning"
                + "_"
                + self.controlFineTuning.ui.comboBox_loss.currentText()
                + "_"
                + self.controlFineTuning.ui.comboBox_opt.currentText()
                + "_"
                + self.controlFineTuning.ui.comboBox_metrics.currentText()
                + "_"
                + str(self.controlFineTuning.ui.spinBox_epochs.value())
                + "_"
                + str(self.controlFineTuning.ui.spinBox_batch_size.value())
                + "_"
                + trainTestORKFold
                + ".pth"
            )

            self.ui.plainTextEdit_log1.insertPlainText(
                "VAMOS AO PROCESSAMENTO(" + models[x] + ")\n"
            )
            self.model.append(self.get_model(models[x], pretrained=True))

            feature_extractor = FeatureExtractor(self.model[x])

            self.optimizer = self.get_optimizer(
                self.controlFineTuning.ui.comboBox_opt.currentText(), self.model[x]
            )
            
            if self.ui.radio_btn_kf_cnn.isChecked():
                for fold in range(len(train_dataloader)):
                    print(f"Training on fold {fold+1}/{len(train_dataloader)}")

                    for i in range(epoch):
                        num_correct = 0.0
                        num_samples = 0.0
                        losses_train = []
                        accuracies_train = []
                        losses_test = []
                        accuracies_test = []

                        self.model[x].train()

                        for data, label in train_dataloader[fold]:
                            data, label = data.to(device), label.to(device)
                            self.model[x].to(device)
                            targets = self.model[x](data)
                            loss = self.loss_fn(targets, label)
                            accuracies_train.append(
                                label.eq(targets.detach().argmax(dim=1)).float().mean()
                            )
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            losses_train.append(loss.item())

                        testloss = 0.0
                        self.model[x].eval()
                        running_accuracy = 0.0

                        for data, label in test_dataloader[fold]:
                            data, label = data.to(device), label.to(device)
                            self.model[x].to(device)
                            with torch.no_grad():
                                targets = self.model[x](data)
                            running_accuracy += self.accuracy(targets, label)
                            _, predictions = targets.max(1)
                            num_correct += (predictions == label).sum()
                            num_samples += predictions.size(0)
                            loss = self.loss_fn(targets, label)
                            testloss = loss.item() * data.size(0)
                            losses_test.append(loss.item())
                            accuracies_test.append(
                                label.eq(targets.detach().argmax(dim=1)).float().mean()
                            )

                        accurate.append(float(num_correct / num_samples))
                        epochs.append(i)

                        self.loss_monitor.update_loss(models[x], loss)
                        QApplication.processEvents()
                        log = "Acurácia: " + str(float(num_correct / num_samples)) + "\n"
                        self.ui.plainTextEdit_log1.insertPlainText(log)

                        if minvalid_loss is None:
                            minvalid_loss = testloss

                        if minvalid_loss >= testloss:
                            print(
                                f"Test data Decreased({minvalid_loss:.6f}--->{testloss:.6f}) \t Saving The Model"
                            )
                            self.save(self.model[x], nome)
                            minvalid_loss = testloss

                        if fold == 0:
                            print("Salvando Features")
                            self.save_features(feature_extractor, train_dataloader[fold], "train", nome, device)
                            print("Salvamento concluído")
            else:
                for i in range(epoch):
                    num_correct = 0.0
                    num_samples = 0.0
                    losses_train = []
                    accuracies_train = []
                    losses_test = []
                    accuracies_test = []

                    self.model[x].train()

                    for data, label in train_dataloader:
                        data, label = data.to(device), label.to(device)
                        self.model[x].to(device)
                        targets = self.model[x](data)
                        loss = self.loss_fn(targets, label)
                        accuracies_train.append(
                            label.eq(targets.detach().argmax(dim=1)).float().mean()
                        )
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        losses_train.append(loss.item())

                    testloss = 0.0
                    self.model[x].eval()
                    running_accuracy = 0.0

                    for data, label in test_dataloader:
                        data, label = data.to(device), label.to(device)
                        self.model[x].to(device)
                        with torch.no_grad():
                            targets = self.model[x](data)
                        running_accuracy += self.accuracy(targets, label)
                        _, predictions = targets.max(1)
                        num_correct += (predictions == label).sum()
                        num_samples += predictions.size(0)
                        loss = self.loss_fn(targets, label)
                        testloss = loss.item() * data.size(0)
                        losses_test.append(loss.item())
                        accuracies_test.append(
                            label.eq(targets.detach().argmax(dim=1)).float().mean()
                        )

                    accurate.append(float(num_correct / num_samples))
                    epochs.append(i)

                    self.loss_monitor.update_loss(models[x], loss)
                    QApplication.processEvents()
                    log = "Acurácia: " + str(float(num_correct / num_samples)) + "\n"
                    self.ui.plainTextEdit_log1.insertPlainText(log)

                    if minvalid_loss is None:
                        minvalid_loss = testloss

                    if minvalid_loss >= testloss:
                        print(
                            f"Test data Decreased({minvalid_loss:.6f}--->{testloss:.6f}) \t Saving The Model"
                        )
                        self.save(self.model[x], nome)
                        minvalid_loss = testloss

                    if i == 0:
                        print("Salvando Features")
                        self.save_features(feature_extractor, train_dataloader, "train", nome, device)
                        print("Salvamento concluído")


            models.pop(0)
            del self.model[x]

    def newWeights(self, epoch, device, train_dataloader, test_dataloader, models):
        """
        Realiza o processo de ajuste fino (fine-tuning) de um modelo de classificação.

        Parâmetros:
            epoch: O número de épocas de treinamento.
            device: O dispositivo de execução (por exemplo, 'cuda' para GPU ou 'cpu' para CPU).
            train_dataloader: O dataloader de treinamento que fornece os dados de treinamento.
            test_dataloader: O dataloader de teste que fornece os dados de teste.
            models: Uma lista de modelos a serem finetunados.

        Retorna:
            Nenhum valor de retorno.

        Exemplo:
            epoch = 10
            device = 'cuda'
            train_dataloader = [...]
            test_dataloader = [...]
            models = ['resnet50', 'vgg16']
            fineTuning(epoch, device, train_dataloader, test_dataloader, models)
        """
        trainTestORKFold = None
        if self.ui.radio_btn_tt_cnn.isChecked():
            trainTestORKFold = "tt" + self.ui.spinBox_cnn_train.text()
        elif self.ui.radio_btn_kf_cnn.isChecked():
            trainTestORKFold = "kf" + self.ui.radio_btn_kf_cnn.text()

        accurate = []
        epochs = []
        self.loss_monitor.set_model_names(models)

        for j in range(len(models)):
            contador_treino = 0
            x = 0
            minvalid_loss = None
            nome = (
                models[x]
                + "_"
                + "New-Weights"
                + "_"
                + self.controlNewWeights.ui.comboBox_loss.currentText()
                + "_"
                + self.controlNewWeights.ui.comboBox_optimizer.currentText()
                + "_"
                + self.controlNewWeights.ui.comboBox_metric.currentText()
                + "_"
                + str(self.controlNewWeights.ui.spinBox_epochs.value())
                + "_"
                + str(self.controlNewWeights.ui.spinBox_batch_size.value())
                + "_"
                + trainTestORKFold
                + ".pth"
            )

            self.ui.plainTextEdit_log1.insertPlainText(
                "VAMOS AO PROCESSAMENTO(" + models[x] + ")\n"
            )
            self.model.append(self.get_model(models[x], pretrained=False))

            feature_extractor = FeatureExtractor(self.model[x])

            self.optimizer = self.get_optimizer(
                self.controlNewWeights.ui.comboBox_optimizer.currentText(), self.model[x]
            )
            
            if self.ui.radio_btn_kf_cnn.isChecked():
                for fold in range(len(train_dataloader)):
                    print(f"Training on fold {fold+1}/{len(train_dataloader)}")

                    for i in range(epoch):
                        num_correct = 0.0
                        num_samples = 0.0
                        losses_train = []
                        accuracies_train = []
                        losses_test = []
                        accuracies_test = []

                        self.model[x].train()

                        for data, label in train_dataloader[fold]:
                            data, label = data.to(device), label.to(device)
                            self.model[x].to(device)
                            targets = self.model[x](data)
                            loss = self.loss_fn(targets, label)
                            accuracies_train.append(
                                label.eq(targets.detach().argmax(dim=1)).float().mean()
                            )
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            losses_train.append(loss.item())

                        testloss = 0.0
                        self.model[x].eval()
                        running_accuracy = 0.0

                        for data, label in test_dataloader[fold]:
                            data, label = data.to(device), label.to(device)
                            self.model[x].to(device)
                            with torch.no_grad():
                                targets = self.model[x](data)
                            running_accuracy += self.accuracy(targets, label)
                            _, predictions = targets.max(1)
                            num_correct += (predictions == label).sum()
                            num_samples += predictions.size(0)
                            loss = self.loss_fn(targets, label)
                            testloss = loss.item() * data.size(0)
                            losses_test.append(loss.item())
                            accuracies_test.append(
                                label.eq(targets.detach().argmax(dim=1)).float().mean()
                            )

                        accurate.append(float(num_correct / num_samples))
                        epochs.append(i)

                        self.loss_monitor.update_loss(models[x], loss)
                        QApplication.processEvents()
                        log = "Acurácia: " + str(float(num_correct / num_samples)) + "\n"
                        self.ui.plainTextEdit_log1.insertPlainText(log)

                        if minvalid_loss is None:
                            minvalid_loss = testloss

                        if minvalid_loss >= testloss:
                            print(
                                f"Test data Decreased({minvalid_loss:.6f}--->{testloss:.6f}) \t Saving The Model"
                            )
                            self.save(self.model[x], nome)
                            minvalid_loss = testloss

                        if fold == 0:
                            print("Salvando Features")
                            self.save_features(feature_extractor, train_dataloader[fold], "train", nome, device)
                            print("Salvamento concluído")
            else:
                for i in range(epoch):
                    num_correct = 0.0
                    num_samples = 0.0
                    losses_train = []
                    accuracies_train = []
                    losses_test = []
                    accuracies_test = []

                    self.model[x].train()

                    for data, label in train_dataloader:
                        data, label = data.to(device), label.to(device)
                        self.model[x].to(device)
                        targets = self.model[x](data)
                        loss = self.loss_fn(targets, label)
                        accuracies_train.append(
                            label.eq(targets.detach().argmax(dim=1)).float().mean()
                        )
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        losses_train.append(loss.item())

                    testloss = 0.0
                    self.model[x].eval()
                    running_accuracy = 0.0

                    for data, label in test_dataloader:
                        data, label = data.to(device), label.to(device)
                        self.model[x].to(device)
                        with torch.no_grad():
                            targets = self.model[x](data)
                        running_accuracy += self.accuracy(targets, label)
                        _, predictions = targets.max(1)
                        num_correct += (predictions == label).sum()
                        num_samples += predictions.size(0)
                        loss = self.loss_fn(targets, label)
                        testloss = loss.item() * data.size(0)
                        losses_test.append(loss.item())
                        accuracies_test.append(
                            label.eq(targets.detach().argmax(dim=1)).float().mean()
                        )

                    accurate.append(float(num_correct / num_samples))
                    epochs.append(i)

                    self.loss_monitor.update_loss(models[x], loss)
                    QApplication.processEvents()
                    log = "Acurácia: " + str(float(num_correct / num_samples)) + "\n"
                    self.ui.plainTextEdit_log1.insertPlainText(log)

                    if minvalid_loss is None:
                        minvalid_loss = testloss

                    if minvalid_loss >= testloss:
                        print(
                            f"Test data Decreased({minvalid_loss:.6f}--->{testloss:.6f}) \t Saving The Model"
                        )
                        self.save(self.model[x], nome)
                        minvalid_loss = testloss

                    if i == 0:
                        print("Salvando Features")
                        self.save_features(feature_extractor, train_dataloader, "train", nome, device)
                        print("Salvamento concluído")


            models.pop(0)
            del self.model[x]

    
    def layers(self, modelname):
        """
        Retorna uma lista com as camadas de um determinado modelo.

        Parâmetros:
            modelname (str): O nome do modelo.

        Retorna:
            list: Uma lista contendo as camadas do modelo.

        """
        model = self.get_model(modelname, True)
        return self.nodos(model)


    def save(self, model, nome):
        """
        Salva o estado de um modelo em um arquivo.

        Parâmetros:
            model: O modelo a ser salvo.
            nome (str): O nome do arquivo de saída.

        """

        path = self.output + '/' + nome[:-4]
        os.makedirs(path, exist_ok=True) 
        torch.save(model.state_dict(), os.path.join(path, nome))
        print("SALVANDO EM:\n", path)



    def save_features(self, feature_extractor, dataloader, dataset_type, model_name, device):
        """
        Salva as características extraídas de um modelo em um arquivo CSV.

        Parâmetros:
            feature_extractor: O extrator de características.
            dataloader: O dataloader contendo os dados.
            dataset_type (str): O tipo de conjunto de dados (e.g., 'train', 'test').
            model_name (str): O nome do modelo.
            device: O dispositivo de execução (CPU ou GPU).
        """

        features = []
        labels = []

        for data, label in dataloader:
            data = data.to(device)
            with torch.no_grad():
                batch_features = feature_extractor.extract_features(data)
            features.append(batch_features)
            labels.append(label)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        features_np = features.cpu().numpy()
        labels_np = labels.numpy()

        # Convert the numpy arrays to pandas DataFrames
        features_df = pd.DataFrame(features_np)
        labels_df = pd.DataFrame(labels_np)

        path = os.path.join(self.output, model_name[:-4])
        save_dir = os.path.join(path)
        save_dir1 = os.path.join(path, f"{dataset_type}/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_dir1):
            os.makedirs(os.path.join(save_dir, f"{dataset_type}/"))

        # Save features and labels as CSV files
        features_df.to_csv(os.path.join(save_dir, f"{dataset_type}/{dataset_type}_features.csv"), index=False)
        labels_df.to_csv(os.path.join(save_dir, f"{dataset_type}/{dataset_type}_labels.csv"), index=False)


    def save_featuresOriginal(self, features, labels, dataset_type, model_name, contador):
        features_np = np.concatenate([f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f for f in features])
        labels_np = np.concatenate([l.detach().cpu().numpy() if isinstance(l, torch.Tensor) else l for l in labels])

        # Convert the numpy arrays to pandas DataFrames
        features_df = pd.DataFrame(features_np)
        labels_df = pd.DataFrame(labels_np)

        path = os.path.join(self.output, model_name[:-4], dataset_type)
        save_dir = os.path.join(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save features and labels as CSV files
        features_df.to_csv(os.path.join(save_dir, f"{dataset_type}_features_{contador}.csv"), index=False)
        labels_df.to_csv(os.path.join(save_dir, f"{dataset_type}_labels_{contador}.csv"), index=False)    

    def save_featuresFUNFACPU(self, features, labels, dataset_type, model_name, contador):
        
        features_np = np.concatenate([f.detach().numpy() if isinstance(f, torch.Tensor) else f for f in features])
        labels_np = np.concatenate([l.detach().numpy() if isinstance(l, torch.Tensor) else l for l in labels])

        # Convert the numpy arrays to pandas DataFrames
        features_df = pd.DataFrame(features_np)
        labels_df = pd.DataFrame(labels_np)

        path = self.output+'/'+model_name[:-4]+'/'+dataset_type+'/'
        save_dir = os.path.join(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save features and labels as CSV files
        features_df.to_csv(os.path.join(save_dir, f"{dataset_type}_features_{contador}.csv"), index=False)
        labels_df.to_csv(os.path.join(save_dir, f"{dataset_type}_labels_{contador}.csv"), index=False)



    
    def save_features1(self,features,labels,train_test,nome,control):
        path = self.output+'/'+nome[:-4]+'/'+train_test+'/'
        os.makedirs(path, exist_ok = True) 
        size_name=path+'size_images.csv'
        dir = path
        if(control==0):
            for file in os.scandir(dir):
                os.remove(file.path)
            #size_f=features.shape
            features_df = pd.DataFrame(features)
            labels_df = pd.DataFrame(labels, columns=['labels'])
            features_df.to_csv(size_name, index=False)
            labels_df.to_csv('labels.csv', index=False)
        
        feature_name=path+'f_'+str(control)+'.csv'


        #with open(feature_name, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
        #    csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
        #    for x in features:
        #        csv_writer.writerow(x)

        #labels=labels.reshape(-1)
        #features=features.reshape(-1)

        #df = pd.DataFrame(features)
        #dl = pd.DataFrame(labels)
        
        #label_name=path+'l_'+str(control)+'.csv' 
        
        #df.to_csv(feature_name,index=False)
        #dl.to_csv(label_name,index=False)
        
    def nodos(self,model):
        nodes, _ = get_graph_node_names(model)
        return nodes

    def feature_extraction(self,model):
        nodes, _ = get_graph_node_names(model)
        retorno=nodes
        feature_extractor = create_feature_extractor(
	        model, return_nodes=[retorno[-2]])
        return feature_extractor

    def open_browse1(self, line, tab_name, type_folder):
        #path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a source directory:', expanduser('~'))
        path = QtWidgets.QFileDialog.getExistingDirectory(self)
        self.ui.line_input_1.setText(path)
        # getOpenFileName(self, 'Open a file', '', 'All Files (*.*)')
        if path != '':
            if tab_name == 'cnn' and type_folder=='input_folder':
                #print(dirname)
                image_datasets = {
                    x: datasets.ImageFolder(
                        os.path.join(path, x), 
                        transform=self.data_transforms[x]
                    )
                    for x in [self.TRAIN, self.VAL, self.TEST]
                }
                self.image_datasets=image_datasets
                print('\n\n',self.image_datasets['test'],'\n\n')
                #print(path.title())
                #print('\n\n',self.TRAIN,'\n\n')
                self.dataloaders = {
                    x: torch.utils.data.DataLoader(
                        self.image_datasets[x], batch_size=8,
                        shuffle=True, num_workers=4
                    )
                    for x in [self.TRAIN, self.VAL, self.TEST]
                }
                self.dataset_sizes = {x: len(self.image_datasets[x]) for x in [self.TRAIN, self.VAL, self.TEST]}
                self.class_names = self.image_datasets[self.TRAIN].classes


    def test(self):
        print("Test before training")
        self.eval_model(nn.CrossEntropyLoss())


    
    # def search_files(self, list_ui, path):
    #     files = [f for f in os.listdir(path) if f[-4:] == '.csv']
    #     if list_ui == 'classif':
    #         self.data['datasets'] = {f: path+f for f in files}
    #         self.add_items(self.ui.listWidget_datasets, files)
    #         self.add_log(f'{len(files)} datasets were found')

    

    def checkedItems(self, list):
        """
        Retorna os itens selecionados em uma lista com caixas de seleção.

        Parâmetros:
            list: A lista com caixas de seleção.

        Retorna:
            selected (list): Uma lista contendo os itens selecionados.
        """

        selected = []

        for index in range(list.count()):
            item = list.item(index)
            if item.checkState() == QtCore.Qt.Checked:
                selected.append(item.text())

        return selected

    
    def get_optimizer(self, optimizer_name, model):
        """
        Retorna o otimizador de acordo com o nome especificado.

        Parâmetros:
            optimizer_name (str): O nome do otimizador desejado.
            model: O modelo para o qual o otimizador será criado.

        Retorna:
            optimizer: O otimizador correspondente.
        """

        optimizer = None

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters())
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

        return optimizer

    
    def get_model(self, model_name, pretrained):
        """
        Retorna um modelo pré-treinado da biblioteca PyTorch Vision.

        Parâmetros:
            model_name (str): O nome do modelo desejado.
            pretrained (bool): Indica se o modelo pré-treinado deve ser carregado.

        Retorna:
            model: O modelo pré-treinado carregado.
        """

        model = torch.hub.load('pytorch/vision:v0.10.0', model_name.lower(), pretrained)
        return model


    def dt_transforms(self):
        """
        Retorna uma sequência de transformações para pré-processamento de dados de imagem.

        Retorna:
            transform: A sequência de transformações.
        """
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform


    def select_loss(self, loss_name):
        """
        Seleciona e retorna uma função de perda com base no nome especificado.

        Args:
            loss_name (str): O nome da função de perda.

        Returns:
            loss: A função de perda selecionada.
        """
        
        if loss_name == "BCEWithLogitsLoss":
            loss = torch.nn.BCEWithLogitsLoss()
        elif loss_name == "CrossEntropyLoss":
            loss = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Nome de função de perda inválido.")

        return loss


    def otimizador(self):
        """
        Retorna um otimizador do tipo SGD com os parâmetros especificados.

        Returns:
            optimizer: Otimizador do tipo SGD.
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        return optimizer


    def eval(self):
        self.model.eval()


    def train_one_epoch(self,epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.dataloaders[self.TRAIN]):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i%20==0:
                print("TA RODANDO")
            if i % 1000 == 999:
                last_loss = running_loss
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.dataloaders[self.TRAIN]) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return running_loss
    def run_epochs(self,epochs):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0

        EPOCHS = epochs

        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, writer)

            # We don't need gradients on to do reporting
            self.model.train(False)

            running_vloss = 0.0
            for i, vdata in enumerate(self.dataloaders[self.VAL]):
                vinputs, vlabels = vdata
                voutputs = self.model(vinputs)
                vloss = self.loss_fn(voutputs, vlabels)
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1   
    
    def plot_graph(self,x,y):
        grafico=PlotWidget()
        layout = QGridLayout()
        self.ui.widget_monitor.setLayout(layout)
        grafico.plot(x,y, pen=None, symbol='o')
        layout.addWidget(grafico)

    def print_output(self, s):
        print(s)
    
    def populate_list_widget_exibition(self, folder_path):
        """
        Popula um QListWidget com opções provenientes de subpastas de um diretório específico.

        Args:
            folder_path (str): O caminho para o diretório que contém as subpastas com as opções.

        Returns:
            None
        """
        # Carregar o QListWidget
        list_widget = self.ui.list_saved_models

        # Classe personalizada de item com checkbox
        class CheckboxListItem(QListWidgetItem):
            def __init__(self, text):
                super().__init__(text)
                self.setFlags(self.flags() | Qt.ItemIsUserCheckable)
                self.setCheckState(Qt.Unchecked)

        # Função para verificar se uma pasta contém a subpasta "results"
        def has_results_folder(path):
            results_folder = os.path.join(path, "results")
            return os.path.isdir(results_folder)

        # Função para obter o nome do arquivo CSV em uma pasta "results"
        def get_csv_files(path):
            results_folder = os.path.join(path, "results")
            csv_files = [name for name in os.listdir(results_folder) if name.endswith(".csv")]
            return csv_files

        # Função para popular o QListWidget com as opções
        def populate(widget, path):
            # Limpar a lista antes de popular novamente
            widget.clear()

            # Obter as subpastas de primeiro nível
            subfolders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

            # Adicionar as opções ao QListWidget
            for dir_name in subfolders:
                model_folder = os.path.join(path, dir_name)
                if has_results_folder(model_folder):
                    csv_files = get_csv_files(model_folder)
                    for csv_file in csv_files:
                        option = f"{dir_name}_{csv_file}"
                        item = CheckboxListItem(option)
                        widget.addItem(item)

        # Popular o QListWidget com as opções
        populate(list_widget, folder_path)


    def populate_list_widget(self, folder_path):
        """
        Popula um QListWidget com as subpastas de primeiro nível de um diretório específico.

        Args:
            folder_path (str): O caminho para o diretório que contém as subpastas.

        Returns:
            None
        """
        # Carregar o QListWidget
        list_widget = self.ui.listWidget_datasets

        # Classe personalizada de item com checkbox
        class CheckboxListItem(QListWidgetItem):
            def __init__(self, text):
                super().__init__(text)
                self.setFlags(self.flags() | Qt.ItemIsUserCheckable)
                self.setCheckState(Qt.Unchecked)

        # Função para popular o QListWidget com as subpastas de primeiro nível
        def populate(widget, path):
            # Limpar a lista antes de popular novamente
            widget.clear()

            # Obter subpastas de primeiro nível
            subfolders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

            # Adicionar as subpastas com checkboxes ao QListWidget
            for dir_name in subfolders:
                item = CheckboxListItem(dir_name)
                widget.addItem(item)

        # Popular o QListWidget com as subpastas de primeiro nível
        populate(list_widget, folder_path)

    

    def classification(self,features,labels,classificador):
        pass
    

    def load_data_csv(self, file_path, labels_path, test_size=None, kfold=False, n_splits=None, random_state=None):
        """
        Carrega dados de um arquivo CSV juntamente com os rótulos correspondentes.

        Args:
            file_path (str): O caminho para o arquivo CSV contendo os dados.
            labels_path (str): O caminho para o arquivo CSV contendo os rótulos.
            test_size (float or int, optional): O tamanho do conjunto de teste, em termos absolutos (int) ou como uma fração do conjunto de dados (float). Apenas necessário se `kfold` for False. Default é None.
            kfold (bool, optional): Indica se o conjunto de dados deve ser dividido em folds usando validação cruzada k-fold. Default é False.
            n_splits (int, optional): O número de folds a serem gerados. Apenas necessário se `kfold` for True. Default é None.
            random_state (int or RandomState, optional): O estado do gerador de números aleatórios. Apenas necessário se `kfold` for True. Default é None.

        Returns:
            tuple or generator: Um objeto que contém os dados carregados, dependendo dos argumentos fornecidos.
                - Se `kfold` for True, retorna um objeto gerador que produz os índices de treinamento e teste para cada fold, juntamente com os recursos e rótulos completos.
                - Se `kfold` for False, retorna quatro objetos que contêm os recursos e rótulos de treinamento e teste, respectivamente.

        """
        data = pd.read_csv(file_path)
        labels = pd.read_csv(labels_path)  # Carrega o arquivo de rótulos
        
        features = data.iloc[:, :-1]
        labels = labels.iloc[:, 0]  # Considera apenas a primeira coluna do arquivo de rótulos
        
        if kfold:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = kf.split(features)
            return splits, features, labels
        else:
            train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size, random_state=random_state)
            return train_features, test_features, train_labels, test_labels

    
    def import_classifier(self, classifier_name):
        """
        Importa e retorna um classificador de aprendizado de máquina com base no nome especificado.

        Args:
            classifier_name (str): O nome do classificador a ser importado.

        Returns:
            classifier: Um objeto do classificador importado.

        Raises:
            ValueError: Se o nome do classificador não for suportado.

        """
        if classifier_name == "MLP":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(max_iter=100)
        elif classifier_name == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier()
        elif classifier_name == "SVM":
            from sklearn.svm import SVC
            return SVC()
        else:
            raise ValueError("Classifier '{}' not supported.".format(classifier_name))

    def save_cm(self, cm, labels, save_folder, classifier_name):
        
        # Criar o diretório, caso não exista
        os.makedirs(save_folder, exist_ok=True)

        # Definindo as classes (rótulos) das categorias
        classes = np.unique(labels)

        # Plotando a matriz de confusão como um gráfico
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Matriz de Confusão {classifier_name}')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predito')
        plt.ylabel('Real')

        # Preenchendo a matriz de confusão com os valores
        thresh = cm.max() / 2.
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(f'{save_folder}/{classifier_name}_cm.png')

    def save_results_to_csv(self, results, save_folder, classifier_name):
        """
        Salva os resultados de avaliação em um arquivo CSV.

        Args:
            results (list): Uma lista de dicionários contendo os resultados de avaliação.
            save_folder (str): O diretório de salvamento dos arquivos CSV.
            classifier_name (str): O nome do classificador associado aos resultados.

        """
        for i, result in enumerate(results):
            result_df = pd.DataFrame(result.items(), columns=['Metric', 'Value'])
            filename = f'{save_folder}/{classifier_name}.csv'
            
            # Criar o diretório, caso não exista
            os.makedirs(save_folder, exist_ok=True)
            
            result_df.to_csv(filename, index=False)


    def train_classifier(self, classifier_names, dataset_csv, folder_name, kfold):
        """
        Treina classificadores e avalia seu desempenho usando métricas.

        Args:
            classifier_names (str or list): O(s) nome(s) do(s) classificador(es) a ser(em) treinado(s).
            dataset_csv: Os dados do conjunto de dados em formato CSV.
            folder_name (str): O nome do diretório de saída para salvar os resultados.
            kfold (bool, optional): Indica se deve ser usado o K-Fold cross-validation. O padrão é False.

        """
        

        # Verificar se classifier_names é uma string e transformá-la em uma lista
        if isinstance(classifier_names, str):
            classifier_names = [classifier_names]
        
        if kfold:
            splits, features, labels = dataset_csv
            list_splits=list(splits)

        # Obter o diretório de saída
        save_folder = self.ui.line_output_2.text()
        
        #Contador auxiliar
        qtd_classifier=len(classifier_names)
        qtd_aux=1

        # Loop sobre os nomes dos classificadores
        for classifier_name in classifier_names:
            self.ui.plainTextEdit_log2.insertPlainText(f"Processando {classifier_name} \n")
            self.ui.plainTextEdit_log2.update()

            self.ui.widget_3.insertPlainText(f"{classifier_name}:\n")
            self.ui.widget_3.update()
            
            # Importar o classificador
            classifier = self.import_classifier(classifier_name)
            
            # Lista para armazenar os resultados de avaliação
            results = []
            
            if kfold:
                # Se for usado K-Fold cross-validation
                #splits, features, labels = dataset_csv
                #splits=list(splits1)
                elementos = list_splits
                splits1 = (valor for valor in elementos)
                #print(splits)
                #print(features)
                #print(labels)


                # Loop sobre as divisões do conjunto de dados
                for train_index, test_index in splits1:
                    #print("RODOU")
                    train_features, test_features = features.iloc[train_index], features.iloc[test_index]
                    train_labels, test_labels = labels.iloc[train_index], labels.iloc[test_index]
                    
                    # Treinar o classificador usando train_features e train_labels
                    classifier.fit(train_features, train_labels)
                    
                    # Testar o classificador usando test_features e test_labels
                    predictions = classifier.predict(test_features)
                    accuracy = classifier.score(test_features, test_labels)
                    
                    # Calcular métricas
                    kappa = cohen_kappa_score(test_labels, predictions)
                    precision = precision_score(test_labels, predictions)
                    recall = recall_score(test_labels, predictions)
                    f1 = f1_score(test_labels, predictions)
                    cm = confusion_matrix(test_labels, predictions)
                    tn, fp, fn, tp = cm.ravel()
                    especificidade = tn / (tn + fp)

                    # Imprimir métricas
                    #print("Accuracy:", accuracy)
                    #print("Cohen's Kappa Score:", kappa)
                    #print("Precision:", precision)
                    #print("Recall:", recall)
                    if(self.ui.comboBox_metric.currentText()=="Accuracy"):
                        self.ui.widget_3.insertPlainText(f"Acuracia: {accuracy}\n")
                        self.ui.widget_3.update()
                    else:
                        self.ui.widget_3.insertPlainText(f"Kappa: {kappa}\n")
                        self.ui.widget_3.update()
                    result = {
                        "Acuracia": accuracy,
                        "Kappa": kappa,
                        "Sensibilidade": recall,
                        "Especificidade": especificidade,
                        "F1-Score": f1
                    }
                    
                    # Adicionar o dicionário de resultados à lista de resultados
                    results.append(result)

                
            else:
                # Se não for usado K-Fold cross-validation
                train_features, test_features, train_labels, test_labels = dataset_csv
                
                for epoch in range(5):
                    # Laço de treinamento personalizado para o classificador específico com 'epochs'
                    #classifier.partial_fit(train_features, train_labels, classes=np.unique(train_labels))
                    # Treinar o classificador usando train_features e train_labels
                    classifier.fit(train_features, train_labels)
                
                # Testar o classificador usando test_features e test_labels
                predictions = classifier.predict(test_features)
                accuracy = classifier.score(test_features, test_labels)
                
                # Calcular métricas
                kappa = cohen_kappa_score(test_labels, predictions)
                precision = precision_score(test_labels, predictions)
                recall = recall_score(test_labels, predictions)
                f1 = f1_score(test_labels, predictions)
                cm = confusion_matrix(test_labels, predictions)
                tn, fp, fn, tp = cm.ravel()
                especificidade = tn / (tn + fp)

                # Imprimir métricas
                print("Acuracia:", accuracy)
                print("Kappa:", kappa)
                print("Sensibilidade:", recall)
                print("Especificidade:", especificidade)
                print("F1-Score", f1)
                
                if(self.ui.comboBox_metric.currentText()=="Accuracy"):
                        self.ui.widget_3.insertPlainText(f"Acurácia: {accuracy}\n")
                        self.ui.widget_3.update()
                else:
                    self.ui.widget_3.insertPlainText(f"Kappa: {kappa}\n")
                    self.ui.widget_3.update()
                
                result = {
                    "Acuracia": accuracy,
                    "Kappa": kappa,
                    "Sensibilidade": recall,
                    "Especificidade": especificidade,
                    "F1-Score": f1
                }
                
                # Adicionar o dicionário de resultados à lista de resultados
                results.append(result)
            
            #print(results[0])
            # Caminho para o diretório do classificador
            classifier_folder = save_folder + f"/{folder_name}" + "/results/"
            
            # Salvar os resultados em um arquivo CSV
            if kfold:
                self.save_cm(cm,test_labels,classifier_folder,classifier_name)
                self.save_results_to_csv([results[0]], classifier_folder, classifier_name)
            else:
                self.save_cm(cm,test_labels,classifier_folder,classifier_name)
                self.save_results_to_csv(results, classifier_folder, classifier_name)
            if(qtd_classifier==qtd_aux):
                self.ui.plainTextEdit_log2.insertPlainText("Processamento Finalizado!")
                self.ui.plainTextEdit_log2.update()
            qtd_aux+=1
        self.ui.comboBox_metric.setEnabled(True)
        self.ui.plainTextEdit_log2.update()




    # Tela de exibição

    def converter_path_caminho(self, caminho_original):
        """
        Converte um caminho original em um nome de modelo e nome de arquivo CSV concatenados.

        Args:
            caminho_original (str): O caminho original a ser convertido.

        Returns:
            str: O nome do modelo e nome do arquivo CSV concatenados.

        """

        # Substituir barras invertidas por barras normais e dividir o caminho em partes
        partes = caminho_original.replace('\\', '/').split("/")
        
        # Obter o nome do modelo da penúltima parte do caminho
        nome_modelo = partes[-3]
        
        # Obter o nome do arquivo CSV da última parte do caminho, removendo a extensão .csv
        nome_csv = partes[-1].replace('.csv', '')
        
        # Concatenar o nome do modelo e nome do arquivo CSV com um underscore
        return f"{nome_modelo}_{nome_csv}"



    def updateTableWithCSVs(self, table: QTableWidget, csv_paths: list):
        """
        Atualiza uma tabela com os dados de vários arquivos CSV.

        Args:
            table (QTableWidget): A tabela a ser atualizada.
            csv_paths (list): Uma lista contendo os caminhos dos arquivos CSV.

        """

        # Limpar a tabela
        table.clear()
        all_data = []
        selected_metrics = self.checkedItems(self.ui.list_metrics)
        qtd_rows=len(selected_metrics)+1
        if "precision_recall_curve" in selected_metrics:
            qtd_rows -= 1
        if "roc_curve" in selected_metrics:
            qtd_rows -= 1
        if "Matriz de Confusao" in selected_metrics:
            qtd_rows -= 1

        #print(selected_metrics)
        #print("Recall" in selected_metrics)


        for path in csv_paths:
            # Abrir o arquivo CSV atual
            with open(path, 'r') as file:
                # Ler os dados do arquivo CSV usando o módulo csv
                reader = csv.reader(file)
                # Converter os dados em uma lista
                data = list(reader)
                # Adicionar os dados do CSV atual à lista de dados
                nome = self.converter_path_caminho(path)
                all_data.append((nome, data))  # Adicionar uma tupla com o nome do arquivo e os dados à lista
            if "Matriz de Confusao" in selected_metrics:
                novo_path = path[:-4]
                novo_path += "_cm.png"
                image = Image.open(novo_path)
                image.show()

        # Obter o número de linhas e colunas dos dados
        num_csvs = len(all_data)
        rowsNew=num_csvs*qtd_rows
        rows = sum(len(data) for _, data in all_data)
        columns = len(all_data[0][1][0]) if rows > 0 else 0

        #print(rowsNew)

        # Definir o número de linhas e colunas da tabela
        #print(rows)
        table.setRowCount(rowsNew)
        table.setColumnCount(columns + 1)  # Adicionar uma coluna extra para o nome do arquivo

        # Preencher a tabela com os dados dos CSVs
        current_row = 0
        current_aux = 0
        for csv_index, (csv_name, csv_data) in enumerate(all_data):
            # Preencher a célula com o nome do arquivo
            name_item = QTableWidgetItem(csv_name)
            table.setItem(current_row, 0, name_item)
            table.setSpan(current_row, 0, (qtd_rows), 1)  # Expandir a célula do nome do arquivo por todas as linhas dos dados

            # Preencher as células com os dados do CSV atual
            for row, csv_row in enumerate(csv_data):
                #print(csv_row[0])
                
                for column, value in enumerate(csv_row):
                    #print(csv_row[0])
                    if(csv_row[0] == "Metric" or (csv_row[0] in selected_metrics)):
                        #print(current_aux,current_aux + row, column + 1)
                        item = QTableWidgetItem(value)
                        table.setItem(current_aux, column + 1, item)
                        if(column==1):
                            current_aux+=1
                    else:
                        pass
            #print(len(csv_data))
            current_row += (qtd_rows)

        for column in range(table.columnCount()):
            table.setColumnWidth(column, 150)  # Define uma largura mínima de 150 pixels para todas as colunas


    def convertPath(self, string: str, path: str) -> str:
        """
        Converte a string e o caminho fornecidos em um novo caminho modificado.

        Args:
            string (str): A string a ser convertida.
            path (str): O caminho base.

        Returns:
            str: O novo caminho modificado.

        """

        # Encontrar o último '_' na string
        last_underscore_index = string.rfind('_')

        # Extrair o valor após o último '_'
        gdm_value = string[last_underscore_index + 1:]

        # Remover o final após o último '_'
        string_without_end = string[:last_underscore_index]

        # Construir o novo caminho
        new_path = os.path.join(path, string_without_end, 'results', gdm_value)

        return new_path

    def reset_tablewidget(self,table_widget=None):
        if table_widget==None:
            table_widget=self.ui.tableWidget
        # Remove o conteúdo das células
        table_widget.clearContents()
        # Define o número de linhas como zero
        table_widget.setRowCount(0)

    def run_exib(self):
        """
        Função para exibir os resultados com base nos modelos selecionados e nas métricas selecionadas.

        A função realiza as seguintes etapas:
        1. Obtém os modelos salvos selecionados do QListWidget.
        2. Obtém as métricas selecionadas do QListWidget.
        3. Cria uma lista vazia para armazenar os caminhos modificados.
        4. Itera sobre os caminhos dos modelos selecionados e converte cada caminho utilizando a função convertPath.
        5. Adiciona os caminhos modificados à lista.
        6. Atualiza a tabela com os arquivos CSV dos caminhos modificados.

        Parameters:
            self (objeto): Instância do objeto que chama a função.
        
        Returns:
            None
        """
        selected_saved_models = self.checkedItems(self.ui.list_saved_models)
        selected_metrics = self.checkedItems(self.ui.list_metrics)
        paths = []
        
        # Converter os caminhos dos modelos selecionados
        for path in selected_saved_models:
            modified_path = self.convertPath(path, self.ui.line_input_3.text())
            paths.append(modified_path)
        
        # Atualizar a tabela com os arquivos CSV dos caminhos modificados
        self.updateTableWithCSVs(self.ui.tableWidget, paths)


    def run_classificacao(self):
        """
        Função para executar a classificação com base nos modelos selecionados e nos conjuntos de dados selecionados.

        A função realiza as seguintes etapas:
        1. Esvazia a memória da GPU e coleta o lixo.
        2. Limpa o campo de texto "plainTextEdit_log2".
        3. Obtém os modelos de classificação selecionados da QListWidget.
        4. Obtém os conjuntos de dados selecionados da QListWidget.
        5. Verifica se a pasta de entrada e a pasta de saída foram selecionadas e se pelo menos um modelo e um conjunto de dados foram selecionados. Caso contrário, exibe mensagens de erro apropriadas.
        6. Verifica se o tipo de treinamento é Treino-Teste ou K-Fold Cross Validation.
        7. Se o tipo de treinamento for Treino-Teste:
            a. Obtém os caminhos do arquivo CSV de treinamento e dos rótulos de treinamento com base no conjunto de dados selecionado.
            b. Calcula o tamanho do conjunto de teste com base no valor do slider "horizontalSlider_classifi".
            c. Carrega os dados CSV usando a função load_data_csv.
            d. Chama a função train_classifier para treinar os classificadores nos dados de treinamento.
        8. Se o tipo de treinamento for K-Fold Cross Validation:
            a. Obtém os caminhos do arquivo CSV de treinamento e dos rótulos de treinamento com base no conjunto de dados selecionado e no número de splits definido no spin box "spinBox_classifi_kf".
            b. Carrega os dados CSV usando a função load_data_csv com o parâmetro kfold definido como True.
            c. Chama a função train_classifier para treinar os classificadores usando a validação cruzada K-Fold.
        9. Exibe a mensagem "Tudo certo!".

        Parameters:
            self (objeto): Instância do objeto que chama a função.
        
        Returns:
            None
        """
        torch.cuda.empty_cache()
        gc.collect()
        self.ui.plainTextEdit_log2.clear()
        selected_classificadores = self.checkedItems(self.ui.listWidget_models)
        selected_datasets = self.checkedItems(self.ui.listWidget_datasets)
        
        # Verifica se os campos obrigatórios foram preenchidos
        if self.ui.line_input_2.text() == '':
            self.ui.plainTextEdit_log2.insertPlainText("Por favor, selecione a pasta de entrada!\n")
        elif self.ui.line_output_2.text() == '':
            self.ui.plainTextEdit_log2.insertPlainText("Por favor, selecione a pasta de saída!\n")
        elif len(selected_classificadores) == 0:
            self.ui.plainTextEdit_log2.insertPlainText("Por favor, selecione ao menos um modelo!\n")
        elif len(selected_datasets) == 0:
            self.ui.plainTextEdit_log2.insertPlainText("Por favor, selecione ao menos um dataset!\n")
        elif not (self.ui.radio_btn_tt_classifi.isChecked() or self.ui.radio_btn_kf_classifi.isChecked()):
            self.ui.plainTextEdit_log2.insertPlainText("Configure o Treino e Teste ou K-Fold!\n")
        else:
            dataset_csv = None
            
            # Verifica o tipo de treinamento selecionado
            if self.ui.radio_btn_tt_classifi.isChecked():
                # Treino-Teste
                del dataset_csv
                file_path_csv = Path(self.ui.line_input_2.text() + "/" + selected_datasets[0] + "/train/train_features.csv")
                label_path_csv = Path(self.ui.line_input_2.text() + "/" + selected_datasets[0] + "/train/train_labels.csv")
                test_size = (100 - self.ui.horizontalSlider_classifi.value()) / 100
                
                # Carrega os dados CSV
                dataset_csv = self.load_data_csv(file_path_csv, label_path_csv, test_size)
                
                self.ui.plainTextEdit_log2.insertPlainText("Processando os Classificadores!\n")
                self.ui.comboBox_metric.setEnabled(False)

                # Treina os classificadores
                self.thread=QThread()
                thread=threading.Thread(target=self.train_classifier,args=(selected_classificadores, dataset_csv, selected_datasets[0], False,))
                thread.start()


                #self.train_classifier(selected_classificadores, dataset_csv, selected_datasets[0], False)
            else:
                # K-Fold Cross Validation
                del dataset_csv
                file_path_csv = Path(self.ui.line_input_2.text() + "/" + selected_datasets[0] + "/train/train_features.csv")
                label_path_csv = Path(self.ui.line_input_2.text() + "/" + selected_datasets[0] + "/train/train_labels.csv")
                splits = int(self.ui.spinBox_classifi_kf.text())
                
                # Carrega os dados CSV
                dataset_csv = self.load_data_csv(file_path_csv, label_path_csv, n_splits=splits, kfold=True)
                
                # Treina os classificadores usando K-Fold
                self.thread=QThread()
                thread=threading.Thread(target=self.train_classifier,args=(selected_classificadores, dataset_csv, selected_datasets[0], True,))
                thread.start()
                #self.train_classifier(selected_classificadores, dataset_csv, selected_datasets[0], True)



    def run(self):
        torch.cuda.empty_cache()
        gc.collect()
        #self.train_test()
        #self.load_dataloader(16)
        if(self.model!= [] or self.model != None):
            del self.model
            self.model=[]
        self.ui.plainTextEdit_log1.clear()
        selected_CNN = self.checkedItems(self.ui.listWidget_arch)
        if(self.ui.radio_btn_tt_cnn.isChecked()==False and self.ui.radio_btn_kf_cnn.isChecked()==False):
            self.ui.plainTextEdit_log1.insertPlainText("Configure o Treino e Teste ou K-Fold!\n")
        elif(self.ui.line_input_1.text() == ''):
            self.ui.plainTextEdit_log1.insertPlainText("Adicione a pasta de entrada!\n")
        elif(self.ui.line_output_1.text() == ''):
            self.ui.plainTextEdit_log1.insertPlainText("Adicione a pasta de saída!\n")
        elif(len(selected_CNN)==0):
            self.ui.plainTextEdit_log1.insertPlainText("Selecione ao menos 1 modelo!\n")
        elif(self.ui.radio_btn_ft.isChecked()==False and self.ui.radio_btn_nw.isChecked()==False and self.ui.radio_btn_pt.isChecked()==False):
            self.ui.plainTextEdit_log1.insertPlainText("Selecione o setup de execução!\n")
        else:
            device=self.device()
            if(self.ui.radio_btn_ft.isChecked()):
                if(self.ui.radio_btn_tt_cnn.isChecked()):
                    del self.image_datasets
                    del self.labels
                    del self.dataset_sizes 
                    del self.class_names
                    train,test=self.train_test()
                else:
                    train,test=self.kfold_split(int(self.ui.spinBox_cnn_kf.text()))
                print("\n\n\n")
                print(train)
                print(test)
                print("\n\n\n")
                epochs=self.controlFineTuning.ui.spinBox_epochs.value()
                batch_size=self.controlFineTuning.ui.spinBox_batch_size.value()
                #self.model=self.get_model(selected_CNN[i],pretrained=True)
                self.loss_fn=self.select_loss(self.controlFineTuning.ui.comboBox_loss.currentText())
                train_dataloader,test_dataloader=self.load_dataloader(train,test,batch_size)

                print("\n\n\n")
                print(train_dataloader)
                print(test_dataloader)
                print("\n\n\n")

                del train
                del test
                #outputs=[]
                for x in selected_CNN:
                    #loop = QtCore.QEventLoop()
                    #widget = self.controlOutput
                    #widget.show()
                    #widget.closed.connect(loop.quit)
                    #loop.exec_()
                    print(self.controlOutput.ui.comboBox.currentIndex())
                    #while self.controlOutput.ui.comboBox.currentIndex() == -1:
                    #   pass 
                    #outputs.append(self.layers(x))
                self.thread=QThread()
                thread=threading.Thread(target=self.fineTuning,args=(epochs,device,train_dataloader,test_dataloader,selected_CNN,))
                thread.start()
                del train_dataloader
                del test_dataloader
                del epochs
                del device
                del selected_CNN

                #self.worker=Worker(self.fineTuning,args=(epochs,))
                #self.worker.moveToThread(self.thread)
                #self.thread.started.connect(self.worker.run)
                #self.worker.finished.connect(self.thread.quit)
                #self.worker.finished.connect(self.worker.deleteLater)
                #self.thread.finished.connect(self.thread.deleteLater)
                #self.thread.start()
                #self.fineTuning(epochs)
                
            #Se selecionado New Weights
            if(self.ui.radio_btn_nw.isChecked()):
                if(self.ui.radio_btn_tt_cnn.isChecked()):
                    del self.image_datasets
                    del self.labels
                    del self.dataset_sizes 
                    del self.class_names
                    train,test=self.train_test()
                else:
                    train,test=self.kfold_split(int(self.ui.spinBox_cnn_kf.text()))
                print("\n\n\n")
                print(train)
                print(test)
                print("\n\n\n")
                epochs=self.controlNewWeights.ui.spinBox_epochs.value()
                batch_size=self.controlNewWeights.ui.spinBox_batch_size.value()
                #self.model=self.get_model(selected_CNN[i],pretrained=True)
                self.loss_fn=self.select_loss(self.controlNewWeights.ui.comboBox_loss.currentText())
                train_dataloader,test_dataloader=self.load_dataloader(train,test,batch_size)

                print("\n\n\n")
                print(train_dataloader)
                print(test_dataloader)
                print("\n\n\n")

                del train
                del test
                #outputs=[]
                for x in selected_CNN:
                    #loop = QtCore.QEventLoop()
                    #widget = self.controlOutput
                    #widget.show()
                    #widget.closed.connect(loop.quit)
                    #loop.exec_()
                    print(self.controlOutput.ui.comboBox.currentIndex())
                    #while self.controlOutput.ui.comboBox.currentIndex() == -1:
                    #   pass 
                    #outputs.append(self.layers(x))
                self.thread=QThread()
                thread=threading.Thread(target=self.newWeights,args=(epochs,device,train_dataloader,test_dataloader,selected_CNN,))
                thread.start()
                del train_dataloader
                del test_dataloader
                del epochs
                del device
                del selected_CNN
        if(self.ui.radio_btn_pt.isChecked()):
            epochs=10
            batch_size=8
            selected_CNN = self.checkedItems(self.ui.listWidget_arch)
            self.model=self.get_model(selected_CNN[i],pretrained=True)
            self.loss_fn=self.select_loss("CrossEntropyLoss")
            self.optimizer="Adam"
            self.load_dataloader(batch_size)
            print("VAMOS AO PROCESSAMENTO..\n")
            self.pretrained(epochs)
            


    def UiComponents(self,epochs,models,values):
        
        # creating a widget object
        widget = QWidget()
 
        # creating a new label
        label = QLabel()
 
        # making it multiline
        #label.setWordWrap(True)
 
        x = range(0, 1)
 
        # create plot window object
        plt = pg.plot()
 
        # showing x and y grids
        plt.showGrid(x = True, y = True)
 
        # adding legend
        plt.addLegend()
 
        # set properties of the label for y axis
        plt.setLabel('left', 'Vertical Values', units ='y')
 
        # set properties of the label for x axis
        plt.setLabel('bottom', 'Horizontal Values', units ='s')
 
        # setting horizontal range
        plt.setXRange(0, 10)
 
        # setting vertical range
        plt.setYRange(0, 20)

        for x in range(1):
                line = plt.plot([0,1,2,3,4], [2,5,3,7,9], pen ='g', symbol ='x', symbolPen ='g', symbolBrush = 0.2, name ='green')
        # plotting line in green color
        # with dot symbol as x, not a mandatory field
        #line1 = plt.plot(x, y, pen ='g', symbol ='x', symbolPen ='g', symbolBrush = 0.2, name ='green')
 
        # plotting line2 with blue color
        # with dot symbol as o
        #line2 = plt.plot(x, y2, pen ='b', symbol ='o', symbolPen ='b', symbolBrush = 0.2, name ='blue')
 
        # getting data  of the line 1
        #value = line1.getData()
 
        # setting text to the label
        #label.setText("_________Line1_Data_______: " + str(value))
 
        # Creating a grid layout
        layout = QGridLayout()
 
        # setting this layout to the widget
        self.ui.widget_monitor.setLayout(layout)
 
        # adding label to the layout
        #layout.addWidget(label, 1, 0)
 
        # plot window goes on right side, spanning 3 rows
        layout.addWidget(plt)
 
        # setting this widget as central widget of the main window
        #self.setCentralWidget(self.ui.widget_monitor)

        
    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    thread=QThread()
    application.moveToThread(thread)
    application.show()
    sys.exit(app.exec_())
