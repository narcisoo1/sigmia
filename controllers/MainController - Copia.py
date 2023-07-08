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
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, precision_recall_curve, roc_curve, auc

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




# Bar Graph class
class BarGraphItem(pg.BarGraphItem):
 
    # constructor which inherit original
    # BarGraphItem
    def __init__(self, *args, **kwargs):
        pg.BarGraphItem.__init__(self, *args, **kwargs)
 
    # creating a mouse double click event
    def mouseDoubleClickEvent(self, e):
 
        # setting scale
        self.setScale(0.2)

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    finished = pyqtSignal()
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done




class ApplicationWindow(QtWidgets.QMainWindow,QObject):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.loss_monitor = LossMonitor()
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.coreCNN = CoreCNN()
        self.ui.widget_monitor.setLayout(QVBoxLayout())
        self.ui.widget_monitor.layout().addWidget(self.loss_monitor)
        self.use_gpu = torch.cuda.is_available()
        self.dataset_sizes=None
        self.image_datasets=None
        self.dataloaders=None
        self.trainDataloader=None
        self.testDataloader=None
        self.model=[]
        self.data_dir = None
        self.data_iter=None
        self.dataset = None
        self.labels=None
        self.output=None
        self.p_train=None
        self.INPUT='input'
        self.TRAIN = 'train'
        self.VAL = 'val'
        self.TEST = 'test'
        self.data_transforms=self.dt_transforms()
        self.class_names=None
        self.optimizer=None
        self.loss_fn=torch.nn.CrossEntropyLoss()
        self.finished=None
        self.data = {}
        self.translate_ui()
        self.init_signals()
        self.gpu_disponivel()
        #list_widget = self.populate_list_widget()
        #self.ui.listWidget_datasets = list_widget
        self.threadpool = QThreadPool()

    def init_signals(self):
        self.ui.stackedWidget.setCurrentIndex(0)
        

        self.tabs = [self.ui.btn_tab_CNN, self.ui.btn_tab_Classf, self.ui.btn_tab_MC]
        self.logs = [self.ui.plainTextEdit_log1, self.ui.plainTextEdit_log2]

        self.controlNewWeights = ControlNewWeights()
        self.controlFineTuning = ControlFineTuning()
        self.controlPreTrainedWindow = ControlPreTrainedWindow()
        self.controlOutput = OutputController()

        self.ui.btn_browser_input_1.clicked.connect(
            lambda: self.open_browse(self.ui.line_input_1, 'cnn', 'input_folder'))
        self.ui.btn_browser_input_2.clicked.connect(
            lambda: self.open_browse(self.ui.line_input_2, 'classif', 'input_folder'))
        self.ui.btn_browser_input_3.clicked.connect(
            lambda: self.open_browse(self.ui.line_input_3, 'exib', 'input_folder'))
        self.ui.btn_browser_output_1.clicked.connect(
            lambda: self.open_browse(self.ui.line_output_1, 'cnn', 'output_folder'))
        self.ui.btn_browser_output_2.clicked.connect(
            lambda: self.open_browse(self.ui.line_output_2, 'classif', 'output_folder'))
        self.ui.btn_browser_output_3.clicked.connect(
            lambda: self.open_browse(self.ui.line_output_3, 'exib', 'output_folder'))

        self.tabs = [self.ui.btn_tab_CNN, self.ui.btn_tab_Classf, self.ui.btn_tab_MC]
        self.ui.btn_tab_CNN.clicked.connect(lambda: self.charge_tab(0))
        self.ui.btn_tab_Classf.clicked.connect(lambda: self.charge_tab(1))
        self.ui.btn_tab_MC.clicked.connect(lambda: self.charge_tab(2))

        self.ui.conf_btn_nw.clicked.connect(lambda: self.controlNewWeights.show())
        self.ui.conf_btn_ft.clicked.connect(lambda: self.controlFineTuning.show())
        self.ui.conf_btn_pt.clicked.connect(lambda: self.controlPreTrainedWindow.show())

        ## RUN
        self.ui.btn_cnn_run.clicked.connect(lambda: self.run())
        self.ui.btn_classifi_run.clicked.connect(lambda: self.run_classificacao())
        self.ui.btn_exib_run.clicked.connect(lambda: self.run_exib())
        

        ## CNN
        self.charged_init = False
        self.ui.horizontalSlider_cnn.valueChanged.connect(self.onChangeSliderCNN)
        self.ui.spinBox_cnn_train.valueChanged.connect(self.onChangeSpinBoxCNNTrain)
        self.ui.spinBox_cnn_test.valueChanged.connect(self.onChangeSpinBoxCNNTest)

        self.radio_setup_cnn_selected = 0
        self.btns_setup = [self.ui.conf_btn_nw, self.ui.conf_btn_ft, self.ui.conf_btn_pt]
        self.ui.radio_btn_nw.clicked.connect(lambda: self.onClickedRadioSetup(0))
        self.ui.radio_btn_ft.clicked.connect(lambda: self.onClickedRadioSetup(1))
        self.ui.radio_btn_pt.clicked.connect(lambda: self.onClickedRadioSetup(2))

        self.radio_tt_cnn_selected = 0
        self.btsns_tt_cnn = [self.ui.radio_btn_tt_cnn, self.ui.radio_btn_tt_classifi]
        self.ui.radio_btn_tt_cnn.clicked.connect(lambda: self.onClickedRadioTT(0))
        self.ui.radio_btn_kf_cnn.clicked.connect(lambda: self.onClickedRadioTT(1))

        # List datasets
        self.ui.listWidget_datasets.itemDoubleClicked.connect(self.onItemDoubleClicked)


    def translate_ui(self):
        #self.ui.listWidget_arch = QtWidgets.QListWidget(self)
        self.core = json.load(open('controllers/configs.json'))

        _translate = QtCore.QCoreApplication.translate
        for name, _ in self.core['applications'].items():
            #print(_translate("MainWindow", name))
            item = QtWidgets.QListWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setText(_translate("MainWindow", name))
            self.ui.listWidget_arch.addItem(item)
        
        self.ui.listWidget_models.clear()
        for name, _ in self.core['classifications'].items():
            #print(_translate("MainWindow", name))
            item = QtWidgets.QListWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setText(_translate("MainWindow", name))
            self.ui.listWidget_models.addItem(item)

        for name, _ in self.core['metrics'].items():
            self._append_items(self.ui.list_metrics, name)

    def onClickedRadioSetup(self, radio_n):
        self.radio_setup_cnn_selected = radio_n
        for i in range(3):
            if i == radio_n:
                self.btns_setup[i].setEnabled(True)
            else:
                self.btns_setup[i].setEnabled(False)

    def onClickedRadioTT(self, radio_n):
        self.radio_tt_cnn_selected = 0
        if radio_n == 0:
            self.ui.spinBox_cnn_train.setEnabled(True)
            self.ui.spinBox_cnn_test.setEnabled(True)
            self.ui.horizontalSlider_cnn.setEnabled(True)
            self.ui.spinBox_cnn_kf.setEnabled(False)
        elif radio_n == 1:
            self.ui.spinBox_cnn_train.setEnabled(False)
            self.ui.spinBox_cnn_test.setEnabled(False)
            self.ui.horizontalSlider_cnn.setEnabled(False)
            self.ui.spinBox_cnn_kf.setEnabled(True)

    def onChangeSliderCNN(self):
        if not self.charged_init:
            self.charged_init = True
            self.ui.spinBox_cnn_train.setValue(self.ui.horizontalSlider_cnn.value())
            self.ui.spinBox_cnn_test.setValue(abs(self.ui.horizontalSlider_cnn.value() - 100))
            self.charged_init = False

    def onChangeSpinBoxCNNTrain(self):
        if not self.charged_init:
            self.charged_init = True
            self.ui.horizontalSlider_cnn.setValue(self.ui.spinBox_cnn_train.value())
            self.ui.spinBox_cnn_test.setValue(abs(self.ui.spinBox_cnn_train.value() - 100))
            self.charged_init = False

    def onChangeSpinBoxCNNTest(self):
        if not self.charged_init:
            self.charged_init = True
            self.ui.horizontalSlider_cnn.setValue(abs(self.ui.spinBox_cnn_test.value() - 100))
            self.ui.spinBox_cnn_train.setValue(abs(self.ui.spinBox_cnn_test.value() - 100))
            self.charged_init = False

    def add_items(self, list_widget: QtWidgets.QListWidget, items: list):
        _translate = QtCore.QCoreApplication.translate
        self.reset_list(list_widget)
        for item in items:
            self._append_items(list_widget, item)

    def _append_items(self, list_widget: QtWidgets.QListWidget, item: str):
        _translate = QtCore.QCoreApplication.translate
        _item = QtWidgets.QListWidgetItem()
        _item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        _item.setCheckState(QtCore.Qt.Unchecked)
        _item.setText(_translate("MainWindow", item))
        list_widget.addItem(_item)

    def reset_list(self, list_widget: QtWidgets.QListWidget):
        list_widget.clear()
        self._append_items(list_widget, 'All')

    def onItemDoubleClicked(self, item):
        self.add_log(f'Debug - {item.text()} double\n')

    def charge_tab(self, index):
        for i in range(len(self.tabs)):
            self.ui.stackedWidget.setCurrentIndex(index)
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
        for plainText in self.logs:
            plainText.insertPlainText(text)

    def gpu_disponivel(self):
        for x in range(torch.cuda.device_count()):
            self.ui.comboBox_gpu.addItem(torch.cuda.get_device_name(x))
    
    def model_layers(self,model):
        nodes, _ = get_graph_node_names(model)
        #retorno=nodes
        for x in nodes:
            self.ui.comboBox_gpu_2.addItem(torch.cuda.get_device_name(x))
    
    def device(self):
        if(self.ui.comboBox_gpu.currentText() == "Do Not Use"):
            return "cpu"
        else:
            for x in range(torch.cuda.device_count()):
                if (self.ui.comboBox_gpu.currentText() == torch.cuda.get_device_name(x)):
                    device="cuda:"+str(x)
                    return str("cuda:"+str(x))

    def open_browse (self, line, tab_name, type_folder):
        #path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a source directory:', expanduser('~'))
        path = QtWidgets.QFileDialog.getExistingDirectory(self)
        # getOpenFileName(self, 'Open a file', '', 'All Files (*.*)')
        if path != '':
            if tab_name == 'cnn' and type_folder=='input_folder':
                self.ui.line_input_1.setText(path)
                self.input=path
            if tab_name == 'cnn' and type_folder=='output_folder':
                self.ui.line_output_1.setText(path)
                self.output=path
            
            if tab_name== 'classif' and type_folder=='input_folder':
                self.ui.line_input_2.setText(path)
                self.populate_list_widget(path)
            if tab_name== 'classif' and type_folder=='output_folder':
                self.ui.line_output_2.setText(path)
            
            if tab_name== 'exib' and type_folder=='input_folder':
                self.ui.line_input_3.setText(path)
                self.populate_list_widget_exibition(path)
            if tab_name== 'exib' and type_folder=='output_folder':
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
    def load_dataloader(self,train,test,batch_size):
        trainDataloader=torch.utils.data.DataLoader(train, batch_size)
        testDataloader=torch.utils.data.DataLoader(test, batch_size)
        #samples, labels = iter(self.trainDataloader).next()
        #train_features, train_labels = next(iter(self.trainDataloader))
        #print(train_features,train_labels)
        #print("Tamanho: ",len(labels[:24]),'\n')
        print(len(trainDataloader))
        print(len(testDataloader))
        return trainDataloader,testDataloader

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

    def kfold_split(self,n_splits):
        path=self.ui.line_input_1.text()
        image_datasets = datasets.ImageFolder(os.path.join(path),transform=self.data_transforms)
        
        kf = KFold(n_splits=n_splits, shuffle=True)
        splits = kf.split(image_datasets)
        
        train_splits = []
        test_splits = []
        
        for train_index, test_index in splits:
            train_dataset = torch.utils.data.Subset(image_datasets, train_index)
            test_dataset = torch.utils.data.Subset(image_datasets, test_index)
            train_splits.append(train_dataset)
            test_splits.append(test_dataset)
        
        return train_splits, test_splits

    def train_test(self):
        #print(self.INPUT)
        path=self.ui.line_input_1.text()
        image_datasets = datasets.ImageFolder(os.path.join(path),transform=self.data_transforms)
        #print(image_datasets)
        self.p_train=self.ui.horizontalSlider_cnn.value()/100
        print('\n\n',self.p_train,'\n\n')
        train_size = int(self.p_train * len(image_datasets))
        test_size = int(len(image_datasets)-train_size)
        train, test = torch.utils.data.random_split(image_datasets, [train_size, test_size])
        #print(len(image_datasets['input']))
        self.image_datasets=image_datasets
        self.labels=image_datasets.classes
        self.dataset_sizes = train_size+test_size
        self.class_names = self.labels
        print(self.labels)
        print(train)
        print(test)
        return train,test



    def accuracy(self,predictions, labels):
        classes = torch.argmax(predictions, dim=1)
        return torch.mean((classes == labels).float())

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
        trainTestORKFold = None
        if self.ui.radio_btn_tt_cnn.isChecked():
            trainTestORKFold = "tt" + self.ui.spinBox_cnn_train.text()
        elif self.ui.radio_btn_kf_cnn.isChecked():
            trainTestORKFold = "kf" + self.ui.radio_btn_kf_cnn.text()

        accurate = []
        epochs = []
        #self.loss_monitor = LossMonitor(models, self.ui.widget_monitor)  # Passe o widget para a classe LossMonitor
        # Criação do QLabel e definição do texto
        #self.label.setText("Novo texto")
        #self.ui.widget_monitor.setWindowTitle("UEEEEEEEEEEEEEEEEEEEEEE")
        # Criação do QLabel dentro do QWidget
        self.loss_monitor.set_model_names(models)

        #QApplication.processEvents()
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

                self.loss_monitor.update_loss(models[x],loss)
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

    def fineTuningOriginal(self, epoch, device, train_dataloader, test_dataloader, models):
        trainTestORKFold = None
        if self.ui.radio_btn_tt_cnn.isChecked():
            trainTestORKFold = "tt" + self.ui.spinBox_cnn_train.text()
        elif self.ui.radio_btn_kf_cnn.isChecked():
            trainTestORKFold = "kf" + self.ui.radio_btn_kf_cnn.text()

        accurate = []
        epochs = []

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

            for i in range(epoch):
                num_correct = 0.0
                num_samples = 0.0
                losses_train = []
                accuracies_train = []
                losses_test = []
                accuracies_test = []
                features=[]
                labels=[]

                self.model[x].train()

                for data, label in train_dataloader:
                    data, label = data.to(device), label.to(device)
                    self.model[x].to(device)
                    targets = self.model[x](data)
                    loss = self.loss_fn(targets, label)
                    accuracies_train.append(
                        label.eq(targets.detach().argmax(dim=1)).float().mean()
                    )
                    #print(device)
                    if device == 'cpu':
                        labels.append(label.numpy())
                        features.append(feature_extractor.extract_features(data))
                    else:
                        labels.append(label.cpu().numpy())
                        features.append(feature_extractor.extract_features(data))
                    #labels.append(label.numpy())
                    #data.to(device)
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

                if i == 0:
                    self.save_features(
                        features, labels, "train", nome, contador_treino
                    )
                del features
                del labels
                contador_treino += 1

                accurate.append(float(num_correct / num_samples))
                epochs.append(i)

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

            models.pop(0)
            del self.model[x]

    
    def layers(self,modelname):
        model=self.get_model(modelname,True)
        return self.nodos(model)

    def save(self,model,nome):
        path = self.output+'/'+nome[:-4]
        os.makedirs(path, exist_ok = True) 
        torch.save(model.state_dict() , os.path.join(path,nome))
        print("SALVANDO EM: \n",path)


    def save_features(self, feature_extractor, dataloader, dataset_type, model_name, device):
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
        save_dir1 = os.path.join(path,f"{dataset_type}/")
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

    

    def checkedItems(self,list):
        selected=[]
        for index in range(list.count()):
            item = list.item(index)
            if item.checkState() == QtCore.Qt.Checked:
                selected.append(item.text())
        return selected
    
    def get_optimizer(self,optimizer_name,model):
        if(optimizer_name=="Adam"):
            otimizador = torch.optim.Adam(model.parameters())
        if(optimizer_name=="SGD"):
            otimizador = torch.optim.SGD(model.parameters(),lr=1e-2)
        return otimizador
    
    def get_model(self,model_name,pretrained):
        model = torch.hub.load('pytorch/vision:v0.10.0', model_name.lower(), pretrained)
        return model

    def dt_transforms(self):
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform

    def select_loss(self,loss_name):
        if (loss_name=="BCEWithLogitsLoss"):
            loss=torch.nn.BCEWithLogitsLoss()
        else: 
            if (loss_name=="CrossEntropyLoss"):
                loss=torch.nn.CrossEntropyLoss()
        return loss

    def otimizador(self):
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
    
    def populate_list_widget_exibition(self,folder_path):
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

    def populate_list_widget(self,folder_path):
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
            #print(f"Folder Path: {path}")
            # Limpar a lista antes de popular novamente
            widget.clear()

            # Obter subpastas de primeiro nível
            subfolders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

            # Adicionar as subpastas com checkboxes ao QListWidget
            for dir_name in subfolders:
                #print(f"Folder Path: {dir_name}")
                item = CheckboxListItem(dir_name)
                widget.addItem(item)

        # Popular o QListWidget com as subpastas de primeiro nível
        populate(list_widget, folder_path)
    

    def classification(self,features,labels,classificador):
        pass
    
    # Função para carregar o arquivo CSV e dividir em treino e teste ou k-fold
    def load_data_csv(self, file_path, labels_path, test_size=None, kfold=False, n_splits=None, random_state=None):
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
    
    def import_classifier(self,classifier_name):
        if classifier_name == "GDM":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression()
        elif classifier_name == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier()
        elif classifier_name == "SVM":
            from sklearn.svm import SVC
            return SVC()
        else:
            raise ValueError("Classifier '{}' not supported.".format(classifier_name))

    def save_results_to_csv(self, results, save_folder, classifier_name):
        for i, result in enumerate(results):
            result_df = pd.DataFrame(result.items(), columns=['Metric', 'Value'])
            filename = f'{save_folder}/{classifier_name}.csv'
            
            # Criar o diretório, caso não exista
            os.makedirs(save_folder, exist_ok=True)
            
            result_df.to_csv(filename, index=False)

    def train_classifier(self, classifier_names, dataset_csv, folder_name,kfold=False):
        if isinstance(classifier_names, str):
            classifier_names = [classifier_names]
            
        save_folder = self.ui.line_output_2.text()
        for classifier_name in classifier_names:
            classifier = self.import_classifier(classifier_name)
            results = []
            
            if kfold:
                splits, features, labels = dataset_csv
                for train_index, test_index in splits:
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
                    
                    # Imprimir métricas
                    print("Accuracy:", accuracy)
                    print("Cohen's Kappa Score:", kappa)
                    print("Precision:", precision)
                    print("Recall:", recall)
                    
                    result = {
                        "Accuracy": accuracy,
                        "Cohen's Kappa Score": kappa,
                        "Precision": precision,
                        "Recall": recall
                    }
                    results.append(result)
            
            else:
                train_features, test_features, train_labels, test_labels = dataset_csv
                
                # Treinar o classificador usando train_features e train_labels
                classifier.fit(train_features, train_labels)
                
                # Testar o classificador usando test_features e test_labels
                predictions = classifier.predict(test_features)
                accuracy = classifier.score(test_features, test_labels)
                
                # Calcular métricas
                kappa = cohen_kappa_score(test_labels, predictions)
                precision = precision_score(test_labels, predictions, average='micro')
                recall = recall_score(test_labels, predictions, average='micro')
                
                # Imprimir métricas
                print("Accuracy:", accuracy)
                print("Cohen's Kappa Score:", kappa)
                print("Precision:", precision)
                print("Recall:", recall)
                
                result = {
                    "Accuracy": accuracy,
                    "Cohen's Kappa Score": kappa,
                    "Precision": precision,
                    "Recall": recall
                }
                results.append(result)
            
            classifier_folder = save_folder+f"/{folder_name}"+"/results/"
            self.save_results_to_csv(results, classifier_folder, classifier_name)

    def train_classifierOriginal(self, classifier_name, dataset_csv, kfold=False):
        classifier = self.import_classifier(classifier_name)
        results = []
        if kfold:
            splits, features, labels = dataset_csv
            for train_index, test_index in splits:
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
                
                # Calcular Precision-Recall Curve
                precision_curve, recall_curve, _ = precision_recall_curve(test_labels, predictions)
                
                # Calcular ROC Curve
                fpr, tpr, _ = roc_curve(test_labels, predictions)
                roc_auc = auc(fpr, tpr)
                
                # Imprimir métricas
                print("Accuracy:", accuracy)
                print("Cohen's Kappa Score:", kappa)
                print("Precision:", precision)
                print("Recall:", recall)
                
                result = {
                    "Accuracy": accuracy,
                    "Cohen's Kappa Score": kappa,
                    "Precision": precision,
                    "Recall": recall
                }
                results.append(result)


        else:
            train_features, test_features, train_labels, test_labels = dataset_csv
            
            # Treinar o classificador usando train_features e train_labels
            classifier.fit(train_features, train_labels)
            
            # Testar o classificador usando test_features e test_labels
            predictions = classifier.predict(test_features)
            accuracy = classifier.score(test_features, test_labels)
            
            # Calcular métricas
            kappa = cohen_kappa_score(test_labels, predictions)
            precision = precision_score(test_labels, predictions,average='micro')
            recall = recall_score(test_labels, predictions,average='micro')
            
            # Calcular Precision-Recall Curve
            #precision_curve, recall_curve, _ = precision_recall_curve(test_labels, predictions)
            
            # Calcular ROC Curve
            #fpr, tpr, _ = roc_curve(test_labels, predictions)
            #roc_auc = auc(fpr, tpr)
            
            # Imprimir métricas
            print("Accuracy:", accuracy)
            print("Cohen's Kappa Score:", kappa)
            print("Precision:", precision)
            print("Recall:", recall)

            result = {
                "Accuracy": accuracy,
                "Cohen's Kappa Score": kappa,
                "Precision": precision,
                "Recall": recall
            }
            results.append(result)
    

    # Tela de exibição

    def converter_path_caminho(self,caminho_original):
        partes = caminho_original.replace('\\', '/')
        partes = partes.split("/")
        nome_modelo = partes[-3]
        #print(nome_modelo)
        nome_csv = partes[-1].replace('.csv', '')
        return f"{nome_modelo}_{nome_csv}"


    def updateTableWithCSVs(self, table: QTableWidget, csv_paths: list):
        # Limpar a tabela
        table.clear()
        all_data = []

        for path in csv_paths:
            # Abrir o arquivo CSV atual
            with open(path, 'r') as file:
                # Ler os dados do arquivo CSV usando o módulo csv
                reader = csv.reader(file)
                # Converter os dados em uma lista
                data = list(reader)
                # Adicionar os dados do CSV atual à lista de dados
                nome=self.converter_path_caminho(path)
                #print(nome)
                all_data.append((nome, data))  # Adicionar uma tupla com o nome do arquivo e os dados à lista

        # Obter o número de linhas e colunas dos dados
        num_csvs = len(all_data)
        rows = sum(len(data) for _, data in all_data)
        columns = len(all_data[0][1][0]) if rows > 0 else 0

        # Definir o número de linhas e colunas da tabela
        table.setRowCount(rows)
        table.setColumnCount(columns + 1)  # Adicionar uma coluna extra para o nome do arquivo

        # Preencher a tabela com os dados dos CSVs
        current_row = 0
        for csv_index, (csv_name, csv_data) in enumerate(all_data):
            # Preencher a célula com o nome do arquivo
            name_item = QTableWidgetItem(csv_name)
            table.setItem(current_row, 0, name_item)
            table.setSpan(current_row, 0, len(csv_data), 1)  # Expandir a célula do nome do arquivo por todas as linhas dos dados

            # Preencher as células com os dados do CSV atual
            for row, csv_row in enumerate(csv_data):
                for column, value in enumerate(csv_row):
                    item = QTableWidgetItem(value)
                    table.setItem(current_row + row, column + 1, item)

            current_row += len(csv_data)

        for column in range(table.columnCount()):
            table.setColumnWidth(column, 150)  # Define uma largura mínima de 150 pixels para todas as colunas

    def convertPath(self,string: str, path: str) -> str:
        # Encontrar o último '_' na string
        last_underscore_index = string.rfind('_')

        # Extrair o valor após o último '_'
        gdm_value = string[last_underscore_index + 1:]

        # Remover o final após o último '_'
        string_without_end = string[:last_underscore_index]

        # Construir o novo caminho
        new_path = os.path.join(path, string_without_end, 'results', gdm_value)

        return new_path

    def run_exib(self):
        selected_saved_models = self.checkedItems(self.ui.list_saved_models)
        selected_metrics = self.checkedItems(self.ui.list_metrics)
        paths=[]
        for path in selected_saved_models:
            paths.append(self.convertPath(path,self.ui.line_input_3.text()))
        self.updateTableWithCSVs(self.ui.tableWidget,paths)

    def run_classificacao(self):
        torch.cuda.empty_cache()
        gc.collect()
        self.ui.plainTextEdit_log2.clear()
        selected_classificadores = self.checkedItems(self.ui.listWidget_models)
        selected_datasets = self.checkedItems(self.ui.listWidget_datasets)
        if(self.ui.line_input_2.text() == '' ):
            self.ui.plainTextEdit_log2.insertPlainText("Por favor selecione a pasta de entrada!\n")
        elif(self.ui.line_output_2.text() == '' ):
            print("Por favor selecione a pasta de saída!")
        elif(len(selected_classificadores) == 0 ):
            print("Por favor selecione ao menos um modelo!")
        elif(len(selected_datasets) == 0 ):
            print("Por favor selecione ao menos um dataset!")
        elif(self.ui.radio_btn_tt_classifi.isChecked()==False and self.ui.radio_btn_kf_classifi.isChecked()==False):
            self.ui.plainTextEdit_log2.insertPlainText("Configure o Treino e Teste ou K-Fold!\n")
        else:
            dataset_csv=None
            if self.ui.radio_btn_tt_classifi.isChecked():
                del dataset_csv
                file_path_csv = Path(self.ui.line_input_2.text() + "/" + selected_datasets[0] + "/train/train_features.csv")
                label_path_csv = Path(self.ui.line_input_2.text() + "/" + selected_datasets[0] + "/train/train_labels.csv")
                test_size=(100-(self.ui.horizontalSlider_classifi.value()))/100
                print(file_path_csv) 
                dataset_csv = self.load_data_csv(file_path_csv, label_path_csv, test_size)
                self.train_classifier(selected_classificadores,dataset_csv,selected_datasets[0],False)
            else:
                del dataset_csv
                file_path_csv = Path(self.ui.line_input_2.text() + "/" + selected_datasets[0] + "/train/train_features_0.csv")
                label_path_csv = Path(self.ui.line_input_2.text() + "/" + selected_datasets[0] + "/train/train_labels_0.csv")
                splits=int(self.ui.spinBox_classifi_kf.text())
                dataset_csv = self.load_data_csv(file_path_csv, label_path_csv, n_splits=splits, kfold=True)
                self.train_classifier(selected_classificadores,dataset_csv,selected_datasets[0],True)
            #print(dataset_csv)
            print("Tudo certo!")


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
                
                
            if(self.ui.radio_btn_nw.isChecked()):
                epochs=self.controlNewWeights.ui.spinBox_epochs.value()
                batch_size=self.controlNewWeights.ui.spinBox_batch_size.value()
                selected_CNN = self.checkedItems(self.ui.listWidget_arch)
                self.model=self.get_model(selected_CNN[i],pretrained=True)
                self.loss_fn=self.select_loss(self.controlNewWeights.ui.comboBox_loss.currentText())
                self.optimizer=self.get_optimizer(self.controlNewWeights.ui.comboBox_optimizer.currentText())
                self.load_dataloader(batch_size)
                print("VAMOS AO PROCESSAMENTO..\n")
                self.newweights(epochs)
                print("NewWeights")
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
