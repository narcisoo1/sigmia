import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.model_type = self._get_model_type(model)

    def _get_model_type(self, model):
        if isinstance(model, models.AlexNet):
            return "AlexNet"
        elif isinstance(model, models.ResNet):
            return "ResNet"
        elif isinstance(model, models.GoogLeNet):
            return "GoogLeNet"
        else:
            raise ValueError("Modelo não suportado.")

    def extract_features(self, inputs):
        if self.model_type == "AlexNet":
            features = self.model.features(inputs)
        elif self.model_type == "ResNet":
            features = self.model.conv1(inputs)
            features = self.model.bn1(features)
            features = self.model.relu(features)
            features = self.model.maxpool(features)

            features = self.model.layer1(features)
            features = self.model.layer2(features)
            features = self.model.layer3(features)
            features = self.model.layer4(features)

            features = self.model.avgpool(features)

        elif self.model_type == "GoogLeNet":
            features = self.model.conv1(inputs)
            features = self.model.maxpool1(features)
            features = self.model.conv2(features)
            features = self.model.conv3(features)
            features = self.model.maxpool2(features)
            features = self.model.inception3a(features)
            features = self.model.inception3b(features)
            features = self.model.maxpool3(features)
            features = self.model.inception4a(features)
            features = self.model.inception4b(features)
            features = self.model.inception4c(features)
            features = self.model.inception4d(features)
            features = self.model.inception4e(features)
            features = self.model.maxpool4(features)
            features = self.model.inception5a(features)
            features = self.model.inception5b(features)
            features = self.model.avgpool(features)

        features = torch.flatten(features, 1)

        return features