import sys
import os
from os.path import expanduser
from pathlib import Path
import json
import pickle
import time
import copy
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


a="Adam"

print(torch.optim.Adam())