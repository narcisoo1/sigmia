import torch
from torch import nn
import torch.nn.functional as fun
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as num
 

transform = transforms.Compose([
                                 transforms.ToTensor()
])
 

traindata = datasets.MNIST('', train = True, transform = transform, download = True)
traindata,testdata = random_split(traindata,[50000,10000])

train_loader = DataLoader(traindata, batch_size=32)
test_loader = DataLoader(testdata, batch_size=32)
 
# Building Our Mode
class network(nn.Module):

    def __init__(self):
        super(network,self).__init__()
        self.fc = nn.Linear(28*28, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
 

    def forward(self, y):
        y = y.view(y.shape[0],-1)   
        y = fun.relu(self.fc(y))
        y = fun.relu(self.fc1(y))
        y = self.fc2(y)
        return y
 
models = network()
if torch.cuda.is_available():
    models = models.cuda()
 
criterions = nn.CrossEntropyLoss()
optimizers = torch.optim.SGD(models.parameters(), lr = 0.01)
 
epoch = 7
minvalid_loss = num.inf

for i in range(epoch):
    trainloss = 0.0
    models.train()     
    for data, label in train_loader:
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        
        optimizers.zero_grad()
        targets = models(data)
        loss = criterions(targets,label)
        loss.backward()
        optimizers.step()
        trainloss += loss.item()
    
    testloss = 0.0
    models.eval()    
    for data, label in test_loader:
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        
        targets = models(data)
        loss = criterions(targets,label)
        testloss = loss.item() * data.size(0)

    print(f'Epoch {i+1} \t\t Training data: {trainloss / len(train_loader)} \t\t Test data: {testloss / len(test_loader)}')
    if minvalid_loss > testloss:
        print(f'Test data Decreased({minvalid_loss:.6f}--->{testloss:.6f}) \t Saving The Model')
        minvalid_loss = testloss

        torch.save(models.state_dict(), 'saved_model.pth')