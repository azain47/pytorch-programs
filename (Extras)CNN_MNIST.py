from torchvision import datasets
import torchvision.transforms as transforms
import torch
from torch import utils as torchutils
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

train_data = datasets.MNIST(root='./datasets', train=True,transform=transforms.ToTensor(),
                                   download=True)

test_data = datasets.MNIST(root='./datasets',train=False,transform=transforms.ToTensor(),
                                  download=True)

batch_size = 128

train_loader = torchutils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle = True)

test_loader = torchutils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle = False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# number of classes or targets of the dataset
K = len(set(train_data.targets.numpy()))

# convolutional neural network model

class CNN(nn.Module):
    def __init__(self, K) :
        super(CNN,self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=128*2*2,out_features=512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,K)
        )

    def forward(self,X):
        out = self.conv_layers(X)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out
    
model = CNN(K)

model.to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
def batch_training(model, criterion, optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss= []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)

            loss = criterion(outputs,targets)
            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())
        
        train_loss = np.mean(train_loss)

        model.eval()

        test_loss = []

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs,targets)

            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        train_losses[it] = train_loss
        test_losses[it] = test_loss

        dt = datetime.now()-t0

        print(f'Epoch:{it+1}, Time taken:{dt}, Train loss:{train_loss:.4f}, \
              Test Loss:{test_loss:.4f}')
        
    return train_losses, test_losses

train_losses, test_losses =  batch_training(model, criterion, optimizer,train_loader, test_loader, epochs=10)

plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

model.eval()
correct = 0
total = 0

for data in train_loader:
    images,labels = data 
    images,labels = images.to(device),labels.to(device)

    outputs = model(images)
        
    _,predicted = torch.max(outputs,1)

    correct+= (predicted == labels).sum().item()
    total += labels.shape[0]

train_accuracy = correct/total

correct = 0
total = 0

for data in test_loader:
    images,labels = data 
    images,labels = images.to(device),labels.to(device)

    outputs = model(images)
    
    _,predicted = torch.max(outputs,1)

    correct+= (predicted == labels).sum().item()
    total += labels.shape[0]

test_accuracy = correct/total

print(f'Train Accuracy : {train_accuracy*100}, Test Accuracy : {test_accuracy*100}')

x_test = test_data.data.numpy()
y_test = test_data.targets.numpy()
y_pred = np.array([])

for data in test_loader:
    images,targets = data 
    images,targets = images.to(device),targets.to(device)

    outputs = model(images)
    
    _,predicted = torch.max(outputs,1)
    
    y_pred = np.concatenate(( y_pred, predicted.cpu().numpy()))

incorrectPredIdx = np.where(y_pred!=y_test)[0]

print(f'\nIncorrectly Predicted Images :{incorrectPredIdx.shape[0]}')

x = int(input('\nHow many samples do u wish to see? type 0 to exit.. : '))
# y_test = y_test.astype(np.uint8)
# y_pred = y_pred.astype(np.uint8)
while True:
    if(x>0):
        n = np.random.choice(incorrectPredIdx)

        plt.imshow(x_test[n].reshape(28,28), cmap = 'gray')
        plt.title(f' True Label : {y_test[n]}, Prediction : {y_pred[n]}')
        plt.show()
        x-=1

    else:
        break
