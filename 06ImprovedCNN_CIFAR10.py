import torch
from torch import utils as torchutils
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

training_transform = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0,translate=(0.1,0.1)),
    transforms.ToTensor()]
)
train_data = datasets.CIFAR10(root='./datasets', train=True,transform=training_transform,
                                   download=True)

test_data = datasets.CIFAR10(root='./datasets',train=False,transform=transforms.ToTensor(),
                                  download=True)

batch_size = 128

train_loader = torchutils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle = True)

test_loader = torchutils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle = False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# number of classes or targets of the dataset
K = len(set(train_data.targets))

# convolutional neural network model

class CNN(nn.Module):
    def __init__(self, K) :
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024,K)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        x = F.dropout(x,p=0.5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,p=0.5)
        x = self.fc2(x)
        return x
    
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
train_losses=[]
test_losses=[]
#check if a model already exists, if it does, we dont train the model.
try:
    model = torch.load('CIFARClassifier.pt', map_location=device)    
    model.to(device)
    while True:
        
        x = input('Existing Model found, do you want to use the existing one or wish to Re-Train the model ? (1 = Use Existing , 2 = Re-Train)...\n')
        
        if x == str(1):
            print('Using Existing Model.... \n')
            break

        elif x == str(2):
            print('Training the model.... \n')
            train_losses, test_losses =  batch_training(model, criterion, optimizer,train_loader, test_loader, epochs=80)
            break

        else:
            print('Enter valid response')

except FileNotFoundError as err:    
    print('Existing Model not found, Training the model, saving it as ''CIFARClassifier.pt'' .... \n')    
    train_losses, test_losses =  batch_training(model, criterion, optimizer,train_loader, test_loader, epochs=80)
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.legend()
    plt.show()
    torch.save(model,'CIFARClassifier.pt')

model.eval()
# correct = 0
# total = 0

# for data in train_loader:
#     images,labels = data 
#     images,labels = images.to(device),labels.to(device)

#     outputs = model(images)
        
#     _,predicted = torch.max(outputs,1)

#     correct+= (predicted == labels).sum().item()
#     total += labels.shape[0]

# train_accuracy = correct/total

# correct = 0
# total = 0

# for data in test_loader:
#     images,labels = data 
#     images,labels = images.to(device),labels.to(device)

#     outputs = model(images)
    
#     _,predicted = torch.max(outputs,1)

#     correct+= (predicted == labels).sum().item()
#     total += labels.shape[0]

# test_accuracy = correct/total

# print(f'Train Accuracy : {train_accuracy*100}, Test Accuracy : {test_accuracy*100}')

labels = '''airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck '''.split("\n")

x_test = test_data.data
y_test = test_data.targets
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
y_test = np.array(y_test).astype(np.uint8)
y_pred = np.array(y_pred).astype(np.uint8)
while True:
    if(x>0):
        n = np.random.choice(incorrectPredIdx)

        plt.imshow(x_test[n].reshape(32,32,3))
        plt.title(f' True Label : {labels[y_test[n]]}, Prediction : {labels[y_pred[n]]}')
        plt.show()
        x-=1

    else:
        break

from torchsummary import summary

print(summary(model,(3,32,32)))