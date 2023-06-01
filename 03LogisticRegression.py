import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# breast cancer dataset
data = load_breast_cancer()

#inputs and targets
X,Y = data['data'], data['target']

#split data into training and test sets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)

N, D = X_train.shape

#preprocess the data, scale them to their mean and standard deviation. 
#apply zero mean unit variance
from sklearn.preprocessing import StandardScaler

 # apply standard scaling
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

model = nn.Sequential(
    nn.Linear(D,1),
    nn.Sigmoid()
)

#convert into tensors
X_train_std = torch.from_numpy(X_train_std.astype(np.float32))

X_test_std  = torch.from_numpy(X_test_std.astype(np.float32))

Y_train = torch.from_numpy(Y_train.astype(np.float32).reshape(-1,1))

Y_test = torch.from_numpy(Y_test.astype(np.float32).reshape(-1,1))

#loss function and optimizer
criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters())

epochs = 700

train_losses = np.zeros(epochs)
test_losses = np.zeros(epochs)
# train_acc = np.zeros(epochs)
# test_acc = np.zeros(epochs)
#train the network
print('model training....')

for i in range(epochs):
    model.zero_grad()

    outputs = model(X_train_std)
    loss = criterion(outputs,Y_train)

    loss.backward()
    optimizer.step()

    outputs_test = model(X_test_std)
    loss_test = criterion(outputs_test,Y_test)

    train_losses[i] = loss.item()
    test_losses[i] = loss_test.item()
    
    # Accuracy Calculation per iteration
    # with torch.no_grad():
    #     o_train = outputs
    #     o_train = np.round(o_train.numpy())
    #     train_acc[i] = np.mean(o_train == Y_train.numpy())

    #     o_test = outputs_test
    #     o_test = np.round(o_test.numpy())
    #     test_acc[i] = np.mean(o_test == Y_test.numpy())

    # if((i+1)%20==0): print(f'Train accuracy / iter :{train_acc[i]:.4f} and Test accuracy/iter:{test_acc[i]:.4f}')

    if((i+1)%20==0): print(f'Epoch : {i+1} / {epochs}, Train Loss : {loss.item():.4f}, Test Loss : {loss_test.item():.4f}')
    

#accuracy of training vs testing as a mean, after training
with torch.no_grad():
    o_train = model(X_train_std)
    o_train = np.round(o_train.numpy())
    acc_train = np.mean(o_train == Y_train.numpy())

    o_test = model(X_test_std)
    o_test = np.round(o_test.numpy())
    acc_test = np.mean(o_test == Y_test.numpy())

print(f'Train accuracy:{acc_train:.4f} and Test accuracy:{acc_test:.4f}')

plt.plot(train_losses, label = 'Training Loss')
plt.plot(test_losses, label = 'Test losses')
plt.legend()
plt.show()

#save the model in our favoured name and format
torch.save(model,'breast_cancer_logistic_regression.pt')

### Saving and loading the model using state dictionary instead of direct model

#torch.save(model.state_dict(),'breast_cancer_logistic_regression.pt')
#recreate the first model to load in our original model
# model2 = nn.Sequential(
#     nn.Linear(D,1),
#     nn.Sigmoid()
# )

#load in the model
# model2.load_state_dict(torch.load('breast_cancer_logistic_regression.pt'))

#load the model using torch.load() function
model2 = torch.load('breast_cancer_logistic_regression.pt')

#check if it's the same as our first model
for params1,params2 in zip(model.parameters(),model2.parameters()):
    print(params1 == params2)

#map the accuracies and compare it to our original accuracies
print('\n Loaded model accuracies :\n')

with torch.no_grad():
    o_train = model2(X_train_std)
    o_train = np.round(o_train.numpy())
    acc_train = np.mean(o_train == Y_train.numpy())

    o_test = model2(X_test_std)
    o_test = np.round(o_test.numpy())
    acc_test = np.mean(o_test == Y_test.numpy())

print(f'Train accuracy:{acc_train:.4f} and Test accuracy:{acc_test:.4f}')

## plot the accuracies
# plt.plot(train_acc, label = 'Training acc')
# plt.plot(test_acc, label = 'Test acc')
# plt.legend()
# plt.show()