import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('moore.csv',header=None).values
X = data[:,0].reshape(-1,1)
Y = data[:,1].reshape(-1,1)

Y = np.log(Y)

mx = X.mean()
sx = X.std()
my = Y.mean()
sy = Y.std()

X = (X-mx)/sx
Y = (Y-my)/sy
plt.scatter(X,Y)
plt.show()

inputs = torch.from_numpy(X.astype(np.float32))

target = torch.from_numpy(Y.astype(np.float32))

model = nn.Linear(1,1)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(),lr = 0.3,momentum=0.7)

epochs = 150

losses = []
for i in range(epochs):
    model.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs,target)

    losses.append(loss.item())

    loss.backward()
    optimizer.step()

    if((i+1)%20==0): print(f'Epoch : {i+1} / {epochs}, Loss : {loss.item():.4f}')

plt.plot(losses)
plt.show()

predicted = model(inputs).detach().numpy()
plt.scatter(X,Y, label='Original Data')
plt.plot(X,predicted, label ='Fitted Data')
plt.legend()
plt.show()

weights = model.weight.data.numpy()
bias = model.bias.data.numpy()
#a = weights[0,0] *sy/sx

print(weights,bias)