import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

N = 20

X = np.random.random(N) * 10 - 7

Y = 0.7 * X - 4.2 + np.random.randn(N) 

plt.scatter(X,Y)
plt.show()

model = nn.Linear(1,1)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(),lr = 0.03)

X = X.reshape(N,1)

Y = Y.reshape(N,1)

inputs = torch.from_numpy(X.astype(np.float32))

target = torch.from_numpy(Y.astype(np.float32))

epochs = 150

losses = []
for i in range(epochs):
    model.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs,target)

    losses.append(loss.item())

    loss.backward()
    optimizer.step()

    print(f'Epoch : {i+1} / {epochs}, Loss : {loss.item():.4f}')

plt.plot(losses)
plt.show()

predicted = model(inputs).detach().numpy()
plt.scatter(X,Y, label='Original Data')
plt.plot(X,predicted, label ='Fitted Data')
plt.legend()
plt.show()

weights = model.weight.data.numpy()
bias = model.bias.data.numpy()

print(weights,bias)