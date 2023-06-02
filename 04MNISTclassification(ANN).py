from torchvision import datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

train_data = datasets.MNIST(root='./datasets/', train=True, transform=transforms.ToTensor(),
                           download=True)
test_data = datasets.MNIST(root='./datasets/', train=False, transform=transforms.ToTensor(), 
                           download=True)

print("Loading the data...\n")
batch_Size = 128 

Train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size = batch_Size, shuffle = True)

Test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size = batch_Size, shuffle = False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
        
    model = nn.Sequential(
        nn.Linear(784,128),
        nn.ReLU(),
        nn.Linear(128,10)
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters())
    
    epochs = 10

    train_losses = np.zeros(epochs)
    test_losses =np.zeros(epochs)

    for i in range(epochs):

        train_loss=[]

        # calculate losses and train the model from the batchloader data 
        for inputs,targets in Train_loader:
            
            # move data to the gpu since our model is now running on gpu hehe

            inputs,targets = inputs.to(device),targets.to(device)
            
            #reshape the data / flatten the 28x28 data into a long vector
            inputs = inputs.view(-1,784)
            
            #the classic model training steps 
            # forward - backward - optimize
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs,targets)

            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
        #take the mean of all the losses and append it to the loss per epoch, but in reality the losses are still
        #different for all the batches. so your final mean loss will still not be indicative of actual test loss value
        #but could be a good approximation. 
            
        train_losses[i] = np.mean(train_loss)

        # the same thing, but for testing batch. we calculate losses for test batch, but dont optimize our 
        # model based on the loss.
        
        test_loss = []

        for inputs,targets in Test_loader:
            inputs,targets = inputs.to(device),targets.to(device)
            inputs = inputs.view(-1,784)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs,targets)
            
            test_loss.append(loss.item())
        
        test_losses[i] = np.mean(test_loss)

        print(f'Epoch : {i+1}/{epochs}, Train Loss : {train_losses[i]} , Test Loss : {test_losses[i]}')

    plt.plot(train_losses,label='Train Losses')
    plt.plot(test_losses,label='Test Losses')
    plt.legend()
    plt.show()
    
    torch.save(model,'MNISTClassifier.pt')

#check if a model already exists, if it does, we dont train the model.
try:
    model = torch.load('MNISTClassifier.pt', map_location=device)    
    model.to(device)
    while True:
        
        x = input('Existing Model found, do you want to use the existing one or wish to Re-Train the model ? (1 = Use Existing , 2 = Re-Train)...\n')
        
        if x == str(1):
            print('Using Existing Model.... \n')
            break

        elif x == str(2):
            print('Training the model.... \n')
            train()
            break

        else:
            print('Enter valid response')

except FileNotFoundError as err:    
    print('Existing Model not found, Training the model, saving it as ''MNISTClassifier.pt'' .... \n')    
    train()

print('\nCalculating Accuracies depending on training and testing data...\n')
#Accuracy calculations with batches and epoch

correct = 0
total = 0

with torch.no_grad():
    for data in Train_loader:
        images,labels = data 
        images,labels = images.to(device),labels.to(device)
        
        #this is needed as we want to flatten the 28*28 pixels/images into a long vector
        images = images.view(-1,784)

        outputs = model(images)
        
        _,predicted = torch.max(outputs,1)

        correct+= (predicted == labels).sum().item()
        total += labels.shape[0]

train_accuracy = correct/total

correct = 0
total = 0

with torch.no_grad():
    for data in Test_loader:
        images,labels = data 
        images,labels = images.to(device),labels.to(device)
        
        #this is needed as we want to flatten the 28*28 pixels/images into a long vector
        images = images.view(-1,784)

        outputs = model(images)
        
        _,predicted = torch.max(outputs,1)

        correct+= (predicted == labels).sum().item()
        total += labels.shape[0]

test_accuracy = correct/total

print(f'Train Accuracy : {train_accuracy*100}, Test Accuracy : {test_accuracy*100}')


# Now to see where the model makes the inaccuracies, by using the testing data and checking the data against
# the true labels and predicted labels.

print("\nNow Let's see where did our model get confused, by comparing the predicted label of an image and true label of an image... ")

x_test = test_data.data.numpy()
y_test = test_data.targets.numpy()
y_pred = np.array([])

with torch.no_grad():
    for data in Test_loader:
        images,labels = data 
        images,labels = images.to(device),labels.to(device)
        
        #this is needed as we want to flatten the 28*28 pixels/images into a long vector
        images = images.view(-1,784)

        outputs = model(images)
        
        _,predicted = torch.max(outputs,1)
        
        y_pred = np.concatenate(( y_pred, predicted.cpu().numpy()))

incorrectPredIdx = np.where(y_pred!=y_test)[0]

print(f'\nIncorrectly Predicted Images :{incorrectPredIdx.shape[0]}')

x = int(input('\nHow many samples do u wish to see? type 0 to exit.. : '))

while True:
    if(x>0):
        n = np.random.choice(incorrectPredIdx)

        plt.imshow(x_test[n], cmap = 'gray')
        plt.title(f' True Label : {y_test[n]}, Prediction : {y_pred[n]}')
        plt.show()
        x-=1

    else:
        break