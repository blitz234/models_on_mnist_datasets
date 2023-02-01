# %%
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# %%
from torch import optim
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = torch.flatten(images,start_dim=1, end_dim= -1)
    
        # TODO: Training pass
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline


images, labels = next(iter(trainloader))

img = images[0].view(1,784)

with torch.no_grad():
    logps = model.forward(img)

ps = torch.exp(logps)

result = pd.DataFrame(data = ps.numpy().reshape(10,1), columns = ['Probability'])
result['number'] = range(10)

fig = plt.figure(figsize= [12,3])

plt.subplot(1,2,1)
sns.barplot(data= result, y = 'Probability', x = 'number')

plt.subplot(1,2,2)
plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r');


# %%



