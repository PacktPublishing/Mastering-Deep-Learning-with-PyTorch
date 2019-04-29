import numpy as np
from sklearn.datasets import make_circles
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import matplotlib.pyplot as plt

torch.manual_seed(7388)
#Number of sampled data points we need to generate.
n_samples = 1500

class makeCircles(data.Dataset):
    def __init__(self, n_samples=1500):
        X, y = make_circles(n_samples=n_samples, noise=.05)
        self.inputs = torch.tensor(X, dtype=torch.float)
        self.labels = torch.tensor(np.reshape(y, (len(y), 1)), dtype=torch.float)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class Net(nn.Module):
    def __init__(self, n_features, n_neurons):
        super(Net, self).__init__()
        self.hidden_1 = nn.Linear(in_features=n_features, out_features=n_neurons)
        self.hidden_2 = nn.Linear(in_features=n_neurons, out_features=n_neurons)
        self.out_layer = nn.Linear(in_features=n_neurons, out_features=1)

    def forward(self, X):
        out = F.relu(self.hidden_1(X))
        out = F.relu(self.hidden_2(out))
        out = torch.sigmoid(self.out_layer(out))
        return out


trainData = makeCircles(n_samples=n_samples)
trainLoader = data.DataLoader(dataset=trainData,
                              batch_size=1500,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=False,
                              shuffle=True)


model = Net(n_features=2, n_neurons=20)

# Setup the loss function. We are currently using Binary Cross Entropy
# You can also use torch.nn.BCEWithLogitsLoss and remove the Sigmoid
# layer from the model as this is already included in the loss function.
criterion = torch.nn.BCELoss(reduction='mean')

# Setup the optimizer to determine the parameters for the neural network
# to do binary classification. Do play around this other optimizers.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# How many epochs should be used for the model training?
num_epochs = 500

# At what frequency should we print the current loss.
print_freq = 10

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(trainLoader):
        # Calculate the output based on the current network parameters.
        labels_pred = model(data)

        # Calculate the error, based on the previously defined criterion.
        # We used binary cross entropy as the loss (aka error) criteria.
        loss = criterion(labels_pred, target)

        # Print the Epochs and the errors.
        if epoch % print_freq == 0:
            print(' Epoch {epoch:3.0f} Loss {loss:.4f}'
                  .format(epoch=epoch, loss=loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Plotting the Decision Region and the blobs
x1_min, x1_max = -1.5, 1.5
x2_min, x2_max = -1.5, 1.5
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                     np.arange(x2_min, x2_max, 0.1))

Z = model(torch.tensor((np.c_[x1.ravel(), x2.ravel()]), dtype=torch.float))

# Convert the model predictions to numpy and reshape it to in line with
# the input parameters
Z = Z.detach().numpy().reshape(x1.shape)

# Extract the input arrarys and the labels
X, y = [], []
for x in range(n_samples):
    inputs, label = trainData.__getitem__(x)
    X.append(inputs.numpy().tolist())
    y.append(label.numpy().tolist())

# Reshape the y back into a single numpy array
X = np.array(X)
y = np.concatenate(np.array(y), axis=0)

plt.figure(figsize=(8,8))
# Draw the region's decision boundary
plt.contourf(x1, x2, Z, levels=0, cmap='coolwarm')

# # Plot the blobs on the chart
plt.scatter(X[:, 0],
            X[:, 1],
            c=y,
            cmap='Dark2',
            edgecolor='k')
plt.show()
