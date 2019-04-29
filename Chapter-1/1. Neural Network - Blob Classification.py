import numpy as np
from sklearn.datasets import make_blobs
import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid
import matplotlib.pyplot as plt



# Setup the random seed to keep the data reproducible.
torch.manual_seed(7388)

#Number of sampled data points we need to generate.
n_samples = 1500
X,y = make_blobs(n_samples=n_samples, centers=2, random_state=7388)

# we have data in just two dimensions. So we need to use input size as 2
input_size = 2

# This is the number of neuron in the hidden layer. We can play around with this number.
# More neurons and more layers(not included in this code), means a complex combination
# of the input parameters.
hidden_size = 2

# Since we are generating a binary class, we need to identify to which blob (class) the
# point belongs to.
output_size = 1

y = np.reshape(y, (len(y),1))
inputs = torch.tensor(X, dtype=torch.float)
labels = torch.tensor(y, dtype=torch.float)

# We write a simple sequential two layer neural network model.
model = Sequential(
    Linear(in_features=input_size, out_features=hidden_size),
    ReLU(),
    Linear(in_features=input_size, out_features=output_size),
    Sigmoid()
)

# Setup the loss function. We are currently using Binary Cross Entropy
# You can also use torch.nn.BCEWithLogitsLoss and remove the Sigmoid
# layer from the model as this is already included in the loss function.
criterion = torch.nn.BCELoss(reduction='mean')

# Setup the optimizer to determine the parameters for the neural network
# to do binary classification. Do play around this other optimizers.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# How many epochs should be used for the model training?
num_epochs = 30

# At what frequency should we print the current loss.
print_freq = 10

for epoch in range(num_epochs):

    # Calculate the output based on the current network parameters.
    labels_pred = model(inputs)

    # Calculate the error, based on the previously defined criterion.
    # We used binary cross entropy as the loss (aka error) criteria.
    loss = criterion(labels_pred, labels)

    # Print the Epochs and the errors.
    if epoch % print_freq == 0:
        print(' Epoch {epoch:3.0f} Loss {loss:.4f}'
              .format(epoch=epoch, loss=loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plotting the Decision Region and the blobs
x1_min, x1_max = -15, 15
x2_min, x2_max = -15, 15
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                     np.arange(x2_min, x2_max, 0.1))

Z = model(torch.tensor((np.c_[x1.ravel(), x2.ravel()]), dtype=torch.float))

# Convert the model predictions to numpy and reshape it to in line with
# the input parameters
Z = Z.detach().numpy().reshape(x1.shape)

# Reshape the y back into a single numpy array
y = np.concatenate(y, axis=0)

plt.figure(figsize=(8,8))
# Draw the region's decision boundary
plt.contourf(x1, x2, Z, levels=0, cmap='coolwarm')

# Plot the blobs on the chart
plt.scatter(X[:, 0],
            X[:, 1],
            c=y,
            cmap='Dark2',
            edgecolor='k')
plt.show()


