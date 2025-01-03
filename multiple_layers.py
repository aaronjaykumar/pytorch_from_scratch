import nn
import numpy as np
from matplotlib import pyplot as plt


# -- Define the neural network model --
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x, self)
        x = self.sigmoid1(x, self)
        x = self.fc2(x, self)
        x = self.sigmoid2(x, self)
        x = self.fc3(x, self)
        x = self.sigmoid3(x, self)
        return x


# -- Parameters --
num_epochs = 50000  # Number of epochs for training
beta = 0.1  # Learning rate
N_training = 1000  # Number of samples for training

# -- Create an instance of the model --
input_size = 2
output_size = 1
model = SimpleNN(input_size, 6, 6, output_size)

# -- Generate random training data --
np.random.seed(42)
x = 2 * np.random.rand(2, N_training) - 1
label = np.ones((1, N_training)) * (np.sqrt(x[0] ** 2 + x[1] ** 2) + 0.1 * np.random.randn(N_training) > 0.5)


# -- Define the loss function --
criterion = nn.BCELoss()

# -- Plot --
plt.figure(1)
y0_idx = (label[0, :] == 0).nonzero()[0]
y1_idx = (label[0, :] == 1).nonzero()[0]
plt.scatter(x[0, y0_idx], x[1, y0_idx], marker='x', color='b', label='Label 0')
plt.scatter(x[0, y1_idx], x[1, y1_idx], marker='x', color='r', label='Label 1')
circle1 = plt.Circle((0, 0), 0.5, edgecolor='black', fill=False, label='Ground Truth (unknown)')
ax = plt.gcf().gca()
ax.add_patch(circle1)
plt.title('Training Dataset')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.legend()
plt.axis('square')


loss_vec = []
for epoch in range(num_epochs):
    # -- Forward propagation --
    outputs = model(x)

    # -- Compute loss --
    loss = criterion(label, outputs, model)
    loss_vec.append(loss)  # We track the loss to plot it later on

    # -- Backpropagation --
    model.zero_grad()
    model.backward()

    # -- Update Weights --
    model.step(beta)


plt.figure(3)
plt.clf()
plt.plot(loss_vec)
plt.xlabel('Epoch')
plt.ylabel('BCE')
plt.title('BCE v. Epoch')
plt.grid()

plt.figure(2)
plt.clf()
outputs = np.round(model(x))
y0_idx = (outputs[0, :] == 0).nonzero()[0]
y1_idx = (outputs[0, :] == 1).nonzero()[0]
plt.scatter(x[0, y0_idx], x[1, y0_idx], marker='o', color='b', label='Label 0')
plt.scatter(x[0, y1_idx], x[1, y1_idx], marker='o', color='r', label='Label 1')
circle1 = plt.Circle((0, 0), 0.5, edgecolor='black', fill=False, label='Ground Truth (unknown)')
ax = plt.gcf().gca()
ax.add_patch(circle1)
plt.title('Classification after Training')
plt.axis('square')

x0_min, x0_max = x[0, :].min() - 1, x[0, :].max() + 1
x1_min, x1_max = x[1, :].min() - 1, x[1, :].max() + 1
x0_grid, x1_grid = np.meshgrid(np.arange(x0_min, x0_max, 0.05), np.arange(x1_min, x1_max, 0.05))
xx = np.vstack([x0_grid.reshape((1, -1)), x1_grid.reshape((1, -1))])
a = model(xx).reshape(x0_grid.shape)
plt.contour(x0_grid, x1_grid, a, [0.5], colors=('orange',))
plt.legend()
plt.show()


