import nn
import numpy as np
from matplotlib import pyplot as plt

# Number of training epochs
num_epochs = 5000


# Initialise layers
layer1 = nn.Linear(2, 1)
layer2 = nn.Sigmoid()
criterion = nn.BCELoss()

# Get training data
x = np.random.rand(1, 1000)
theta = np.random.rand(1, 1000)*0.5*np.pi
x = np.concatenate([np.sin(theta)*x, np.cos(theta)*x])
label = np.ones((1, 1000))*(np.sqrt(x[0]**2+x[1]**2) > 0.5)
x = x+0.02*np.random.randn(2, 1000)

loss = []

# Plot before training
plt.figure(1)
y0_idx = (label[0, :] == 0).nonzero()[0]
y1_idx = (label[0, :] == 1).nonzero()[0]
plt.scatter(x[0, y0_idx], x[1, y0_idx], marker='x', color='b', label='Actual Label 0')
plt.scatter(x[0, y1_idx], x[1, y1_idx], marker='x', color='r', label='Actual Label 1')
x_plot = np.linspace(0, 0.5, 100)
y_plot = np.sqrt(1/4-x_plot**2)
plt.plot(x_plot, y_plot, label='Ground Truth (unknown)')
x_plot = np.array([0, -layer1.b[0, 0]/layer1.w[0, 0]])
y_plot = -(layer1.b[0, 0]+layer1.w[0, 0]*x_plot)/layer1.w[0, 1]
plt.plot(x_plot, y_plot, label='Initial Decision Boundary')
plt.axis('equal')
plt.title('Classification before training')
plt.legend()
plt.grid()

for epoch in range(num_epochs):

    output = layer2(layer1(x))

    # --------------------------------------------------------------------------------------------------

    # loss = loss[0, 0] if len(loss.shape) > 1 else loss
    loss.append(criterion(label, output)) # loss = criterion(x, output)

    grad_a = layer2.gradient(criterion.gradient())
    grad_a = layer1.gradient(grad_a)
    layer1.update_weights(0.1)

    # --------------------------------------------------------------------------------------------------

# Plot after training
plt.figure(2)
plt.subplot(1, 2, 1)
y0_idx = (label[0, :] == 0).nonzero()[0]
y1_idx = (label[0, :] == 1).nonzero()[0]
plt.scatter(x[0, y0_idx], x[1, y0_idx], marker='x', color='b', label='Actual Label 0')
plt.scatter(x[0, y1_idx], x[1, y1_idx], marker='x', color='r', label='Actual Label 1')
x_plot = np.linspace(0, 0.5, 100)
y_plot = np.sqrt(1/4-x_plot**2)
plt.plot(x_plot, y_plot, label='Ground Truth')
x_plot = np.array([0, -layer1.b[0, 0]/layer1.w[0, 0]])
y_plot = -(layer1.b[0, 0]+layer1.w[0, 0]*x_plot)/layer1.w[0, 1]
plt.plot(x_plot, y_plot, label='Initial Decision Boundary')
x_plot = np.array([0, -layer1.b[0, 0]/layer1.w[0, 0]])
y_plot = -(layer1.b[0, 0]+layer1.w[0, 0]*x_plot)/layer1.w[0, 1]
plt.plot(x_plot, y_plot, label='Learned Decision Boundary', color='g')
plt.axis('equal')
plt.title('Classification after training')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
y0_idx = (np.round(output[0, :]) == 0).nonzero()[0]
y1_idx = (np.round(output[0, :]) == 1).nonzero()[0]
plt.scatter(x[0, y0_idx], x[1, y0_idx], marker='o', color='b', label='Label 0')
plt.scatter(x[0, y1_idx], x[1, y1_idx], marker='o', color='r', label='Label 1')
plt.plot(x_plot, y_plot, label='Learned Decision Boundary', color='g')
plt.axis('equal')
plt.legend()
plt.grid()

plt.show()
print(layer1.w)
print(layer1.b)