import nn
import numpy as np
from matplotlib import pyplot as plt


# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x, self)
        x = self.sigmoid1(x, self)
        x = self.fc2(x, self)
        x = self.sigmoid2(x, self)
        x = self.fc3(x, self)
        return x


# -- Parameters --
num_epochs = 10000  # Number of epochs for training
beta = 0.03  # Learning rate
lbd = 0.002 # Weight penalty
N_training = 20  # Number of samples for training

input_size = 1
output_size = 1
model = SimpleNN(input_size, 15, 20, output_size)

np.random.seed(42)
x = np.random.randn(1, N_training)
y = 3*x+4+np.sqrt(0.9)*np.random.randn(1, N_training)

criterion = nn.MSELoss()

loss_vec = []
for epoch in range(num_epochs):
    outputs = model(y)

    loss = criterion(x, outputs, model)
    loss_vec.append(loss)

    model.zero_grad()
    model.backward()
    model.step(beta, lbd)

plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(y[0], x[0], 'x', label='Pilots (Channel Input)')
plt.plot(np.sort(y)[0], model(np.sort(y))[0], label='Trained Weights')
plt.plot(3*np.sort(x)[0]+4, np.sort(x)[0], 'black', label='Ground truth (unknown)')
bottom, top = plt.ylim()
plt.xlabel('Channel Output')
plt.ylabel('Equalizer Output')
plt.legend()
plt.ylim(bottom, top)
plt.title('Equalizer Input v. Output (after training)')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(loss_vec, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.title('Loss Over Epoch')
plt.grid()

plt.show()
