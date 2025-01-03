import nn
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)
layer1 = nn.Linear(1, 1)
layer2 = nn.Sigmoid()
criterion = nn.MSELoss()

x = np.random.rand(1, 1000)
y = -np.log(1/x-1)+np.sqrt(0.01)*np.random.randn(1, 1000)

num_epochs = 1000
loss_vec = []

# plt.figure(1)
# plt.plot(y[0], x[0], 'x', label='Pilots (Channel Input)')
# bottom, top = plt.ylim()
init_output = layer2(layer1(np.sort(y)))[0]
# plt.plot(np.sort(y)[0], init_output, label='Initial Weights')
# plt.xlabel('Channel Output')
# plt.ylabel('Equalizer Output')
# plt.legend()
# plt.ylim(bottom, top)
# plt.title('Equalizer Input v. Output (before training)')
# plt.grid()
# plt.show()

for epoch in range(num_epochs):

    output = layer2(layer1(y))

    # --------------------------------------------------------------------------------------------------

    # loss = loss[0, 0] if len(loss.shape) > 1 else loss
    loss_vec.append(criterion(x, output)) # loss = criterion(x, output)

    grad_a = layer2.gradient(criterion.gradient())
    grad_a = layer1.gradient(grad_a)
    layer1.update_weights(0.2)
    
    # --------------------------------------------------------------------------------------------------


plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(y[0], x[0], 'x', label='Pilots (Channel Input)')
bottom, top = plt.ylim()
plt.plot(np.sort(y)[0], init_output, label='Initial Weights')
plt.plot(np.sort(y)[0], layer2(layer1(np.sort(y)))[0], label='Trained Weights')
plt.xlabel('Channel Output')
plt.ylabel('Equalizer Output')
plt.legend()
plt.ylim(bottom, top)
plt.title('Equalizer Input v. Output (after training)')
plt.grid()


plt.subplot(1, 2, 2)
plt.plot(loss_vec, label = 'Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.title('Loss Over Epoch')
plt.grid()

plt.show()
