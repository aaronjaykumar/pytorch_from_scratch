import nn
import numpy as np
from matplotlib import pyplot as plt


layer1 = nn.Linear(1, 1)
criterion = nn.MSELoss()

x = np.random.randn(1, 1000)
y = 3*x+4+np.sqrt(0.9)*np.random.randn(1, 1000)

num_epochs = 1000
loss_vec = []
w_vec = []
b_vec = []

# plt.figure(1)
# plt.plot(y[0], x[0], 'x', label='Pilots (Channel Input)')
# bottom, top = plt.ylim()
init_output = layer1(np.sort(y))[0]
# plt.plot(np.sort(y)[0], layer1(np.sort(y))[0], label='Initial Weights')
# plt.xlabel('Channel Output')
# plt.ylabel('Equalizer Output')
# plt.legend()
# plt.ylim(bottom, top)
# plt.title('Equalizer Input v. Output (before training)')
# plt.grid()

for epoch in range(num_epochs):

    output = layer1(y)

    # --------------------------------------------------------------------------------------------------

    # loss = loss[0, 0] if len(loss.shape) > 1 else loss
    loss_vec.append(criterion(x, output)) # loss = criterion(x, output)
    w_vec.append(layer1.w[0, 0])
    b_vec.append(layer1.b[0, 0])

    grad_a = layer1.gradient(criterion.gradient())
    layer1.update_weights(0.06)

    # --------------------------------------------------------------------------------------------------

w_opt = np.cov(y,x)[0,1]*(1-1/x.shape[1])/np.var(y)
b_opt = np.mean(x)-w_opt*np.mean(y)
mse_opt = np.var(x)-np.var(y)*w_opt**2

# --------------------------------------------------------------------------------------------------

plt.figure(1)
plt.subplot(2, 2, 1)
plt.plot(y[0], x[0], 'x', label='Pilots (Channel Input)')
bottom, top = plt.ylim()
plt.plot(np.sort(y)[0], init_output, label='Initial Weights')
plt.plot(np.sort(y)[0], layer1(np.sort(y))[0], label='Trained Weights')
plt.xlabel('Channel Output')
plt.ylabel('Equalizer Output')
plt.legend()
plt.ylim(bottom, top)
plt.title('Equalizer Input v. Output (after training)')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(loss_vec, label = 'Loss')
plt.plot([0, num_epochs-1], [mse_opt, mse_opt], label='MMSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.title('Comparison of Loss and LMMSE')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(w_vec, label='Weight')
plt.plot([0, num_epochs-1], [w_opt, w_opt], label = 'Optimal Weight')
plt.xlabel('Epoch')
plt.ylabel('Weight')
plt.legend()
plt.title('Comparison of Weight and LMMSE Weight')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(b_vec, label='Bias')
plt.plot([0, num_epochs-1], [b_opt, b_opt], label='Optimal Bias')
plt.xlabel('Epoch')
plt.ylabel('Bias')
plt.legend()
plt.title('Comparison of Bias and LMMSE Bias')
plt.grid()

plt.show()
