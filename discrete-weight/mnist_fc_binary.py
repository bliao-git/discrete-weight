
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__) # 1.4.1
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from utils import load_mnist_data
x_train, y_train, x_validate, y_validate, x_test, y_test = load_mnist_data()


# In[3]:


import network, train, utils
from layers import ReluLayer, BinaryFullyConnectedLayer, BatchNormLayer


# In[4]:


nn = network.NeuralNetwork(in_size=[None, 784], n_out_classes=10,
                           loss_func=utils.smooth_hinge_loss)

nn.reset_graph()

# Hidden FC-1
nn.add_layer(BinaryFullyConnectedLayer(out_dim=2048))
nn.add_layer(BatchNormLayer(axes=[0]))
nn.add_layer(ReluLayer())

# Hidden FC-2
nn.add_layer(BinaryFullyConnectedLayer(out_dim=2048))
nn.add_layer(BatchNormLayer(axes=[0]))
nn.add_layer(ReluLayer())

# Hidden FC-3
nn.add_layer(BinaryFullyConnectedLayer(out_dim=2048))
nn.add_layer(BatchNormLayer(axes=[0]))
nn.add_layer(ReluLayer())

# Output SVM layer (linear part)
nn.add_layer(BinaryFullyConnectedLayer(out_dim=10))
nn.add_layer(BatchNormLayer(axes=[0]))

nn.finalize()


# # Optimizer

# In[5]:


data_train = (x_train, y_train)
opt = train.Trainer(nn, data_train)


# In[6]:


opt.set_rho(0.5)
opt.set_ema_rates(0.999)


# In[7]:


losses_and_accs_train = []
losses_and_accs_valid = []
losses_and_accs_test = []

n_epochs = 250

for t in range(n_epochs):
    print('Epoch: ', t)
    opt.train_epoch(batch_size=100, ema_decay=0.95, n_output=10, verbose=True)
    
    losses_and_accs_train.append(
        opt.loss_and_accuracy((x_train, y_train), max_batch=400, inference=True))
    losses_and_accs_test.append(
        opt.loss_and_accuracy((x_test, y_test), max_batch=400, inference=True))
    losses_and_accs_valid.append(
        opt.loss_and_accuracy((x_validate, y_validate), max_batch=400, inference=True))
    
    print('Train loss/acc: ', losses_and_accs_train[-1],
          'Test loss/acc: ', losses_and_accs_test[-1])

losses_and_accs_train = np.asarray(losses_and_accs_train)
losses_and_accs_valid = np.asarray(losses_and_accs_valid)
losses_and_accs_test = np.asarray(losses_and_accs_test)


# In[11]:


print('Train: ', opt.loss_and_accuracy((x_train, y_train), inference=True))
print('Valid: ', opt.loss_and_accuracy((x_validate, y_validate), inference=True))
print('Test: ', opt.loss_and_accuracy((x_test, y_test), inference=True))


# In[12]:


best_epoch = np.argmax(losses_and_accs_valid[:,1]) + 1
print('Best epoch: ', best_epoch)
print('Train acc: ', losses_and_accs_train[best_epoch-1, 1])
print('Valid acc: ', losses_and_accs_valid[best_epoch-1, 1])
print('Test acc: ', losses_and_accs_test[best_epoch-1, 1])


# In[13]:


losses_and_accs = np.concatenate(
    [np.asarray(losses_and_accs_train),
     np.asarray(losses_and_accs_valid),
     np.asarray(losses_and_accs_test)], axis=1)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
ax1.semilogy(losses_and_accs[:,0], '-o', label='Train loss')
ax1.semilogy(losses_and_accs[:,2], '-o', label='Valid loss')
ax1.semilogy(losses_and_accs[:,4], '-o', label='Test loss')

ax2.plot(losses_and_accs[:,1], '-o', label='Train acc')
ax2.plot(losses_and_accs[:,3], '-o', label='Valid acc')
ax2.plot(losses_and_accs[:,5], '-o', label='Test acc')

for ax in [ax1,ax2]:
    ax.legend()

ax2.set_ylim(0.1,1)
    
print('Final results: ', losses_and_accs[-1])

