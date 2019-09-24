

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__) # 1.4.1
%matplotlib inline
```

    1.4.0



```python
from utils import load_mnist_data
x_train, y_train, x_validate, y_validate, x_test, y_test = load_mnist_data()
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz


    
```python
import network, train, utils
from layers import ReluLayer, BinaryFullyConnectedLayer, BatchNormLayer
```


```python
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
```

# Optimizer


```python
data_train = (x_train, y_train)
opt = train.Trainer(nn, data_train)
```


```python
opt.set_rho(0.5)
opt.set_ema_rates(0.999)
```


```python
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
```

    Epoch:  0
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99900001, 0.99900001, 0.99900001, 0.99900001]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.911945, 0.47
    Iter: 55 of 550 || Estimated train loss/acc: 0.755030, 0.55
    Iter: 110 of 550 || Estimated train loss/acc: 0.657861, 0.65
    Iter: 165 of 550 || Estimated train loss/acc: 0.567390, 0.70
    Iter: 220 of 550 || Estimated train loss/acc: 0.509903, 0.78
    Iter: 275 of 550 || Estimated train loss/acc: 0.413838, 0.88
    Iter: 330 of 550 || Estimated train loss/acc: 0.403023, 0.80
    Iter: 385 of 550 || Estimated train loss/acc: 0.343300, 0.86
    Iter: 440 of 550 || Estimated train loss/acc: 0.309755, 0.92
    Iter: 495 of 550 || Estimated train loss/acc: 0.233674, 0.97
    Train loss/acc:  (0.21959417028860612, 0.95318181037902827) Test loss/acc:  (0.22122448980808257, 0.95200000286102293)
    Epoch:  1
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99905002, 0.99905002, 0.99905002, 0.99905002]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.201606, 0.95
    Iter: 55 of 550 || Estimated train loss/acc: 0.171083, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.138359, 0.98
    Iter: 165 of 550 || Estimated train loss/acc: 0.172491, 0.96
    Iter: 220 of 550 || Estimated train loss/acc: 0.190417, 0.92
    Iter: 275 of 550 || Estimated train loss/acc: 0.147556, 0.96
    Iter: 330 of 550 || Estimated train loss/acc: 0.113190, 0.97
    Iter: 385 of 550 || Estimated train loss/acc: 0.101167, 0.98
    Iter: 440 of 550 || Estimated train loss/acc: 0.090006, 0.97
    Iter: 495 of 550 || Estimated train loss/acc: 0.092897, 0.98
    Train loss/acc:  (0.055756806920875202, 0.97276364673267712) Test loss/acc:  (0.062099439799785612, 0.96479999542236328)
    Epoch:  2
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99909753, 0.99909753, 0.99909753, 0.99909753]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.057969, 0.98
    Iter: 55 of 550 || Estimated train loss/acc: 0.059813, 0.97
    Iter: 110 of 550 || Estimated train loss/acc: 0.058974, 0.97
    Iter: 165 of 550 || Estimated train loss/acc: 0.037885, 0.98
    Iter: 220 of 550 || Estimated train loss/acc: 0.028064, 0.99
    Iter: 275 of 550 || Estimated train loss/acc: 0.058828, 0.94
    Iter: 330 of 550 || Estimated train loss/acc: 0.075775, 0.93
    Iter: 385 of 550 || Estimated train loss/acc: 0.055172, 0.98
    Iter: 440 of 550 || Estimated train loss/acc: 0.018564, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.021440, 0.99
    Train loss/acc:  (0.021088304126804524, 0.97963637612082743) Test loss/acc:  (0.027513275519013405, 0.96989999771118163)
    Epoch:  3
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99914265, 0.99914265, 0.99914265, 0.99914265]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.014801, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.023518, 0.99
    Iter: 110 of 550 || Estimated train loss/acc: 0.037891, 0.98
    Iter: 165 of 550 || Estimated train loss/acc: 0.021641, 0.99
    Iter: 220 of 550 || Estimated train loss/acc: 0.029676, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.038901, 0.96
    Iter: 330 of 550 || Estimated train loss/acc: 0.064996, 0.96
    Iter: 385 of 550 || Estimated train loss/acc: 0.027413, 0.97
    Iter: 440 of 550 || Estimated train loss/acc: 0.014045, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.015474, 0.99
    Train loss/acc:  (0.024984702027656815, 0.98321819478815253) Test loss/acc:  (0.030645575225353241, 0.97630000352859492)
    Epoch:  4
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9991855, 0.9991855, 0.9991855, 0.9991855]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.025405, 0.97
    Iter: 55 of 550 || Estimated train loss/acc: 0.147129, 0.94
    Iter: 110 of 550 || Estimated train loss/acc: 0.014044, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.009423, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.019476, 0.98
    Iter: 275 of 550 || Estimated train loss/acc: 0.005164, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.007904, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.019390, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.033187, 0.97
    Iter: 495 of 550 || Estimated train loss/acc: 0.018417, 0.99
    Train loss/acc:  (0.017213639525527305, 0.98032728715376416) Test loss/acc:  (0.022928962986916303, 0.97289999961853024)
    Epoch:  5
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99922621, 0.99922621, 0.99922621, 0.99922621]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.017266, 0.98
    Iter: 55 of 550 || Estimated train loss/acc: 0.003514, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.010203, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.001922, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.004339, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.013119, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.012338, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.014905, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.016133, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.009950, 0.99
    Train loss/acc:  (0.013236556596715342, 0.98369092377749356) Test loss/acc:  (0.020272168051451444, 0.97360000371932987)
    Epoch:  6
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9992649, 0.9992649, 0.9992649, 0.9992649]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.009951, 0.98
    Iter: 55 of 550 || Estimated train loss/acc: 0.009252, 0.99
    Iter: 110 of 550 || Estimated train loss/acc: 0.016633, 0.99
    Iter: 165 of 550 || Estimated train loss/acc: 0.027358, 0.95
    Iter: 220 of 550 || Estimated train loss/acc: 0.045113, 0.94
    Iter: 275 of 550 || Estimated train loss/acc: 0.011536, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.013000, 0.98
    Iter: 385 of 550 || Estimated train loss/acc: 0.016109, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.022790, 0.97
    Iter: 495 of 550 || Estimated train loss/acc: 0.006541, 1.00
    Train loss/acc:  (0.014158240844580261, 0.98474546735936941) Test loss/acc:  (0.022350023183971644, 0.97279999494552616)
    Epoch:  7
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99930167, 0.99930167, 0.99930167, 0.99930167]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.017199, 0.98
    Iter: 55 of 550 || Estimated train loss/acc: 0.008291, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.013216, 0.98
    Iter: 165 of 550 || Estimated train loss/acc: 0.019141, 0.99
    Iter: 220 of 550 || Estimated train loss/acc: 0.007034, 0.99
    Iter: 275 of 550 || Estimated train loss/acc: 0.010614, 0.98
    Iter: 330 of 550 || Estimated train loss/acc: 0.005373, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.034667, 0.95
    Iter: 440 of 550 || Estimated train loss/acc: 0.009949, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.008822, 0.99
    Train loss/acc:  (0.021470717442306605, 0.98296365044333722) Test loss/acc:  (0.028241724185645582, 0.97330001115798948)
    Epoch:  8
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9993366, 0.9993366, 0.9993366, 0.9993366]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.008092, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.003486, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.003596, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.016591, 0.99
    Iter: 220 of 550 || Estimated train loss/acc: 0.030925, 0.97
    Iter: 275 of 550 || Estimated train loss/acc: 0.003640, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.008664, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.009063, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.022190, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.005870, 1.00
    Train loss/acc:  (0.010068809914995324, 0.98989091873168944) Test loss/acc:  (0.01718910062685609, 0.97780000209808349)
    Epoch:  9
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99936974, 0.99936974, 0.99936974, 0.99936974]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.003684, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.012943, 0.99
    Iter: 110 of 550 || Estimated train loss/acc: 0.007193, 0.99
    Iter: 165 of 550 || Estimated train loss/acc: 0.002828, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.007984, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000738, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.009482, 0.99
    Iter: 385 of 550 || Estimated train loss/acc: 0.005101, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.001568, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.012453, 0.98
    Train loss/acc:  (0.0080405182408338251, 0.99154546260833742) Test loss/acc:  (0.01633010513149202, 0.97810000896453853)
    Epoch:  10
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99940127, 0.99940127, 0.99940127, 0.99940127]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.004393, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.002777, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.002188, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000617, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.004359, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.004530, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.004249, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.001730, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.003113, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.006564, 0.99
    Train loss/acc:  (0.008647616151720286, 0.99121819019317625) Test loss/acc:  (0.018116874713450672, 0.97880000352859498)
    Epoch:  11
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99943119, 0.99943119, 0.99943119, 0.99943119]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.002549, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.016430, 0.99
    Iter: 110 of 550 || Estimated train loss/acc: 0.010799, 0.98
    Iter: 165 of 550 || Estimated train loss/acc: 0.014479, 0.99
    Iter: 220 of 550 || Estimated train loss/acc: 0.002724, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.004946, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.024389, 0.97
    Iter: 385 of 550 || Estimated train loss/acc: 0.005924, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.007331, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.004555, 1.00
    Train loss/acc:  (0.011735386186364022, 0.98565455870194874) Test loss/acc:  (0.020244639348238707, 0.97209999799728397)
    Epoch:  12
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99945962, 0.99945962, 0.99945962, 0.99945962]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.018257, 0.98
    Iter: 55 of 550 || Estimated train loss/acc: 0.033999, 0.93
    Iter: 110 of 550 || Estimated train loss/acc: 0.007618, 0.99
    Iter: 165 of 550 || Estimated train loss/acc: 0.019932, 0.96
    Iter: 220 of 550 || Estimated train loss/acc: 0.012297, 0.98
    Iter: 275 of 550 || Estimated train loss/acc: 0.003067, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.016801, 0.97
    Iter: 385 of 550 || Estimated train loss/acc: 0.006619, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.004651, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.003170, 1.00
    Train loss/acc:  (0.0065556557798250158, 0.99256364345550541) Test loss/acc:  (0.014673550035804511, 0.98150001049041746)
    Epoch:  13
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99948663, 0.99948663, 0.99948663, 0.99948663]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.004679, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.002011, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.002796, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.033842, 0.96
    Iter: 220 of 550 || Estimated train loss/acc: 0.017767, 0.97
    Iter: 275 of 550 || Estimated train loss/acc: 0.008251, 0.99
    Iter: 330 of 550 || Estimated train loss/acc: 0.004303, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.006827, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.006838, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.005311, 1.00
    Train loss/acc:  (0.0099804181453179222, 0.9908000087738037) Test loss/acc:  (0.018553729858249424, 0.97770000457763673)
    Epoch:  14
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99951231, 0.99951231, 0.99951231, 0.99951231]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.010235, 0.99
    Iter: 55 of 550 || Estimated train loss/acc: 0.003619, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.003482, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000827, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.002754, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.002017, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.012426, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.024620, 0.96
    Iter: 440 of 550 || Estimated train loss/acc: 0.003286, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.005326, 0.99
    Train loss/acc:  (0.005837950979105451, 0.99336364269256594) Test loss/acc:  (0.013695523594506084, 0.982300009727478)
    Epoch:  15
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99953669, 0.99953669, 0.99953669, 0.99953669]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000770, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.003503, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.001218, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000890, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.007842, 0.99
    Iter: 275 of 550 || Estimated train loss/acc: 0.007238, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.002228, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.001563, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.002395, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.006979, 0.99
    Train loss/acc:  (0.0081325466702268888, 0.98958182768388225) Test loss/acc:  (0.017441397844813765, 0.97520000696182252)
    Epoch:  16
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99955988, 0.99955988, 0.99955988, 0.99955988]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.001226, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.007569, 0.99
    Iter: 110 of 550 || Estimated train loss/acc: 0.002799, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000259, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.018318, 0.98
    Iter: 275 of 550 || Estimated train loss/acc: 0.005960, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.017837, 0.98
    Iter: 385 of 550 || Estimated train loss/acc: 0.009090, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.008695, 0.98
    Iter: 495 of 550 || Estimated train loss/acc: 0.001898, 1.00
    Train loss/acc:  (0.0052993664238601924, 0.99445455074310307) Test loss/acc:  (0.015692401262931525, 0.97890000820159917)
    Epoch:  17
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99958187, 0.99958187, 0.99958187, 0.99958187]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.009579, 0.99
    Iter: 55 of 550 || Estimated train loss/acc: 0.001145, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.003490, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.001110, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.014401, 0.99
    Iter: 275 of 550 || Estimated train loss/acc: 0.012229, 0.98
    Iter: 330 of 550 || Estimated train loss/acc: 0.008592, 0.98
    Iter: 385 of 550 || Estimated train loss/acc: 0.007235, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000302, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.011254, 0.97
    Train loss/acc:  (0.0053370300528000703, 0.99452727794647222) Test loss/acc:  (0.014980534398928285, 0.97900000333786008)
    Epoch:  18
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99960279, 0.99960279, 0.99960279, 0.99960279]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.012432, 0.98
    Iter: 55 of 550 || Estimated train loss/acc: 0.001272, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.002253, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.005820, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.006976, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.016775, 0.98
    Iter: 330 of 550 || Estimated train loss/acc: 0.012388, 0.99
    Iter: 385 of 550 || Estimated train loss/acc: 0.009414, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.007919, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.011769, 0.98
    Train loss/acc:  (0.0043607576995749369, 0.99505455017089839) Test loss/acc:  (0.013653411278501153, 0.98160001277923581)
    Epoch:  19
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99962264, 0.99962264, 0.99962264, 0.99962264]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.008136, 0.99
    Iter: 55 of 550 || Estimated train loss/acc: 0.000599, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.008015, 0.99
    Iter: 165 of 550 || Estimated train loss/acc: 0.001234, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000725, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000448, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.007762, 0.98
    Iter: 385 of 550 || Estimated train loss/acc: 0.007591, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.002666, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.005060, 1.00
    Train loss/acc:  (0.0060244872023097491, 0.99267273426055913) Test loss/acc:  (0.017454042006283997, 0.97620000123977657)
    Epoch:  20
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99964154, 0.99964154, 0.99964154, 0.99964154]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.001928, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.003779, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.007142, 0.99
    Iter: 165 of 550 || Estimated train loss/acc: 0.001228, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.005330, 0.99
    Iter: 275 of 550 || Estimated train loss/acc: 0.010504, 0.98
    Iter: 330 of 550 || Estimated train loss/acc: 0.006353, 0.99
    Iter: 385 of 550 || Estimated train loss/acc: 0.000200, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.007709, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.001630, 1.00
    Train loss/acc:  (0.011583831499923359, 0.98758182959123086) Test loss/acc:  (0.023103590849786998, 0.9754999995231628)
    Epoch:  21
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99965948, 0.99965948, 0.99965948, 0.99965948]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.001716, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.004113, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.007557, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.004395, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.001719, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.006559, 0.99
    Iter: 330 of 550 || Estimated train loss/acc: 0.004289, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.005139, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.002869, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.009772, 0.99
    Train loss/acc:  (0.0043326154783029447, 0.99481818675994871) Test loss/acc:  (0.013091333652846515, 0.98230001211166385)
    Epoch:  22
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99967653, 0.99967653, 0.99967653, 0.99967653]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.006497, 0.99
    Iter: 55 of 550 || Estimated train loss/acc: 0.000890, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.001110, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.001093, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.012972, 0.99
    Iter: 275 of 550 || Estimated train loss/acc: 0.000767, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.004052, 0.99
    Iter: 385 of 550 || Estimated train loss/acc: 0.006943, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.000759, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.004671, 1.00
    Train loss/acc:  (0.0040700938412919644, 0.99525454998016361) Test loss/acc:  (0.013961764278355986, 0.98130000829696651)
    Epoch:  23
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99969268, 0.99969268, 0.99969268, 0.99969268]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.001643, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.002046, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000298, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.007586, 0.99
    Iter: 220 of 550 || Estimated train loss/acc: 0.002295, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000729, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.027851, 0.97
    Iter: 385 of 550 || Estimated train loss/acc: 0.005483, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.002833, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.000188, 1.00
    Train loss/acc:  (0.0031644251375374469, 0.99629091262817382) Test loss/acc:  (0.012734872358851134, 0.98350001335144044)
    Epoch:  24
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99970806, 0.99970806, 0.99970806, 0.99970806]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.001825, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000291, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.003284, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000844, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.002971, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.001656, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000300, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.015473, 0.98
    Iter: 440 of 550 || Estimated train loss/acc: 0.002388, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.004698, 1.00
    Train loss/acc:  (0.0047791255714202466, 0.99465455055236818) Test loss/acc:  (0.015873626973479986, 0.97870000600814822)
    Epoch:  25
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99972266, 0.99972266, 0.99972266, 0.99972266]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.004469, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.010765, 0.99
    Iter: 110 of 550 || Estimated train loss/acc: 0.006034, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.003182, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.007017, 0.99
    Iter: 275 of 550 || Estimated train loss/acc: 0.001634, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.005824, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.002307, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000977, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.007161, 0.99
    Train loss/acc:  (0.002590512240051546, 0.99734545707702638) Test loss/acc:  (0.013674850948154927, 0.98270001411437991)
    Epoch:  26
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99973655, 0.99973655, 0.99973655, 0.99973655]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000985, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.001525, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000337, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.003440, 0.99
    Iter: 220 of 550 || Estimated train loss/acc: 0.001760, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.002674, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.002490, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.013369, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.006496, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.001941, 1.00
    Train loss/acc:  (0.0039305292941968547, 0.99552727699279786) Test loss/acc:  (0.013857077443972229, 0.98130001068115236)
    Epoch:  27
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99974972, 0.99974972, 0.99974972, 0.99974972]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.005380, 0.99
    Iter: 55 of 550 || Estimated train loss/acc: 0.001093, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.023553, 0.97
    Iter: 165 of 550 || Estimated train loss/acc: 0.002458, 0.99
    Iter: 220 of 550 || Estimated train loss/acc: 0.000579, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000039, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.016173, 0.97
    Iter: 385 of 550 || Estimated train loss/acc: 0.015228, 0.98
    Iter: 440 of 550 || Estimated train loss/acc: 0.005514, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.005432, 0.99
    Train loss/acc:  (0.0024731323719871315, 0.99725454807281499) Test loss/acc:  (0.013565937881357968, 0.98280001163482666)
    Epoch:  28
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99976224, 0.99976224, 0.99976224, 0.99976224]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000980, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000118, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.003853, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.001165, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.001930, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.003035, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.006728, 0.99
    Iter: 385 of 550 || Estimated train loss/acc: 0.003848, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.007891, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.001073, 1.00
    Train loss/acc:  (0.0016057633134980941, 0.99834545612335202) Test loss/acc:  (0.011234099753201008, 0.98560000896453859)
    Epoch:  29
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9997741, 0.9997741, 0.9997741, 0.9997741]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.005275, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.006800, 0.99
    Iter: 110 of 550 || Estimated train loss/acc: 0.000302, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000510, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.001663, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.002164, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.002277, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.002976, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.005994, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.000342, 1.00
    Train loss/acc:  (0.0016882762106516483, 0.99852727413177489) Test loss/acc:  (0.012443752381950618, 0.98320001125335699)
    Epoch:  30
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99978542, 0.99978542, 0.99978542, 0.99978542]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.002311, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.002904, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.004738, 0.99
    Iter: 165 of 550 || Estimated train loss/acc: 0.002849, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000212, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.001606, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.001438, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000213, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.003667, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000054, 1.00
    Train loss/acc:  (0.0012270642231768845, 0.99883636474609372) Test loss/acc:  (0.011844017170369625, 0.98430001020431523)
    Epoch:  31
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99979615, 0.99979615, 0.99979615, 0.99979615]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000116, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000361, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.004864, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000006, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.002383, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.008194, 0.99
    Iter: 330 of 550 || Estimated train loss/acc: 0.006371, 0.99
    Iter: 385 of 550 || Estimated train loss/acc: 0.000602, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.001180, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000241, 1.00
    Train loss/acc:  (0.0020894545391836965, 0.99783636569976808) Test loss/acc:  (0.01305000969208777, 0.98290001153945927)
    Epoch:  32
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99980634, 0.99980634, 0.99980634, 0.99980634]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.002727, 0.99
    Iter: 55 of 550 || Estimated train loss/acc: 0.005243, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000015, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.004002, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000082, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.001448, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.001951, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000409, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000066, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.004916, 0.99
    Train loss/acc:  (0.0018316392129583453, 0.9978000020980835) Test loss/acc:  (0.012271883729845286, 0.9838000106811523)
    Epoch:  33
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.999816, 0.999816, 0.999816, 0.999816]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.001717, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.007440, 0.99
    Iter: 110 of 550 || Estimated train loss/acc: 0.000652, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000225, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.002619, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.003670, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.003088, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.005726, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.003914, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.005252, 0.99
    Train loss/acc:  (0.0014890264422336423, 0.99856363773345946) Test loss/acc:  (0.011620128480717539, 0.98440001249313358)
    Epoch:  34
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99982518, 0.99982518, 0.99982518, 0.99982518]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000273, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.003170, 0.99
    Iter: 110 of 550 || Estimated train loss/acc: 0.000431, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000557, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000198, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000015, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000517, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.003602, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000384, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000335, 1.00
    Train loss/acc:  (0.0027218072661410338, 0.99658182144165042) Test loss/acc:  (0.014408098924905062, 0.98250000715255736)
    Epoch:  35
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99983394, 0.99983394, 0.99983394, 0.99983394]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000030, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.002312, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000343, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.004952, 0.99
    Iter: 220 of 550 || Estimated train loss/acc: 0.000047, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000106, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.003145, 0.99
    Iter: 385 of 550 || Estimated train loss/acc: 0.001106, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000193, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.001939, 1.00
    Train loss/acc:  (0.0011608771066213112, 0.99869091033935542) Test loss/acc:  (0.011704950039857068, 0.98440001010894773)
    Epoch:  36
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99984223, 0.99984223, 0.99984223, 0.99984223]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000403, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000212, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000605, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.001841, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000116, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000055, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.009005, 0.99
    Iter: 385 of 550 || Estimated train loss/acc: 0.002307, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.002247, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000045, 1.00
    Train loss/acc:  (0.0011335600700112991, 0.99876363754272457) Test loss/acc:  (0.012407982838340104, 0.98370001077651981)
    Epoch:  37
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99985009, 0.99985009, 0.99985009, 0.99985009]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000952, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.001576, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000859, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000064, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000629, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000545, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000370, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.003153, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000013, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.002500, 1.00
    Train loss/acc:  (0.00077239775079016742, 0.99920000076293947) Test loss/acc:  (0.012243734055664391, 0.98400001049041752)
    Epoch:  38
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9998576, 0.9998576, 0.9998576, 0.9998576]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000346, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.002745, 0.99
    Iter: 110 of 550 || Estimated train loss/acc: 0.000461, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000024, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.004230, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.005922, 0.99
    Iter: 330 of 550 || Estimated train loss/acc: 0.000011, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.005562, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000316, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.001450, 1.00
    Train loss/acc:  (0.0015306885421821129, 0.99832727432250978) Test loss/acc:  (0.012991520294453949, 0.98310000658035279)
    Epoch:  39
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9998647, 0.9998647, 0.9998647, 0.9998647]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.007372, 0.99
    Iter: 55 of 550 || Estimated train loss/acc: 0.002078, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000525, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.001128, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000080, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.005715, 0.99
    Iter: 330 of 550 || Estimated train loss/acc: 0.000016, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000984, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.001423, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.010102, 0.99
    Train loss/acc:  (0.00077102700750401713, 0.99912727355957032) Test loss/acc:  (0.010782843078486622, 0.9857000088691712)
    Epoch:  40
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99987149, 0.99987149, 0.99987149, 0.99987149]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.001703, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.001120, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000153, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000489, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000242, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000027, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.005070, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000813, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.001477, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000113, 1.00
    Train loss/acc:  (0.0012031575934749773, 0.99907272815704351) Test loss/acc:  (0.011573909157887101, 0.9849000120162964)
    Epoch:  41
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99987793, 0.99987793, 0.99987793, 0.99987793]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000029, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.003104, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000887, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000835, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.001276, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.002628, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000216, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000033, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000055, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000077, 1.00
    Train loss/acc:  (0.00057259018822417246, 0.99954545497894287) Test loss/acc:  (0.011281418523867614, 0.98480001211166379)
    Epoch:  42
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99988401, 0.99988401, 0.99988401, 0.99988401]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.001055, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000312, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000152, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000034, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000599, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.002081, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.010558, 0.99
    Iter: 385 of 550 || Estimated train loss/acc: 0.002582, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.000412, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.001071, 1.00
    Train loss/acc:  (0.00066523144496241833, 0.99929090976715085) Test loss/acc:  (0.012826594847720117, 0.98430001020431523)
    Epoch:  43
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99988979, 0.99988979, 0.99988979, 0.99988979]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000273, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000990, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.001549, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.002610, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.001037, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000012, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.001131, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.002802, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.001828, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.002906, 1.00
    Train loss/acc:  (0.0014312292644436556, 0.99843636512756351) Test loss/acc:  (0.011919951643794774, 0.98510001182556151)
    Epoch:  44
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99989527, 0.99989527, 0.99989527, 0.99989527]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000670, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000493, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000928, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000033, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000103, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000175, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000785, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000025, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.009873, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.000117, 1.00
    Train loss/acc:  (0.00052872116082157427, 0.99961818218231202) Test loss/acc:  (0.012508343514055014, 0.98470001220703129)
    Epoch:  45
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99990052, 0.99990052, 0.99990052, 0.99990052]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.002542, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000415, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.001804, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.003455, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000344, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.002579, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000008, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000773, 1.00
    Train loss/acc:  (0.00038108477439932772, 0.99972727298736574) Test loss/acc:  (0.011925081044901163, 0.98420001029968263)
    Epoch:  46
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99990547, 0.99990547, 0.99990547, 0.99990547]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000010, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.001189, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.001889, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.001271, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000706, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000016, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000055, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.001267, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.004933, 0.99
    Iter: 495 of 550 || Estimated train loss/acc: 0.000802, 1.00
    Train loss/acc:  (0.00087674262404313847, 0.99894545555114744) Test loss/acc:  (0.012120698363287375, 0.98480000972747805)
    Epoch:  47
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99991018, 0.99991018, 0.99991018, 0.99991018]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000365, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.001803, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.003096, 0.99
    Iter: 165 of 550 || Estimated train loss/acc: 0.000246, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000072, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.006473, 0.99
    Iter: 330 of 550 || Estimated train loss/acc: 0.000061, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000415, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.002085, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000297, 1.00
    Train loss/acc:  (0.00042600262219071863, 0.99952727317810064) Test loss/acc:  (0.012960814471589401, 0.98400001287460326)
    Epoch:  48
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99991465, 0.99991465, 0.99991465, 0.99991465]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000441, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.002791, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.008617, 0.99
    Iter: 165 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.002203, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000571, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000089, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000195, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.00037064700933637015, 0.99963636398315425) Test loss/acc:  (0.011035626312950625, 0.98640001058578486)
    Epoch:  49
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99991894, 0.99991894, 0.99991894, 0.99991894]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.001030, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000003, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000049, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000239, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.003461, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.001965, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000035, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.003295, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000083, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000066, 1.00
    Train loss/acc:  (0.0007975596157749268, 0.99934545516967777) Test loss/acc:  (0.013159294473007321, 0.98520001411437985)
    Epoch:  50
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99992299, 0.99992299, 0.99992299, 0.99992299]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000221, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000796, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000019, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000408, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.003296, 0.99
    Iter: 275 of 550 || Estimated train loss/acc: 0.000019, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000697, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.007189, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.018255, 0.98
    Iter: 495 of 550 || Estimated train loss/acc: 0.000987, 1.00
    Train loss/acc:  (0.00054511256327652356, 0.99940000057220457) Test loss/acc:  (0.011570891859009861, 0.98520001173019411)
    Epoch:  51
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99992687, 0.99992687, 0.99992687, 0.99992687]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000046, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000018, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000508, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.015139, 0.97
    Iter: 275 of 550 || Estimated train loss/acc: 0.007634, 0.99
    Iter: 330 of 550 || Estimated train loss/acc: 0.000404, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000120, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000076, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000045, 1.00
    Train loss/acc:  (0.0001550045025857633, 0.99989090919494628) Test loss/acc:  (0.011086334954015911, 0.9860000085830688)
    Epoch:  52
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999305, 0.9999305, 0.9999305, 0.9999305]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000258, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000027, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.001216, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000483, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.003416, 0.99
    Iter: 275 of 550 || Estimated train loss/acc: 0.000454, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000045, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000005, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.002600, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.009425, 0.99
    Train loss/acc:  (0.00037389595388803671, 0.99974545478820798) Test loss/acc:  (0.012927538801450282, 0.98560001134872433)
    Epoch:  53
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99993396, 0.99993396, 0.99993396, 0.99993396]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000092, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.001570, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000148, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000491, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000045, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000531, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000135, 1.00
    Train loss/acc:  (0.00024438353802720897, 0.99978181838989255) Test loss/acc:  (0.011408595119137317, 0.9863000130653381)
    Epoch:  54
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99993724, 0.99993724, 0.99993724, 0.99993724]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000068, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000848, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000034, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000071, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000154, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000782, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.001380, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.001154557398668575, 0.99938181877136234) Test loss/acc:  (0.01399712534621358, 0.9838000106811523)
    Epoch:  55
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999404, 0.9999404, 0.9999404, 0.9999404]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.008628, 0.99
    Iter: 55 of 550 || Estimated train loss/acc: 0.000108, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.006627, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.003664, 0.99
    Iter: 220 of 550 || Estimated train loss/acc: 0.000208, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000016, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000496, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000290, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000004, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000237, 1.00
    Train loss/acc:  (0.00030365150867741323, 0.99980000019073489) Test loss/acc:  (0.011811075091827661, 0.98590001344680789)
    Epoch:  56
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99994338, 0.99994338, 0.99994338, 0.99994338]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000092, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.001326, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000698, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000248, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000254, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.001218, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000027, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000951, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000218, 1.00
    Train loss/acc:  (0.00013368433115968126, 0.99987272739410404) Test loss/acc:  (0.0100351877277717, 0.98750001192092896)
    Epoch:  57
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99994624, 0.99994624, 0.99994624, 0.99994624]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000005, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000242, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000521, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000194, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000063, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000339, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000224, 1.00
    Train loss/acc:  (0.00011113579712489173, 0.99987272739410404) Test loss/acc:  (0.010839169319951906, 0.9871000099182129)
    Epoch:  58
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99994892, 0.99994892, 0.99994892, 0.99994892]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000004, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000016, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000149, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000015, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000270, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (3.9502378394879722e-05, 0.99998181819915777) Test loss/acc:  (0.010379838612279854, 0.98830000877380375)
    Epoch:  59
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99995148, 0.99995148, 0.99995148, 0.99995148]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000003, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000188, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000082, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000051, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (7.07589308894476e-05, 0.99992727279663085) Test loss/acc:  (0.011328750140964985, 0.98680001020431518)
    Epoch:  60
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99995393, 0.99995393, 0.99995393, 0.99995393]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000077, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000121, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000153, 1.00
    Train loss/acc:  (8.1896703007980812e-05, 0.99992727279663085) Test loss/acc:  (0.0113037429866381, 0.98760000944137571)
    Epoch:  61
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99995625, 0.99995625, 0.99995625, 0.99995625]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000549, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000061, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000029, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000010, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.002420, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000006, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000281, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.001860, 1.00
    Train loss/acc:  (2.0687965203662692e-05, 1.0) Test loss/acc:  (0.010492468263546471, 0.98780001163482667)
    Epoch:  62
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99995846, 0.99995846, 0.99995846, 0.99995846]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000055, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000173, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000057, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000549, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.001263, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000297, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (8.848681373977388e-05, 0.99990909099578862) Test loss/acc:  (0.010972618005944242, 0.9871000099182129)
    Epoch:  63
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99996054, 0.99996054, 0.99996054, 0.99996054]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.002588, 0.99
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000224, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000050, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000327, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000031, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000050, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (2.4615579951896359e-05, 0.99998181819915777) Test loss/acc:  (0.010620292754611, 0.98810001134872438)
    Epoch:  64
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99996251, 0.99996251, 0.99996251, 0.99996251]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000272, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.001175, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000152, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.00012973912088537858, 0.99992727279663085) Test loss/acc:  (0.012366285539465026, 0.98730001211166385)
    Epoch:  65
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99996436, 0.99996436, 0.99996436, 0.99996436]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000438, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.002725, 0.99
    Iter: 110 of 550 || Estimated train loss/acc: 0.000004, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000053, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000227, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000026, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000029, 1.00
    Train loss/acc:  (2.0692408240499536e-05, 1.0) Test loss/acc:  (0.011105921956695965, 0.98710001230239863)
    Epoch:  66
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99996614, 0.99996614, 0.99996614, 0.99996614]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000163, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000251, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000084, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000018, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000122, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (8.8789971099588038e-05, 0.99989090919494628) Test loss/acc:  (0.011558122240421654, 0.98640001296997071)
    Epoch:  67
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99996781, 0.99996781, 0.99996781, 0.99996781]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000094, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000012, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Train loss/acc:  (4.2660778007205719e-05, 0.99994545459747319) Test loss/acc:  (0.010986463834706228, 0.9871000099182129)
    Epoch:  68
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99996942, 0.99996942, 0.99996942, 0.99996942]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000371, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000099, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000245, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000036, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000044, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000316, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000094, 1.00
    Train loss/acc:  (2.1156003622722479e-05, 0.99998181819915777) Test loss/acc:  (0.010157760528381915, 0.98780001163482667)
    Epoch:  69
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99997097, 0.99997097, 0.99997097, 0.99997097]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000003, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000029, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000012, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000020, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000005, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (4.7924739981376371e-06, 1.0) Test loss/acc:  (0.010504422364756466, 0.98760001182556156)
    Epoch:  70
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999724, 0.9999724, 0.9999724, 0.9999724]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000005, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000035, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (3.7975105399565025e-06, 1.0) Test loss/acc:  (0.011059504728764295, 0.98760000944137571)
    Epoch:  71
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99997377, 0.99997377, 0.99997377, 0.99997377]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000081, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000039, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (2.3572387762142584e-06, 1.0) Test loss/acc:  (0.011993469389853999, 0.98730001211166385)
    Epoch:  72
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99997509, 0.99997509, 0.99997509, 0.99997509]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000018, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000105, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000092, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (7.4960401186465545e-08, 1.0) Test loss/acc:  (0.010992941427975893, 0.98850001096725459)
    Epoch:  73
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99997634, 0.99997634, 0.99997634, 0.99997634]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000008, 1.00
    Train loss/acc:  (6.0019090390348782e-07, 1.0) Test loss/acc:  (0.01093670808011666, 0.9884000110626221)
    Epoch:  74
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99997753, 0.99997753, 0.99997753, 0.99997753]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000084, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000039, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000007, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000008, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000005, 1.00
    Train loss/acc:  (4.9857119613008004e-08, 1.0) Test loss/acc:  (0.011280695850255143, 0.9886000108718872)
    Epoch:  75
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99997866, 0.99997866, 0.99997866, 0.99997866]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000115, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000044, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000016, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000061, 1.00
    Train loss/acc:  (2.8153798749289392e-08, 1.0) Test loss/acc:  (0.01074959704419598, 0.98850001096725459)
    Epoch:  76
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99997973, 0.99997973, 0.99997973, 0.99997973]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000003, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000056, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000110, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000003, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (4.2919523159732557e-09, 1.0) Test loss/acc:  (0.011295433390987455, 0.98830001115798949)
    Epoch:  77
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99998075, 0.99998075, 0.99998075, 0.99998075]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (2.11736402310304e-08, 1.0) Test loss/acc:  (0.011007333085872232, 0.98800001144409177)
    Epoch:  78
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999817, 0.9999817, 0.9999817, 0.9999817]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000090, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000740, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (4.8708897355192506e-06, 1.0) Test loss/acc:  (0.01245534031848365, 0.98720001220703124)
    Epoch:  79
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999826, 0.9999826, 0.9999826, 0.9999826]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (2.3307498188594103e-06, 1.0) Test loss/acc:  (0.011876166112851933, 0.98760001182556156)
    Epoch:  80
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99998349, 0.99998349, 0.99998349, 0.99998349]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000054, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000007, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (5.4999180254937263e-06, 1.0) Test loss/acc:  (0.011197654935531319, 0.98760001182556156)
    Epoch:  81
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99998432, 0.99998432, 0.99998432, 0.99998432]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000535, 1.00
    Train loss/acc:  (2.0104146401936616e-07, 1.0) Test loss/acc:  (0.01164014648173179, 0.98910001039505002)
    Epoch:  82
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999851, 0.9999851, 0.9999851, 0.9999851]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000005, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000004, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.004560, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000126, 1.00
    Train loss/acc:  (1.4066739822737873e-07, 1.0) Test loss/acc:  (0.011738255778327584, 0.98770001173019406)
    Epoch:  83
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99998581, 0.99998581, 0.99998581, 0.99998581]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000007, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000053, 1.00
    Train loss/acc:  (4.3739352556506426e-07, 1.0) Test loss/acc:  (0.012070542778819799, 0.98770001173019406)
    Epoch:  84
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99998653, 0.99998653, 0.99998653, 0.99998653]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.5338551239355267e-06, 1.0) Test loss/acc:  (0.012551085143350064, 0.98780000925064082)
    Epoch:  85
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99998719, 0.99998719, 0.99998719, 0.99998719]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000088, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.0863788123639428e-09, 1.0) Test loss/acc:  (0.011954073000149492, 0.98790001153945928)
    Epoch:  86
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99998784, 0.99998784, 0.99998784, 0.99998784]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000408, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000008, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.011427123214220956, 0.98870001077651981)
    Epoch:  87
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99998844, 0.99998844, 0.99998844, 0.99998844]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000003, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000126, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.011752545260824263, 0.9886000108718872)
    Epoch:  88
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99998903, 0.99998903, 0.99998903, 0.99998903]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000004, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000118, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000007, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.011733894636854529, 0.9884000110626221)
    Epoch:  89
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99998957, 0.99998957, 0.99998957, 0.99998957]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000164, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.003012, 0.99
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (6.2308761450615505e-08, 1.0) Test loss/acc:  (0.012352065434970427, 0.98830001115798949)
    Epoch:  90
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999011, 0.99999011, 0.99999011, 0.99999011]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000054, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000271, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012264727410802152, 0.98770001173019406)
    Epoch:  91
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999058, 0.99999058, 0.99999058, 0.99999058]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000017, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000236, 1.00
    Train loss/acc:  (5.7250064155699544e-07, 1.0) Test loss/acc:  (0.012140786466188729, 0.98760001182556156)
    Epoch:  92
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999106, 0.99999106, 0.99999106, 0.99999106]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000037, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (8.8596303529894523e-09, 1.0) Test loss/acc:  (0.01245253070956096, 0.98850001096725459)
    Epoch:  93
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999154, 0.99999154, 0.99999154, 0.99999154]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.0126598876924254, 0.98790001153945928)
    Epoch:  94
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999195, 0.99999195, 0.99999195, 0.99999195]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000014, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000182, 1.00
    Train loss/acc:  (0.00023290897828478666, 0.99974545478820798) Test loss/acc:  (0.01391069207340479, 0.9857000088691712)
    Epoch:  95
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999237, 0.99999237, 0.99999237, 0.99999237]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000012, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000546, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000146, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000004, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000059, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (3.6695036711555738e-06, 1.0) Test loss/acc:  (0.011941929315216839, 0.98780001163482667)
    Epoch:  96
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999273, 0.99999273, 0.99999273, 0.99999273]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000125, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000107, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000340, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000011, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000005, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000098, 1.00
    Train loss/acc:  (8.2875484355553069e-07, 1.0) Test loss/acc:  (0.012005377640016378, 0.98740001201629635)
    Epoch:  97
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999309, 0.99999309, 0.99999309, 0.99999309]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000389, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000041, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000581, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000021, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (8.36040617002709e-07, 1.0) Test loss/acc:  (0.011788416868075729, 0.98770001173019406)
    Epoch:  98
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999344, 0.99999344, 0.99999344, 0.99999344]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000004, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000014, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000060, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.3046819062300198e-06, 1.0) Test loss/acc:  (0.011436931490898132, 0.98810001134872438)
    Epoch:  99
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999938, 0.9999938, 0.9999938, 0.9999938]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000008, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000075, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000056, 1.00
    Train loss/acc:  (5.7978518518873237e-07, 1.0) Test loss/acc:  (0.01166791420429945, 0.98850001096725459)
    Epoch:  100
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999941, 0.9999941, 0.9999941, 0.9999941]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000264, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.8345925292867998e-08, 1.0) Test loss/acc:  (0.012112339087761938, 0.98770001173019406)
    Epoch:  101
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999944, 0.9999944, 0.9999944, 0.9999944]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000020, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000007, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (8.7661861801858648e-08, 1.0) Test loss/acc:  (0.011924231350421905, 0.98760001182556156)
    Epoch:  102
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999947, 0.9999947, 0.9999947, 0.9999947]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000058, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000021, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000420, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000015, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (2.5217375557738046e-07, 1.0) Test loss/acc:  (0.012371595615422847, 0.98790001153945928)
    Epoch:  103
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999493, 0.99999493, 0.99999493, 0.99999493]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000082, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (2.0024313760752027e-07, 1.0) Test loss/acc:  (0.011933063140317017, 0.98770001173019406)
    Epoch:  104
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999517, 0.99999517, 0.99999517, 0.99999517]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000073, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000003, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000006, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (5.6363172585737297e-09, 1.0) Test loss/acc:  (0.011730759582133033, 0.9884000110626221)
    Epoch:  105
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999541, 0.99999541, 0.99999541, 0.99999541]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000093, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (2.9770291637925601e-08, 1.0) Test loss/acc:  (0.011972023452117356, 0.98890001058578492)
    Epoch:  106
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999565, 0.99999565, 0.99999565, 0.99999565]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000004, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.7053285329314928e-10, 1.0) Test loss/acc:  (0.011567028798308456, 0.98890001058578492)
    Epoch:  107
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999589, 0.99999589, 0.99999589, 0.99999589]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (2.3069023055458358e-10, 1.0) Test loss/acc:  (0.011452432809455786, 0.98890001058578492)
    Epoch:  108
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999607, 0.99999607, 0.99999607, 0.99999607]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000061, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000131, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000020, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.4834930342741163e-09, 1.0) Test loss/acc:  (0.011547335656359792, 0.98950001001358034)
    Epoch:  109
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999624, 0.99999624, 0.99999624, 0.99999624]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000097, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000003, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000032, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.2787443103108714e-07, 1.0) Test loss/acc:  (0.012050704080611468, 0.98900001049041752)
    Epoch:  110
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999642, 0.99999642, 0.99999642, 0.99999642]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000024, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000030, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012116760057397187, 0.98890001058578492)
    Epoch:  111
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999966, 0.9999966, 0.9999966, 0.9999966]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000003, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (3.844342328110625e-09, 1.0) Test loss/acc:  (0.01159476074622944, 0.98940001010894774)
    Epoch:  112
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999678, 0.99999678, 0.99999678, 0.99999678]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.6403553424672943e-09, 1.0) Test loss/acc:  (0.012039645398035646, 0.98890001058578492)
    Epoch:  113
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999696, 0.99999696, 0.99999696, 0.99999696]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000004, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000011, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000161, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.011982596403977369, 0.98850001096725459)
    Epoch:  114
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999714, 0.99999714, 0.99999714, 0.99999714]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000037, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (2.464692161910617e-07, 1.0) Test loss/acc:  (0.012106774672513154, 0.9884000110626221)
    Epoch:  115
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999726, 0.99999726, 0.99999726, 0.99999726]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012243294983054511, 0.98780001163482667)
    Epoch:  116
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999738, 0.99999738, 0.99999738, 0.99999738]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012484909103659448, 0.98820001125335688)
    Epoch:  117
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999975, 0.9999975, 0.9999975, 0.9999975]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012800600576447323, 0.98750001192092896)
    Epoch:  118
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999762, 0.99999762, 0.99999762, 0.99999762]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000040, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000144, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012436779232057269, 0.9884000110626221)
    Epoch:  119
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999774, 0.99999774, 0.99999774, 0.99999774]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000011, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000065, 1.00
    Train loss/acc:  (8.2678004466949702e-09, 1.0) Test loss/acc:  (0.01252465003868565, 0.9884000110626221)
    Epoch:  120
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999785, 0.99999785, 0.99999785, 0.99999785]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000030, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.7255730530120092e-08, 1.0) Test loss/acc:  (0.012437920302752446, 0.98880000829696657)
    Epoch:  121
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999797, 0.99999797, 0.99999797, 0.99999797]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000005, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012231204709969461, 0.98850001096725459)
    Epoch:  122
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999809, 0.99999809, 0.99999809, 0.99999809]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000049, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000038, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000026, 1.00
    Train loss/acc:  (2.7694337156638292e-07, 1.0) Test loss/acc:  (0.012395337726920842, 0.98870001077651981)
    Epoch:  123
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999821, 0.99999821, 0.99999821, 0.99999821]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012340340908849612, 0.98830001115798949)
    Epoch:  124
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999833, 0.99999833, 0.99999833, 0.99999833]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000083, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012915439154021442, 0.98850001096725459)
    Epoch:  125
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999839, 0.99999839, 0.99999839, 0.99999839]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012871925523504614, 0.9884000110626221)
    Epoch:  126
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999845, 0.99999845, 0.99999845, 0.99999845]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000025, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000027, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000142, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.00015088058705425118, 0.99996363639831543) Test loss/acc:  (0.01510023371432908, 0.98750000953674322)
    Epoch:  127
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999851, 0.99999851, 0.99999851, 0.99999851]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000902, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.6362453359099823e-08, 1.0) Test loss/acc:  (0.013168688458390533, 0.98850001096725459)
    Epoch:  128
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999857, 0.99999857, 0.99999857, 0.99999857]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000006, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000010, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (4.5812660325530239e-08, 1.0) Test loss/acc:  (0.013582405597117032, 0.98870001077651981)
    Epoch:  129
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999863, 0.99999863, 0.99999863, 0.99999863]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (6.699693503006446e-10, 1.0) Test loss/acc:  (0.012822430653719153, 0.98910001039505002)
    Epoch:  130
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999869, 0.99999869, 0.99999869, 0.99999869]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000024, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.1850314184481579e-10, 1.0) Test loss/acc:  (0.012606365553856448, 0.98830001115798949)
    Epoch:  131
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999875, 0.99999875, 0.99999875, 0.99999875]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000023, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012201601847627898, 0.98880001068115231)
    Epoch:  132
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999881, 0.99999881, 0.99999881, 0.99999881]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012555700202983643, 0.98920001029968263)
    Epoch:  133
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999887, 0.99999887, 0.99999887, 0.99999887]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000032, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.001554, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012343623139045121, 0.98920001029968263)
    Epoch:  134
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999893, 0.99999893, 0.99999893, 0.99999893]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000273, 1.00
    Train loss/acc:  (9.8222006768496203e-07, 1.0) Test loss/acc:  (0.0121664876813702, 0.98800000905990604)
    Epoch:  135
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999899, 0.99999899, 0.99999899, 0.99999899]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (7.1330445776270191e-08, 1.0) Test loss/acc:  (0.012473095237764938, 0.98900000810623168)
    Epoch:  136
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999905, 0.99999905, 0.99999905, 0.99999905]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013371344481943197, 0.98890001058578492)
    Epoch:  137
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999911, 0.99999911, 0.99999911, 0.99999911]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000009, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012714260480133816, 0.98890001058578492)
    Epoch:  138
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999917, 0.99999917, 0.99999917, 0.99999917]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000004, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000009, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013021509901791432, 0.98900001049041752)
    Epoch:  139
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999923, 0.99999923, 0.99999923, 0.99999923]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.01340569136547856, 0.98910001039505002)
    Epoch:  140
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999928, 0.99999928, 0.99999928, 0.99999928]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000127, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000018, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012665327855793294, 0.98900001049041752)
    Epoch:  141
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.99999934, 0.99999934, 0.99999934, 0.99999934]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000022, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.01266661117253534, 0.98920001029968263)
    Epoch:  142
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000091, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012425601400673258, 0.98930001020431524)
    Epoch:  143
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000022, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012814316526055336, 0.98900001049041752)
    Epoch:  144
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000037, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013119520447216928, 0.98870001077651981)
    Epoch:  145
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012631174109410495, 0.98870001077651981)
    Epoch:  146
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000045, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012826764075507526, 0.98880001068115231)
    Epoch:  147
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000141, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012956261744445783, 0.98850001096725459)
    Epoch:  148
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013088258104398847, 0.9884000110626221)
    Epoch:  149
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000005, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.01304901408236347, 0.98880001068115231)
    Epoch:  150
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000033, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013011242395550652, 0.98810001134872438)
    Epoch:  151
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000027, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000017, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013349312944264967, 0.98820001125335688)
    Epoch:  152
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013020126400515437, 0.98780001163482667)
    Epoch:  153
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000007, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013178688879124821, 0.98920001029968263)
    Epoch:  154
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000005, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.001178, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.2747543424880674e-12, 1.0) Test loss/acc:  (0.01370969930634601, 0.98890001058578492)
    Epoch:  155
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000048, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.001404, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.01318696452464792, 0.98850001096725459)
    Epoch:  156
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000035, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000003, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013261024505918612, 0.98880001068115231)
    Epoch:  157
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000102, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013731279365674708, 0.98810001134872438)
    Epoch:  158
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.6452156614832347e-09, 1.0) Test loss/acc:  (0.013292643930763006, 0.98830001115798949)
    Epoch:  159
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000353, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012951093353331089, 0.98880001068115231)
    Epoch:  160
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000730, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (2.2063995857024564e-08, 1.0) Test loss/acc:  (0.013125602924264967, 0.98830001115798949)
    Epoch:  161
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000006, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.9896446743091061e-08, 1.0) Test loss/acc:  (0.013075534109957517, 0.98800001144409177)
    Epoch:  162
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000015, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013311932166106998, 0.98870001077651981)
    Epoch:  163
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012994135078042746, 0.98910001039505002)
    Epoch:  164
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013039596348535269, 0.98870001077651981)
    Epoch:  165
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000016, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000013, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013016397680621595, 0.9884000110626221)
    Epoch:  166
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.012994115937035531, 0.98900001049041752)
    Epoch:  167
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.01330056447419338, 0.9884000110626221)
    Epoch:  168
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.01311938174534589, 0.98900001049041752)
    Epoch:  169
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013287704470567405, 0.9886000108718872)
    Epoch:  170
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013827487407252193, 0.98880001068115231)
    Epoch:  171
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000042, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013476304346695542, 0.98900001049041752)
    Epoch:  172
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014167584725655616, 0.98800001144409177)
    Epoch:  173
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000079, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (4.6905293412568096e-08, 1.0) Test loss/acc:  (0.013486339459195733, 0.98930001020431524)
    Epoch:  174
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000168, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (3.1563790385420856e-09, 1.0) Test loss/acc:  (0.013514292002655566, 0.98910001039505002)
    Epoch:  175
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000102, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (9.5085719790817655e-09, 1.0) Test loss/acc:  (0.013328963275998831, 0.98930001020431524)
    Epoch:  176
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000004, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000021, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (3.4046363659647546e-09, 1.0) Test loss/acc:  (0.013292238623835146, 0.98880001068115231)
    Epoch:  177
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000169, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.1654242784557441e-08, 1.0) Test loss/acc:  (0.013595059897297687, 0.98880001068115231)
    Epoch:  178
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013525265553034842, 0.98890001058578492)
    Epoch:  179
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000065, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013802076012943872, 0.98900001049041752)
    Epoch:  180
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000004, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.01336697836406529, 0.98880001068115231)
    Epoch:  181
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000037, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000005, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013330068900249898, 0.98890001058578492)
    Epoch:  182
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000026, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013592992625199259, 0.98900001049041752)
    Epoch:  183
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013825392870930955, 0.98930001020431524)
    Epoch:  184
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013597155828028918, 0.98910001039505002)
    Epoch:  185
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000019, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013712305976887364, 0.98900001049041752)
    Epoch:  186
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000549, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.6375431435087443e-10, 1.0) Test loss/acc:  (0.014408509635686642, 0.98890001058578492)
    Epoch:  187
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013856285489164294, 0.98870001077651981)
    Epoch:  188
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000018, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (2.9456889685455031e-08, 1.0) Test loss/acc:  (0.014500588388182222, 0.98820001125335688)
    Epoch:  189
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014876540252007544, 0.98790001153945928)
    Epoch:  190
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014388929465785622, 0.9884000110626221)
    Epoch:  191
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000170, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013823164887726306, 0.98870001077651981)
    Epoch:  192
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000277, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013950576866045594, 0.98830001115798949)
    Epoch:  193
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000609, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014049122140277178, 0.98820001125335688)
    Epoch:  194
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014972754828631878, 0.98790001153945928)
    Epoch:  195
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (8.3883668677034704e-08, 1.0) Test loss/acc:  (0.014401345396790929, 0.98790001153945928)
    Epoch:  196
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014795748431224639, 0.98830001115798949)
    Epoch:  197
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000141, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000006, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014888077178038656, 0.98800001144409177)
    Epoch:  198
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000077, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013973338978830725, 0.98850001096725459)
    Epoch:  199
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014700054170098155, 0.98870001077651981)
    Epoch:  200
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000016, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014437559342477471, 0.98880001068115231)
    Epoch:  201
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000034, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000105, 1.00
    Train loss/acc:  (2.6703328330768271e-08, 1.0) Test loss/acc:  (0.014686542842537165, 0.98790001153945928)
    Epoch:  202
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014982930143596605, 0.98880000829696657)
    Epoch:  203
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014518056379165501, 0.98890001058578492)
    Epoch:  204
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014420784628018737, 0.9886000108718872)
    Epoch:  205
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000122, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000014, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014071373925544322, 0.98890001058578492)
    Epoch:  206
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013655172341968865, 0.98910001039505002)
    Epoch:  207
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000023, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014434289305936545, 0.98910001039505002)
    Epoch:  208
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014264352126047016, 0.98920001029968263)
    Epoch:  209
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013739179950207472, 0.98880001068115231)
    Epoch:  210
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000047, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (4.2494155588263478e-09, 1.0) Test loss/acc:  (0.014451348884031177, 0.98870001077651981)
    Epoch:  211
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000047, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.01364478810224682, 0.98910001039505002)
    Epoch:  212
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (2.1122447254707697e-08, 1.0) Test loss/acc:  (0.014008996111806482, 0.98970000982284545)
    Epoch:  213
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000017, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013639693660661579, 0.98890001058578492)
    Epoch:  214
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013911664127372205, 0.98940001010894774)
    Epoch:  215
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014060142717789858, 0.98890001058578492)
    Epoch:  216
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000009, 1.00
    Train loss/acc:  (4.2857671835670108e-10, 1.0) Test loss/acc:  (0.014122546084690839, 0.98870001077651981)
    Epoch:  217
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000027, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000003, 1.00
    Train loss/acc:  (2.3610201341481034e-08, 1.0) Test loss/acc:  (0.013766875038854778, 0.98980000972747806)
    Epoch:  218
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000011, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013594473842531443, 0.98960000991821284)
    Epoch:  219
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000009, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014230334274470806, 0.98850001096725459)
    Epoch:  220
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000123, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014426081340643578, 0.9884000110626221)
    Epoch:  221
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014597990079237207, 0.98900001049041752)
    Epoch:  222
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000005, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000003, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013900818792171777, 0.98870001077651981)
    Epoch:  223
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.01405782983172685, 0.98850001096725459)
    Epoch:  224
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.015039328997954727, 0.98790001153945928)
    Epoch:  225
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014474328886599323, 0.98820000886917114)
    Epoch:  226
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014347574063576759, 0.9886000108718872)
    Epoch:  227
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000221, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000001, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014375900746781554, 0.9886000108718872)
    Epoch:  228
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (9.2602559479101641e-10, 1.0) Test loss/acc:  (0.014628416683990508, 0.9884000110626221)
    Epoch:  229
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.015123838297222391, 0.98900001049041752)
    Epoch:  230
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014234763078857213, 0.9884000110626221)
    Epoch:  231
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000007, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014346323357895017, 0.9886000108718872)
    Epoch:  232
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.01453981630736962, 0.9886000108718872)
    Epoch:  233
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014132892314810306, 0.98870001077651981)
    Epoch:  234
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000070, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013509177596461086, 0.98940001010894774)
    Epoch:  235
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013536442084346163, 0.98810001134872438)
    Epoch:  236
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000014, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000006, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014001422225264832, 0.98850001096725459)
    Epoch:  237
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013766511282883585, 0.98880001068115231)
    Epoch:  238
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000308, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000005, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000565, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000002, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013769737053662539, 0.98820001125335688)
    Epoch:  239
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000004, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000085, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (1.744128593889565e-07, 1.0) Test loss/acc:  (0.014110289842355996, 0.98810001134872438)
    Epoch:  240
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (2.9801700624044646e-08, 1.0) Test loss/acc:  (0.013684774318535346, 0.9884000110626221)
    Epoch:  241
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000018, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000005, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (9.6251925159859562e-08, 1.0) Test loss/acc:  (0.013692731101746175, 0.98800001144409177)
    Epoch:  242
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013682023409055546, 0.98880001068115231)
    Epoch:  243
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000022, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000008, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014580316731007769, 0.98810001134872438)
    Epoch:  244
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (7.6463782699482348e-10, 1.0) Test loss/acc:  (0.013826967184431851, 0.9886000108718872)
    Epoch:  245
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014045552238821984, 0.9886000108718872)
    Epoch:  246
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013399644359014928, 0.98840000867843625)
    Epoch:  247
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000433, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000019, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013908767006359995, 0.98850001096725459)
    Epoch:  248
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000008, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.013919577725076964, 0.98820000886917114)
    Epoch:  249
    ========= Begin epoch =========
    batch_size = 100
    EMA rates:
    [0.9999994, 0.9999994, 0.9999994, 0.9999994]
    rho:
    [0.5, 0.5, 0.5, 0.5]
    Iter: 0 of 550 || Estimated train loss/acc: 0.000015, 1.00
    Iter: 55 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 110 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 165 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 220 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 275 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 330 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 385 of 550 || Estimated train loss/acc: 0.000009, 1.00
    Iter: 440 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Iter: 495 of 550 || Estimated train loss/acc: 0.000000, 1.00
    Train loss/acc:  (0.0, 1.0) Test loss/acc:  (0.014516581057105214, 0.98790001153945928)



```python
print('Train: ', opt.loss_and_accuracy((x_train, y_train), inference=True))
print('Valid: ', opt.loss_and_accuracy((x_validate, y_validate), inference=True))
print('Test: ', opt.loss_and_accuracy((x_test, y_test), inference=True))
```

    Train:  (0.0, 1.0)
    Valid:  (0.011915930546820164, 0.99099999666213989)
    Test:  (0.014516581781208515, 0.9879000186920166)



```python
best_epoch = np.argmax(losses_and_accs_valid[:,1]) + 1
print('Best epoch: ', best_epoch)
print('Train acc: ', losses_and_accs_train[best_epoch-1, 1])
print('Valid acc: ', losses_and_accs_valid[best_epoch-1, 1])
print('Test acc: ', losses_and_accs_test[best_epoch-1, 1])
```

    Best epoch:  209
    Train acc:  1.0
    Valid acc:  0.992400007248
    Test acc:  0.9892000103



```python
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
```

    Final results:  [ 0.          1.          0.01191593  0.99100001  0.01451658  0.98790001]



![png](mnist_fc_binary_files/mnist_fc_binary_10_1.png)

