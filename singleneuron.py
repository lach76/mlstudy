import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

class SingleNeuron(object):
    def __init__(self):
        self._w = 0
        self._b = 0
        self._w_grad = 0
        self._b_grad = 0
        self._x = 0

    def set_params(self, w, b):
        self._w = w
        self._b = b

    def forpass(self, x):
        self._x = x
        _y_hat = self._w * self._x + self._b
        return _y_hat

    def backprop(self, err):
        m = len(self._x)
        self._w_grad = 0.1 * np.sum(err * self._x) / m
        self._b_grad = 0.1 * np.sum(err * 1) / m

    def update_grad(self):
        self.set_params(self._w + self._w_grad, self._b + self._b_grad)

diabetes = datasets.load_diabetes()
print(diabetes.data.shape, diabetes.target.shape)
print(diabetes.target[:10])
print(diabetes.data[:5])

n1 = SingleNeuron()
n1.set_params(5, 1)
for i in range(30000):
    y_hat = n1.forpass(diabetes.data[:, 2])
    error = diabetes.target - y_hat
    n1.backprop(error)
    n1.update_grad()

print('Final W', n1._w)
print('Final b', n1._b)
