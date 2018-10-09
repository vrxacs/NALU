import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
from scipy.stats import logistic
from matplotlib import cm

def weight_function(x, y):
	return np.tanh(x) * logistic.cdf(y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-10.0, 10.0, 0.1)
X, Y = np.meshgrid(x, y)
zs = np.array([weight_function(x, y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z, cmap=cm.copper_r)

ax.set_xlabel('tanh')
ax.set_ylabel('sigmoid')

plt.show()