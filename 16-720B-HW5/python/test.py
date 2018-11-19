import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

fig = plt.figure()
ax = fig.add_subplot(111)
# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

x = np.arange(-5, 5, 0.05)
h1 = ax.plot(x, sigmoid(x), label='sigmoid(x)', color='g')
h2 = ax.plot(x, sigmoid(x)*(1-sigmoid(x)), label='d/dx sigmoid(x)', color='b')
h3 = ax.plot(x, tanh(x), label='tanh(x)', color='r')
h4 = ax.plot(x, 1-tanh(x)**2, label='d/dx tanh(x)', color='y')
ax.legend(handles=h1+h2+h3+h4)
plt.show()