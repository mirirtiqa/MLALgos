import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid

x = np.array([
      [200.0,17.0],
      [123.5,12.0],
      [223.0,15.0]
])
y = np.array([1,0,1])

#forward prop

def sigmoid(z):
  return 1/(1+np.exp(-z))

def dense(a_in,W,b,f):
  units = W.shape[1]
  a_out = np.zeros(units)
  for j in range(units):
    w = W[:,j]
    z = np.dot(a_in,w) + b[j]
    a_out[j] = f(z)

  return a_out


a_in = np.array([1.0,1.0])
W = np.array([
    [1,2,3],
    [2,3,2]
])
b = np.array([1,2,3])
def identity(val):
  return val
print(dense(a_in,W,b,identity))

def my_sequential(x,W1,b1,W2,b2,f):
  a1 = dense(x,W1,b1,f)
  a2 = dense(a1,W2,b2,f)
  return a2

def predict(X,W1,b1,W2,b2,f):
  m = X.shape[0]
  p = np.zeros((m,1))
  for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2,f)
  return p



W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

norm_l = tf.keras.layers.Normalization(axis=-1)

X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]]) 
norm_l.adapt(X_tst) 

X_tstn = norm_l(X_tst)  # remember to normalize
predictions = predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp,sigmoid)

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
  if predictions[i]>=0.5:
    yhat[i] = 1
  else:
    yhat[i] = 0

print(yhat)


#vectorised implementation of dense func and sigmoid
def vecDense(A_in,W,B,G):
  Z = np.matmul(A_in,W) + B
  A_out = G(Z)
  return A_out


#training with tensorflow

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


model = Sequential([
    Dense(units=25,activation='sigmoid'),
    Dense(units=15,activation='sigmoid'),
    Dense(units=1,activation='sigmoid'),
])

from tensorflow.keras.losses import BinaryCrossentropy
model.compile(loss=BinaryCrossentropy())

model.fit(X,Y,epochs=100)