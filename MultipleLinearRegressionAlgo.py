import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def predict_single_loop(x,w,b):
  n = x.shape[0]
  p_i = 0
  for i in range(n):
    p = p + x[i]*w[i]
  p +=b
  return p

def compute_cost(X,y,w,b):
  m = X.shape[0]
  cost = 0.0
  for i in range(m):
    f_wb_i = np.dot(X[i],w) +b 
    cost = cost + (f_wb_i - y[i])**2
  cost = cost/(2*m)
  return cost

cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')

def compute_gradient(X,y,w,b):
  m,n = X.shape
  dj_w = np.zeros((n,))
  dj_b =0.
  for i in range(m):
    error = (np.dot(X[i],w) + b) - y[i]
    for j in range(n):
      dj_w[j] = dj_w[j] + error*X[i,j]
    dj_b = dj_b + error
  dj_w = dj_w/m
  dj_b = dj_b/m
  return dj_b,dj_w

tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

import copy, math

def gradient_descent(X,y,w_init,b_init,compute_cost,compute_gradient,alpha,iter):
  J_history = []
  w = copy.deepcopy(w_init)  #avoid modifying global w within function
  b = b_init
 
  for i in range(iter):
    dj_b,dj_w = compute_gradient(X,y,w,b)
    w = w - alpha * dj_w
    b = b - alpha * dj_b
  

    if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost(X, y, w, b))

    if i % math.ceil(iter / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
  return w, b, J_history



initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")