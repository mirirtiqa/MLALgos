import numpy as np
import matplotlib.pyplot as plt
import copy, math
def plot_decision_boundary(X, y, w, b):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])

def sigmoid(z):
  op = 1/(1 + np.exp(-z))
  return op

def compute_cost_logistic(X,y,w,b):
  m = X.shape[0]
  cost = 0.0
  for i in range(m):
    z_i = np.dot(X[i],w) + b
    f_wb_i = sigmoid(z_i)
    loss = (-y[i]*np.log(f_wb_i)) - ((1-y[i])*np.log(1- f_wb_i))
    cost +=loss
  cost = cost/m
  return cost


def gradient(X,y,w,b):
  m = X.shape[0]
  n = X.shape[1]
  dj_w = np.zeros((n,))
  dj_b =0.
  for i in range(m):
    z = sigmoid(np.dot(X[i],w) + b)
    error = z - y[i]
    for j in range(n):
      dj_w[j] =dj_w[j] + error*X[i,j]
    dj_b += error

  dj_w = dj_w/m
  dj_b = dj_b/m
  return dj_b,dj_w


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

b_init = 1
w_init = np.array([ 1,1])


initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 3.0e-5
# run gradient descent
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost_logistic, gradient,
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape

# Plot the decision boundary
plot_decision_boundary(X_train, y_train, w_final, b_final)