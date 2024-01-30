import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

print(x_train.shape)
print(y_train.shape)

m = x_train.shape[0]
print(m)

plt.scatter(x_train,y_train,marker= 'o', c ='r')

plt.title("Housing Prices")
plt.ylabel('price')
plt.xlabel('Size')
plt.show()

w = 200
b = 100

def compute_model_output(x,w,b):
  m = x.shape[0]
  f_wb = np.zeros(m)
  for i in range(m):
    f_wb[i] = w * x[i] + b
  return f_wb


def getCost(f_wb, y,m):
  sum_cost =0
  for i in range(m):
    cost = (f_wb[i] - y[i])**2
    sum_cost +=cost
  total_cost = (0.5/m) * sum_cost
  return total_cost


tmp_f_wb = compute_model_output(x_train,w,b)
cost_func_data = getCost(tmp_f_wb,y_train,m)



print(tmp_f_wb)
print(cost_func_data)

cost_func_data = np.zeros(5)
for i in range(5):
  tmp_f_wb = compute_model_output(x_train,i*100,b)
  cost_func_data[i] = getCost(tmp_f_wb,y_train,m)

print(tmp_f_wb)
print(cost_func_data)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

w_param = np.array([0, 1, 2, 3, 4])

plt.plot(w_param,cost_func_data, c='b',label='Cost Function')

# Plot the data points
# plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Cost Function data for different values of w")
# Set the y-axis label
plt.ylabel('Cost Function value')
# Set the x-axis label
plt.xlabel('w')
plt.legend()
plt.show()

#gradient descent for linear regression


x_train = np.array([1.0,2.0,3.0,4.0,5.0,6.0])
y_train = np.array([300.0,500.0,800.0,1000.0,950.0,1500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")



def square_error(x,y,w,b,m):
  sum_cost = 0
  for i in range(m):
    f_wb = w*x[i] + b
    cost = (f_wb - y[i])**2
    sum_cost +=cost

  totalCost = 1/(2*m) * sum_cost
  return totalCost





def derivative(x,y,w,b,m):
  djb =0
  djw =0
  for i in range(m):
    f_wb = w*x[i] + b
    innerTermWi = (f_wb - y[i]) * x[i]
    innerTermBi = (f_wb - y[i])
    djb = djb + innerTermBi
    djw = djw + innerTermWi
  djw= djw/m
  djb = djb/m
  return djb,djw






def gradientDescent(x,y,alpha,w_init,b_init,iter,derivative,costFunction,m):
  w = w_init
  b = b_init

  J_history = []
  p_history = []
  for i in range(iter):
    djb, djw = derivative(x,y,w,b,m)

    b = b - alpha * djb
    w = w - alpha * djw



    if i<100000:
      # prevent resource exhaustion
      J_history.append( costFunction(x, y, w , b,m))
      p_history.append([w,b])

  return b,w, J_history, p_history

b,w,J_history, p_history = gradientDescent(x_train,y_train,1.0e-2,0,0,10000, derivative, square_error,2)

print(b,w)



# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_history[:100])
ax2.plot(1000 + np.arange(len(J_history[1000:])), J_history[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()



