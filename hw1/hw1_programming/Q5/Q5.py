import numpy as np
from matplotlib import pyplot as plt
from numpy.random import randint

d = 100  # dimensions of data
n = 1000  # number of data points
X = np.random.normal(0, 1, size=(n, d))
w_true = np.random.normal(0, 1, size=(d, 1))
y = X.dot(w_true) + np.random.normal(0, 0.5, size=(n, 1))


# print(X.shape)  # (1000, 100)
# print(y.shape)  # (1000, 1)


# ==================Problem Set 5.1=======================
print("==================Problem Set 5.1=======================")

def compute_loss(X, y, w):
    """
    compute squared error loss
    Input:
      X (ndarray (m,n)): m examples, n features
      y (ndarray (m,)) : m target values
      w (ndarray (n,)) : n weight parameters
    Output:
      loss (scalar): loss
    """
    m = X.shape[0]
    loss = 0.0
    for i in range(m):
        pred = np.dot(X[i], w)
        loss += (pred - y[i]) ** 2
    return loss[0]


# print(w_ls.shape)  # (100, 1) => transpose: (1, 100)
# print(X[0].shape)  # (100, ) = (1, 100)
w_ls = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
F_w_ls = compute_loss(X, y, w_ls)
print(f"Total squared error is {F_w_ls} for w_ls on training set")

w_0 = np.zeros((d,))
F_w_0 = compute_loss(X, y, w_0)
print(f"Total squared error is {F_w_0} for w_0 (all 0 vector) on training set")

# create test set and compute loss
X_test = np.random.normal(0, 1, size=(n, d))
y_Test = X_test.dot(w_true) + np.random.normal(0, 0.5, size=(n, 1))
F_w_ls_test = compute_loss(X_test, y_Test, w_ls)
print(f"Total squared error is {F_w_ls_test} for w_ls on test set")

# TODO: comment
# After deriving the w_ls, I compared the total squared error of the test set and that of the training

# ==================Problem Set 5.2=======================
print("==================Problem Set 5.2=======================")

# TODO: check this correctness
def compute_gradient(X, y, w):
    """
    Computes the gradient for linear regression
    Input:
      X (ndarray (m,n)): m examples, n features
      y (ndarray (m,)) : m target values
      w (ndarray (n,)) : n weight parameters
    Output:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
    """
    m, n = X.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    for i in range(m):
        err = np.dot(X[i], w) - y[i]
        dj_dw += 2 * err * X[i]
    return dj_dw


def gradient_descent(X, y, w_in, compute_gradient, compute_loss, lr, num_iters):
    """
    perform gradient descent with learning rate lr for num_iters using the compute_gradient function
    Input:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      compute_gradient    : function to compute the gradient
      compute_loss        : function to compute loss (used for plotting)
      lr (float)          : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
    Output:
      w (ndarray (n,)) : Updated values of parameters
      """
    # An array to store cost J and w's at each iteration primarily for graphing later
    w = np.copy(w_in)
    cost_array = []
    for i in range(num_iters):
        # Calculate the gradient
        dj_dw = compute_gradient(X, y, w)
        # Update w using gradient descent
        w = w - lr * dj_dw
        curr_loss = compute_loss(X, y, w)
        # print(f"curr_loss for {i} iteration with lr={lr}: ", curr_loss)
        cost_array.append(curr_loss)
    return w, cost_array


lr1 = 0.00005
lr2 = 0.0005
lr3 = 0.0007
num_iter = 20
w_0 = np.zeros((d,))
w_1, cost_array_1 = gradient_descent(X, y, w_0, compute_gradient, compute_loss, lr1, num_iter)
w_2, cost_array_2 = gradient_descent(X, y, w_0, compute_gradient, compute_loss, lr2, num_iter)
w_3, cost_array_3 = gradient_descent(X, y, w_0, compute_gradient, compute_loss, lr3, num_iter)

print(f"Final Total squared error using GD after 20 iterations with learning rate = {lr1} is {cost_array_1[-1]}")
print(f"Final Total squared error using GD after 20 iterations with learning rate = {lr2} is {cost_array_2[-1]}")
print(f"Final Total squared error using GD after 20 iterations with learning rate = {lr3} is {cost_array_3[-1]}")


# plot them on the same graph
plt.figure()
plt.plot(range(1, 21), cost_array_1, color='r', label="lr=0.00005")
plt.plot(range(1, 21), cost_array_2, color='g', label="lr=0.0005")
plt.plot(range(1, 21), cost_array_3, color='b', label="lr=0.0007")
plt.xlabel("k (num of iterations)")
plt.ylabel("loss")
plt.legend(loc="best")
plt.title('loss for gradient descent')
plt.show()


# ==================Problem Set 5.3=======================
print("==================Problem Set 5.3=======================")
def compute_stochastic_gradient(X, y, w):
    """
    Computes the stochastic gradient for linear regression
    Input:
      X (ndarray (m,n)): m examples, n features
      y (ndarray (m,)) : m target values
      w (ndarray (n,)) : n weight parameters
    Output:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
    """
    d, n = X.shape  # (number of examples, number of features)
    rand = randint(0, n)
    return 2 * (np.dot(X[rand], w) - y[rand]) * X[rand].reshape(n, 1)

lr1 = 0.0005
lr2 = 0.005
lr3 = 0.01
num_iter = 1000
w_0 = np.zeros((d,))
w_1, cost_array_1 = gradient_descent(X, y, w_0, compute_stochastic_gradient, compute_loss, lr1, num_iter)
print(f"Final Total squared error using SGD after 1000 iterations with learning rate = {lr1} is {cost_array_1[-1]}")
w_2, cost_array_2 = gradient_descent(X, y, w_0, compute_stochastic_gradient, compute_loss, lr2, num_iter)
w_3, cost_array_3 = gradient_descent(X, y, w_0, compute_stochastic_gradient, compute_loss, lr3, num_iter)

print(f"Final Total squared error using SGD after 1000 iterations with learning rate = {lr2} is {cost_array_2[-1]}")
print(f"Final Total squared error using SGD after 1000 iterations with learning rate = {lr3} is {cost_array_3[-1]}")


plt.figure()
plt.plot(range(1, 1001), cost_array_1, color='r', label="lr=0.0005")
plt.plot(range(1, 1001), cost_array_2, color='g', label="lr=0.005")
plt.plot(range(1, 1001), cost_array_3, color='b', label="lr=0.01")
plt.xlabel("k (num of iterations)")
plt.ylabel("loss")
plt.legend(loc="best")
plt.title('loss for stochastic gradient descent')
plt.show()

