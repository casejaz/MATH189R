import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg, sparse, misc
from sklearn.preprocessing import OneHotEncoder
# assumes data loaded into X_bin_* and y_bin_*


Data_Train = np.asarray(pd.read_csv('mnist_train.csv', header=None, engine='python'))
X_Train = Data_Train[:, 1:]
Y_Train = Data_Train[:, 0]
Limited_Data_Train = np.asarray([row for row in Data_Train if row[0]==0 or row[0] == 1])
X_Train_Bin = Limited_Data_Train[ : , 1:]
Y_Train_Bin = Limited_Data_Train[ : , 0]


Data_Test = np.asarray(pd.read_csv('mnist_test.csv', header=None, engine='python'))
X_Test = Data_Test[ : , 1:]
Y_Test = Data_Test[ : , 0]
Limited_Data_Test = np.asarray([row for row in Data_Test if row[0]==0 or row[0] == 1])
X_Test_Bin = Limited_Data_Test[ : , 1:]
Y_Test_Bin = Limited_Data_Test[ : , 0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_likelihood(X, y_bool, theta, reg=1e-6):
	X = np.matrix(X)
	mu = sigmoid(X*theta/10)
	mu = np.asarray(mu)
	mu[~y_bool] = 1- mu[~y_bool]
	return np.log(mu).sum() - reg *np.linalg.norm(theta)/2


def grad_log_likelihood(X, y, theta, reg=1e-6):
	X_Transpose = np.transpose(X)
	y = np.matrix(y)
	y = np.transpose(y)
	return X_Transpose * (sigmoid(X * theta) - y) + reg * theta


def gradient_descent(X,y, reg=1000, lr=1e-3, tol=1e-6, max_iters=300, verbose=False, print_freq=5):
	X = np.matrix(X)
	y = y.astype(bool)

	theta = np.zeros(X.shape[1])
	theta = np.matrix(theta)
	theta = np.transpose(theta)

	objective = [log_likelihood(X,y,theta, reg=reg)]
	grad = grad_log_likelihood(X,y,theta,reg=reg)

	while len(objective)-1 <= max_iters and np.linalg.norm(grad) > tol:
		if (len(objective) - 1) % print_freq == 0:
			print('[i={}] likelihood: {}. grad norm: {}'.format(
				len(objective) - 1, objective[-1], np.linalg.norm(grad),
			))

		grad = grad_log_likelihood(X,y,theta,reg=reg) / X.shape[0]
		theta = theta-lr*grad
		objective.append(log_likelihood(X,y,theta,reg=reg))

	print('[i={}] done. grad norm = {:0.2f}'.format(
		len(objective)-1, np.linalg.norm(grad)
	))

	return theta, objective


def newton_log_likelihood(X, y_bool, theta, reg=1e-6):
	X = np.matrix(X)
	# do not need to normalize the date
	mu = sigmoid(X*theta)
	mu = np.asarray(mu)
	mu[~y_bool] = 1- mu[~y_bool]
	return np.log(mu).sum() - reg *np.linalg.norm(theta)/2



def newton_step(X, y, theta, reg=1e-6):
	X = np.matrix(X)
	mu = np.asarray([m.item(0) for m in sigmoid(X * theta)])
	return linalg.cho_solve(		
		linalg.cho_factor(X.transpose() * sparse.diags(mu * (1 - mu), 0) * X + reg * sparse.eye(X.shape[1])),
		grad_log_likelihood(X, y, theta, reg=reg),
	)


def newton_method(
	X, y,
	reg=1e-6, tol=1e-6, max_iters=300,
	print_freq=5,
	):
	y = y.astype(bool)

	theta = np.zeros(X.shape[1])
	theta = np.matrix(theta)
	theta = np.transpose(theta)

	objective = [newton_log_likelihood(X, y, theta, reg=reg)]
	step = newton_step(X, y, theta, reg=reg)
	
	while len(objective)-1 <= max_iters and np.linalg.norm(step) > tol:
		if (len(objective)-1) % print_freq == 0:
			print('[i={}] likelihood: {}. step norm: {}'.format(
				len(objective)-1, objective[-1], np.linalg.norm(step)
			))

		step = newton_step(X, y, theta, reg=reg)
		theta -= step
		objective.append(newton_log_likelihood(X, y, theta, reg=reg))
	
	print('[i={}] done. step norm = {:0.2f}'.format(
		len(objective)-1, np.linalg.norm(step)
	))

	return theta, objective

#theta, objective = gradient_descent(X_Train_Bin,Y_Train_Bin, reg=100, max_iters=500)
#theta2, objective2 = newton_method(X_Train_Bin, Y_Train_Bin, reg=100, max_iters=500)

# plt.xlim([-20,500])
# plt.ylim([-9000, 200])
# plt.ylabel('Likelihood with Gaussian Prior')
# plt.xlabel('Iterations')
# plt.title('MNIST Binary Problem')
# plt.plot(range(len(objective)), objective)
# plt.plot(range(len(objective2)), objective2)
# plt.show()




def softmax(x):
	s = np.exp(x - np.max(x, axis = 1))
	return s / np.sum(s, axis=1)


def log_softmax(x):
	return x - misc.logsumexp(x, axis=1)


def softmax_log_likelihood(X, y_one_hot, W, reg=1e-6):
	X = np.matrix(X)
	W = np.matrix(W)
	W_Transpose = np.transpose(W)

	mu = X * W
	return np.sum(mu[y_one_hot] - misc.logsumexp(mu, axis =1)) - reg * np.einsum('ij,ji->', W_Transpose, W)/2


def soft_grad_log_likelihood(X, y_one_hot, W, reg=1e-6):
	X = np.matrix(X)
	X_Transpose = np.transpose(X)
	W = np.matrix(W)
	mu = X * W
	mu = np.exp(mu- np.max(mu, axis=1))
	mu = mu / np.sum(mu, axis=1)
	return X_Transpose * (mu-y_one_hot) + reg*W


def softmax_grad(
	X, y, reg=1e-6, lr=1e-8, tol=1e-6,
	max_iters=300, batch_size=256,
	verbose=False, print_freq=5):

	enc = OneHotEncoder()
	y_one_hot = enc.fit_transform(y.copy().reshape(-1,1)).astype(bool).toarray()
	W = np.zeros((X.shape[1], y_one_hot.shape[1]))
	ind = np.random.randint(0, X.shape[0], size=batch_size)
	objective = [softmax_log_likelihood(X[ind], y_one_hot[ind], W, reg=reg)]
	grad = soft_grad_log_likelihood(X[ind], y_one_hot[ind], W, reg=reg)

	while len(objective)-1 <= max_iters and np.linalg.norm(grad) > tol:
		if verbose and (len(objective)-1) % print_freq == 0: print('[i={}] likelihood: {}. grad norm: {}'.format(
			len(objective)-1, objective[-1], np.linalg.norm(grad)
			))

		ind = np.random.randint(0, X.shape[0], size=batch_size)
		grad = soft_grad_log_likelihood(X[ind], y_one_hot[ind], W, reg=reg)
		W = W - lr * grad

		objective.append(softmax_log_likelihood(X[ind], y_one_hot[ind], W, reg=reg))

	print('[i={}] done. grad norm = {:0.2f}'.format(
		len(objective)-1, np.linalg.norm(grad)
		))

	return W, objective


# W, objective = softmax_grad(X_Train, Y_Train, reg=1000, max_iters = 500)
# plt.plot(range(len(objective)), objective)
# plt.xlim([-50,500])
# plt.ylim([-600, 30])
# plt.ylabel('Stochastic Log-likelihood')
# plt.xlabel('Iterations')
# plt.title('Softmax Regression')
# plt.show()
# print X_Test*W
# print Y_Test


# Limit data to only 2500 data points (first 2500 points)
X_Train_C = X_Train[ :2500, :]
Y_Train_C = Y_Train[ :2500]
X_Test_C = X_Test[ :2500, :] 
Y_Test_C = Y_Test[ :2500]


def predict_knn(X_test, X_train, y_train, k=10):
	y_pred = [0] * X_test.shape[0]

	for i in range(X_test.shape[0]):
		imgi = X_test[i]
		ind = np.argpartition(1. / np.linalg.norm(X_train - np.transpose(imgi[:,np.newaxis]), axis=1), -k)[-k:]
		y_pred[i] = np.argmax(np.bincount(y_train[ind]))

	return y_pred


def y_pred(X_test, theta):
	y_prediction = X_Test_Bin*theta
	for i in range(len(y_prediction)):
		if y_prediction[i] > 0.:
			y_prediction[i] = 1
		else:
			y_prediction[i] = 0
	return y_prediction

#theta2, objective2 = newton_method(X_Train_Bin, Y_Train_Bin, reg=100, max_iters=500)
y_prediction = predict_knn(X_Test_C, X_Train_C, Y_Train_C, k=1)
numTrue = 0
for i in range(len(y_prediction)):
	if y_prediction[i] == Y_Test_C.item(i):
		numTrue += 1

accuracy = float(numTrue)/ float(len(Y_Test_C))
print accuracy 