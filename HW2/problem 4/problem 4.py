import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
df = pd.read_csv('https://math189r.github.io/hw/data/online_news_popularity/online_news_popularity.csv', sep =", ", engine='python')
X, y = df[[col for col in df.columns if col not in ['url', 'shares', 'cohort']]], np.log(df.shares).reshape(-1,1)
X = np.hstack((np.ones_like(y), X))

#
def objective(X, y, w, reg=1e-6):
	print "Hello from objective"
	X_M = np.matrix(X)
	y_M = np.matrix(y)
	w_M = np.matrix(w)
	err = X_M * w_M - y_M
	err = float(err.T * err)
	print "I am alive!"
	return (err + reg * np.abs(w_M).sum())/len(y_M)

#
def grad_objective(X, y, w):
	print "Hello from grad_objective"
	X_M = np.matrix(X)
	y_M = np.matrix(y)
	w_M = np.matrix(w)
	return X_M.T * (X_M * w_M - y_M) / len(y_M)

def prox(x, gamma):
	print "Hello from prox"

	for index in range(len(x)):

		value = x.item(index)
		if abs(value) <= gamma:
			x.itemset(index,0.)
		elif value > gamma:
			value -= gamma
			x.itemset(index,value)
		elif value < -gamma:
			value += gamma
			x.itemset(index,value)
		print "Hello from for loop!"
	return x

def lasso_grad(X, y, reg=1e-6, lr=1e-3, tol=1e-6, max_iters=300, batch_size=256, eps=1e-5, verbose=True, print_freq=5):
	print "Hello lasso_grad"
	X_M = np.matrix(X)
	y_M = np.matrix(y)
	y_M = y_M.reshape(-1,1)
	w_M = np.linalg.solve(X_M.T * X_M, X_M.T * y_M)
	ind = np.random.randint(0, X_M.shape[0], size=batch_size)
	obj = [objective(X_M[ind], y_M[ind], w_M, reg=reg)]
	grad = grad_objective(X_M[ind], y_M[ind], w_M)
	while (len(obj)-1 <= max_iters) and (np.linalg.norm(grad) > tol):
		if verbose and (len(obj)-1) % print_freq == 0:
			print("[i={}] objective: {}. sparsity = {:0.2f}".format(len(obj)-1, obj[-1], (np.abs(w_M) < reg*lr).mean()))
		ind = np.random.randint(0, X_M.shape[0], size=batch_size)
		grad = grad_objective(X_M[ind], y_M[ind], w_M)
		w_M = prox(w_M - lr * grad, reg * lr)
		obj.append(objective(X_M[ind], y_M[ind], w_M, reg=reg))
	if verbose:
		print("[i={}] done. sparsity = {:0.2f}".format(len(obj)-1, (np.abs(w_M) < reg*lr).mean()))
	return w_M, obj

def lasso_path(X, y, reg_min=1e-8, reg_max=10, regs=10, **grad_args):
	print "Hello from lasso_grad"
	X_M = np.matrix(X)
	y_M = np.matrix(y)

	W = np.zeros((X_M.shape[1], regs))
	tau = np.linspace(reg_min, reg_max, regs)
	for i in range(regs):
		W[:,i] = lasso_grad(X_M, y_M, reg=1/tau[i], max_iters=1000, batch_size=1024, **grad_args)[0].flatten()
	return tau, W

tau, W = lasso_path(X, y, reg_min=1e-15, reg_max=0.02, regs=10, lr=1e-12)
p = plt.plot(tau, W.T)
#plt.title("Lasso Path")
#plt.xlabel("tau")
#plt.ylabel("w")
#plt.show()
#a = np.array(df.columns)[np.argsort(-W[:,9])[:5]+1]


lr = 1e-12
reg = 1e5
w, obj = lasso_grad(X, y, reg=reg, lr=lr, eps=1e-2, max_iters=2500, batch_size=1024, verbose=True, print_freq=250)
plt.title("rLasso Objective Convergence")
plt.ylabel("Stochastic Objective")
plt.xlabel("Iteration")
plt.plot(obj)
plt.show()
