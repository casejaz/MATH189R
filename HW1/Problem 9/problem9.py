import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
from collections import defaultdict
df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/' 'iris.csv', sep=',', engine='python')
plt.style.use('ggplot')

def discriminant_analysis(X, y, linear = False, reg = 0.0):
	labels = np.unique(y)
	print labels
	mu = {}
	cov = {}
	pi = {}
	for label in labels:
		pi[label] = (y == label).mean()
		mu[label] = X[y == label].mean(axis = 0)
		diff = X[y == label] - mu[label]
		diff_M = np.matrix(diff)
		cov[label] = np.transpose(diff_M) * diff_M /(y == label).sum()

	if linear:
		cov = sum((y == label).sum() * cov[label] for label in labels)
		cov = cov / y.shape[0]
		cov = reg * np.diag(np.diag(cov)) + (1-reg)*cov
	return pi, mu, cov

def normal_density(X, mu, cov):
	#predict class probability
	diff = X - mu
	diff_M = np.matrix(diff)
	diff_T = np.transpose(diff_M)
	return np.exp(-diff_M * np.linalg.inv(cov) * diff_T / 2) / ((2 * np.pi)**(-X.shape[0]/2) * np.sqrt(np.linalg.det(cov)))



def predict_proba(X, pi, mu, cov):
	prob = np.zeros((X.shape[0], len(pi)))
	
	if type(cov) is not dict:
		covariance = cov
		cov = defaultdict(lambda: covariance)
	for i, x in enumerate(X):
		for j in range(len(pi)):
			label = labels[j]

			prob[i,j] = pi[label] * normal_density(x, mu[label], cov[label])
	
	prob = prob / prob.sum(axis=1)[:,np.newaxis]
	
	return prob






	

X = df[['Sepal.Length', 'Petal.Width']].as_matrix()
y = df.Species.as_matrix()
labels = np.unique(y)


pi, mu, cov = discriminant_analysis(
	X,
	y,
	linear = False,
	reg = 0.0
)


for index in range(len(y)):
	if y[index] == 'setosa':
		y[index] = 0
	elif y[index] == 'versicolor':
		y[index] = 1
	else:
		y[index] = 2

print('[linear=True, reg=0.00] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.00] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.05] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.11] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.16] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.21] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.26] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.32] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.37] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.42] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.47] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.53] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.58] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.63] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.68] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.74] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.79] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.84] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.89] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.95] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))
print('[linear=False, reg=0.100] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == y).mean()))