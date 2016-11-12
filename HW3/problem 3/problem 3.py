from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn import metrics

def em(X, k, theta, objective,likelihood, m_step, max_iter=100):
  r = np.ones((X.shape[0], k)) / k
  pi = np.ones(k) / k
  objectives = [objective(X, r, pi, theta)]

  for i in range(max_iter):

    # e-step
    r = likelihood(X, theta) * pi
    r = r / r.sum(axis=1)[:,np.newaxis]

    # m-step
    pi, theta = m_step(X, r)
    objectives.append(objective(X, r, pi, theta))

  return (objectives, r, pi, theta)



def gmm(X, k, prior_alpha, max_iter=100):

  S_0 = np.diag(np.std(X, axis=0)**2) / k**(1/X.shape[1])
  Theta = namedtuple("GMM", "mean cov")
  theta = Theta( mean=X[np.random.randint(0, X.shape[0], (k,))], cov=np.tile(S_0, (k,1,1)))

  denominator = X.shape[0] + prior_alpha.sum() - k
  prior_nu = X.shape[1] + 2

  def likelihood(X, theta):
    p = np.zeros((X.shape[0], k))
    for i in range(k):
      p[:,i] = stats.multivariate_normal.pdf(X, theta.mean[i], theta.cov[i] + 1e-4*np.eye(X.shape[1]))
    print p.shape
    return p


  def m_step(X, r):
    r_sum = r.sum(axis=0)
    pi = (r_sum + prior_alpha - 1) / denominator
    mu = ((X[:,:,np.newaxis] * r[:,np.newaxis,:]).sum(axis=0) / r_sum).T
    sigma = np.zeros((k, X.shape[1], X.shape[1]))
    for i in range(k):
      diff = (X - mu[i]) * np.sqrt(r[:,i])[:,np.newaxis]
      diff_M = np.matrix(diff)
      diff_T = np.transpose(diff_M)

      sigma[i] = (diff_T * diff_M + S_0) / (prior_nu + r_sum[i] + X.shape[1] + 2)

    return pi, Theta(mean=mu, cov=sigma)


  def objective(X, r, pi, theta):
    log_prior = sum(np.log(stats.invwishart.pdf(theta.cov[i], df=prior_nu, scale=S_0)) for i in range(k)) + np.log(stats.dirichlet.pdf(pi, alpha=prior_alpha))
    pi_term = (r * np.log(pi)[np.newaxis,:]).sum()
    likelihood_term = r * np.log(likelihood(X, theta))
    likelihood_term = likelihood_term[r > 1e-12].sum()
    return likelihood_term + pi_term + log_prior


  return em(X, k, theta, objective,likelihood, m_step, max_iter=max_iter)


red = pd.read_csv("winequality-red.csv", sep= ';', engine= 'python')
white = pd.read_csv("winequality-white.csv", sep = ';', engine = 'python')
red = red.as_matrix()
white = white.as_matrix()
X = np.concatenate((red, white), axis = 0)
k=2

obj, r, pi, theta = gmm(X, k, np.ones(2), max_iter = 30)
plt.plot(obj)
plt.xlim([-0.5,30.5])
plt.title("MAP Gaussian Mixture Model")
plt.xlabel("Iteration")
plt.ylabel("Complete Data Log Likelihood")
plt.show()

y = np.zeros((X.shape[0],))
y[:red.shape[0]] = 1
y_pred = np.argmax(r,axis=1)

Confusion_M = metrics.confusion_matrix(y, ~y_pred.astype(bool))

plt.imshow(Confusion_M, interpolation='nearest', cmap=plt.cm.gray_r)
plt.title("GMM Confusion Matrix Plot")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.colorbar()

classes = ["white", "red"]
ticks = np.arange(len(classes))
plt.xticks(ticks, classes)
plt.yticks(ticks, classes)
plt.show()