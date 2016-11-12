from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

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


def bernoullis(X, k, prior_alpha, prior_a, prior_b, max_iter=100):

  S_0 = np.diag(np.std(X, axis=0)**2) / k**(1/X.shape[1])

  Theta = namedtuple("BMM", "mean")
  # compute initial means by partitioning data arbitrarily
  col = int(np.floor(X.shape[0]/k))
  theta = Theta(mean=X[:k*col].reshape(k, -1, X.shape[1]).mean(axis=1))

  def likelihood(X, theta):
    p = np.tile(theta.mean.T, (X.shape[0],1,1) )
    p[X == 0] = 1 - p[X == 0]
    p = p.prod(axis=1)
    return p


  def m_step(X, r):
    denominator = X.shape[0] + prior_alpha.sum() - k

    r_sum = r.sum(axis=0)
    pi = (r_sum + prior_alpha - 1) / denominator
    mu = (((X[:,:,np.newaxis] * r[:,np.newaxis,:]).sum(axis=0) + prior_a - 1) /(r_sum + prior_a + prior_b - 2)).T

    return pi, Theta(mean=mu)


  def objective(X, r, pi, theta):

    log_prior = np.log(stats.beta.pdf(theta.mean, prior_a, prior_b)).sum() + np.log(stats.dirichlet.pdf(pi, alpha=prior_alpha))
    pi_term = (r * np.log(pi)[np.newaxis,:]).sum()
    likelihood_term = r * np.log(likelihood(X, theta))
    likelihood_term = likelihood_term[r > 1e-12].sum()

    return likelihood_term + pi_term + log_prior


  return em(X, k, theta, objective,likelihood, m_step, max_iter=max_iter)

def plot_image(img):
    plt.imshow(img.reshape(28,28), cmap="Greys")
    plt.axis("off")


train = pd.read_csv("mnist_train.csv", header = None)
X = train.iloc[:,1:].as_matrix()
X = (X > X[X > 0].mean()).astype(float)
y = train.iloc[:,0].as_matrix()
del train
np.random.seed(1)
N = int(10000)
subset_ix = np.random.randint(0,X.shape[0],(N,))
X_downsample = X[subset_ix]
k = 10

obj, r, pi, theta = bernoullis(X_downsample, k, np.ones(10),1, 1, max_iter=50)
plt.plot(obj)
plt.title("MAP Bernoulli Mixture Model Convergence Plot")
plt.xlabel("Number of Iteration")
plt.ylabel("Complete Data Log Likelihood")
plt.show()

plt.figure(figsize=(5,2))
for i in range(10):
    plt.subplot(2,5,i+1)
    plot_image(theta.mean[i])

plt.suptitle("Mean Images for Each Component of Mixture of Bernoullis")

plt.show()

