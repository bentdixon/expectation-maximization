# Expectation maximization: iterative algorithm to 
# estimate unknown parameters

# E-step: calculate expected values of hidden variables, given data
# M-step: re-estimates model parameters by maximizing log-likelihood

# Repeat until convergence, i.e., parameter values stop changing or
# LL stops improving

# EM is an extension of MLE to where the variables are hidden
# Posterior probabilities are used in the E-step (Bayesian)

from typing import TypeAlias

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.stats import norm, gaussian_kde

FloatArray: TypeAlias = NDArray[np.float64]

# Generate dataset with two Gaussians
# The viewer would not know the exact parameters of the Gaussians that
# underly the true data

mu1: float = 2       # mean of 1st
sigma1: float = 1    # SD of 1st
mu2: float = -1      # mean of 2nd
sigma2: float = 0.8  # SD of 2nd

X1: FloatArray = np.random.normal(mu1, sigma1, size=200)
X2: FloatArray = np.random.normal(mu2, sigma2, size=600)
X: FloatArray = np.concatenate([X1, X2])

sns.kdeplot(X)
plt.xlabel('X')
plt.ylabel('Density')
plt.title('Density Estimation of X')
plt.show()

# Initialize parameters (initial guesses)
mu1_hat: float = float(np.mean(X1))    # Mean guess of X1
sigma1_hat: float = float(np.std(X1))  # SD guess of X1
mu2_hat: float = float(np.mean(X2))    # Mean guess of X2
sigma2_hat: float = float(np.std(X2))  # SD guess of X2
pi1_hat: float = len(X1) / len(X)      # Proportion X1 contributes
pi2_hat: float = len(X2) / len(X)      # Proportion X2 contributes

# Perform EM
num_epochs: int = 20
log_likelihoods: list[float] = []

# Gamma values are the "responsibilities" of each component
for epoch in range(num_epochs):
    # norm.pdf() returns the height of the pdf at x, assuming a normal distribution
    # returns density, not probability, but relative comparisons are informative => 
    # higher value means x is more typical of that distribution
    gamma1: FloatArray = pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat) # Gamma1[i] is the probability that point i came from Gaussian 1
    gamma2: FloatArray = pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)
    total: FloatArray = gamma1 + gamma2
    gamma1 /= total 
    gamma2 /= total         # Normalizes the likelihood scores. Think of as the denominator in Bayes' formula. 
    # P (group 1 | x) = [P(x | group 1) * P(group 1)] / P(x)
    # evidence, P(x) is total
    # likelihood, P(x | group 1), is norm.pdf(X, mu1_hat, sigma1_hat)
    # prior, P(group 1), is pi1_hat

    # Estimate mean of X1 and X2
    mu1_hat = np.sum(gamma1 * X) / np.sum(gamma1)
    mu2_hat = np.sum(gamma2 * X) / np.sum(gamma2)

    sigma1_hat = np.sqrt(np.sum(gamma1 * (X - mu1_hat)**2) / np.sum(gamma1))
    sigma2_hat = np.sqrt(np.sum(gamma2 * (X - mu2_hat)**2) / np.sum(gamma2))

    # Updating the prior
    pi1_hat = float(np.mean(gamma1))  # Get the mean of all 'soft counts' from each gamma, they become the next prior
    pi2_hat = float(np.mean(gamma2))  # A Gaussian believes to own more of the data gest more weight when 'claiming' an ambiguous point

    log_likelihood: float = np.sum(np.log(pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat) + pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)))
    log_likelihoods.append(log_likelihood)

    # print(f"Epoch: {epoch}")
    # print(f"X1 mean estimate: {mu1_hat}")
    # print(f"X2 mean estimate: {mu2_hat}")
    # print(f"X1 SD estimate: {sigma1_hat}")
    # print(f"X2 SD estimate: {sigma2_hat}")

plt.plot(range(1, num_epochs + 1), log_likelihoods)
plt.xlabel('Epoch')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs. Epoch')
plt.show()

# Show final estimate curves from original data (original smoothed with KDE)
X_sorted: FloatArray = np.sort(X)
density_estimation: FloatArray = (pi1_hat * norm.pdf(X_sorted, mu1_hat, sigma1_hat) + pi2_hat * norm.pdf (X_sorted, mu2_hat, sigma2_hat))

plt.plot(X_sorted, gaussian_kde(X_sorted)(X_sorted), color='green', linewidth=2)
plt.plot(X_sorted, density_estimation, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('Density')
plt.title('Final Density Estimation')
plt.legend(['Kernel Density Estimation', 'Mixture Density'])
plt.show()