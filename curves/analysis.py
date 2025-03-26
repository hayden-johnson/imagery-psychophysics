import numpy as np 
from scipy.stats import multivariate_normal

def compute_SNR(X):
    # mean of var / var of mean 
    S = X.mean(axis=2).var(axis=1)
    N = X.var(axis=2).mean(axis=1)
    SNR = S/N
    return SNR

def compute_d_prime(X, X_hat, targets_per_complexity):
    n_targets = len(X)
    for i in range(0, n_targets, targets_per_complexity):
        x = X[i:i+4].flatten()
        x_hat = X_hat[i:i+4].flatten()

        n_hits, n_signal = 0,0
        n_fa, n_noise = 0, 0
        for correct, resp in zip(x, x_hat):
            if correct == 0:
                n_signal +=1
                if resp == 0:
                    n_hits +=1
            else:
                n_noise +=1
                if resp == 0:
                    n_fa +=1

        H = n_hits / n_signal
        F = n_fa / n_noise 

        d_prime = norm.ppf(H) - norm.ppf(F)
        d_prime_values.append(d_prime)
        
    return np.array(d_prime_values)

