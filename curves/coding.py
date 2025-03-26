import numpy as np
from scipy import spacial, special
import matplotlib.pyplot as plt

def homogeneous_poisson(rate, tmax, bin_size):
    nbins = np.floor(tmax/bin_size).astype(int)
    prob_of_spike = rate * bin_size
    spikes = np.random.rand(nbins) < prob_of_spike
    return spikes * 1

def homogeneous_poisson_generator(n_trials, rate, tmax, bin_size):
    for i in range(n_trials):
        yield homogeneous_poisson(rate, tmax, bin_size)
        
def poisson_prob(k, lam):
    return lam**k/special.factorial(k)*np.exp(-lam)

def generate_rf_grid(n, low, high):
    step = (high-low) / (n)
    start = low + step
    stop = high 
    grid = []
    for x in np.linspace(low, high, num=n+2)[1:-1]:
        for y in np.linspace(low, high, num=n+2)[1:-1]:
            grid.append([x, y])
    return np.array(grid)

## gaussian rf fuction
def gaussian_rf(x, y, strength, spread):
    dist = np.linalg.norm(x-y)
    return strength * np.exp(-(dist)**2 / (2*(spread**2)))

def encoder(stim_loc, rf_grid, strength=1, spread=.1, noise_rate=.1, scale=10):
    signal = np.array([gaussian_rf(rf_loc, stim_loc, strength, spread) for rf_loc in rf_grid])
    noise = np.ones(len(signal)) * noise_rate
    lam = signal + noise
    lam /= lam.sum() # normalize
    return lam * scale