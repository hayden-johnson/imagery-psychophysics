import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.stats import norm
from curves.models import IsotropicGaussian
from curves.core import check_intersections
from curves.rate_distortion import reshape
from curves.probes import generate_custom_probe_set_ext


def run_simulation(targets, knots, probes, n_repeats, tau=7, capacity=2):
    n_targets, n_probes = len(targets), len(probes)
    X = np.ones((n_targets, n_probes, n_repeats)) *-1
    C = np.ones((n_targets, n_probes, n_repeats)) *-1
    R = np.ones((n_targets, n_probes, n_repeats)) *-1
    X_hat = np.ones((n_targets, n_probes, n_repeats)) *-1
    R_hat = np.ones((n_targets, n_probes, n_repeats)) *-1

    for i,(t,k) in enumerate(zip(targets, knots)):
        model = IsotropicGaussian(k, capacity, tau)
        sample = model.sample_curves(n_probes*n_repeats)
        for j,p in enumerate(probes):
            _,n = check_intersections(t, p)
            if n == 0:
                c = .5
            else:
                c = n+.5 if ((i+j)%2 == 0) else n-.5
            r = 1 if n > c else 0
            for k in range(n_repeats):
                X[i, j, k] = n
                C[i, j, k] = c
                R[i, j, k] = r
                _,n_hat = check_intersections(sample[(j*n_repeats) + k], p)
                X_hat[i, j, k] = n_hat
                R_hat[i, j, k] = 1 if n_hat > c else 0
    
    return X, X_hat, R, R_hat



def generate_plots(target_curves, knots, probes, n_sims, n_samples, nc, npc, capacity, tau):
    X = np.zeros((n_sims, 5, nc))
    for i in trange(n_sims):
        _,_,R_vis_sim, R_img_sim = run_simulation(target_curves, 
                                                  knots, 
                                                  probes, 
                                                  n_samples, 
                                                  capacity=capacity, 
                                                  tau=tau)
        A_sim = (R_img_sim == R_vis_sim).astype(int)
        E_sim = R_img_sim - R_vis_sim
        avg_error_sim = [e.mean() for e in split_by_complexity(E_sim, npc)]
        avg_acc_sim = [a.mean() for a in split_by_complexity(A_sim, npc)]
        avg_var_sim = [x.var(axis=2).mean() for x in split_by_complexity(R_img_sim, npc)]
        avg_snr_sim = [compute_SNR(x) for x in split_by_complexity(R_img_sim, npc)]

        R_sim = list(zip(split_by_complexity(R_vis_sim, npc), split_by_complexity(R_img_sim, npc)))
        d_prime_sim = [compute_d_prime(r_vis, r_img) for (r_vis, r_img) in R_sim]

        for j in range(nc):
            X[i][0][j] = avg_error_sim[j]
            X[i][1][j] = avg_acc_sim[j]
            X[i][2][j] = avg_var_sim[j]
            X[i][3][j] = avg_snr_sim[j]
            X[i][4][j] = d_prime_sim[j]
            
    fig, ax = plt.subplots(ncols=5, figsize=(25, 4))
    x = np.arange(nc)
    for i in range(n_sims):
        labels = ['error', 'acc', 'var', 'snr', 'd-prime']
        for j,label in zip(range(5), labels):
            ax[j].plot(x, X[i][j], c='slategrey', alpha=.3)
            ax[j].plot(x, X.mean(axis=0)[j], c='black')
            ax[j].set_title(label)
    plt.savefig('./plot.png')    
    return X
    
        
        
def split_by_complexity(X, npc):
    return np.array([X[i:i+npc] for i in range(0, len(X), npc)])

def compute_SNR(X):
    S = X.mean(axis=2).var(axis=1)
    N = X.var(axis=2).mean(axis=1)
    SNR = S/N
    return SNR.mean()

def compute_d_prime(X, X_hat):
    X = X.flatten()
    X_hat = X_hat.flatten()
    
    n_signal = len(X[X==1])
    n_noise =  len(X[X==0])
    n_hits =  (X_hat[X==1]).sum()
    n_fa =    (X_hat[X==0]).sum()
    
    H = n_hits / n_signal
    F = n_fa / n_noise
    d_prime = norm.ppf(H) - norm.ppf(F)
    return d_prime



if __name__ == '__main__':
    ## load curves and knots
    load_path = f'../stimuli/UNTITLED_2'
    avg_curves = [np.loadtxt(f'{load_path}/selections/{i}_average_targets.txt', delimiter=',') for i in range(10)]
    avg_knots = [np.loadtxt(f'{load_path}/selections/{i}_average_knots.txt', delimiter=',') for i in range(10)]
    
    ## reshape
    curves, knots = [], []
    for tc,kn in zip(avg_curves, avg_knots):
        for t,k in zip(tc, kn):
            knots.append(np.reshape(k, (len(k) // 2, 2)))
            curves.append(reshape(t))
            
            
    probes = generate_custom_probe_set_ext(2, ext=.2, plot=False)
    n_probes = len(probes)
    
    curves_per_complexity = 20
    n_sims, n_samples = 10, 2
    nc, npc = len(avg_curves), curves_per_complexity
    capacity, tau = 11, 7

    X = generate_plots(curves, knots, probes, n_sims, n_samples, nc, npc, capacity, tau)
    #np.savetxt('X.txt', X, delimiter=',')
    
    with open('X.npy', 'wb') as f:
        np.save(f, X, allow_pickle=False)
        