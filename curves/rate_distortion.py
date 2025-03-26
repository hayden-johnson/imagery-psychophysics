import numpy as np
from tqdm import tqdm 
from scipy.stats import pearsonr
from curves.core import check_intersections
from curves.visualization import rd_plot

""" computing distortions """
# return the number of intersections between a probe set and target curve
def get_distortion_profile(probes, curve):
    d = []
    for probe in probes:
        _, n_isecs = check_intersections(curve, probe)
        d.append(n_isecs)
    return np.array(d, dtype=int)

# computes rate distortion profile using pc reconstruction (distortion is in curve space)
def compute_rd_correlation(curves, pcs, recon_kind, dist_kind):
    counter = 0
    d = np.zeros((len(curves), pcs.n_components_))
    for xy in curves:
        for npcs in range(pcs.n_components_):  
            recon = reconstruction(xy, pcs, npcs, kind=recon_kind)
            d[counter, npcs] = distortion(recon, xy, kind=dist_kind) 
        counter +=1 
    return d

# computes rate distortion profile using pc reconstruction (distortion is in intersection space)
def compute_rd_isecs(curves, pcs, probes, recon_kind, dist_kind):
    counter = 0
    d = np.zeros((len(curves), pcs.n_components_))
    for xy in tqdm(curves):
        true_isecs = get_distortion_profile(probes, xy)
        for npcs in range(pcs.n_components_):  
            recon = reconstruction(xy.flatten(order='F'), pcs, npcs, kind=recon_kind)
            recon_isecs = get_distortion_profile(probes, (reshape(recon)))
            d[counter, npcs] = distortion(recon_isecs, true_isecs, kind=dist_kind) 
        counter +=1 
    return d

# use polynomial curve fitting to smooth out the distortion profile
def smooth_distortions(dist, max_knot_number, poly_dim=2, normalize=False, xlabel='fraction of pcs', ylabel='error', title='', plot=True):
    smoothed_distortions = []
    x = np.arange(1,(2*max_knot_number)+1)/(2*max_knot_number)
    for i in range(0, max_knot_number-2):
        y = dist[i, :(max_knot_number*2)]
        p = np.poly1d(np.polyfit(x, y, poly_dim))
        smoothed_distortions.append(p(x))
    smoothed_distortions = np.array(smoothed_distortions)
    if plot:
        rd_plot(smoothed_distortions, max_knot_number, xlabel=xlabel, ylabel=ylabel, title=title)

    return smoothed_distortions

""" reconstruction methods """
# wrapper function for reconstructing curve with different number of pcs
def reconstruction(xy, pcs, npcs, mean_fill=False, kind='component'):
    # call reconstruction method
    if kind == 'component':
        return pca_component_recon(xy, pcs, npcs, mean_fill=mean_fill)
    elif kind == 'threshold':
        return pca_threshold_recon(xy, pcs, npcs, mean_fill=mean_fill)
    else:
        print(f'reconstruction method not found: {reconstruction}')
        return np.array([])
    
# pc reconstruction using component thresholding
def pca_component_recon(vec, pcs, npcs, mean_fill=False):
    proj = vec @ pcs.components_[:npcs,:].T 
    if mean_fill:  ##for missing PCs, used the mean projection value instead of '0'
        proj = np.concatenate((proj, pcs.all_projections_mean_[npcs:]))
    return proj  @ pcs.components_[:len(proj),:]

# pc reconstruction using threshold reconstruction
def pca_threshold_recon(vec, pcs, npcs, mean_fill=False):
    proj = vec @ pcs.components_.T
    indicies = np.argsort(np.absolute(proj))[-npcs:]
    return proj[indicies] @ pcs.components_[indicies]

""" distortion metrics """
# wrapper function for distortion functions
def distortion(x, y, kind):
    if kind == 'r2':
        return r2(x, y)
    elif kind == 'ssq':
        return ssq(x, y)
    elif kind == 'avg_se':
        return avg_se(x, y)
    elif kind == 'avg_ae':
        return avg_ae(x, y)
    else:
        print(f'distortion metric not found: {kind}')
        return -1

# average abs error
def avg_ae(x, y):
    return np.absolute(x-y).mean()

# average square error 
def avg_se(x, y):
    return ((x-y)**2).mean()

# sqrt (sum of squared error) 
def ssq(x, y):
    return np.sqrt(np.sum(x, y))

# 1 - pearson^2
def r2(x, y):
    return 1 - pearsonr(x, y)[0]**2

""" utils """
# reshapes the curves using fortran 
def reshape(XY):
    return np.reshape(XY, (int(len(XY)/2), 2), order='F')
