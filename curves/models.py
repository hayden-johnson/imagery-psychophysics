import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from curves.core import make_curve
from curves.visualization import pretty_picture_of_curve_ax


class IsotropicGaussian(): 
    def __init__(self, knots, C, tau, a=None):
        if a == None:
            k = len(knots)
            a = np.sqrt(np.exp(2* (tau - (C/k) - (2*np.pi) -1)))
        
        self.sigma = np.array([[a, 0],[0, a]])
        self.knots = knots
        self.z = np.array([multivariate_normal(mu, self.sigma) for mu in knots])
                    
    def sample_curves(self, n=1, resolution=300, return_samples=False):
        
        samples = np.array([z.rvs(size=n) for z in self.z])
        z, n, d = samples.shape
        samples = np.transpose(samples, (1, 0, 2))
        curves = []
        for knots in samples:
            curve,_ = make_curve(knots, resolution=resolution)
            curves.append(curve)
        
        if return_samples:
            return curves, samples
        
        return curves
    
    def visualize(self, ax, plot_curve=True, alpha=1, write_to=False):
        curve,_ = make_curve(self.knots)
        x, y = np.mgrid[-1.5:1.5:.01, -1.5:1.5:.01]
        data = np.dstack((x, y))
        for rv in self.z:
            z = rv.pdf(data)
            ax.contour(x, y, z, 3, alpha=.3)
        if plot_curve:
            pretty_picture_of_curve_ax(ax, curve, alpha=alpha, write_to=write_to)
