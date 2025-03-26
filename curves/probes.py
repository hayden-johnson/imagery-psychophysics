import numpy as np 
from math import sqrt 
from curves.core import make_knot_grid, make_curve
from curves.visualization import pretty_picture_of_curve

""" methods for generating probe sets """
def generate_simple_probe_set(grid_size, low=-1, high=1, res=2, plot=True, border=True):
    probe_set = []
    knot_grid = make_knot_grid(grid_size, low, high)

    # generate vertical lines
    start = 0
    stop = grid_size
    if border==False:
        start +=1
        stop -=1
    for i in range(start, stop):
        knots = np.array([knot_grid[i], knot_grid[i+(grid_size*(grid_size-1))]])
        probe,_ = make_curve(knots, resolution=res)
        probe_set.append(probe)
        
    # generate horizontal  lines
    start = 0
    stop = grid_size*grid_size
    if border==False:
        start += grid_size
        stop -= grid_size
    for i in range(start, stop, grid_size):
        knots = np.array([knot_grid[i], knot_grid[i+grid_size-1]])
        probe,_ = make_curve(knots, resolution=res)
        probe_set.append(probe)
        
    # generate diag 1 
    knots = np.array([knot_grid[0], knot_grid[grid_size*grid_size-1]])
    probe,_ = make_curve(knots, resolution=res)
    probe_set.append(probe)

    for i in range(1, grid_size-1, 1):
        knots = np.array([knot_grid[i*grid_size], knot_grid[grid_size*grid_size-1-i]])
        probe,_ = make_curve(knots, resolution=res)
        probe_set.append(probe)
        
        knots = np.array([knot_grid[i], knot_grid[grid_size*(grid_size-i)-1]])
        probe,_ = make_curve(knots, resolution=res)
        probe_set.append(probe)
        
    # generate diag 2
    knots = np.array([knot_grid[grid_size-1], knot_grid[grid_size*(grid_size-1)]])
    probe,_ = make_curve(knots, resolution=res)
    probe_set.append(probe)

    for i in range(1, grid_size-1, 1):
        knots = np.array([knot_grid[i], knot_grid[i*grid_size]])
        probe,_ = make_curve(knots, resolution=res)
        probe_set.append(probe)
        
        knots = np.array([knot_grid[grid_size*(grid_size-i)-1], knot_grid[grid_size*grid_size-i-1]])
        probe,_ = make_curve(knots, resolution=res)
        probe_set.append(probe)
        
    # plot curves   
    if plot:
        colors = ['deeppink']* len(probe_set)
        pretty_picture_of_curve(probe_set, color=colors, xlim=[-1.33, 1.33], ylim=[-1.33, 1.33])

    return probe_set


def generate_custom_probe_set(res=2, low=-1, high=1, plot=True):
    probe_set = []
    # generate vertical lines
    step = (high - low) / res
    print(step)
    #for i in range(low+step, high, step):
    for i in np.arange(low+step, high, step):
        knots = np.array([[i ,low], [i, high]])
        probe,_ = make_curve(knots, resolution=res)
        probe_set.append(probe)

    # generate horizontal lines
    step = (high - low) / res
    for i in np.arange(low+step, high, step):
        knots = np.array([[low, i], [high, i]])
        probe,_ = make_curve(knots, resolution=res)
        probe_set.append(probe)
        
    # generate manual probes (for now)
    knots = np.array([
                    [[-1+sqrt(2), 1], [-1, 1-sqrt(2)]],
                    [[1, -1+sqrt(2)], [1-sqrt(2), -1]],
                    [[1-sqrt(2), 1], [1, 1-sqrt(2)]],
                    [[-1+sqrt(2), -1], [-1, -1+sqrt(2)]]     
                     ])
    for k in knots:
        probe,_ = make_curve(k, resolution=res)
        probe_set.append(probe)
        
    if plot:
        colors = ['deeppink']* len(probe_set)
        pretty_picture_of_curve(probe_set, color=colors, xlim=[-1.33, 1.33], ylim=[-1.33, 1.33])
    
    return probe_set


def generate_custom_probe_set_ext(res, low=-1, high=1, plot=True, ext=0):
    probe_set = []
    # generate vertical lines
    step = (high - low) / res
    for i in np.arange(low+step, high-.0001, step):
        knots = np.array([[i ,low-ext], [i, high+ext]])
        probe,_ = make_curve(knots, resolution=res)
        probe_set.append(probe)

    # generate horizontal lines
    step = (high - low) / res
    for i in np.arange(low+step, high-.0001, step):
        knots = np.array([[low-ext, i], [high+ext, i]])
        probe,_ = make_curve(knots, resolution=res)
        probe_set.append(probe)
        
    # generate manual probes (for now)
    s = 2 - sqrt(2)
    knots = np.array([
                     [[-1+s-ext, 1+ext], [ 1+ext, -1+s-ext]],
                     [[-1-ext, 1-s+ext], [ 1-s+ext, -1-ext]],       
                     [[1-s+ext,  1+ext], [-1-ext, -1+s-ext]],                      
                     [[1+ext,  1-s+ext], [-1+s-ext, -1-ext]],  
                     ])
    for k in knots:
        probe,_ = make_curve(k, resolution=res)
        probe_set.append(probe)
        
               
    if plot:
        colors = ['deeppink']* len(probe_set)
        pretty_picture_of_curve(probe_set, color=colors, xlim=[-1.33, 1.33], ylim=[-1.33, 1.33])
    
    return probe_set

