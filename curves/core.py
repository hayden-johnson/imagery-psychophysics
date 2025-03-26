import numpy as np
from scipy.interpolate import interp1d
from curves.SweepIntersectorLib.SweepIntersector import SweepIntersector

""" core utilities for building curves / knot grids """
# make a grid over the interval X,Y in the range [low, high], shape=(size^2, 2)
def make_knot_grid(size, low, high):
    X,Y = np.meshgrid(np.linspace(low, high, num=size),np.linspace(low, high, num=size))
    grid = np.zeros((size**2, 2))
    grid[:,0] = X.ravel()
    grid[:,1] = Y.ravel()
    return grid

# sample the knots from grid 
def sample_knots(knot_grid, number_of_knot_points = False, closed = False):
    grid_size = knot_grid.shape[0]
    if not number_of_knot_points:
        nkp = knot_grid.shape[0]
    else:
        nkp = number_of_knot_points
        
    dice = np.random.choice(range(grid_size), nkp,replace=nkp > grid_size)
    if closed:
        dice[-1] = dice[0]
    return knot_grid[dice,:]

# interpolate quadratic curves between selected knots
def make_curve(knots, resolution=300, kind='quadratic', clip_pct=0):  
    num_knots = knots.shape[0] 
    if num_knots < 3:
        kind = 'linear'
        #resolution = 2
    else:
        kind = 'quadratic'
        #resolution = 10 * num_knots
    plot_range = np.linspace(0,num_knots-1, num=resolution)
    fx = interp1d(range(num_knots), knots[:,0], kind=kind)
    fy = interp1d(range(num_knots), knots[:,1], kind=kind)
    xy = np.zeros((len(plot_range),2))
    xy[:,0] = fx(plot_range)
    xy[:,1] = fy(plot_range)
    # clipping the end of the curve so that endpoints dont intersect
    if clip_pct > 0:
        clip = int(clip_pct*resolution)
        return xy[clip:-clip], knots
    return xy, knots

# checks the number of knots intersections between two curves, calls find_intersections
def check_intersections(t_curve, p_curve, t_knots=[], p_knots=[]):
    t_p_intersections = find_intersections(t_curve, p_curve)
    num_isecs = len(t_p_intersections)*t_p_intersections.any()
    assert np.issubdtype(num_isecs, np.integer) , \
        f'expected type(num_isec) == int, recieved({type(num_isecs)})'
    if num_isecs > 0:
        assert t_p_intersections.shape == (num_isecs, 2), \
            f'expected shape(t_p_intersections)==(num_isecs, 2), got {t_p_intersections.shape}'

    ## sweep algorithm has trouble finding intersection when the curves shares knots
    ## here we manuallly add shared knots as as intersection
    if len(t_knots) > 0 and len(p_knots) > 1:
        for k1 in t_knots[1:-1]:
            for k2 in p_knots[1:-1]:
                if np.all(k1 == k2):
                    if num_isecs == 0:
                        t_p_intersections = np.array([k1])
                    else:
                        t_p_intersections = np.vstack([t_p_intersections, k1])
                    num_isecs +=1 
    return t_p_intersections, num_isecs


# helper for re-formatting the output of this function
def strip(a_dict):
    ix = []
    for _,isecs in a_dict.items():
        i = isecs[1:-1]
        if len(i) > 1:
            for j in range(len(i)):
                ix.append(np.array(i[j]).squeeze())
        else:
            ix.append(np.array(i).squeeze())
    return np.array(ix)

# uses SweepIntersectorLib to find the coordinates of intersections between 2 curves
def find_intersections(target, probe, self_intersections=False):
    ##package the xy points for target and probe. Each xy point and the point after it define a line segment
    target_seg_list = []
    probe_seg_list = []
    ## creates a list of tuples [([X, Y], [X+1, Y+1]),..,] for all valid X, Y
    for vs, ve in zip(target[0:-1,:], target[1:,:]):
        target_seg_list.append( (tuple(vs),tuple(ve)) )
    for vs, ve in zip(probe[0:-1,:], probe[1:,:]):
        probe_seg_list.append( (tuple(vs),tuple(ve)) )  ##the start/end points can be used as keys
    segList = target_seg_list+probe_seg_list
    # compute probe-target intersections
    isector = SweepIntersector()
    ##TODO: implement proper filtering for pair-wise intersections
    #filter for just the probe/target intersections--the code we use finds self-intersections too.
    ##this assumes that the probe has no self-intersections. If we want more complicated probes we'll have to deal 
    ##with this in a different way.
    # return list of tuples [((seg_start, seg_end),intersections),..,]
    probe_target_intersections = isector.findIntersections(segList)    
    probe_target_intersections = strip({key: probe_target_intersections[key] for key in probe_seg_list if probe_target_intersections[key]})
    # find probe intersections and remove
    isector = SweepIntersector()
    probe_self_intersections = isector.findIntersections(probe_seg_list)
    probe_self_intersections = strip({key: probe_self_intersections[key] for key in probe_seg_list if probe_self_intersections[key]})
    mask = [False if i in probe_self_intersections.tolist() else True for i in probe_target_intersections.tolist()]
    probe_target_intersections = probe_target_intersections[mask]
    #for v in probe_self_intersections:
        # remove self intersection from list
        #if v in probe_target_intersections:
        #    probe_target_intersections = np.delete(probe_target_intersections, v, axis=0)

    # compute target-self intersections. this is not efficient
    if self_intersections:
        isector = SweepIntersector()
        target_self_intersections = isector.findIntersections(target_seg_list)
        target_self_intersections = {key: target_self_intersections[key] for key in target_seg_list if target_self_intersections[key]}
        return probe_target_intersections, target_self_intersections
    else:
        return probe_target_intersections
    