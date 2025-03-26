import os 
import sys
import math
import numpy as np 
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist, pdist
from curves.SweepIntersectorLib.SweepIntersector import SweepIntersector
from curves.visualization import * 

# make a grid over the interval X,Y in the range [low, high], shape=(size^2, 2)
def make_knot_grid(size, low, high):
    X,Y = np.meshgrid(np.linspace(low, high, num=size),np.linspace(low, high, num=size))
    grid = np.zeros((size**2, 2))
    grid[:,0] = X.ravel()
    grid[:,1] = Y.ravel()
    return grid

#interpolate quadratic curves between selected knots
def make_curve(knots, resolution=500, kind='quadratic', clip_pct=0):  
    num_knots = knots.shape[0] 
    if num_knots < 3:
        kind = 'linear'
    plot_range = np.linspace(0,num_knots-1, num = resolution)
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

## sample from the domain st. items that have been sampled more have a lower prob
## goal: every point is sampled t times
def sample_domain(domain, counts, complexity, t, n_samples, eps=1e-4):
    indicies = np.arange(len(domain))
    #probs = (t - counts) + eps
    probs = np.ones(len(domain))
    probs /= probs.sum() # normalize
    selection = [np.random.choice(indicies, complexity, replace=False, p=probs) for _ in range(n_samples)]
    sample = []
    for s in selection:
        sample.append(np.array([domain[i] for i in s]))
    return sample

def calculate_angle(p1, p2, p3):
    # Compute the lengths of the sides of the triangle
    a = math.sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2)
    b = math.sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2)
    c = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) 
    # Use the law of cosines to compute the angle at p2
    cos_angle = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    angle_rad = math.acos(max(min(cos_angle, 1), -1))  # Clamp the value between -1 and 1
    # Convert the angle to degrees
    angle_deg = math.degrees(angle_rad)
    return angle_deg

## return true if constraint in violated 
def check_sequence(x_set, y, too_sharp=10):
    # y_endpoints = [y[0], y[-1]]
    #if len(x_set) == 1:
    #    print(f'y_endpoints: {y_endpoints}')
    #    print(f'x[0]: {x_set[0][0]}')

    ## check the angle between knots is not too small 
    ## prevent 'hairpin' curves 
    if len(y) > 3:
        for i in range(1, len(y)-1, 1):
            if calculate_angle(y[i-1], y[i], y[i+1]) < too_sharp:
                return True

    ## see if y shares a sequence with any of the other curves already in the set 
    for x in x_set:
        x_seq = [x[i:i+2].tolist() for i in range(len(x)-1)]
        y_seq = [y[i:i+2].tolist() for i in range(len(y)-1)]
        # does x share a sequence with y 
        if np.array([s in y_seq for s in x_seq]).any():
            return True
    return False

## creates curves, check for intersections, updates isecs dist
## checks to sees if constraints on isecs are violated
def update_isec_dist(total_isec_dist, curve_set, new_knots, too_close, min_isecs, max_isecs):
    constraint_violated=False
    new_isec_dist = deepcopy(total_isec_dist)
    comp1 = len(new_knots) # complexity of curve 1
    c1, _ = make_curve(new_knots, resolution=500)
    all_isecs=[]
    # find intersections with every curve in curve set
    try:
        for k, c2 in curve_set:
            comp2 = len(k) # complexity of curve 2
            isecs,n_isecs = check_intersections(c1, c2, new_knots, k)
            #c1 = edit_endpoints(c2, c1, isecs)
            new_isec_dist[comp1][n_isecs]+=1
            new_isec_dist[comp2][n_isecs]+=1
            # check to see if constraints have been violated between curves
            # eg. isecs too close, isecs too close to endpoints, endpoints too close
            all_isecs.append(isecs)
            constraint_violated, msg = check_isecs_endpoints(isecs, c1, c2, too_close, min_isecs, max_isecs)
            if constraint_violated:
                return total_isec_dist, c1, [], True, msg 
        #constraint_violated = any(constraints)
        return new_isec_dist, c1, all_isecs, constraint_violated, 'N/A'
    except Exception as e:
        print(f'exception: {e}')
        return total_isec_dist, c1, [], True, 'exception'


# return true is constraints are violated 
def check_isecs_endpoints(isecs, c1, c2, too_close, min_isecs, max_isecs):
    if isinstance(isecs, float) and np.isnan(isecs):
        return True
    if isecs.size > 0:
        const1 = isecs_too_close(isecs, too_close)
        const2 = endpoints_isecs_too_close(isecs, c1, c2, too_close)
    else:
        const1=False
        const2=False
    const3 = isecs.size < min_isecs
    const4 = isecs.size > max_isecs
    const5 = endpoints_too_close(c1, c2, too_close)
    msg = ''
    if const1:
        msg += ' | isecs_too_close'
    if const2:
        msg += ' | endpoints_isecs_too_close'
    if const3:
        msg += f' | min isecs' 
    if const4:
        msg += f' | max isecs' 
    if const5:
        msg += ' | endpoint_too_close'
    return any([const1, const2, const3, const4, const5]), msg


def isecs_too_close(isecs, too_close):
    x,_=isecs.shape
    if x < 2:
        return False
    #print(isecs)
    #print(pdist(isecs, metric='euclidean')) 
    return min(pdist(isecs, metric='euclidean')) < too_close

# checks to see if endpoints of onc curve are too close to the other 
def endpoints_too_close(c1, c2, too_close):
    # check to if c1 endpoints are close the c2
    curves = [c1, c2]
    endpoints = [(c2[0], c2[-1]), (c1[0], c1[-1])]
    for c, points in zip(curves, endpoints):
        for p in points:
            dist = cdist([p], c, metric='euclidean').squeeze().min()
            if(dist < too_close):
                #print(f'too close: {dist}, {too_close}')
                return True
    return False

def endpoints_isecs_too_close(isecs, c1, c2, too_close):
    if isecs.size > 0:
        endpoints = np.array([c2[0], c2[-1], c1[0], c1[-1]])
        dist = cdist(endpoints, isecs, metric='euclidean').squeeze().min()
        if dist < too_close:
            #print(f'too close: {dist}, {too_close}')
            return True
    return False


def check_intersections(t_curve, p_curve, t_knots, p_knots):
    t_p_intersections = find_intersections(t_curve, p_curve)
    #print(t_p_intersections)
    num_isecs = len(t_p_intersections)*t_p_intersections.any()
    assert np.issubdtype(num_isecs, np.integer) , f'expected type(num_isec) == int, recieved({type(num_isecs)})'
    if num_isecs > 0:
        assert t_p_intersections.shape == (num_isecs, 2), f'expected shape(t_p_intersections)==(num_isecs, 2), got {t_p_intersections.shape}'
    #except Exception as e:
    #    print(f'exception: {e}')
    #    return np.array([]), 0

    ## sweep algorithm has trouble finding intersection when the curves shares knots
    ## here we manuallly add shared knots as as intersection
    #print('before loop')
    for k1 in t_knots[1:-1]:
        for k2 in p_knots[1:-1]:
            if np.all(k1 == k2):
                if num_isecs == 0:
                    t_p_intersections = np.array([k1])
                else:
                    #print(f'shape 1: {t_p_intersections.shape}')
                    #print(f'shape 2: {k1.shape}')
                    t_p_intersections = np.vstack([t_p_intersections, k1])
                num_isecs +=1 
    #print('after loop')
    return t_p_intersections, num_isecs

    #return np.array([]), 0

"""
def check_intersections(t_curve, p_curve, t_knots, p_knots):
    try:
        t_p_intersections = strip(find_intersections(t_curve, p_curve))
        num_isecs = len(t_p_intersections)*t_p_intersections.any()
    except Exception as e:
        print(f'exception: {e}')
        return np.array([]), 0
    if type(num_isecs) == list:
        return np.nan, np.nan
    elif num_isecs and (t_p_intersections.shape != (num_isecs, 2)):
        return np.nan, np.nan
    else:
        #TODO: fix self-intersection in find_intersections and remove this
        # if isecs occurs multiple time, probably referes to self intersection
        # delete the isec and update the n_isecs
        isec_list = t_p_intersections.tolist()
        count_dict = {i: isec_list.count(isec_list[i]) for i in range(len(isec_list))}
        for k, v in count_dict.items():
            if v > 1:
                try: 
                    t_p_intersections = np.delete(t_p_intersections, isec_list[k])
                    num_isecs -= v
                except:
                    pass
        return t_p_intersections, num_isecs
"""

# helper for re-formatting the output of this function
def strip(a_dict):
    #print(a_dict)
    #print(f'isecs[1:-1]: {[isecs[1:-1] for seg,isecs in a_dict.items()]}')
    ix = []
    for _,isecs in a_dict.items():
        i = isecs[1:-1]
        #print(isecs)
        if len(i) > 1:
            for j in range(len(i)):
                ix.append(np.array(i[j]).squeeze())
        else:
            ix.append(np.array(i).squeeze())
        #print(isecs[1:-1], len(isecs[1:-1]))
    #ix = np.array([np.array(isecs[1:-1]).squeeze() for seg,isecs in a_dict.items()])
    #print(f'ix: {ix}')
    #if len(a_dict) > 0:
    #    assert ix.shape == (len(a_dict), 2), f'ix shape: {ix.shape}'
    #ix= [isects[1:-1] for seg,isects in a_dict.items()]
    #return np.array(np.array(ix).squeeze(),ndmin=2)
    #print(ix) 
    return np.array(ix)

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

## increment counts based on sampled points
def update_counts(points, counts, grid):
    x = lambda p, g: [1 if np.all(i==p) else 0 for i in g]
    update = np.array([x(p, grid) for p in points]).sum(axis=0)
    return counts + update

## TODO: make this more sophisticated
## simple divergence that return 0 is any number of curve has a set set of curves  > max isecs
def divergence(d, max_isecs=5, max_zeros=5):
    for _, v in d.items():
        #keys = list(v.keys())
        #if len(keys)>0 and max(keys) > max_isecs:
        #    return 0, 'divergence (isecs)'
        for x in range(4):
            if sum([di[x] for di in d.values()]) > max_zeros:
                return 0, 'divergence (zeros)'
    return 1, 'N/A'

"""
##if the endpoints of the probe are too close to the target curve, clip them
def edit_endpoints(A_curve, B_curve, A_B_intersections):
    clip_length = 10
    very_close_to_ix = .01
    ep_too_close_to_curve = .1
    
 
    not_an_intersection = lambda point: (cdist(point, A_B_intersections) > very_close_to_ix).all() if np.any(A_B_intersections) else True
    too_close = lambda point: (cdist(point,A_curve) < ep_too_close_to_curve).any()
    
    ##if the endpoints of B are far away from A_curve, do nothing
    if not too_close(B_curve[(0,-1),:]):
        return B_curve

    ##if the first point in B is not an intersection but is too close to A, clip that point
    first_point = B_curve[None,0, :]
    while not_an_intersection(first_point) and too_close(first_point):
        B_curve[:clip_length, :] = B_curve[clip_length, :]  ##we edit in-place
        clip_length += 1
        first_point = B_curve[None,0, :]
    
    ##if the last point in B is not an intersection but is too close to A, clip that point
    last_point  = B_curve[None,-1,:]
    while not_an_intersection(last_point) and too_close(last_point): 
        B_curve[-clip_length:, :] = B_curve[-clip_length, :] ##we edit in-place
        clip_length += 1
        last_point  = B_curve[None,-1,:]
    
    return B_curve
"""

def no_self_intersections(k):
    c,_ = make_curve(k)
    curve_seg_list = []
    ## creates a list of tuples [([X, Y], [X+1, Y+1]),..,] for all valid X, Y
    for vs, ve in zip(c[0:-1,:], c[1:,:]):
        curve_seg_list.append( (tuple(vs),tuple(ve)) )
        
    isector = SweepIntersector()
    try:
        self_intersections = isector.findIntersections(curve_seg_list)
        self_intersections = {key: self_intersections[key] for key in curve_seg_list if self_intersections[key]}
        return len(self_intersections) == 0
    except:
        return False
    

## save solution data / plots 
def save_solution(knots, M, isecs, isec_dist, grid_counts, d, path, key, max_isec):
    ## plotings ...
    print('saving solution....')
    print('grid counts:')
    print(grid_counts)
    #plot_counts(grid_counts, path, key)
    plot_counts_heatmap(grid_counts, d, path, key)
    plot_isec_dist(isec_dist, path, key, max_isec)
    #plot_isecs(isecs, path, key) ## TODO: fix isecs distribution plot
    n_curves = len(knots)
    complexities = [len(k) for k in knots]
    n_isecs_matrix, isecs_matrix, curve_set = plot_curves(knots, path, key)
    # save the curves to numpy file
    for i,(k,c) in enumerate(zip(knots, curve_set)):
        np.save(f'{path}/curves/curve_{i}.npy', c)
        np.save(f'{path}/curves/knots_{i}.npy', k)

    print(f'n_isecs_matrix: {n_isecs_matrix.shape}')
    print(n_isecs_matrix)
    #print(n_isecs_matrix.shape)
    plot_isecs_heatmap(isecs_matrix, path, key)
    plot_n_isecs_heatmap(n_isecs_matrix, path, key)

    ## save data to csv
    #np.save(grid_counts, f'{path}/{key}_counts.npy')
    counts_df = pd.DataFrame({'counts': grid_counts.ravel()})
    curves_df = pd.DataFrame({'knots': knots}) 
    counts_df.to_csv(f'{path}/{key}_counts.csv', header=False, index=False)
    curves_df.to_csv(f'{path}/{key}_knots.csv', header=False, index=False)
    # TODO: isecs df to csv 

    ## generate the conditions files
    queue_img, target_img, probe_img, visn_img, isecs_img = [],[],[],[],[]
    n_isecs, isecs, target_complexity, probe_complexity = [],[],[],[]
    unique_isec, unique_key = [],[]
    target_id, probe_id = [],[]
    # all combinations of curves except when i == j
    for i in range(n_curves):
        for j in range(n_curves):
            for m in range(M):
                if i != j:
                    queue_img.append(f'{path}/../queues/queue_{i}.png')
                    target_img.append(f'{path}/{key}_curve_{i}_black.png')
                    probe_img.append(f'{path}/{key}_curve_{j}_pink.png')
                    isecs_img.append(f'{path}/{key}_isecs_{i}_{j}.png')
                    visn_img.append(f'{path}/{key}_visn_{i}_{j}.png')
                    n_isecs.append(n_isecs_matrix[i][j])
                    isecs.append(isecs_matrix[i][j])
                    target_id.append(i)
                    probe_id.append(j)
                    target_complexity.append(complexities[i])
                    probe_complexity.append(complexities[j])
                    unique_isec.append(i<j and m==0) # key to help us extract unique intersections
                    unique_key.append(m==0)
    
    conditions = {
        'queue_file': queue_img,
        'target_file': target_img,
        'probe_file': probe_img,
        'visn_file': visn_img,
        'n_isecs': n_isecs,
        'target_id': target_id,
        'probe_id': probe_id,
        'target_complexity': target_complexity,
        'probe_complexity': probe_complexity,
        'unique_isecs' : unique_isec,
        'unique_key': unique_key 
    }
    conditions_df = pd.DataFrame(data=conditions)
    # shuffle order of conditions
    #shuffled_df = conditions_df.sample(frac=1, ignore_index=True)
    #shuffled_df.to_csv(f'{path}/{key}_conditions_df.csv', index=False)
    conditions_df.to_csv(f'{path}/{key}_conditions_df.csv', index=False)
    print('solution saved!')


## recursive CSP function
def backtrack(X, grid, complexity_range, t, w, n_samples, too_close, too_sharp, max_isecs, min_isecs, path):
    # dept = number of curves found
    depth = X['count']
    if (depth==X['total_n_curves']): ## base case
        print('------------------------------------')
        print(f'delta: {w}')
        key = np.random.randint(10000) # generate key, hope it's unique
        print(f'making directory: ./{key}')
        new_path = os.path.join(path, str(key))
        os.mkdir(new_path)
        os.mkdir(os.path.join(new_path, 'plots'))
        os.mkdir(os.path.join(new_path, 'isecs'))
        os.mkdir(os.path.join(new_path, 'curves'))
        save_solution(X['curve_knots'], X['M'], X['isecs'], X['isec_dist'], grid['counts'], grid['d'], new_path, key, max_isecs)
        print('------------------------------------')
        return X, grid, w
    
    ## recursive case
    scores, results, pruned_curve_sample = [], [], []
    c = complexity_range[depth] # get the complexity of the next curve
    ## sample potential curves, prune from set if sample knots violate constraints 
    while len(pruned_curve_sample) == 0:
        curve_sample = sample_domain(grid['points'], grid['counts'], c, t, n_samples)
        #n_samples = max([10, 2**(X['total_n_curves']-depth)])
        #print(f'n_samples: {n_samples}')
        #curve_sample = sample_domain(grid['points'], grid['counts'], c, t, n_samples) 
        for s in curve_sample:
            if check_sequence(X['curve_knots'], s, too_sharp=too_sharp)==False:
                #pruned_curve_sample.append(s) # valid curve
                if no_self_intersections(s):
                    pruned_curve_sample.append(s)
        print(f'pruned {len(curve_sample) - len(pruned_curve_sample)} curves')
        if len(pruned_curve_sample)==0:
            # no valid curves left in set, resample
            print('resampling curves')
    ## for all valid curves
    for v in pruned_curve_sample:
        # update the intersection distribution given with new curve
        new_isec_dist, curve, isecs, constraint_violated, msg_1 = update_isec_dist(X['isec_dist'], list(zip(X['curve_knots'], X['curve_points'])), v, too_close, min_isecs, max_isecs)
        delta, msg_2 = divergence(new_isec_dist, max_isecs=max_isecs)
        # if constraint on curve / intersections are violated, do not recurse on curve
        if constraint_violated or delta==0:
            print(f"recursion terminated at depth: {X['count']} {msg_1} | {msg_2}") 
        else:
            # compute factor to indicate how 'good' the new dist is
            # TODO: try using KL-Divergence from uniform as a score 
            X_copy = deepcopy(X)
            grid_copy = deepcopy(grid)
            X_copy['curve_knots'].append(v)
            X_copy['curve_points'].append(curve)
            X_copy['isec_dist'] = new_isec_dist
            X_copy['count'] +=1
            #if isecs.size!=0:
            X_copy['isecs'].append(isecs)
            grid_copy['counts'] = update_counts(v, grid_copy['counts'], grid['points'])
            ## recursive call
            x,g,w = backtrack(X_copy, grid_copy, complexity_range, t, delta*w, n_samples, too_close, too_sharp, max_isecs, min_isecs, path)
            ## track results
            scores.append(w)
            results.append((x,g,w))

    return [],[],0    
    ## return best result
    #print('search complete')
    #print(f'scores: {scores}')
    #if len(scores) > 0:
    #    max_index = scores.index(max(scores))
    #    return results[max_index]
    #return 0 

# d = dimension of the grid
# t = number of curves per level of complexity 
def generate_curves(d=5, t=5, complexity_range=[3,5,7], too_close=.1, too_sharp=10, n_samples=1000, max_isecs=5, min_isecs=0, path='../stimuli'):
    #print(f'generate: {too_close}')
    print(f'making directory {path}...')
    if not os.path.exists(path):
        os.mkdir(path)
    #complexity_range = np.array(complexity_range).repeat(t)
    isec_dist = {k: defaultdict(int) for k in complexity_range}
    X = {
        'curve_knots': [],
        'curve_points': [],
        'knots': [],
        'isecs': [],
        'isec_dist': isec_dist,
        'count': 0,
        'total_n_curves': len(complexity_range),
        'M':2 # number of repeated trials 
    }
    grid = {
        'points': make_knot_grid(d, -1, 1),
        'counts': np.zeros(d*d),
        'd': d
    }
    X, grid, delta = backtrack(X, grid, complexity_range, t, 1, n_samples, too_close, too_sharp, max_isecs=max_isecs, min_isecs=min_isecs, path=path)


if __name__ == '__main__':
    # create folder to store found stimuli 
    folder = sys.argv[1]
    cr = [3, 3, 4, 4, 5, 5] 
    # main generation function 
    generate_curves(d=10, t=100, complexity_range=cr, n_samples=100, too_close=.02, too_sharp=20, max_isecs=4, min_isecs=0, path=f'../stimuli/{folder}')
