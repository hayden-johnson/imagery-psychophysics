import os
import sys
import math
import random
import pickle
import numpy as np
import pandas as pd 
import warnings
import matplotlib.pyplot as plt
from tqdm import trange
from tqdm import tqdm as log_progress
from sklearn.decomposition import PCA
from sklearn.metrics import auc
import pickle as pk
from curves.core import make_knot_grid, sample_knots, make_curve, check_intersections
from curves.visualization import pretty_picture_of_curve
from curves.rate_distortion import compute_rd_correlation
from curves.rate_distortion import reconstruction, distortion, reshape

def generate_target_curves(knot_grid, 
                           max_knot_number, 
                           curves_per_knot_number, 
                           output_path=None, 
                           save=False):
    
    ## generate a bunch of curves as data for PCA analysis
    total_curves = math.floor((max_knot_number-2))*curves_per_knot_number
    print(f'generating {total_curves} curves...', end=' ')
    
    counter = 0
    all_knots = []
    n_knots = np.ones(total_curves) *-1
    resolution_of_curves = max_knot_number * 10
    X,Y = np.zeros((total_curves, resolution_of_curves)), np.zeros((total_curves, resolution_of_curves))
    for num_knots in log_progress(range(2, max_knot_number, 1)):
        for c in range(curves_per_knot_number):
            knots = sample_knots(knot_grid, num_knots, closed=False)
            # need resolution of ~(10 * n_knots) to make a smooth curves
            xy,_ = make_curve(knots, resolution_of_curves, kind='quadra stic')
            X[counter, :] = xy[:,0]
            Y[counter, :] = xy[:,1]
            n_knots[counter] = num_knots
            all_knots.append(knots)
            counter+=1
            
    XY = np.concatenate((X,Y),axis = 1)
    
    if save and output_path != None:
        target_path = f'{output_path}/targets'
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        np.savetxt(f'{target_path}/all_target_curves.txt', XY, delimiter=',')
        with open('all_knots.pickle', 'wb') as handle:
            pickle.dump(all_knots, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    print('done')
    return XY, all_knots, total_curves

def fit_PCA(XY, max_knot_number=0, output_path=None, save=False):
    print('fitting PCA... ', end='')
    ## PCA on curves: should decay to zero at ~ 2 x max_knot_number
    pca = PCA()
    pca.fit(XY)
    all_projections = XY @ pca.components_.T
    all_projections_mean = np.mean(all_projections, axis=0)
    pca.all_projections_mean_ = all_projections_mean
    if save and output_path != None:
        pca_path = f'{output_path}/pca'
        if not os.path.exists(pca_path):
            os.mkdir(pca_path)
        pk.dump(pca, open(pca_path + '/pca.pkl', 'wb'))
    print('done')
    return pca

def compute_avg_distortions(XY, 
                            pca, 
                            total_curves,
                            curves_per_knot_number, 
                            max_knot_number, 
                            output_path=None):
    
    print('computing average distortion profiles... ')
    distortion_path = f'{output_path}/distortions'
    if not os.path.exists(distortion_path):
        print(f'making path: {distortion_path}')
        os.mkdir(distortion_path)
        
    batch_size = 100
    assert(total_curves % batch_size == 0), f'invalid batch size: ({total_curves}, {batch_size})'
    n_batches = int(total_curves / batch_size)
    #print(f'# of batches: {n_batches}')
    
    counter, batch_number = 0, 0
    d = np.zeros((batch_size, pca.n_components_))
    for idx in log_progress(range(total_curves)):
        xy = XY[idx,:]
        for npcs in range(pca.n_components_):     
            recon = reconstruction(xy, pca, npcs, mean_fill=True)
            recon_curve = reshape(recon)
            curve = reshape(xy)
            d[counter%batch_size, npcs] = distortion(recon, xy, kind='r2') 

        if (counter+1) % batch_size == 0:
            # save data and reset distortions
            #print(f'saving batch: {batch_number}')
            np.savetxt(f'{distortion_path}/distortions_{batch_number}.txt', d, delimiter=',')
            batch_number+=1
            d = np.zeros((batch_size, pca.n_components_))
            
        counter +=1
            
    distortions = np.concatenate([np.loadtxt(f'{distortion_path}/distortions_{i}.txt', delimiter=',') for i in range(n_batches)])
    distortions_sliced = np.array([distortions[start:start+curves_per_knot_number] for start in np.arange(0, total_curves, curves_per_knot_number)])
    #x = [i for i in range(0, total_curves, curves_per_knot_number)]
    #y = [i for i in range(curves_per_knot_number, total_curves+1, curves_per_knot_number)]
    #slices = list(zip(x, y))
    resolution_of_curves = 10*max_knot_number
    avg_distortions=np.zeros((total_curves, 2*resolution_of_curves))
    for i,s in enumerate(distortions_sliced):
        avg_distortions[i] = s.mean(axis=0)
    
    #for i, (start,stop) in enumerate(slices):
    #    avg_distortions[i] = distortions[start:stop].mean(axis=0)
    
    print(f'saving average distortions... ', end='')
    np.savetxt(f'{distortion_path}/avg_distortions.txt', avg_distortions, delimiter=',')
    
    print('done')
    return avg_distortions, distortions

                    
def select_curves_from_profile(curves, 
                               knots, 
                               distortions, 
                               avg_distortions, 
                               total_curves, 
                               curves_per_knot_number, 
                               curves_per_complexity, 
                               output_path=None,
                               save=False):
    curves_sliced = [curves[start:start+curves_per_knot_number] for start in np.arange(0, total_curves, curves_per_knot_number)]
    knots_sliced = [knots[start:start+curves_per_knot_number] for start in np.arange(0, total_curves, curves_per_knot_number)]
    distortions_sliced = [distortions[start:start+curves_per_knot_number] for start in np.arange(0, total_curves, curves_per_knot_number)]
    #curves_sliced = []
    #knots_sliced = []
    #distortions_sliced = []
    #x = [i for i in range(0, total_curves, curves_per_knot_number)]
    #y = [i for i in range(curves_per_knot_number, total_curves+1, curves_per_knot_number)]
    #slices = list(zip(x, y))
    #for i, (start,stop) in enumerate(slices):
    #    curves_sliced.append(curves[start:stop,:])
    #    knots_sliced.append(knots[start:stop])
    #    distortions_sliced.append(distortions[start:stop])

    min_indicies_per_slice = []
    avg_curves = []
    avg_knots = []
    avg_curve_distortions = []

    for curves_slice, knots_slice, distortions_slice, avg_distortions in zip(curves_sliced, knots_sliced, distortions_sliced, avg_distortions):
        diff = np.absolute(distortions_slice - avg_distortions).sum(axis=1)
        indicies,_ = get_n_lowest_values(diff, curves_per_complexity)
        avg_curves.append([curves_slice[x] for x in indicies])
        avg_knots.append([knots_slice[x] for x in indicies])
        avg_curve_distortions.append([distortions_slice[x] for x in indicies])
        min_indicies_per_slice.append(indicies)  

    if save and output_path != None:
        selections_path = f'{output_path}/selections'
        if not os.path.exists(selections_path):
            os.mkdir(selections_path)
        for i,(c,k,d) in enumerate(zip(avg_curves, avg_knots, avg_distortions)):
            np.savetxt(f'{selections_path}/{i}_average_targets.txt', c, delimiter=',')
            knots_flattened = np.array([knot_coords.flatten() for knot_coords in k])
            np.savetxt(f'{selections_path}/{i}_average_knots.txt', knots_flattened, delimiter=',')
            #np.savetxt(f'{selections_path}/average_curve_distortions.txt', d, delimiter=',')
        #with open('avg_knots.pickle', 'wb') as handle:
        #    pickle.dump(avg_knots, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return avg_curves, avg_knots, avg_distortions
                    
def get_n_lowest_values(ls, n):
    indicies = sorted(range(len(ls)), key=lambda sub: ls[sub])[:n]
    values = [ls[x] for x in indicies]
    return indicies, values
                    
def generate_curves(output_path, 
                    size_of_grid, 
                    max_knot_number, 
                    curves_per_knot_number, 
                    curves_per_complexity, 
                    save=False):
    print('generating curves...')
    ## create knot grid, generate target curves
    low, high  = -1, 1
    knot_grid = make_knot_grid(size_of_grid, low, high)
    XY, knots, total_curves = generate_target_curves(knot_grid,
                                                     max_knot_number, 
                                                     curves_per_knot_number,
                                                     output_path=output_path,
                                                     save=save)
    ## fit pca to generated curves
    pca = fit_PCA(XY, max_knot_number=max_knot_number, output_path=output_path, save=True)
    
    ## compute average distortion profiles 
    avg_distortions, distortions = compute_avg_distortions(XY, 
                                                           pca, 
                                                           total_curves,
                                                           curves_per_knot_number, 
                                                           max_knot_number,
                                                           output_path=output_path)

    avg_curves, knots, distortions = select_curves_from_profile(XY, 
                                                                knots, 
                                                                distortions,
                                                                avg_distortions, 
                                                                total_curves,
                                                                curves_per_knot_number, 
                                                                curves_per_complexity,
                                                                output_path=output_path,
                                                                save=save)
    return avg_curves, knots

## helpers
def shuffle(conditions_list):
    while(adjacent(get_target_ids(conditions_list))):
        random.shuffle(conditions_list)
    return conditions_list

def adjacent(ids):
    for i in range(len(ids)-1):
        if ids[i] == ids[i+1]:
            return True
    return False

def get_target_ids(conditions):
    return [df['target_id'].values[0] for df in conditions]

def generate_conditions(targets, 
                        probes, 
                        n_curves_per_complexity, 
                        n_repeats, 
                        n_breaks=2, 
                        target_presentation_rate=2, 
                        output_path=None, 
                        save=False):
    
    ## determine output path
    if save:
        path = f'{output_path}/resources'
        if not os.path.exists(path):
            print(f'making path: {path}')
            os.mkdir(path)

    # create conditions file
    conditions_list = []
    for i, target in enumerate(targets):
        # create mini-conditions for each target curve
        # the split them into 'blocks' according to some presentation rate
        target_id, probe_id = [],[]
        target_img, probe_img, isecs_img = [],[],[]
        n_isecs, isecs, target_complexity, probe_complexity = [],[],[],[]
        comparison, correct_resp = [], []
        for j, probe in enumerate(probes):
            ix, nix = check_intersections(target, probe)
            for r in range(n_repeats):
                target_id.append(i)
                probe_id.append(j)
                n_isecs.append(nix)
                isecs.append(ix)
                tc = (i // n_curves_per_complexity) + 2
                #comp.append(n + ((i + j) % 2))
                target_complexity.append((i // n_curves_per_complexity) + 2)
                probe_complexity.append(2) # all probes are complexity 2 
                # randomly sample comparison from bernouli
                #comp = nix + .5 if np.random.rand() > .5 else nix - .5
                if (i // n_curves_per_complexity) == 0:
                    comp = .5
                else:
                    if (i+j) % 2 == 1 and nix > 0:
                        comp = nix - .5
                    else:
                        comp = nix + .5
                
                comparison.append(comp)
                cr = 'right' if nix > comp else 'left'
                correct_resp.append(cr)
                target_img.append(f'./resources/curve_{i}_black.png')
                probe_img.append(f'./resources/curve_{j}_pink.png')
                isecs_img.append(f'./resources/isecs_{i}_{j}.png')

        conditions = {
            'target_file': target_img,
            'probe_file': probe_img,
            'n_isecs': n_isecs,
            'target_id': target_id,
            'probe_id': probe_id,
            'target_complexity': target_complexity,
            'probe_complexity': probe_complexity,
            'isecs_image': isecs_img,
            'comparator': comparison,
            'correct_resp': correct_resp
        }
        # shuffle each mini-batch 
        conditions_df = pd.DataFrame(data=conditions).sample(frac=1).reset_index(drop=True)
        # make sure probes are not adjacent to each other
        while(adjacent(conditions_df.probe_id.values)):
            conditions_df = conditions_df.sample(frac=1).reset_index(drop=True)
        # split into blocks at some presentation rate
        split_dfs=[]
        for i in range(0, len(conditions_df), target_presentation_rate):
            split_dfs.append(conditions_df.iloc[i:i+target_presentation_rate,:])
        for df in split_dfs:
            conditions_list.append(df)
            
    conditions_list = shuffle(conditions_list)
    df = pd.concat(conditions_list).reset_index(drop=True)
    
    n_trials = len(df)
    break_rate = int(n_trials / (n_breaks+1))
    show_break = [1 if (i+1)%break_rate==0 else 0 for i in range(len(df))]
    show_target = [1 if i%target_presentation_rate==0 else 0 for i in range(len(df))]
    show_break[-1] = 0
    df['show_break'] = show_break
    df['show_target'] = show_target
    
    if save:
        df.to_csv(path + '/conditions_df.csv', index=False)
    
    return df


def save_stimuli(targets, probes, path):
    for i,t in enumerate(targets):
        pretty_picture_of_curve(t, color='black', write_to=f'{path}/curve_{i}_black.png', show=False)

    for i,p in enumerate(probes):
        pretty_picture_of_curve(p, color='deeppink', write_to=f'{path}/curve_{i}_pink.png', show=False, ref_point_color='red')
    
    #pretty_picture_of_curve(targets, color=['black' for _ in range(len(targets))], write_to=f'{path}/plots/all_curves.png', show=False)
    for i, c1 in enumerate(targets):
        for j, c2 in enumerate(probes):
            if i != j:
                isecs, n_isecs = check_intersections(c1, c2) 
                pretty_picture_of_curve([c1,c2], color=['black', 'deeppink'], write_to=f'{path}/isecs_{i}_{j}.png', isecs=isecs, ref_point_color='red', show=False)
                
    pretty_picture_of_curve([], write_to=path + '/blank.png', show=False) 
    pretty_picture_of_curve([], ref_point_color='red', write_to=path + '/blank_red.png', show=False)


warnings.filterwarnings('ignore')
if __name__ == '__main__':
    ## select output folder
    if len(sys.argv) > 1:
        # is version is provided, use as output path
        version = sys.argv[1]
        output_path = f'../stimuli/{version}'
    else:
        # if version is not provied, find unqiue name
        i = 0
        output_path = f'../stimuli/UNTITLED_{i}'
        while os.path.exists(output_path):
            i +=1
            output_path = f'../stimuli/UNTITLED_{i}'
            
    ## create path if it doesn't exist
    if not os.path.exists(output_path):
        print(f'making path: {output_path}... ', end='')
        os.mkdir(output_path)
        print('done')

    ## generate stimuli
    generate_curves(output_path, 10, 30, 500, 20, save=True)
