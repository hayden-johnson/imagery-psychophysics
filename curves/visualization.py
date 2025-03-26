import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from curves.core import make_curve, check_intersections

def rd_plot(d, max_knot_number, normalize_distortion=False, xlabel='fractions of pcs', ylabel='error', title='rd plot', ax=None):
    ## plotting
    if ax==None:
        fig,ax = plt.subplots(figsize=(8,8))
    cmap = cm.get_cmap('viridis_r',lut = max_knot_number-2)
    norm = colors.Normalize(2,max_knot_number)
    #x_range = np.arange(1,pca.n_components_+1)/pca.n_components_
    x_range = np.arange(1,(2*max_knot_number)+1)/(2*max_knot_number)
    cp = cmap.colors#sns.color_palette("rocket", n_colors=len(curve_set_indices)) 
    for i,c in enumerate(cp):
        if normalize_distortion:
            max_distortion = max(d[i,:max_knot_number*2])
            ax.plot(x_range, d[i,:max_knot_number*2] / max_distortion, color=c)
        else:
            ax.plot(x_range, d[i,:max_knot_number*2], color=c)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    #fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.arange(2,max_knot_number+1,2),label='num. knot points, smooth curves', fraction=0.045)
    #ax.legend(loc = 'upper right')
    #plt.show()


def plot_counts_heatmap(counts, d, path, key):
    # should make sure this is correct
    counts = counts.reshape(d,d)
    heatmap = sns.heatmap(counts, linecolor='white', linewidths=.1)
    fig = heatmap.get_figure()
    fig.savefig(f'{path}/plots/{key}_counts_heatmap.png')
    plt.close(fig=fig)

def plot_isecs_heatmap(isecs_matrix, path, key):
    pass

def plot_n_isecs_heatmap(n_isecs_matrix, path, key, savefig=True):
    heatmap = sns.heatmap(n_isecs_matrix, linecolor='white', linewidths=.1, vmin=0, vmax=int(n_isecs_matrix.max()))
    fig = heatmap.get_figure()
    if savefig:
        fig.savefig = (f'{path}/plots/{key}_n_isecs_heatmap.png')
    plt.close(fig=fig)

def plot_isecs(isecs, path, key):
    points=[]
    for s in isecs:
        for i in s:
            if i.size !=0:
                points.append(i)
    points = np.array(points)
    pretty_picture_of_points(points, write_to=f'{path}/plots/{key}_all_isecs.png')

def pretty_picture_of_points(points, xlim=[-1.5, 1.5], ylim=[-1.75, 1.75], write_to=False, dpi=300, show=True, plot_ref_points=True):
    if not show:
        plt.ioff() 
    fig = plt.figure(figsize=(8,8))
    plt.gca().set_xlim(xlim[0],xlim[1])
    plt.gca().set_ylim(ylim[0],ylim[1])
    if plot_ref_points:
        plt.plot(0, 0, markersize=20, marker='o', c='lightblue', alpha=.5)
    plt.scatter(points[:,0], points[:,1], c='y', s=50, alpha=.5)
    if write_to:
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, 
            hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.savefig(write_to, dpi=dpi, transparent = True,bbox_inches = 'tight')#, pad_inches = 0)
        plt.savefig(write_to, dpi=dpi, edgecolor= 'red',bbox_inches = 'tight')#, pad_inches = 0)

    plt.close(fig) 


from matplotlib.offsetbox import OffsetImage, AnnotationBbox
def getImage(path):
    return OffsetImage(plt.imread(path), zoom=.35, alpha=.85)

## curve plotting utility
def pretty_picture_of_curve(curve,
                            xlim=[-1.6, 1.6],
                            ylim=[-1.6, 1.6], 
                            write_to=False, 
                            dpi=500, 
                            color='black',
                            width=3, 
                            show = True, 
                            isecs=np.array([]), 
                            plot_ref_points=True, 
                            ref_point_color='lightblue', 
                            plot_isecs=True, 
                            plot_marker=False, 
                            marker=''
                           ):
    if not show:
        plt.ioff()
    if type(curve) != list:
        curve = [curve]
    if type(color) != list:
        color = [color]
        
    fig,ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if plot_ref_points:
        plt.plot(0, 0, markersize=20, marker='o', c=ref_point_color, alpha=.5)
    if plot_marker:
        ab = AnnotationBbox(getImage(marker), (0, 0), frameon=False)
        ax.add_artist(ab)     
    for c,hue in zip(curve,color):  ##if multiple curves given, will superimpose them on the plot
        _=ax.plot(c[:,0], c[:,1], hue,linewidth=width)
        plt.gca().set_xlim(xlim[0],xlim[1])
        plt.gca().set_ylim(ylim[0],ylim[1])
        # plot intersetions
        if plot_isecs and isecs.any():
            plt.plot(isecs[:,0], isecs[:,1], 'y.', markersize=25, alpha=.5)
        # plot for orientation 

        #plt.scatter(knots[:,0], knots[:,1], s=10, marker='o', color='r', alpha=.5)
    if write_to:
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.savefig(write_to, dpi=dpi, transparent = True,bbox_inches = 'tight')#, pad_inches = 0)
        plt.savefig(write_to, dpi=dpi, edgecolor= 'red',bbox_inches = 'tight',facecolor='white', transparent=False)#, pad_inches = 0)
    if not show:
        plt.close(fig)

## curve plotting utility
def pretty_picture_of_curve_ax(ax,
                            curve,
                            xlim=[-1.5, 1.5],
                            ylim=[-1.75, 1.75], 
                            write_to=False, 
                            dpi=300, 
                            color='black',
                            width=3, 
                            show = True, 
                            isecs=np.array([]), 
                            plot_ref_points=False, 
                            ref_point_color='lightblue', 
                            plot_isecs=True, 
                            plot_marker=False, 
                            marker='',
                            alpha=1
                            ):

    if not show:
        plt.ioff()
    if type(curve) != list:
        curve = [curve]
    if type(color) != list:
        color = [color]
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if plot_ref_points:
        plt.plot(0, 0, markersize=20, marker='o', c=ref_point_color, alpha=.5)
    if plot_marker:
        ab = AnnotationBbox(getImage(marker), (0, 0), frameon=False)
        ax.add_artist(ab)     
    for c,hue in zip(curve,color):  ##if multiple curves given, will superimpose them on the plot
        _=ax.plot(c[:,0], c[:,1], hue,linewidth=width, alpha=alpha)
        plt.gca().set_xlim(xlim[0],xlim[1])
        plt.gca().set_ylim(ylim[0],ylim[1])
        # plot intersetions
        if plot_isecs and isecs.any():
            plt.plot(isecs[:,0], isecs[:,1], 'y.', markersize=25, alpha=.5)
        # plot for orientation 

        #plt.scatter(knots[:,0], knots[:,1], s=10, marker='o', color='r', alpha=.5)
    if write_to:
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.savefig(write_to, dpi=dpi, transparent = True,bbox_inches = 'tight')#, pad_inches = 0)
        plt.savefig(write_to, dpi=dpi, edgecolor= 'red',bbox_inches = 'tight')#, pad_inches = 0)

## plot the number of times each points in grid has been sampled
def plot_counts(counts, path, key):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(counts)), counts, color='royalblue')
    ax.set_ylim([-1, max(counts) + 1])
    ax.set_xlabel('knot')
    ax.set_ylabel('samples')
    fig.savefig(f'{path}/plots/{key}_counts.png')
    plt.close(fig=fig)
    
## plot the distribution of intersections per level of complexity
def plot_isec_dist(total_isec_dist, path, key, max_isecs):
    fig, ax = plt.subplots()
    keys = list(total_isec_dist.keys())
    ds = list(total_isec_dist.values())
    #m = max([max(d.values()) for d in ds])
    x = np.arange(max_isecs+1)
    Y = []
    #print(f'keys: {keys}')
    #print(f'values: {ds}')
    for d in ds:
        Y.append([d[i] for i in x])
    #print(f'len(Y): {len(Y)}')
    #print(Y)
    ax.bar(x - 0.2, Y[0], 0.2, label = f'C = {keys[0]}', color='lightblue')
    #ax.bar(x + 0, Y[1], 0.2, label = f'C = {keys[1]}', color='royalblue')
    #ax.bar(x + 0.2, Y[2], 0.2, label = f'C = {keys[2]}', color='darkblue')
    ax.set_xticks(x)
    ax.set_xlabel("isecs")
    ax.set_ylabel("count")
    ax.legend()
    fig.savefig(f'{path}/plots/{key}_isecs.png')
    plt.close(fig=fig)
    
def plot_curves(knots, path, key):
    n_curves = len(knots)
    curve_set = []
    # for every set of knots, make a curve and plot
    for i, k in enumerate(knots):
        curve, _ = make_curve(k, resolution=500)
        pretty_picture_of_curve(curve, knots[i], color='black', write_to=f'{path}/{key}_curve_{i}_black.png', show=False)
        pretty_picture_of_curve(curve, knots[i], color='deeppink', write_to=f'{path}/{key}_curve_{i}_pink.png', show=False)
        curve_set.append(curve)
    # for every combination of curves, plot the intersection
    n_isec_matrix = [[-1]*n_curves for _ in range(n_curves)]
    isecs_matrix = [[-100, -100]*n_curves for _ in range(n_curves)]
    pretty_picture_of_curve(curve_set, knots, color=['black' for _ in range(len(curve_set))], write_to=f'{path}/plots/{key}_all_curves.png', show=False)
    for i, c1 in enumerate(curve_set):
        #print('')
        for j, c2 in enumerate(curve_set):
            if i != j:
                pretty_picture_of_curve([c1,c2],[knots[i], knots[j]],  color=['black', 'deeppink'], write_to=f'{path}/{key}_visn_{i}_{j}.png', show=False, plot_isecs=False)
                isecs, n_isecs = check_intersections(c1, c2, knots[i], knots[j])
                #print(f' {n_isecs}', end="")
                #print(f'i: {i}, j:{j}')
                isecs_matrix[i][j] = isecs
                n_isec_matrix[i][j] = n_isecs 
                if i > j:
                    pretty_picture_of_curve([c1,c2],[knots[i], knots[j]],  color=['black', 'deeppink'], write_to=f'{path}/isecs/{key}_isecs_{i}_{j}_nisecs_{n_isecs}.png', isecs=isecs, show=False)
                    #isecs_matrix[i][j] = isecs
                    #n_isec_matrix[i][j] = n_isecs
    #print(np.array(n_isec_matrix))
    return np.array(n_isec_matrix), isecs_matrix, curve_set
    