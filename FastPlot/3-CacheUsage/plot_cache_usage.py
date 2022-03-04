from matplotlib import transforms
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from utils import MACHINE_CPU, NETWORKS
from utils import CACHE_SIZE

THREADS_POSITIONS = {
    '1': (0, 0),
    '2': (0, 1),
    '3': (0, 2),
    '4': (0, 3),
    '5': (1, 0),
    '6': (1, 1),
    '7': (1, 2),
    '8': (1, 3),
    # '9': (2, 0),
    # '10': (2, 1),
    # '11': (2, 2),
    # '12': (2, 3)
}

FONTSIZE = {
    'TITLE': 24,
    'SUB-TITLE': 20
}

SUBPLOTS_GRID = {
    4:  {'nrows': 1, 'ncols': 4, 'figsize': (20,5)},
    8:  {'nrows': 2, 'ncols': 4, 'figsize': (20,10)},
    12: {'nrows': 3, 'ncols': 4, 'figsize': (20, 15)}
}

def compute_layer_size(n_elements, element_bytes=None, measurement_unit=None):
    if type(n_elements) == list:
        n_elements = sum(n_elements) / len(n_elements)
    size = n_elements * element_bytes 
    if measurement_unit.lower() == 'b':
        return float(size)
    elif measurement_unit.lower() == 'kb':
        return float(size / 2**10)
    elif measurement_unit.lower() == 'mb':
        return float(size / 2**20)
    elif measurement_unit.lower() == 'gb':
        return float(size / 2**30)

def compute_network_sizes_SP_KB(network: dict):
    network_sizes = {}
    for layer_id, layer_dict in network.items():
        network_sizes[layer_id] = {}
        for tensor_type, tensor_dims in layer_dict.items():
                network_sizes[layer_id][tensor_type] = compute_layer_size(tensor_dims, element_bytes=4, measurement_unit='KB')
    return network_sizes

def plot_network_layers(network: dict, ax, n_threds_title=None):
    df = pd.DataFrame(network)
    df.T.plot(ax=ax, kind='line', rot=0, colormap='tab20c', zorder=3, marker='o')
    ax.set_ylabel('KiloBytes (KB)')
    ax.set_title(n_threds_title+' Threads')
    for axis in [ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    return ax

def plot_cache_sizes(cache_size: dict, ax):
    for cache_level, cache_info in cache_size.items():
        ax.axhline(cache_info['size'], label=cache_level, color=cache_info['color'], linestyle='dashed')
    ax.legend(ncol=2, loc='best')
    return ax

if __name__=='__main__':
    SELECTED_MACHINE = 'BlackYeti'
    # SELECTED_MACHINE = 'Thor'

    nrows, ncols = SUBPLOTS_GRID[len(THREADS_POSITIONS.keys())]['nrows'], SUBPLOTS_GRID[len(THREADS_POSITIONS.keys())]['ncols']
    figsize = SUBPLOTS_GRID[len(THREADS_POSITIONS.keys())]['figsize']

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    SELECTED_NET = 'ALEXNET-Ho'

    for n_threads, pos in THREADS_POSITIONS.items():
        if (len(THREADS_POSITIONS.keys()) == 4): pos = pos[1]
        ax[pos] = plot_network_layers(compute_network_sizes_SP_KB(NETWORKS[SELECTED_NET][n_threads]), ax=ax[pos], n_threds_title=n_threads)

        ax[pos] = plot_cache_sizes(CACHE_SIZE[SELECTED_MACHINE], ax[pos])

        ax[pos].set_yscale('log')
        ax[pos].set_ylim(10**1, 4*10**4)

    plt.tight_layout()

    # Set title
    fig.subplots_adjust(top=0.85)
    title = 'Size (KB) of tensors in each layer of AlexNet, using 1 to {} threads. With respect to L1,L2,L3 cache levels'.format(len(THREADS_POSITIONS))
    plt.text(x=0.5, y=0.94, s=title, ha="center", transform=fig.transFigure, fontsize=FONTSIZE['TITLE'])
    plt.text(x=0.5, y=0.9, s=MACHINE_CPU[SELECTED_MACHINE], ha='center', transform=fig.transFigure, fontsize=FONTSIZE['SUB-TITLE'], color='grey')
    plt.savefig('results_'+SELECTED_NET+'_'+SELECTED_MACHINE+'.pdf')
