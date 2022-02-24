from matplotlib import markers
import pandas as pd 
import matplotlib.pyplot as plt

from utils import NETWORKS
from utils import CACHE_SIZE

THREADS_POSITIONS = {
    '1': (0, 0),
    '2': (0, 1),
    '3': (0, 2),
    '4': (0, 3),
    '5': (1, 0),
    '6': (1, 1),
    '7': (1, 2),
    '8': (1, 3)
}

def compute_layer_size(dimensions: dict, element_bytes=4, measurement_unit='KB', n_threads=1):
    size = element_bytes
    for dim_name, dim_val in dimensions.items():
        if dim_name == 'H':
            size *= (dim_val / n_threads)
        else:
            size *= dim_val
    if measurement_unit.lower() == 'b':
        return float(size)
    elif measurement_unit.lower() == 'kb':
        return float(size / 2**10)
    elif measurement_unit.lower() == 'mb':
        return float(size / 2**20)
    elif measurement_unit.lower() == 'gb':
        return float(size / 2**30)

def compute_network_sizes(network: dict, n_threads=1):
    network_sizes = {}
    for layer_id, layer_dict in network.items():
        network_sizes[layer_id] = {}
        for tensor_type, tensor_dim in layer_dict.items():
            if tensor_type == 'output':
                network_sizes[layer_id][tensor_type] = compute_layer_size(tensor_dim, n_threads=n_threads)
            else:    
                network_sizes[layer_id][tensor_type] = compute_layer_size(tensor_dim)
    return network_sizes

def plot_network_layers(network: dict, ax, n_threds_title=None):
    df = pd.DataFrame(network)
    df.T.plot(ax=ax, kind='line', rot=0, colormap='tab20c', zorder=3, marker='o')
    ax.set_ylabel('KiloBytes (KB)')
    ax.set_title(n_threds_title+' Threads')
    return ax

def plot_cache_sizes(cache_size: dict, ax):
    for cache_level, cache_info in cache_size.items():
        ax.axhline(cache_info['size'], label=cache_level, color=cache_info['color'], linestyle='dashed')
    ax.legend(ncol=2)
    return ax

if __name__=='__main__':
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,10))

    for n_threads, pos in THREADS_POSITIONS.items():
        ax[pos] = plot_network_layers(compute_network_sizes(NETWORKS['ALEXNET'], n_threads=int(n_threads)), ax=ax[pos], n_threds_title=n_threads)

        print(compute_network_sizes(NETWORKS['ALEXNET'], n_threads=int(n_threads)))

        ax[pos] = plot_cache_sizes(CACHE_SIZE, ax[pos])

        ax[pos].set_yscale('log')

    plt.tight_layout()
    plt.savefig('results.pdf')
