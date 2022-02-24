# File that contains the network architectures to plot 

CACHE_SIZE = {
    'L1': {
        'size':     32,       # 32 KB
        'color':    'green',
    },
    'L2': {
        'size':     256,      # 256 KB
        'color':    'orange',
    },
    'L3': {
        'size':     16384,    # 16 MB
        'color':    'red'
    }
}

NETWORKS = {
    'ALEXNET': {
        'layer 0': {
            'input':    {'H': 227, 'W': 227, 'C': 3},
            'kernel':   {'H': 11, 'W': 11, 'C': 3, 'N': 96},
            'output':   {'H': 55, 'W': 55, 'C': 96},
        },
        'layer 1': {
            'input':    {'H': 55, 'W': 55, 'C': 96},
            'kernel':   {'H': 5, 'W': 5, 'C': 96, 'N': 256},
            'output':   {'H': 26, 'W': 26, 'C': 256},
        },
        'layer 2': {
            'input':    {'H': 26, 'W': 26, 'C': 256},
            'kernel':   {'H': 3, 'W': 3, 'C': 256, 'N': 384},
            'output':   {'H': 24, 'W': 24, 'C': 384},
        }, 
        'layer 3': {
            'input':    {'H': 24, 'W': 24, 'C': 384},
            'kernel':   {'H': 3, 'W': 3, 'C': 384, 'N': 384},
            'output':   {'H': 22, 'W': 22, 'C': 384},
        },
        'layer 4': {
            'input':    {'H': 22, 'W': 22, 'C': 384},
            'kernel':   {'H': 3, 'W': 3, 'C': 384, 'N': 256},
            'output':   {'H': 20, 'W': 20, 'C': 256},
        }
    }
}