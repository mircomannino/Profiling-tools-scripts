# File that contains the network architectures to plot 

MACHINE_CPU = {
    'BlackYeti': 'Intel Core i9-9900k', 
    'Thor': 'AMD Ryzen 5990X'
}

CACHE_SIZE = {
    'BlackYeti': {
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
    },

    'Thor': {
        'L1': {
            'size':     32,       # 32 KB
            'color':    'green',
        },
        'L2': {
            'size':     512,      # 512 KB
            'color':    'orange',
        },
        'L3': {
            'size':     32768,    # 32 MB
            'color':    'red'
        }
    }
}

NETWORKS = {
    'ALEXNET-Ho': {
        # 1 Thread
        '1': {
            'layer 0': {
                'input':    154587,
                'kernel':   34848,
                'output':   290400,
            },
            'layer 1': {
                'input':    290400,
                'kernel':   614400,
                'output':   173056,
            },
            'layer 2': {
                'input':    173056,
                'kernel':   884736,
                'output':   221184,
            }, 
            'layer 3': {
                'input':    221184,
                'kernel':   1327104,
                'output':   185856,
            },
            'layer 4': {
                'input':    185856,
                'kernel':   884736,
                'output':   102400,
            }
        },

        # 2 Threads
        '2': {
            'layer 0': {
                'input':    [78315, 81039],
                'kernel':   [34848, 34848],
                'output':   [142560, 147840],
            },
            'layer 1': {
                'input':    [153120, 153120],
                'kernel':   [614400, 614400],
                'output':   [86528, 86528],
            },
            'layer 2': {
                'input':    [93184, 93184],
                'kernel':   [884736, 884736],
                'output':   [110592, 110592],
            }, 
            'layer 3': {
                'input':    [119808, 119808],
                'kernel':   [1327104, 1327104],
                'output':   [92928, 92928],
            },
            'layer 4': {
                'input':    [101376, 101376],
                'kernel':   [884736, 884736],
                'output':   [51200, 51200],
            }
        },

        # 3 Threads
        '3': {
            'layer 0': {
                'input':    [53799, 53799, 56523],
                'kernel':   [34848, 34848, 34848],
                'output':   [95040, 95040, 100320],
            },
            'layer 1': {
                'input':    [100320, 110880, 110880],
                'kernel':   [614400, 614400, 614400],
                'output':   [53248, 59904, 59904],
            },
            'layer 2': {
                'input':    [66560, 66560, 66560],
                'kernel':   [884736, 884736, 884736],
                'output':   [73728, 73728, 73728],
            }, 
            'layer 3': {
                'input':    [82944, 82944, 92160],
                'kernel':   [1327104, 1327104, 1327104],
                'output':   [59136, 59136, 67584],
            },
            'layer 4': {
                'input':    [67584, 76032, 76032],
                'kernel':   [884736, 884736, 884736],
                'output':   [30720, 35840, 35840],
            }
        },

        # 4 Threads
        '4': {
            'layer 0': {
                'input':    [40179, 42903, 42903, 42903],
                'kernel':   [34848, 34848, 34848, 34848],
                'output':   [68640, 73920, 73920, 73920],
            },
            'layer 1': {
                'input':    [79200, 79200, 89760, 89760],
                'kernel':   [614400, 614400, 614400, 614400],
                'output':   [39936, 39936, 46592, 46592],
            },
            'layer 2': {
                'input':    [53248, 53248, 53248, 53248],
                'kernel':   [884736, 884736, 884736, 884736],
                'output':   [55296, 55296, 55296, 55296],
            }, 
            'layer 3': {
                'input':    [64512, 64512, 73728, 73728],
                'kernel':   [1327104, 1327104, 1327104, 1327104],
                'output':   [42240, 42240, 50688, 50688],
            },
            'layer 4': {
                'input':    [59136, 59136, 59136, 59136],
                'kernel':   [884736, 884736, 884736, 884736],
                'output':   [25600, 25600, 25600, 25600],
            }
        },

        # 5 Threads
        '5': {
            'layer 0': {
                'input':    [34731, 34731, 34731, 34731, 34731],
                'kernel':   [34848, 34848, 34848, 34848, 34848],
                'output':   [58080, 58080, 58080, 58080, 58080],
            },
            'layer 1': {
                'input':    [68640, 68640, 68640, 68640, 79200],
                'kernel':   [614400, 614400, 614400, 614400, 614400],
                'output':   [33280, 33280, 33280, 33280, 39936],
            },
            'layer 2': {
                'input':    [39936, 46592, 46592, 46592, 46592],
                'kernel':   [884736, 884736, 884736, 884736, 884736],
                'output':   [36864, 46080, 46080, 46080, 46080],
            }, 
            'layer 3': {
                'input':    [55296, 55296, 55296, 64512, 64512],
                'kernel':   [1327104, 1327104, 1327104, 1327104, 1327104],
                'output':   [33792, 33792, 33792, 42240, 42240],
            },
            'layer 4': {
                'input':    [50688, 50688, 50688, 50688, 50688],
                'kernel':   [884736, 884736, 884736, 884736, 884736],
                'output':   [20480, 20480, 20480, 20480, 20480],
            }
        },

        # 6 Threads
        '6': {
            'layer 0': {
                'input':    [29283, 29283, 29283, 29283, 29283, 32007],
                'kernel':   [34848, 34848, 34848, 34848, 34848, 34848],
                'output':   [47520, 47520, 47520, 47520, 47520, 52800],
            },
            'layer 1': {
                'input':    [58080, 58080, 58080, 58080, 68640, 68640],
                'kernel':   [614400, 614400, 614400, 614400, 614400, 614400],
                'output':   [26624, 26624, 26624, 26624, 33280, 33280],
            },
            'layer 2': {
                'input':    [39936, 39936, 39936, 39936, 39936, 39936],
                'kernel':   [884736, 884736, 884736, 884736, 884736, 884736],
                'output':   [36864, 36864, 36864, 36864, 36864, 36864],
            }, 
            'layer 3': {
                'input':    [46080, 46080, 55296, 55296, 55296, 55296],
                'kernel':   [1327104, 1327104, 1327104, 1327104, 1327104, 1327104],
                'output':   [25344, 25344, 33792, 33792, 33792, 33792],
            },
            'layer 4': {
                'input':    [42240, 42240, 42240, 42240, 50688, 50688],
                'kernel':   [884736, 884736, 884736, 884736, 884736, 884736],
                'output':   [15360, 15360, 15360, 15360, 20480, 20480],
            }
        },

        # 7 Threads
        '7': {
            'layer 0': {
                'input':    [23835, 26559, 26559, 26559, 26559, 26559, 26559],
                'kernel':   [34848, 34848, 34848, 34848, 34848, 34848, 34848],
                'output':   [36960, 42240, 42240, 42240, 42240, 42240, 42240],
            },
            'layer 1': {
                'input':    [47520, 47520, 58080, 58080, 58080, 58080, 58080],
                'kernel':   [614400, 614400, 614400, 614400, 614400, 614400, 614400],
                'output':   [19968, 19968, 26624, 26624, 26624, 26624, 26624],
            },
            'layer 2': {
                'input':    [33280, 33280, 33280, 33280, 39936, 39936, 39936],
                'kernel':   [884736, 884736, 884736, 884736, 884736, 884736, 884736],
                'output':   [27648, 27648, 27648, 27648, 36864, 36864, 36864],
            }, 
            'layer 3': {
                'input':    [46080, 46080, 46080, 46080, 46080, 46080, 55296],
                'kernel':   [1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104],
                'output':   [25344, 25344, 25344, 25344, 25344, 25344, 33792],
            },
            'layer 4': {
                'input':    [33792, 42240, 42240, 42240, 42240, 42240, 42240],
                'kernel':   [884736, 884736, 884736, 884736, 884736, 884736, 884736],
                'output':   [10240, 15360, 15360, 15360, 15360, 15360, 15360],
            }
        },

        # 8 Threads
        '8': {
            'layer 0': {
                'input':    [21111, 23835, 23835, 23835, 23835, 23835, 23835, 23835],
                'kernel':   [34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848],
                'output':   [31680, 36960, 36960, 36960, 36960, 36960, 36960, 36960],
            },
            'layer 1': {
                'input':    [47520, 47520, 47520, 47520, 47520, 47520, 58080, 58080],
                'kernel':   [614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400],
                'output':   [19968, 19968, 19968, 19968, 19968, 19968, 26624, 26624],
            },
            'layer 2': {
                'input':    [33280, 33280, 33280, 33280, 33280, 33280, 33280, 33280],
                'kernel':   [884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736],
                'output':   [27648, 27648, 27648, 27648, 27648, 27648, 27648, 27648],
            }, 
            'layer 3': {
                'input':    [36864, 36864, 46080, 46080, 46080, 46080, 46080, 46080],
                'kernel':   [1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104],
                'output':   [16896, 16896, 25344, 25344, 25344, 25344, 25344, 25344],
            },
            'layer 4': {
                'input':    [33792, 33792, 33792, 33792, 42240, 42240, 42240, 42240],
                'kernel':   [884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736],
                'output':   [10240, 10240, 10240, 10240, 15360, 15360, 15360, 15360],
            }
        },

        # 9 Threads
        '9': {
            'layer 0': {
                'input':    [21111, 21111, 21111, 21111, 21111, 21111, 21111, 21111, 23835],
                'kernel':   [34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848],
                'output':   [31680, 31680, 31680, 31680, 31680, 31680, 31680, 31680, 36960],
            },
            'layer 1': {
                'input':    [36960, 47520, 47520, 47520, 47520, 47520, 47520, 47520, 47520],
                'kernel':   [614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400],
                'output':   [13312, 19968, 19968, 19968, 19968, 19968, 19968, 19968, 19968],
            },
            'layer 2': {
                'input':    [26624, 26624, 26624, 33280, 33280, 33280, 33280, 33280, 33280],
                'kernel':   [884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736],
                'output':   [18432, 18432, 18432, 27648, 27648, 27648, 27648, 27648, 27648],
            },
            'layer 3': {
                'input':    [36864, 36864, 36864, 36864, 36864, 46080, 46080, 46080, 46080],
                'kernel':   [1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104],
                'output':   [16896, 16896, 16896, 16896, 16896, 25344, 25344, 25344, 25344],
            },
            'layer 4': {
                'input':    [33792, 33792, 33792, 33792, 33792, 33792, 33792, 42240, 42240],
                'kernel':   [884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736],
                'output':   [10240, 10240, 10240, 10240, 10240, 10240, 10240, 15360, 15360],
            },
        },

        # 10 Threads
        '10': {
            'layer 0': {
                'input':    [18387, 18387, 18387, 18387, 18387, 21111, 21111, 21111, 21111, 21111],
                'kernel':   [34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848],
                'output':   [26400, 26400, 26400, 26400, 26400, 31680, 31680, 31680, 31680, 31680],
            },
            'layer 1': {
                'input':    [36960, 36960, 36960, 36960, 47520, 47520, 47520, 47520, 47520, 47520],
                'kernel':   [614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400],
                'output':   [13312, 13312, 13312, 13312, 19968, 19968, 19968, 19968, 19968, 19968],
            },
            'layer 2': {
                'input':    [26624, 26624, 26624, 26624, 26624, 26624, 33280, 33280, 33280, 33280],
                'kernel':   [884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736],
                'output':   [18432, 18432, 18432, 18432, 18432, 18432, 27648, 27648, 27648, 27648],
            },
            'layer 3': {
                'input':    [36864, 36864, 36864, 36864, 36864, 36864, 36864, 36864, 46080, 46080],
                'kernel':   [1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104],
                'output':   [16896, 16896, 16896, 16896, 16896, 16896, 16896, 16896, 25344, 25344],
            },
            'layer 4': {
                'input':    [33792, 33792, 33792, 33792, 33792, 33792, 33792, 33792, 33792, 33792],
                'kernel':   [884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736],
                'output':   [10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240],
            },
        },

        # 11 Threads
        '11': {
            'layer 0': {
                'input':    [18387, 18387, 18387, 18387, 18387, 18387, 18387, 18387, 18387, 18387, 18387],
                'kernel':   [34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848],
                'output':   [26400, 26400, 26400, 26400, 26400, 26400, 26400, 26400, 26400, 26400, 26400],
            },
            'layer 1': {
                'input':    [36960, 36960, 36960, 36960, 36960, 36960, 36960, 47520, 47520, 47520, 47520],
                'kernel':   [614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400],
                'output':   [13312, 13312, 13312, 13312, 13312, 13312, 13312, 19968, 19968, 19968, 19968],
            },
            'layer 2': {
                'input':    [26624, 26624, 26624, 26624, 26624, 26624, 26624, 26624, 26624, 33280, 33280],
                'kernel':   [884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736],
                'output':   [18432, 18432, 18432, 18432, 18432, 18432, 18432, 18432, 18432, 27648, 27648],
            },
            'layer 3': {
                'input':    [36864, 36864, 36864, 36864, 36864, 36864, 36864, 36864, 36864, 36864, 36864],
                'kernel':   [1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104],
                'output':   [16896, 16896, 16896, 16896, 16896, 16896, 16896, 16896, 16896, 16896, 16896],
            },
            'layer 4': {
                'input':    [25344, 25344, 33792, 33792, 33792, 33792, 33792, 33792, 33792, 33792, 33792],
                'kernel':   [884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736,],
                'output':   [5120, 5120, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240],
            },
        },

        # 12 Threads
        '12': {
            'layer 0': {
                'input':    [15663, 15663, 15663, 15663, 15663, 18387, 18387, 18387, 18387, 18387, 18387, 18387],
                'kernel':   [34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848, 34848],
                'output':   [21120, 21120, 21120, 21120, 21120, 26400, 26400, 26400, 26400, 26400, 26400, 26400],
            },
            'layer 1': {
                'input':    [36960, 36960, 36960, 36960, 36960, 36960, 36960, 36960, 36960, 36960, 47520, 47520],
                'kernel':   [614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400, 614400],
                'output':   [13312, 13312, 13312, 13312, 13312, 13312, 13312, 13312, 13312, 13312, 19968, 19968],
            },
            'layer 2': {
                'input':    [26624, 26624, 26624, 26624, 26624, 26624, 26624, 26624, 26624, 26624, 26624, 26624],
                'kernel':   [884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736],
                'output':   [18432, 18432, 18432, 18432, 18432, 18432, 18432, 18432, 18432, 18432, 18432, 18432],
            },
            'layer 3': {
                'input':    [27648, 27648, 36864, 36864, 36864, 36864, 36864, 36864, 36864, 36864, 36864, 36864],
                'kernel':   [1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104, 1327104],
                'output':   [8448, 8448, 16896, 16896, 16896, 16896, 16896, 16896, 16896, 16896, 16896, 16896],
            },
            'layer 4': {
                'input':    [25344, 25344, 25344, 25344, 33792, 33792, 33792, 33792, 33792, 33792, 33792, 33792],
                'kernel':   [884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736, 884736],
                'output':   [5120, 5120, 5120, 5120, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240],
            },
        },
    }
}