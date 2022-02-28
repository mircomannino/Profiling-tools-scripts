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
                'input':    78315,
                'kernel':   34848,
                'output':   142560,
            },
            'layer 1': {
                'input':    153120,
                'kernel':   614400,
                'output':   86528,
            },
            'layer 2': {
                'input':    93184,
                'kernel':   884736,
                'output':   110592,
            }, 
            'layer 3': {
                'input':    119808,
                'kernel':   1327104,
                'output':   92928,
            },
            'layer 4': {
                'input':    101376,
                'kernel':   884736,
                'output':   51200,
            }
        },

        # 3 Threads
        '3': {
            'layer 0': {
                'input':    53799,
                'kernel':   34848,
                'output':   95040,
            },
            'layer 1': {
                'input':    100320,
                'kernel':   614400,
                'output':   53248,
            },
            'layer 2': {
                'input':    66560,
                'kernel':   884736,
                'output':   73728,
            }, 
            'layer 3': {
                'input':    82944,
                'kernel':   1327104,
                'output':   59136,
            },
            'layer 4': {
                'input':    67584,
                'kernel':   884736,
                'output':   30720,
            }
        },

        # 4 Threads
        '4': {
            'layer 0': {
                'input':    40179,
                'kernel':   34848,
                'output':   68640,
            },
            'layer 1': {
                'input':    79200,
                'kernel':   614400,
                'output':   39936,
            },
            'layer 2': {
                'input':    53248,
                'kernel':   884736,
                'output':   55296,
            }, 
            'layer 3': {
                'input':    64512,
                'kernel':   1327104,
                'output':   42240,
            },
            'layer 4': {
                'input':    59136,
                'kernel':   884736,
                'output':   25600,
            }
        },

        # 5 Threads
        '5': {
            'layer 0': {
                'input':    34731,
                'kernel':   34848,
                'output':   58080,
            },
            'layer 1': {
                'input':    68640,
                'kernel':   614400,
                'output':   33280,
            },
            'layer 2': {
                'input':    39936,
                'kernel':   884736,
                'output':   36864,
            }, 
            'layer 3': {
                'input':    55296,
                'kernel':   1327104,
                'output':   33792,
            },
            'layer 4': {
                'input':    50688,
                'kernel':   884736,
                'output':   20480,
            }
        },

        # 6 Threads
        '6': {
            'layer 0': {
                'input':    29283,
                'kernel':   34848,
                'output':   47520,
            },
            'layer 1': {
                'input':    58080,
                'kernel':   614400,
                'output':   26624,
            },
            'layer 2': {
                'input':    39936,
                'kernel':   884736,
                'output':   36864,
            }, 
            'layer 3': {
                'input':    46080,
                'kernel':   1327104,
                'output':   25344,
            },
            'layer 4': {
                'input':    42240,
                'kernel':   884736,
                'output':   15360,
            }
        },

        # 7 Threads
        '7': {
            'layer 0': {
                'input':    23835,
                'kernel':   34848,
                'output':   36960,
            },
            'layer 1': {
                'input':    47520,
                'kernel':   614400,
                'output':   19968,
            },
            'layer 2': {
                'input':    33280,
                'kernel':   884736,
                'output':   27648,
            }, 
            'layer 3': {
                'input':    46080,
                'kernel':   1327104,
                'output':   25344,
            },
            'layer 4': {
                'input':    33792,
                'kernel':   884736,
                'output':   10240,
            }
        },

        # 8 Threads
        '8': {
            'layer 0': {
                'input':    21111,
                'kernel':   34848,
                'output':   31680,
            },
            'layer 1': {
                'input':    47520,
                'kernel':   614400,
                'output':   19968,
            },
            'layer 2': {
                'input':    33280,
                'kernel':   884736,
                'output':   27648,
            }, 
            'layer 3': {
                'input':    36864,
                'kernel':   1327104,
                'output':   16896,
            },
            'layer 4': {
                'input':    33792,
                'kernel':   884736,
                'output':   10240,
            }
        },
    },


    'ALEXNET-Hf': {
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
                'input':    [150501, 151182],
                'kernel':   [15840, 19008],
                'output':   290400,
            },
            'layer 1': {
                'input':    [274560, 279840],
                'kernel':   [245760, 368640],
                'output':   173056,
            },
            'layer 2': {
                'input':    [159744, 166400],
                'kernel':   [294912, 589824],
                'output':   221184,
            }, 
            'layer 3': {
                'input':    [202752, 211968],
                'kernel':   [442368, 884736],
                'output':   185856,
            },
            'layer 4': {
                'input':    [168960, 177408],
                'kernel':   [294912, 589824],
                'output':   102400,
            }
        },

        # 3 Threads
        '3': {
            'layer 0': {
                'input':    [112365, 149820, 149820],
                'kernel':   [9504, 12672, 12672],
                'output':   290400,
            },
            'layer 1': {
                'input':    [137280, 274560, 274560],
                'kernel':   [122880, 173056, 173056],
                'output':   173056,
            },
            'layer 2': {
                'input':    159744,
                'kernel':   294912,
                'output':   221184,
            }, 
            'layer 3': {
                'input':    202752,
                'kernel':   442368,
                'output':   185856,
            },
            'layer 4': {
                'input':    168960,
                'kernel':   294912,
                'output':   102400,
            }
        },

        # 4 Threads
        '4': {
            'layer 0': {
                'input':    [74910, 112365, 112365, 112365],
                'kernel':   [6336, 9504, 9504, 9504],
                'output':   290400,
            },
            'layer 1': {
                'input':    [137280, 137280, 137280, 274560],
                'kernel':   [122880, 122880, 122880, 245760],
                'output':   173056,
            },
            'layer 2': {
                'input':    159744,
                'kernel':   294912,
                'output':   221184,
            }, 
            'layer 3': {
                'input':    202752,
                'kernel':   442368,
                'output':   185856,
            },
            'layer 4': {
                'input':    168960,
                'kernel':   294912,
                'output':   102400,
            }
        },
    }
}