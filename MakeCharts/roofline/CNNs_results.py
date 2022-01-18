# All the results for each CNN network benchmark

ALEX_NET_27A = {
    'info': {
        'Compiler': 'ICC',
        'Vector':   'YES',
        'Parallel': 'NO'
    },
    'results': {
        'order-1': {'GFLOPS_s': 0.97, 'OI': 0.035},
        'order-2': {'GFLOPS_s': 30.8, 'OI': 0.16},
        'order-3': {'GFLOPS_s': 20.8, 'OI': 0.16},
        'order-4': {'GFLOPS_s': 27.29, 'OI': 0.16},
        'order-5': {'GFLOPS_s': 2.29, 'OI': 0.16},
    }
}

ALEX_NET_27B  = {
    'info': {
        'Compiler': 'ICC',
        'Vector':   'NO',
        'Parallel': 'NO'
    },
    'results': {
        'order-1': {'GFLOPS_s': 1.01, 'OI': 0.037},
        'order-2': {'GFLOPS_s': 3.89, 'OI': 0.12},
        'order-3': {'GFLOPS_s': 3.77, 'OI': 0.12},
        'order-4': {'GFLOPS_s': 3.84, 'OI': 0.12},
        'order-5': {'GFLOPS_s': 2.4, 'OI': 0.16},
    }
}

ALEX_NET_27C = {
    'info': {
        'Compiler': 'POLLY',
        'Vector':   'YES',
        'Parallel': 'NO'
    },
    'results': {
        'order-1': {'GFLOPS_s': 0.2, 'OI': 0.03},
        'order-2': {'GFLOPS_s': 23.21, 'OI': 0.16},
        'order-3': {'GFLOPS_s': 17.51, 'OI': 0.16},
        'order-4': {'GFLOPS_s': 20.67, 'OI': 0.16},
        'order-5': {'GFLOPS_s': 2.21, 'OI': 0.12},
    }
}
