# All the results for each CNN network benchmark

ALEX_NET_27A = {
    'info': {
        'Compiler': 'ICC',
        'Vector':   'YES',
        'Parallel': 'NO'
    },
    'results': {
        'order-1': {'GFLOPS_s': 0.4548, 'OI': 0.167},
        'order-2': {'GFLOPS_s': 8.17, 'OI': 0.167},
        'order-3': {'GFLOPS_s': 5.5674, 'OI': 0.167},
        'order-4': {'GFLOPS_s': 6.4542, 'OI': 0.167},
        'order-5': {'GFLOPS_s': 0.4738, 'OI': 0.167},
    }
}

ALEX_NET_27B  = {
    'info': {
        'Compiler': 'ICC',
        'Vector':   'NO',
        'Parallel': 'NO'
    },
    'results': {
        'order-1': {'GFLOPS_s': 0.5686, 'OI': 0.167},
        'order-2': {'GFLOPS_s': 0.8048, 'OI': 0.125},
        'order-3': {'GFLOPS_s': 0.767, 'OI': 0.125},
        'order-4': {'GFLOPS_s': 0.7888, 'OI': 0.125},
        'order-5': {'GFLOPS_s': 0.4964, 'OI': 0.167},
    }
}

ALEX_NET_27C = {
    'info': {
        'Compiler': 'LLVM-POLLY',
        'Vector':   'YES',
        'Parallel': 'NO'
    },
    'results': {
        'order-1': {'GFLOPS_s': 0.7292, 'OI': 0.167},
        'order-2': {'GFLOPS_s': 8.1064, 'OI': 0.167},
        'order-3': {'GFLOPS_s': 6.6012, 'OI': 0.167},
        'order-4': {'GFLOPS_s': 7.1374, 'OI': 0.167},
        'order-5': {'GFLOPS_s': 0.468, 'OI': 0.143},
    }
}


ALEX_NET_28A = {
    'info': {
        'Compiler': 'ICC',
        'Vector':   'YES',
        'Parallel': '8-threads'
    },
    'results': {
        'order-1': {'GFLOPS_s': 22.378, 'OI': 0.167, 'shift': 1},
        'order-2': {'GFLOPS_s': 405.3124, 'OI': 0.167, 'shift': 1},
        'order-3': {'GFLOPS_s': 405.3124, 'OI': 0.167, 'shift': 1.1},
        'order-4': {'GFLOPS_s': 202.6562, 'OI': 0.167, 'shift': 1},
        'order-5': {'GFLOPS_s': 30.6372, 'OI': 0.167, 'shift': 1},
    }
}

ALEX_NET_28B  = {
    'info': {
        'Compiler': 'ICC',
        'Vector':   'NO',
        'Parallel': '8-threads'
    },
    'results': {
        'order-1': {'GFLOPS_s': 20.4574, 'OI': 0.167, 'shift': 1},
        'order-2': {'GFLOPS_s': 45.035, 'OI': 0.125, 'shift': 1},
        'order-3': {'GFLOPS_s': 50.664, 'OI': 0.125, 'shift': 1},
        'order-4': {'GFLOPS_s': 50.664, 'OI': 0.125, 'shift': 1.},
        'order-5': {'GFLOPS_s': 30.8372, 'OI': 0.167, 'shift': 1},
    }
}

ALEX_NET_28C = {
    'info': {
        'Compiler': 'LLVM-POLLY',
        'Vector':   'YES',
        'Parallel': 'NO'
    },
    'results': {
        'order-1': {'GFLOPS_s': 24.2, 'OI': 0.167, 'shift': 1},
        'order-2': {'GFLOPS_s': 405.2, 'OI': 0.167, 'shift': 1},
        'order-3': {'GFLOPS_s': 405.2, 'OI': 0.167, 'shift': 1.1},
        'order-4': {'GFLOPS_s': 202.6, 'OI': 0.167, 'shift': 1.2},
        'order-5': {'GFLOPS_s': 29.4, 'OI': 0.167, 'shift': 1},
    }
}
