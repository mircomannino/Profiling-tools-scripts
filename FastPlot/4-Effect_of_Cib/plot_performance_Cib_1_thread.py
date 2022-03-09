from xxlimited import foo
import pandas as pd
import matplotlib.pyplot as plt

CONVOLUTION_INFO = {
    'Hi': 34, 'Wi': 34, 'Ci': 128,
    'Hf': 3, 'Wf': 3,
    'Wo': 32, 'Ho': 32
}

RESULTS_BLOCKED_1_THREADS = {
    # Cob = 64
    '64': {
        # Cib values
        '128': 14.18, '64': 12.98, '32': 12.59, '16': 12.44, '8': 11.19, '4': 11.84, '2': 12.59, 'non-blocked': 7.756
    },
    # Cob = 128
    '128': {
        # Cib values
        '128': 22.56, '64': 19.87, '32': 17.63, '16': 17.33, '8': 17.49, '4': 15.45, '2': 16.46, 'non-blocked': 11.25 
    },
    # Cob = 256
    '256': {
        # Cib values
        '128': 38.75, '64': 34.95, '32': 30.29, '16': 26.65, '8': 26.11, '4': 26.66, '2': 23.71, 'non-blocked': 20.33
    },
    # Cob = 512
    '512': {
        # Cib values
        '128': 63.12, '64': 60.97, '32': 58.62, '16': 50.89, '8': 45.56, '4': 43.20, '2': 43.74, 'non-blocked': 34.88
    },
    # Cob = 1024
    '1024': {
        # Cib values
        '128': 108.83, '64': 108.64, '32': 107.05, '16': 101.17, '8': 90.68, '4': 81.37, '2': 78.33, 'non-blocked': 68.71
    },
    # Cob = 2048
    '2048': {
        # Cib values
        '128': 216.13, '64': 203.49, '32': 202.76, '16': 199.71, '8': 193.13, '4': 173.08, '2': 156.15, 'non-blocked': 158.217
    },
}



if __name__=='__main__':
    # Plot performance 1 thread
    # performance = pd.DataFrame(RESULTS_BLOCKED_1_THREADS)
    # performance.T.plot(kind='bar', rot=0)
    # plt.show()

    # Plot memory footprint
    Cob_list = RESULTS_BLOCKED_1_THREADS.keys()
    fig, ax = plt.subplots(ncols=1, nrows=(len(Cob_list)))
    for i, Cob in enumerate(Cob_list):
        footprints = {}
        footprints['INPUT'] = {}
        footprints['KERNEL'] = {}
        footprints['OUTPUT'] = {}
        for Cib in RESULTS_BLOCKED_1_THREADS[Cob].keys():
            if Cib=='non-blocked': continue
            footprints['INPUT'][Cib] = CONVOLUTION_INFO['Hi'] * CONVOLUTION_INFO['Wi'] * int(Cib)
            footprints['KERNEL'][Cib] = CONVOLUTION_INFO['Hf'] * CONVOLUTION_INFO['Wf'] * int(Cob) * int(Cib)
            footprints['OUTPUT'][Cib] = CONVOLUTION_INFO['Ho'] * CONVOLUTION_INFO['Wo'] * int(Cob)
        footprints = pd.DataFrame(footprints)
        footprints.plot(ax=ax[i], kind='bar', rot=0)
        ax[i].set_title('Cob: ' + Cob)
        ax[i].set_yscale('log')
    plt.subplots_adjust(hspace=0.5)
    plt.show()