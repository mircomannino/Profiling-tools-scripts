import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import numpy as np

if __name__=='__main__':

    output_tensor = np.zeros(shape=(290400,10980), dtype=bool)

    # Dummy fill
    # n_iter_dummy = 0
    # n_lines = 1000
    # for i, row in enumerate(output_tensor):
    #     for j, el in enumerate(row):
    #         output_tensor[i][j] = n_iter_dummy
    #     n_lines -= 1
    #     if n_lines == 0:
    #         n_lines = 1000
    #         n_iter_dummy += 10

    fig, ax = plt.subplots()

    ax = sns.heatmap(
        output_tensor,
        vmin=0
    )


    plt.savefig('./results.pdf')