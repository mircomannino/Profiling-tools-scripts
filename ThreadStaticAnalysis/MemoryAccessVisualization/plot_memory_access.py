from matplotlib import markers
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import numpy as np


def heatmap():
    output_size = 290400
    output_tensor = np.zeros(shape=(int(output_size/(10)), 10), dtype=bool)

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
    )


    plt.savefig('./heatmap.pdf')

def yx_plot():
    output_size = 290400

    file_path='/home/mannino/Desktop/Git/EfficientConvolution/testMemAccess.txt'

    with open(file_path, 'r') as file:
        for i in range(0, 9): next(file)    # Skip first 9 rows

        x = []
        y = []
        for line in file:
            line_splitted = line.split()
            # if len(line_splitted) != 2: print(line_splitted)
            x.append(line_splitted[0])
            y.append(line_splitted[1])              

    plt.scatter(x, y, marker='o')

    plt.savefig('yx_plot.png')

if __name__=='__main__':

    yx_plot()