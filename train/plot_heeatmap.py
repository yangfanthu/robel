import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

import pdb

if __name__ == "__main__":
    data = np.load('rarl-result.npy', allow_pickle=True)
    data = data.item()
    data_keys = list(data.keys())
    data_keys.sort()
    matrix = np.ones((9,9))
    for data_key in data_keys:
        i = int(data_key[1])
        j = int(data_key[4])
        matrix[i,j] = data[data_key]
        matrix[j,i] = data[data_key]
    for i in range(9):
        matrix[i,i] = 1
    sbn.heatmap(matrix, cmap='GnBu')
    plt.title('success rate under different damage cases for RSAC',fontsize=13)
    plt.savefig('rarl_heatmap.pdf')
    
    # plt.show()
