import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

import pdb

if __name__ == "__main__":
    data = np.load('rarl_result_sim.npy', allow_pickle=True)
    data = data.item()
    data_keys = list(data.keys())
    data_keys.sort()
    matrix = np.ones((9,9))
    for data_key in data_keys:
        i = int(data_key[1])
        j = int(data_key[4])
        print(data[data_key])
        matrix[i,j] = data[data_key]
        matrix[j,i] = data[data_key]
    # ARL
    # matrix[0,0] = 0.8
    # matrix[1,1] = 1.0
    # matrix[2,2] = 0.9
    # matrix[3,3] = 1.0
    # matrix[4,4] = 0.9
    # matrix[5,5] = 0.9
    # matrix[6,6] = 1.0
    # matrix[7,7] = 0.7
    # matrix[8,8] = 1.0

    #sac
    matrix[0,0] = 0.5
    matrix[1,1] = 0.9
    matrix[2,2] = 1.0
    matrix[3,3] = 0.3
    matrix[4,4] = 0.0
    matrix[5,5] = 0.5
    matrix[6,6] = 0.0
    matrix[7,7] = 0.
    matrix[8,8] = 0.1

    # rarl sim
    matrix[0,0] = 1
    matrix[1,1] = 1
    matrix[2,2] = 1.0
    matrix[3,3] = 1
    matrix[4,4] = 1
    matrix[5,5] = 1
    matrix[6,6] = 1
    matrix[7,7] = 1
    matrix[8,8] = 1
    #sac sim
    # matrix[0,0] = 1
    # matrix[1,1] = 1.0
    # matrix[2,2] = 1
    # matrix[3,3] = 0.9
    # matrix[4,4] = 0.8
    # matrix[5,5] = 1
    # matrix[6,6] = 0
    # matrix[7,7] = 1
    # matrix[8,8] = 0
    
    sbn.heatmap(matrix, cmap='GnBu')
    plt.title('success rate under different damage cases for RSAC in sim',fontsize=13)
    plt.savefig('rarl_heatmap_sim.pdf')
    
    # plt.show()
