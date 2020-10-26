import matplotlib.pyplot as plt
import numpy as np
import glob
import math

import pdb
if __name__ == "__main__":
    data_path = glob.glob("updated_results/*.npy")
    data_path.sort()
    data = []
    for this_data_path in data_path:
        data.append(math.pi - abs(np.load(this_data_path)))
    sac_dataset = data[:10]
    rarl_dataset = data[10:]
    plt.xlim([-1,51])
    plt.ylim([-0.2, 3.3])
    color_table = ['palegreen', 'greenyellow', 'lawngreen', 'powderblue','paleturquoise','turquoise','orange', 'wheat', 'goldenrod','r']
    # for i,rarl in enumerate(rarl_dataset):
    #     if i == 9:
    #         continue
    #     if len(rarl) > 50:
    #         rarl = rarl[:50]
    #     plt.plot(rarl, c = color_table[i])
    # plt.plot(rarl_dataset[9], 'k--')

    for i, sac in enumerate(sac_dataset):
        if i == 9:
            continue
        if len(sac) > 50:
            sac = sac[:50]
        if sac[0] > 2.5:
            sac[0] = 0
        plt.plot(sac, c = color_table[i])
    plt.plot(sac_dataset[9], 'k--')


    plt.xlabel("timestep", fontsize=18)
    plt.ylabel("valve angle", fontsize=18)
    other = [5,4,4]
    # plt.legend()
    handle1, = plt.plot(other, c='k')
    handle2, = plt.plot(other, c='k', linestyle='--')
    plt.legend(handles=[handle1,handle2],labels=['damaged','complete'],loc='upper left')
    plt.title("Valve Angles under Different Damages for SAC", fontsize=17)
    plt.savefig('sac.pdf')
    # plt.show()
