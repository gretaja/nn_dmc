import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pyvibdmc as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from h11o6.models.egnn.h11o6_egnn_training import EGNNModelLN


def cart_to_pot_fast(ckpt_file, coords_np, batch_size=1024):
    """
    Optimized CPU evaluation of geometries in batches.
    coords_np: np.ndarray of shape (N,17,3)
    Returns: np.ndarray of energies in a.u.
    """

    atom_types = torch.tensor([8.0, 1.0, 1.0, 8.0, 1.0, 1.0,
    8.0, 1.0, 1.0, 8.0, 1.0, 1.0,
    8.0, 1.0, 1.0, 8.0, 1.0], dtype=torch.float32)

    model = EGNNModelLN(n_atoms=17, atom_types_list=atom_types, hidden_dim=64, num_layers=3, n_freqs=3, coord_rescale=0.01)
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mu = ckpt["mu"]
    sigma = ckpt["sigma"]

    # Pre-allocate atom types tensor

    _atom_types_tensor = atom_types.unsqueeze(0).unsqueeze(-1)  # (1,17,1)

    N_total = coords_np.shape[0]
    energies = np.empty((N_total,), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, N_total, batch_size):
            end = min(start + batch_size, N_total)
            batch = torch.tensor(coords_np[start:end], dtype=torch.float32)  # (B,17,3)
            B = batch.shape[0]

            # expand atom types once per batch
            z = _atom_types_tensor.expand(B, -1, -1)  # (B,17,1)

            y_norm = model(batch, z).squeeze(-1)  # (B,)

            # store energies in-place
            energies[start:end] = (y_norm * sigma + mu).numpy()

    return energies  # convert to a.u.

def calc_nn_test_errors(ckpt_file,cds_data,y_data):

    cds_test = np.load(cds_data)
    y_test = np.load(y_data)
    energies_test = 10**y_test - 100

    output = cart_to_pot_fast(ckpt_file, cds_test)

    test_errors = output - energies_test

    average_error = np.mean(test_errors)

    test_MAE = np.mean(np.abs(test_errors))

    return energies_test, output, test_MAE, average_error


def plot_2d_pred_errors(ckpt_file,cds_data,y_data):
    bin_width = 1400
    bin_height = 120
    xlim = 60000
    ylim = 2000
    x_ticks = np.arange(0,75000,15000)
    
    energies_test, output, test_MAE, average_error = calc_nn_test_errors(ckpt_file,cds_data,y_data)

    test_errors = output.detach().numpy() - energies_test.detach().numpy()

    fig, ax = plt.subplots()

    h = ax.hist2d(energies_test.detach().numpy(),test_errors,bins=[np.arange(0,xlim+bin_width,bin_width),np.arange(-ylim,ylim+bin_height,bin_height)],norm=LogNorm(vmin=1),cmap = 'viridis')

    fig.colorbar(h[3], ax=ax)

    plt.hlines(0,0,xlim,color = 'white',linewidth = 1.5)

    steps = np.arange(0,41,3)

    error_pairs = []
    for i in range(len(test_errors)):
        error_pairs.append([energies_test.detach().numpy()[i],test_errors[i]])
        
    bins = []
    for i in range(len(h[1])-1):
        bin_elements = []
        for pair in error_pairs:
            if h[1][i] < pair[0] and h[1][i+1] > pair[0]:
                bin_elements.append(pair[1])
        bins.append(bin_elements)
        
    bin_stats = []
    for k in bins:
        bin_stats.append([np.mean(k),np.std(k)])

    for j in steps:
        #mid_bin = (h[1][steps[j]]+h[1][steps[j+1]])/2
        plt.vlines(h[1][j],bin_stats[j][0]-bin_stats[j][1],bin_stats[j][0]+bin_stats[j][1],color = 'magenta',linewidth = 3)
        plt.scatter(h[1][j],bin_stats[j][0],color = 'magenta')

    j = steps[7]
    plt.vlines(h[1][j],bin_stats[j][0]-bin_stats[j][1],bin_stats[j][0]+bin_stats[j][1],color = 'black',linewidth = 3)
    plt.scatter(h[1][j],bin_stats[j][0],color = 'black')
        
    plt.xlim(0,xlim)
    plt.xlabel('E(MOB-ML) (/cm$^{-1}$)',fontsize=16)
    plt.ylabel(r'E(NN) - E(MOB-ML) (/cm$^{-1}$)',fontsize=16)
    plt.ylim(-ylim,ylim)
    
    plt.xticks(x_ticks)

    plt.show()

    print('MAE: {0:0.2f}, average error: {1:0.2f}'.format(test_MAE,average_error))


def plot_pred_errors(ckpt_file,cds_data,y_data):

    y_test_regular, output_regular, _, _ = calc_nn_test_errors(ckpt_file,cds_data,y_data)

    plt.rcdefaults()
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(5, 5))
    x = np.linspace(0,100000,1000)
    plt.plot(x,x,linestyle = 'dashed',color = 'black')
    plt.scatter(y_test_regular,output_regular,color = 'rebeccapurple')
    plt.xlabel(r'E(MOB-ML) (/cm$^{-1}$)',fontsize=16)
    plt.ylabel(r'E(NN) (/cm$^{-1}$)',fontsize=16)
    plt.xlim(0,100000)
    plt.ylim(0,100000)

    plt.xticks(np.arange(0, 120000, 20000))
    plt.yticks(np.arange(0, 120000, 20000))

    plt.show()