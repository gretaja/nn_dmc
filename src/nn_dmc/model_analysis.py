"""Functions for NN model error analyis"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pyvibdmc as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def calc_nn_test_errors(system,model,x_data,y_data):
    if system == 'h5o3':
        input_size = 28
        hidden_size = 360

    elif system == 'h7o4':
        input_size = 55
        hidden_size = 540

    elif system == 'h9o5':
        input_size = 91
        hidden_size = 360

    elif system == 'ohh2':
        input_size = 6
        hidden_size = 60

    elif system == 'h11o6':
        input_size = 136
        hidden_size = 450

    elif system == 'h2o':
        input_size = 3
        hidden_size = 30

    else:
        raise ValueError('not a valid system name')

    X_test_read = np.load(x_data)
    y_test_read = np.load(y_data)
    
    X_test = torch.tensor(X_test_read, dtype=torch.float32)
    y_test = torch.tensor(y_test_read, dtype=torch.float32)
    y_test_regular = torch.tensor([(10**(i))-100 for i in y_test])

    output_size = 1

    testmodel = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True),
                        nn.BatchNorm1d(hidden_size),
                        nn.SiLU(),
                        nn.Dropout(),
                        nn.Linear(hidden_size, hidden_size, bias=True),
                        nn.BatchNorm1d(hidden_size),
                        nn.SiLU(),
                        nn.Dropout(),
                        nn.Linear(hidden_size, hidden_size, bias=True),
                        nn.BatchNorm1d(hidden_size),
                        nn.SiLU(),
                        nn.Dropout(),
                        nn.Linear(hidden_size, output_size,bias=True),
                        nn.ReLU()

    )

    # Load the model's state dictionary from the saved file
    testmodel.load_state_dict(torch.load(model,map_location=torch.device('cpu')))
    testmodel.eval()

    output = testmodel(X_test)

    output_regular = torch.tensor([(10**(j))-100 for j in output])

    MAE = nn.L1Loss()

    test_MAE = MAE(output_regular, y_test_regular)

    test_errors = output_regular.detach().numpy() - y_test_regular.detach().numpy()

    average_error = np.mean(test_errors)

    return y_test_regular, output_regular, test_MAE, average_error


def plot_2d_pred_errors(system,model,x_data,y_data):
    if system == 'h5o3':
        bin_width = 600
        bin_height = 50
        xlim = 25000
        ylim = 800

    elif system == 'h7o4':
        bin_width = 850
        bin_height = 60
        xlim = 35000
        ylim = 1000

    elif system == 'h9o5':
        bin_width = 1200
        bin_height = 120
        xlim = 50000
        ylim = 2000

    elif system == 'ohh2':
        bin_width = 250
        bin_height = 15
        xlim = 10000
        ylim = 250

    elif system == 'h11o6':
        bin_width = 1400
        bin_height = 120
        xlim = 60000
        ylim = 2000

    elif system == 'h2o':
        bin_width = 225
        bin_height = 10
        xlim = 9000
        ylim = 150

    else:
        raise ValueError('not a valid system name')
    
    y_test_regular, output_regular, test_MAE, average_error = calc_nn_test_errors(system,model,x_data,y_data)

    test_errors = output_regular.detach().numpy() - y_test_regular.detach().numpy()

    fig, ax = plt.subplots()

    h = ax.hist2d(y_test_regular.detach().numpy(),test_errors,bins=[np.arange(0,xlim+bin_width,bin_width),np.arange(-ylim,ylim+bin_height,bin_height)],norm=LogNorm(vmin=1),cmap = 'viridis')

    fig.colorbar(h[3], ax=ax)

    plt.hlines(0,0,xlim,color = 'white',linewidth = 1.5)

    steps = np.arange(0,41,3)

    error_pairs = []
    for i in range(len(test_errors)):
        error_pairs.append([y_test_regular.detach().numpy()[i],test_errors[i]])
        
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

    plt.show()

    print('MAE: {0:0.2f}, average error: {1:0.2f}'.format(test_MAE,average_error))


def plot_pred_errors(system,model,x_data,y_data):

    y_test_regular, output_regular, test_MAE, average_error = calc_nn_test_errors(system,model,x_data,y_data)

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
    #plt.title(f'Energies for H$_7$O$_4^-$ MOBML NN: 0$\%$ dropout, 0 decay, 540 nodes, 3 hidden')
    plt.xticks(np.arange(0, 120000, 20000))
    plt.yticks(np.arange(0, 120000, 20000))

    plt.show()