"""Functions for DMC simulation analysis"""

import numpy as np
import pyvibdmc as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def check_for_holes(file_path, threshold):
    """
    Parses a file to find the timesteps where "Lowest energy walker" falls below a threshold.

    Args:
        file_path (str): Path to the output file.
        threshold (float): The threshold value.

    Returns:
        list: A list of tuples (timestep, value) for timesteps where the value falls below the threshold.
    """
    results = []
    current_timestep = None

    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace from the line
            line = line.strip()

            # Check if the line indicates a new timestep
            if line.startswith("Time step"):
                try:
                    # Extract the timestep number
                    current_timestep = int(line.split()[2])
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not extract timestep from line: {line}. Error: {e}")

            # Check if the line contains "Average energy of ensemble"
            elif "Lowest energy walker" in line:
                try:
                    # Extract the value after "Average energy of ensemble:"
                    parts = line.split(":")
                    value_part = parts[1].split()[0]  # Extract the numeric value
                    value = float(value_part)

                    # Check if the value falls below the threshold
                    if value < threshold and current_timestep is not None and current_timestep > 0:
                        results.append((current_timestep, value))

                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping line due to error: {e}")

    if results:
        return results[0]
    else:
        return 0, 0
    
def check_for_holes_2(file_path, threshold):
    """
    Parses a file to find the timesteps where "Lowest energy walker" falls below a threshold.

    Args:
        file_path (str): Path to the output file.
        threshold (float): The threshold value.

    Returns:
        list: A list of tuples (timestep, value) for timesteps where the value falls below the threshold.
    """
    results = []
    current_timestep = None

    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace from the line
            line = line.strip()

            # Check if the line indicates a new timestep
            if line.startswith("Time step"):
                try:
                    # Extract the timestep number
                    current_timestep = int(line.split()[2])
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not extract timestep from line: {line}. Error: {e}")

            # Check if the line contains "Average energy of ensemble"
            elif "Average energy of ensemble:" in line:
                try:
                    # Extract the value after "Average energy of ensemble:"
                    parts = line.split(":")
                    value_part = parts[1].split()[0]  # Extract the numeric value
                    value = float(value_part)

                    # Check if the value falls below the threshold
                    if value < threshold and current_timestep is not None and current_timestep > 0:
                        results.append((current_timestep, value))

                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping line due to error: {e}")

    if results:
        return results[0]
    else:
        return 0, 0
    
def parse_log_file(file_path):
    lowest = []
    highest = []
    average = []

    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace from the line
            line = line.strip()

            # Check if the line contains lowest energy info
            if "Lowest energy walker" in line:
                try:
                    parts = line.split(":")
                    value_part = parts[1].split()[0]  # Extract the numeric value
                    value = float(value_part)

                    lowest.append(value)

                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping line due to error: {e}")

            # Check if the line contains highest energy info
            elif "Highest energy walker" in line:
                try:
                    parts = line.split(":")
                    value_part = parts[1].split()[0]  # Extract the numeric value
                    value = float(value_part)

                    highest.append(value)
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping line due to error: {e}")

            # Check if the line contains average energy info
            elif "Average energy" in line:
                try:
                    parts = line.split(":")
                    value_part = parts[1].split()[0]  # Extract the numeric value
                    value = float(value_part)

                    average.append(value)
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping line due to error: {e}")

            else:
                pass

    return highest, lowest, average

def plot_log_stats(file_path,ymax=None):
    highest, lowest, average = parse_log_file(file_path)

    timesteps = np.arange(1,len(highest)+1,1)
    plt.plot(timesteps,highest,label='Highest')
    plt.plot(timesteps,lowest,label='Lowest')
    plt.plot(timesteps,average,label='Average')

    plt.legend()
    plt.xlabel('Timestep (a.u.)')
    plt.ylabel(r'Potential Energy (cm$^{-1}$)')

    if ymax != None:
        plt.ylim(0,ymax)
    plt.show()


def plot_1d_dists(data_path,dist_type,atom_indices,name,cumulative=False,bins=None,xlims=None,ylims=None,stats=False):
    cds = np.load(f'{data_path}_cds.npy')
    cds = pv.Constants.convert(cds,'angstroms',to_AU=False) # Conversion of cds to angstroms
    dws = np.load(f'{data_path}_dws.npy')

    analyzer = pv.AnalyzeWfn(cds)

    dist_total = []
    dws_total = []

    for i in range(len(atom_indices)):

        if dist_type == 'bond_length':
            dist = analyzer.bond_length(atom_indices[i][0],atom_indices[i][1])
            x_label_end = 'Distance ($\AA$)'
        elif dist_type == 'bond_angle':
            dist = analyzer.bond_angle(atom_indices[i][0],atom_indices[i][1],atom_indices[i][2])
            dist = np.rad2deg(dist)
            x_label_end = 'Angle (degrees)'
        elif dist_type == 'dihedral':
            dist = analyzer.dihedral(atom_indices[i][0],atom_indices[i][1],atom_indices[i][2],atom_indices[i][3])
            dist = np.rad2deg(dist)
            x_label_end = 'Angle (degrees)'
        else:
            return TypeError('Not a valid distribution type')
        
        if cumulative == False:
            if bins == None:
                n, bin = np.histogram(dist,weights = dws, bins = 100, density = True)
            else:
                n, bin = np.histogram(dist,weights = dws, bins = bins, density = True)

            bin_centers = (bin[:-1] + bin[1:]) / 2

            plt.plot(bin_centers, n, label = f'{atom_indices[i][0]}-{atom_indices[i][1]}')

        else:
            dist_total.append(dist)
            dws_total.append(dws)

    if cumulative == True:

        dist_total = np.concatenate(dist_total)
        dws_total = np.concatenate(dws_total)

        if bins == None:
            n, bin = np.histogram(dist_total,weights = dws_total, bins = 100, density = True)
        else:
            n, bin = np.histogram(dist_total,weights = dws_total, bins = bins, density = True)

        bin_centers = (bin[:-1] + bin[1:]) / 2

        plt.plot(bin_centers, n)

    if xlims != None:
        plt.xlim(xlims[0],xlims[1])

    if ylims != None:
        plt.ylim(ylims[0],ylims[1])

    plt.legend()
    plt.xlabel(rf'{name} {x_label_end}')
    plt.ylabel('Probability Amplitude')
    
    plt.show()
    plt.clf()

def plot_mult_dists(systems,data_paths,dist_type,atom_indices,name,
                    cumulative=False,bins=None,xlims=None,ylims=None,colors=None,linestyles=None):
    for p in range(len(systems)):
        cds = np.load(f'{data_paths[p]}_cds.npy')
        cds = pv.Constants.convert(cds,'angstroms',to_AU=False) # Conversion of cds to angstroms
        dws = np.load(f'{data_paths[p]}_dws.npy')

        analyzer = pv.AnalyzeWfn(cds)

        dist_total = []
        dws_total = []

        for i in range(len(atom_indices[p])):

            if dist_type == 'bond_length':
                dist = analyzer.bond_length(atom_indices[p][i][0],atom_indices[p][i][1])
                x_label_end = 'Distance ($\AA$)'
            elif dist_type == 'bond_angle':
                dist = analyzer.bond_angle(atom_indices[p][i][0],atom_indices[p][i][1],atom_indices[i][2])
                dist = np.rad2deg(dist)
                x_label_end = 'Angle (degrees)'
            elif dist_type == 'dihedral':
                dist = analyzer.dihedral(atom_indices[p][i][0],atom_indices[p][i][1],atom_indices[i][2],atom_indices[i][3])
                dist = np.rad2deg(dist)
                x_label_end = 'Angle (degrees)'
            else:
                return TypeError('Not a valid distribution type')
            
            if cumulative == False:
                if bins == None:
                    n, bin = np.histogram(dist,weights = dws, bins = 100, density = True)
                else:
                    n, bin = np.histogram(dist,weights = dws, bins = bins, density = True)

                bin_centers = (bin[:-1] + bin[1:]) / 2

                plt.plot(bin_centers, n, label = f'{systems[p]}: {atom_indices[p][i][0]}-{atom_indices[p][i][1]}')

            else:
                dist_total.append(dist)
                dws_total.append(dws)

        if cumulative == True:

            dist_total = np.concatenate(dist_total)
            dws_total = np.concatenate(dws_total)

            if bins == None:
                n, bin = np.histogram(dist_total,weights = dws_total, bins = 100, density = True)
            else:
                n, bin = np.histogram(dist_total,weights = dws_total, bins = bins, density = True)

            bin_centers = (bin[:-1] + bin[1:]) / 2

            if colors == None:
                if linestyles == None:
                    plt.plot(bin_centers, n, label=f'{systems[p]}')
                else:
                    plt.plot(bin_centers, n, label=f'{systems[p]}',linestyle=linestyles[p])
            else:
                if linestyles == None:
                    plt.plot(bin_centers, n, label=f'{systems[p]}',color=colors[p])
                else:
                    plt.plot(bin_centers, n, label=f'{systems[p]}',color=colors[p],linestyle=linestyles[p])

    if xlims != None:
        plt.xlim(xlims[0],xlims[1])

    if ylims != None:
        plt.ylim(ylims[0],ylims[1])

    plt.legend(frameon=False)
    plt.xlabel(rf'{name} {x_label_end}')
    plt.ylabel('Probability Amplitude')
    
    plt.show()
    plt.clf()

def calc_dist_stats(data_paths,dist_type,atom_indices):

    exp_list = []
    fwhm_list = []

    for p in range(len(data_paths)):

        cds = np.load(f'{data_paths[p]}_cds.npy')
        cds = pv.Constants.convert(cds,'angstroms',to_AU=False) # Conversion of cds to angstroms
        dws = np.load(f'{data_paths[p]}_dws.npy')

        analyzer = pv.AnalyzeWfn(cds)

        dist_total = []
        dws_total = []
        for i in range(len(atom_indices)):

            if dist_type == 'bond_length':
                dist = analyzer.bond_length(atom_indices[i][0],atom_indices[i][1])
            elif dist_type == 'bond_angle':
                dist = analyzer.bond_angle(atom_indices[i][0],atom_indices[i][1],atom_indices[i][2])
                dist = np.rad2deg(dist)
            elif dist_type == 'dihedral':
                dist = analyzer.dihedral(atom_indices[i][0],atom_indices[i][1],atom_indices[i][2],atom_indices[i][3])
                dist = np.rad2deg(dist)
            else:
                return TypeError('Not a valid distribution type')
            
            dist_total.append(dist)
            dws_total.append(dws)

        dist_total = np.concatenate(dist_total)
        dws_total = np.concatenate(dws_total)

        total_exp_val = analyzer.exp_val(operator=dist_total, dw=dws_total)
        exp_list.append(total_exp_val)

        n, bins = np.histogram(dist_total,bins=100,weights=dws_total,density=True)
        # Step 2: Find the peak of the histogram
        max_bin_height = np.max(n)
        max_bin_index = np.argmax(n)

        # Step 3: Calculate the half maximum
        half_max = max_bin_height / 2.0

        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Find left crossing point
        left_index = np.where(n[:max_bin_index] < half_max)[0][-1]
        left_half_max_x = bin_centers[left_index] + (half_max - n[left_index]) * (bin_centers[left_index + 1] - bin_centers[left_index]) / (n[left_index + 1] - n[left_index])

        # Find right crossing point
        right_index = np.where(n[max_bin_index:] < half_max)[0][0] + max_bin_index
        right_half_max_x = bin_centers[right_index - 1] + (half_max - n[right_index - 1]) * (bin_centers[right_index] - bin_centers[right_index - 1]) / (n[right_index] - n[right_index - 1])

        # Step 5: Calculate the FWHM
        fwhm = right_half_max_x - left_half_max_x
        fwhm_list.append(fwhm)

    return np.mean(exp_list), np.std(exp_list), np.mean(fwhm_list), np.std(fwhm_list)
        