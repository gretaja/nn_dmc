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
