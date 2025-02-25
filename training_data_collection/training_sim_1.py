import numpy as np

#included for running in mobml-collab directory on klone
import sys
pyvib = '/home/packages'
sys.path.insert(0, pyvib)

import pyvibdmc as pv
from pyvibdmc import potential_manager as pm
from pyvibdmc.simulation_utilities.mpi_imp_samp_manager import MPI_Potential, MPI_ImpSampManager

system_name = 'h3o2' #replace with system of interest

#the atom list should match the order of the atoms in the mobml minimum energy structure files
if system_name == 'h3o2':
    atom_list = ['H','O','H','O','H']
elif system_name == 'h5o2':
    atom_list = ['H','H','H','O','H','H','O']
elif system_name == 'h5o3':
    atom_list = ['O','H','O','H','H','O','H','H']
elif system_name == 'h7o4':
    atom_list = ['O','H','O','H','H','O','H','H','O','H','H']
elif system_name == 'h9o5':
    atom_list = ['O','H','O','H','H','O','H','H','O','H','H','O','H','H']

if __name__ == '__main__': #if using multiprocessing on windows / mac, you need to encapsulate using this line
    #this directory is part of the one you copied that is outside of pyvibdmc.
    pot_dir = '/home/scripts/potential/' 
    py_file = f'{system_name}_pyscfpot.py'
    pot_func = 'scf_pot' 

    #The Potential object assumes you have already made a .so file and can successfully call it from Python
    pot = MPI_Potential(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir)

    # Starting Structure
    coords = np.load(f'{system_name}_mobml_min.npy')

    start_coord = np.expand_dims(coords,axis=0) # Make it (1 x num_atoms x 3)
    sim_num = 0
    myDMC = pv.DMC_Sim(sim_name=f'{system_name}_{sim_num}',
                            output_folder=f'/home/output/{system_name}/500w_1000ts_1dt',
                            weighting='discrete', #or 'continuous'. 'continuous' keeps the ensemble size constant.
                            num_walkers=500, #number of geometries exploring the potential surface
                            num_timesteps=1000, #how long the simulation will go. (num_timesteps * delta_t atomic units of time)
                            equil_steps=500, #how long before we start collecting wave functions
                            chkpt_every=500, #checkpoint the simulation every "chkpt_every" time steps
                            wfn_every=999, #collect a wave function every "wfn_every" time steps
                            desc_wt_steps=300, #number of time steps you allow for descendant weighting per wave function
                            atoms=atom_list,
                            delta_t=1, #the size of the time step in atomic units
                            potential=pot,
                            start_structures=start_coord, #can provide a single geometry, or an ensemble of geometries
                            DEBUG_save_training_every=1, #collect walkers at every time step
                            masses=None #can put in artificial masses, otherwise it auto-pulls values from the atoms string
    )
    myDMC.run()