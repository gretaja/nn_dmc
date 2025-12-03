# Neural Networks for DMC (nn_dmc)
Python scripts for training and evaluating neural network potentials for use in DMC simulations.

Potentials developed for OH<sup>-</sup>(H<sub>2</sub>O)<sub>1-3</sub> and H<sub>3</sub>O<sup>-</sup> utilize a feedforward neural network framework, while recent work in developing potentials for OH<sup>-</sup>(H<sub>2</sub>O)<sub>5</sub> utilize an equivariant graph neural network framework. All models utilize a PyTorch framework.

Publications: [**OH<sup>-</sup>(H<sub>2</sub>O)<sub>1-3</sub>**](https://pubs.acs.org/doi/10.1021/acs.jpca.4c08826), [**H<sub>3</sub>O<sup>-</sup>**](https://pubs.acs.org/doi/10.1021/acs.jpca.5c06590)

Dependencies: torch, numpy, matplotlib, pyvibdmc

# Contents:
 - nn_training: Python scripts for initializing a feedforward NN (train_nn_model) or equivariant graph NN (train_egnn_model) model for training in a HPC environment using GPUs

 - model_evaluation: Python scripts for evaluating the trained feedforward NN (nn_potential) or equivariant graph NN (egnn_potential) on-the-fly, like in a production-run DMC simulation

 - training_data_collection: Python scripts for the small DMC simulations used to generate training geometries and corresponding potential energies from the MOB-ML reference potential

 - src/nn_dmc: Python scripts I would like to eventually turn into a pip-installable package for all NN DMC related research tasks. Currently includes:

    * model_analysis: functions for calculating test errors and creating corresponding visualizations of these errors for the feedforward NNs

    * egnn_model_analysis: functions for calculating test errors and creating corresponding visualizations of these errors for the equivariant graph NNs

    * molec_descriptors: functions to calculate the Coulomb matrix molecular descriptor from Cartesian coordinates and the various sorting procedures used to account for permutational invariance of like-atoms and like-molecules

    * sim_analysis: useful functions for analysis of DMC data, like identifying "holes" in the simulation, plotting projections of the probability amplitude on to chosen geometric coordinates, calculating statistics among multiple idependent simulations of the same system, and comparing simulation data between different molecular systems or different coordinates

    * misc: other potentially useful functions that don't really fit anywhere else
