import os, sys

import numpy as np

import ase, ase.io
from ase import Atoms

__JULIA__ = False
try:
    from julia import Julia
    # Avoid precompile issues
    jl = Julia(compiled_modules=False)  
    
    # Import a Julia module
    from julia import Main

    # current_dir = os.path.dirname(__file__)
    # print(os.path.dirname(__file__))
    Main.include("/home/antonio/IOcp2k/Modules/time_correlation.jl")
    # Main.include(os.path.join(os.path.dirname(__file__), "time_correlation.jl"))
    __JULIA__ = True
except:
    __JULIA__ = False
    raise NotImplementedError("A pure python version is not available. Try to run it with python-jl")


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

import scipy, scipy.optimize, scipy.fft

import copy

from mpi4py import MPI

import time

ANG2_PS_TO_SI = 1e-8
ANG_PS_TO_SI = 1e+2
KELVIN_TO_EV = 8.6173e-5
_e_charge_ = 1.602176634

matplotlib.use('tkagg')
class Conductivity:
    """
    GET CONDUCTIVITY FROM ASE ATOMS
    ===============================
    """
    
    def __init__(self, **kwargs):
        """
        DEFINE THE ATTRIBUTES OF THE CLASS
        ==================================
        """
        #ase atoms object
        self.atoms = None
        
        # PICOSECOND
        self.dt = None
        
        # Tempeartue in Kelvin
        self.T = None
        
        #  Volume in Angstrom3
        self.V = None

        # all the volumes
        self.volumes = None
        
        # atomic types with oxidation charges
        self.types_q = None

        #int the number of snapshots
        self.N = None

        # The time correlations
        self.correlations = None

        # The errors
        self.error_correlations = None

        # All the currents
        self.js = None


        # Setup the attribute control
        self.__total_attributes__ = [item for item in self.__dict__.keys()]
        # This must be the last attribute to be setted
        self.fixed_attributes = True 

        # Setup any other keyword given in input (raising the error if not already defined)
        for key in kwargs:
            self.__setattr__(key, kwargs[key])
        
    def init(self, ase_atoms = None, dt = 0.5, T = 300, types_q = {"Na" : +1, "Cl" : -1}):
        """
        INITIALIZE
        ==========

        Paramters:
        ----------
            -ase_atoms: list of ase atoms objects
            -dt: time step in FEMPTOSECOND
            -T: temperature in Kelvin
            -types_q: dictioary
        """
        if ase_atoms is None:
            raise ValueError("Provide ase atoms in input")

        self.atoms = ase_atoms

        # in picosecond
        self.dt = dt * 1e-3

        self.T = T

        self.V = 0.

        self.types_q = types_q

        self.N = len(self.atoms)

        # in picosecond
        self.t = np.arange(self.N) * self.dt

        # in Angs3
        self.volumes = np.zeros(self.N, dtype = type(dt))

        # The correlations with the errors
        self.correlations = np.zeros(self.N, dtype = type(dt))

        self.error_correlations = np.zeros(self.N, dtype = type(dt))

        # Get all the currents in Angstrom/picosecond
        self.js = np.zeros((self.N, 3), dtype = type(self.dt))


    
    def get_Vcom_from_ase_atoms(self, index):
        """
        GET THE CENTER OF MASS VELOCITIES
        =================================
    
        Parameters:
        -----------
            -index: int, the index of the corresponding atoms object
    
        Returns:
        --------
            -V_com: np.array with shape 3, the center of mass velocity in ANGSROM/PICOSECOND
        """
        # Get the masses
        masses = np.asarray(self.atoms[index].get_masses())
    
        # Get the center of mass position
        V_com = np.einsum('i, ia -> a', masses, self.atoms[index].get_velocities()[:,:]) /np.sum(masses)
    
        return V_com

        
    def get_selected_atoms_qs(self, index):
        """
        SELECT ATOMIC INDICES FROM ASE ATOMS
        ====================================

        Parameters:
        -----------
            -index: int, the index of the corresponding atoms object

        Returns;
        --------
            -selected_atoms: np.array with the indices of the selected atoms from self.types_q
            -selected_qs: np.array with the ox charges correpsonding to the selected atoms
        """
        # Get the total number of atoms
        N_at_tot = len(self.atoms[index].positions[:,0])
        
        # Get all the chemical symbols
        chem_symb = np.asarray(self.atoms[index].get_chemical_symbols()).ravel()
        
        # Select only the atoms I want
        selected_atoms = []
        # with the corresponign ox charges
        selected_qs    = []
        
        for target_atom in self.types_q.keys():
            # Select the atomic indices corresponding to the atoms that I want
            my_sel = np.arange(N_at_tot)[np.isin(chem_symb, [target_atom])]
            # Select the atoms contributing to the ionic conductivity
            selected_atoms.append(my_sel)
            # Get the corresponding oxidations charges
            selected_qs.append([self.types_q[target_atom]] * len(my_sel))
            
        # Transform everything in array
        selected_atoms = np.asarray(selected_atoms).ravel()
        selected_qs    = np.asarray(selected_qs, dtype = int).ravel()

        return selected_atoms, selected_qs

    def get_current_from_ase_atoms(self, index, debug = False, subtract_vcom =True, return_volume = False):
        """
        GET THE CURRENT
        ===============
    
        Get the total current for the single snapshopts ANGSTROM/PICOSECOND
    
        ..math: J(t) = \sum_{i=1}^{N} q_{i} v_{i}(t)
        
        Parameters:
        -----------
            -index: int, the index of the corresponding atoms object
    
        Returns:
        --------
            -J: np.array with shape (N_at_selected, 3), the current for each selected atomic types in the simulation box
        """
        # Select the atoms that contribute to the conductivity with the ox charges corresponind to them
        selected_atoms, selected_qs = self.get_selected_atoms_qs(index)
        
        if debug:
            print('Selected atoms', chem_symb[selected_atoms])
            print('Selected oxchs', selected_qs)
    
        # Get the velocities of the selected atoms
        selected_velocities = self.atoms[index].get_velocities()[selected_atoms,:]
        
        if subtract_vcom:
            # Subtract the COM velocities Angstrom/Picosecond
            selected_velocities[:,:] -= self.get_Vcom_from_ase_atoms(index)
    
        J = np.einsum('i, ia -> ia', selected_qs, selected_velocities)
    
        if debug:
            print('q', selected_qs)
            print('J', J)
            print('V', selected_velocities)
    
        J_all = np.einsum('ia -> a', J)

        if return_volume:
            return J_all, self.atoms[index].get_volume()
            
        return J_all

    def get_time_correlation_fft(self):
        """
        RETURN THE TIME CORREALTION FUNCTION USING FFT
        ==============================================
        ======
        """
        # Duplicate in size but set all zeros
        new_js = np.zeros((2 * self.N, 3), dtype = type(self.js[0,0]))
        # Set the inital values and the others will be zeros
        new_js[:self.N,:] = self.js[:,:]
        
        # The J J correlation in Fourier
        J_omega  = scipy.fft.fft(new_js[:,:], axis = 0)
        
        # Get the modulus squre and perform the dot product
        J2_omega = np.einsum('ia, ia -> i', np.conjugate(J_omega), J_omega)
        
        # Perform the inverse Fourier transform
        C_t = scipy.fft.ifft(J2_omega)

        # Check the imaginary part
        if np.imag(C_t).max() > 1e-5:
            print("Discarting imaginary part...")
        # Apply the normalization
        normalization = np.arange(self.N, 0, -1)
        
        C_t_real = np.real(C_t)[:self.N] /normalization
        
        return C_t_real

    def get_current_current_correlation_function(self, use_julia = True):
        """
        GET THE CURRENT CURRENT AUTOCORRELATION FUNCTION
        ================================================
        """
        # Range on all the snapshots to get the currents
        for i in range(self.N):
            self.js[i,:], self.volumes[i] = self.get_current_from_ase_atoms(i, return_volume = True)

        if use_julia and __JULIA__:
            # Get the julia windowed average (BRUTE FORCE)
            self.correlations, self.error_correlations = Main.get_time_correlation_vector(self.js, self.N)
        else:
            pass
            # self.correlations, self.error_correlations = self.get_time_correlation_fft()
    
        self.V = np.average(self.volumes)


    def plot_current_current_correlation_function(self):
        """
        PLOT THE CURRENT CURRENT CORRELATION FUNCTION
        =============================================
        """

        if np.all(self.correlations == 0):
            self.get_current_current_correlation_function()
            
        fig = plt.figure(figsize=(8, 8))

        gs = gridspec.GridSpec(1, 2, figure = fig)
        
        ax = fig.add_subplot(gs[0,0])
        ax.set_title("Current-current time correlation {}".format(self.types_q.keys()))
        # PICOSCOND and ANGSTROM^2
        ax.errorbar(self.t, self.correlations, yerr = self.error_correlations, lw = 3, color = "k")
        
        ax.set_xlabel('Time [ps]', size = 15)
        ax.set_ylabel('$C_{JJ}(t)$ [Angstrom$^2$/ps$^2$]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)

        ax = fig.add_subplot(gs[0,1])
        ax.set_title("Current-current time correlation {}".format(self.types_q.keys()))
        J_omega  = scipy.fft.fft(self.correlations[:], axis = 0)
        omega    = scipy.fft.fftfreq(self.N, self.dt * 1e-12)
        # PICOSCOND and ANGSTROM^2
        ax.plot(omega, np.real(J_omega), lw = 3, color = "k")
        
        ax.set_xlabel('$\\omega$', size = 15)
        ax.set_ylabel('$C_{JJ}(\\omega)$ [Angstrom$^2$/ps$^2$]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # plt.legend()
        plt.tight_layout()

        plt.show()


    def plot_sigma(self, up_to_t = None):
        """
        PLOT THE CONDUCTIVITY
        =====================
        """
        if np.all(self.correlations == 0):
            self.get_current_current_correlation_function()
            
        fig = plt.figure(figsize=(8, 8))

        factor =  1e+3 * _e_charge_ /(self.V * 3 * self.T)

        JJ_norm = self.correlations * factor
        err_JJ_norm = self.error_correlations * factor

        if up_to_t is None:
            up_to_t = self.t[-1] /2

        # Now integrate the conductivity
        mask = self.t < up_to_t
        integrated_sigma = np.sum(JJ_norm[mask]) * self.dt
        
        # Plot everyhing
        gs = gridspec.GridSpec(1, 1, figure = fig)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title("Conductivity for {}".format(self.types_q.keys()))
        # PICOSCOND and ANGSTROM^2
        ax.errorbar(self.t, JJ_norm, yerr = err_JJ_norm, lw = 3, color = "k")
        ax.fill_between(self.t[mask], JJ_norm.min(), JJ_norm[mask], alpha=0.3, color='gray', label ='$\\sigma$' + '={:.1e} Si/m'.format(integrated_sigma))
        ax.set_xlabel('Time [ps]', size = 15)
        ax.set_ylabel('$\\frac{<J(t)J(0)>}{3VkT}$ [Si/(m ps)]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        plt.legend(fontsize = 15)
        plt.tight_layout()

        plt.show()


    def test_implementation2(self):
        """
        A TEST FUNCTION
        ===============
        """
        new_js = np.zeros((2 * self.N, 3))
        new_js[:self.N,:] = self.js[:,:]
        # The J J correlation in Fourier
        J_omega  = scipy.fft.fft(new_js[:,:], axis = 0)

        # Get the square
        J2_omega = np.einsum('ia, ia -> i', np.conjugate(J_omega), J_omega)
        C_t = scipy.fft.ifft(J2_omega)
    
        if np.imag(C_t).max() > 1e-5:
            print("Discarting imaginary part...")
        normalization = np.arange(self.N, 0, -1)
        C_t_real = np.real(C_t[:self.N]) /normalization
    
        matplotlib.use('tkagg')
        # Create base plot
        fig, ax1 = plt.subplots()
        
        ax1.errorbar(self.t, self.correlations, yerr = self.error_correlations, lw = 3, color = "k", label = "Julia")
        ax1.legend()
        
        ax2 = ax1#.twinx()
        mask = self.t < self.t[-1]/2
        ax2.plot(self.t, C_t_real, color = 'red', label = "python Fourier")
        ax2.legend(loc = 'upper left')
        plt.show()


        
    def test_implementation(self):
        """
        A TEST FUNCTION
        ===============

        We compare the result of the windowed average in julia with FFT of python
        """
        # The J J correlation in Fourier
        J_omega  = scipy.fft.fft(self.js[:,:], axis = 0)
        
        # Get the modulus squre and perform the dot product
        J2_omega = np.einsum('ia, ia -> i', np.conjugate(J_omega), J_omega)
        
        # Perform the inverse Fourier transform
        C_t = scipy.fft.ifft(J2_omega)

        # Check the imaginary part
        if np.imag(C_t).max() > 1e-5:
            raise ValueError("Discarting imaginary part...")
        # Apply the normalization
        normalization = np.arange(self.N, 0, -1)
        # Get the final result
        C_t_real = np.real(C_t) /normalization
    
        matplotlib.use('tkagg')
        # Create base plot
        fig, ax1 = plt.subplots()
        
        ax1.errorbar(self.t, self.correlations, yerr = self.error_correlations, lw = 3, color = "k", label = "Julia")
        ax1.legend()
        
        ax2 = ax1#.twinx()
        mask = self.t < self.t[-1]/2
        ax2.plot(self.t[mask], C_t_real[mask], color = 'red', lw = 3, ls = ":", label = "python Fourier")
        ax2.legend(loc = 'upper left')
        plt.show()
