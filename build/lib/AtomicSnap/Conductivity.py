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

import AtomicSnap
from AtomicSnap import AtomicSnapshots

import scipy, scipy.optimize, scipy.fft

import copy

import time

ANG2_PS_TO_SI = 1e-8
ANG_PS_TO_SI = 1e+2
KELVIN_TO_EV = 8.6173e-5
_e_charge_   = 1.602176634

THZ_TO_CM = 33

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

        # all the volumes
        self.volumes = None

        # get the temperatures in Kelvin
        self.temperatures = None
        
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
        
    def init(self, ase_atoms = None, dt = 0.5, temperatures = None, types_q = None):
        """
        INITIALIZE
        ==========

        Paramters:
        ----------
            -ase_atoms: list of ase atoms objects
            -dt: time step in FEMPTOSECOND
            -temperatures: temperatures in Kelvin
            -types_q: dictionary, example {"Na" : +1, "Cl" : -1}
        """
        if ase_atoms is None:
            raise ValueError("Provide ase atoms in input")

        self.atoms = ase_atoms

        # in picosecond
        self.dt = dt * 1e-3

        # All the temperature in Kelvin
        self.temperatures = temperatures

        # Get atomic types with the ox charges
        self.types_q = types_q

        # The steps of the trajectory
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
        # Get all the masses
        masses = np.asarray(self.atoms[index].get_masses())
    
        # Get the center of mass velocity in ANGSTROM/PICOSECOND
        V_com = np.einsum('i, ia -> a', masses, self.atoms[index].get_velocities()[:,:]) /np.sum(masses)
    
        return V_com

        
    def get_selected_atoms_qs(self, index):
        """
        SELECT ATOMIC INDICES FROM ASE ATOMS
        ====================================

        Select the atoms using the dictioanry self.types_q

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
        
        # Get all the chemical symbols as array
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
            print(target_atom)
            print(my_sel)
            print([self.types_q[target_atom]] * len(my_sel))
            print()
            
        # Transform everything in array
        selected_atoms = np.asarray(selected_atoms).ravel()
        selected_qs    = np.asarray(selected_qs, dtype = int).ravel()

        return selected_atoms, selected_qs

    def get_current_from_ase_atoms(self, index, debug = False, subtract_vcom = True, return_volume = False):
        """
        GET THE CURRENT
        ===============
    
        Get the total current for the single snapshopts ANGSTROM/PICOSECOND
    
        ..math: J(t) = \sum_{i=1}^{N} q_{i} v_{i}(t)
        
        Parameters:
        -----------
            -index: int, the index of the corresponding atoms object
            -debug: bool
            -subtract_vcom: bool, if True we subtract the COM velocity
            -return_volume: bool, if True we return the volume in ANGSTROM3 (useful for computing the average volume)
    
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

        # Mutliply the velocitivies with the charges
        J = np.einsum('i, ia -> ia', selected_qs, selected_velocities)
    
        if debug:
            print('sel atoms', selected_atoms)
            print('q', selected_qs)
            print('J', J)
            print('V', selected_velocities)
    
        J_all = np.einsum('ia -> a', J)

        if return_volume:
            return J_all, self.atoms[index].get_volume()
            
        return J_all

    def get_time_correlation_fft(self, normalize = False):
        """
        RETURN THE TIME CORREALTION FUNCTION USING FFT
        ==============================================

        Alternative implementation of the windowed average to get the time correlation function

        Here we use scipy FFT and IFFT. 
        
        We use the padding of the signal, i.e. we double the trajectory and we set the second half to zero

        Parameters:
        -----------
            -normalize: bool, if True we impose the normalization, useful to compare with the Julia implementation

        Returns:
        --------
            -C_t_real: np.array with shape self.N: the time correaltion function of the dipole
        """
        print(' We compute the dipole dipole time correlation function using PYTHON FFT!')
        # Duplicate in size but set all zeros
        j = np.zeros((2 * self.N, 3), dtype = type(self.js[0,0]))
        
        # Set the inital values and the others will be zeros
        j[:self.N,:] = self.js[:,:]
        
        # The J J correlation in Fourier
        j_omega  = scipy.fft.fft(j[:,:], axis = 0)
        
        # Get the modulus square and perform the dot product
        j2_omega = np.einsum('ia, ia -> i', np.conjugate(j_omega), j_omega)
        
        # Perform the inverse Fourier transform
        C_t = scipy.fft.ifft(j2_omega)

        # Check the imaginary part
        if np.imag(C_t).max() > 1e-5:
            warnings.warn(" WARNING: Discarting imaginary part...")

        if normalize:
            # Apply the normalization
            print(' Normalizing the correlation function\n')
            normalization = np.arange(self.N, 0, -1)
            C_t_real = np.real(C_t)[:self.N] /normalization
        else:
            print(' NOT normalizing the correlation function\n')
            C_t_real = np.real(C_t)[:self.N]
        
        return C_t_real

    def set_correlations(self, use_julia = False, python_normalize = False):
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
            # Use pyhon
            self.correlations = self.get_time_correlation_fft(normalize = python_normalize) 
            self.error_correlations = np.zeros(self.N)


    def get_spectra(self, correlations, delta = None, pad = False):
        """
        GET THE CORRELATION FUNCTION SPECTRA
        ====================================

        Get the FT of the current current correlation function

        ..math: C(\omega) = \int dt \exp{i \omega t - delta t} \left\langle j(t) j(0) \right\rangle

        Parameters:
        -----------
            -correlations: np.array: the time correlations
            -delta: the smearing in cm-1
            -pad: bool if True we pad the correlations in input
            
        Returns:
        --------
            -omega: np.array, the frequenceis in cm-1
            -sigma_spectra: np.array, the intensity of the sigma spectra
            
        """
        print("\n\nC[OMEGA] | We are computing the dynamical conductivity!")
        # Get the leght of the time correlation
        length = len(correlations)
        
        # The fft has dimension (eAngstrom)^2
        new_correlations = np.zeros(length * 2, dtype = type(correlations[0]))

        if pad:
            print("C[OMEGA] | PAD")
            new_correlations[:length] = correlations[:]
            new_correlations[0] = 0.5 * new_correlations[0]
        else:
            print("C[OMEGA] | NO PAD")
            new_correlations[:length] = correlations[:][::-1]
            new_correlations[length:] = correlations[:]
            # new_correlations[0] = 0.5 * new_correlations[0]

        
        # Picosecond
        t = np.arange(2 * length) * self.dt
        if delta is None:
            print("C[OMEGA] | No smearing!\n")
            new_correlations_omega  = scipy.fft.fft(new_correlations) 
        else:
            # Picoseconds
            tau = 1/(delta * THZ_TO_CM**-1)
            print("C[OMEGA] | Using a smearing of {:.2f} cm-1 {:.2f} ps\n".format(delta, tau))
            new_correlations_omega  = scipy.fft.fft(new_correlations * np.exp(- t/tau)) 
        print()
        
        # The freq are in Thz
        omega     = scipy.fft.fftfreq(2 * length, self.dt)
        # Go in cm-1
        omega *= THZ_TO_CM

        return omega, np.imag(new_correlations_omega) 




    def plot_correlation_function(self, omega_min_max = [0, 5000], smearing = None, pad = False, save_data = False):
        """
        PLOT THE CURRENT CURRENT CORRELATION FUNCTION
        =============================================
        """
        if np.all(self.correlations == 0):
            self.set_correlations()
            
        fig = plt.figure(figsize=(10, 5))

        gs = gridspec.GridSpec(1, 2, figure = fig)
        
        ax = fig.add_subplot(gs[0,0])
        ax.set_title("Current-current time correlation {}".format(self.types_q.keys()))
        # PICOSCOND and ANGSTROM^2/PICOSCOND2
        ax.errorbar(self.t, self.correlations, yerr = self.error_correlations, lw = 3, color = "k")
        
        ax.set_xlabel('Time [ps]', size = 15)
        ax.set_ylabel('$C_{JJ}(t)$ [Angstrom$^2$/ps$^2$]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)

        ax = fig.add_subplot(gs[0,1])

        # Get the FOurier tranform of the conductivity
        omega, sigma = self.get_spectra(self.correlations, delta = smearing, pad = pad)
        if omega_min_max is None:
            # PICOSCOND and ANGSTROM^2
            ax.plot(omega, sigma, lw = 3, color = "purple")
        else:
            mask = (omega_min_max[0] < omega) & (omega < omega_min_max[1])
            # PICOSCOND and ANGSTROM^2
            ax.plot(omega[mask], (sigma)[mask], lw = 3, color = "purple")
        
        ax.set_xlabel('$\\omega$ [cm$^{-1}$]', size = 15)
        ax.set_ylabel('$C_{JJ}(\\omega)$ [Angstrom$^2$/ps$^2$]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        plt.tight_layout()
        plt.show()

        
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 2, figure = fig)
        ax = fig.add_subplot(gs[0,0])
        # PICOSCOND and ANGSTROM^2
        ax.errorbar(self.t, self.correlations, yerr = self.error_correlations, lw = 3, color = "k")
        ax.set_xlabel('Time [ps]', size = 15)
        ax.set_ylabel('$C_{JJ}(t)$ [Angstrom$^2$/ps$^2$]', size = 15)
        ax.tick_params(axis = 'both', labelsize = 12)
        print()

        t_integrations = np.arange(0, self.t[-1], 0.5)[1:]
        all_sigma = np.zeros(len(t_integrations), dtype = type(self.dt))
        for it, tmax in enumerate(t_integrations):
            ax.axvline(tmax)
            mask_integration = self.t < tmax
            conv = self.get_factor(mask_integration)
            sigma = np.sum(self.correlations[mask_integration]) * self.dt * conv
            all_sigma[it] = sigma
            print("\nCONV = {}".format(conv))
            print("INTEGRAL up to t {:.1f} ps | {:.4f}    Si/m".format(tmax, sigma))
        print()
        gs = gridspec.GridSpec(1, 2, figure = fig)
        ax = fig.add_subplot(gs[0,1])
        # PICOSCOND and Si/m
        ax.plot(t_integrations, all_sigma, 'd', lw = 3)
        
        ax.set_xlabel('Time [ps]', size = 15)
        ax.set_ylabel('$\\sigma$ [Si/m]', size = 15)
        ax.tick_params(axis = 'both', labelsize = 12)
        plt.tight_layout()
        plt.show()

        if save_data:
            data = {"x" : list(omega), "y" : list(sigma), "delta" : smearing}
            AtomicSnapshots.save_dict_to_json(data_file, data)


    def get_factor(self, mask):
        """
        GET THE PREFACTOR TO COMPUTE THE CONDUCTIVITY IN SI UNITS
        =========================================================

        We compute the current current corerelation function in Angstrom2/picosecond2
        """
        V_av = np.average(self.volumes[mask])
        T_av = np.average(self.temperatures[mask])
        ########### FACTOR to GET THE CONDUCTIVITY IN Si/m ###########
        factor =  (1e+3 * _e_charge_) /(V_av * 3 * T_av * KELVIN_TO_EV)
        ###############################################################
        
        return factor



    
    # DEBUG

    def test_implementation(self):
        """
        A TEST FUNCTION
        ===============

        Here we compare the correlation function obtained with Julia and the one of python using FFT
        """
        
        # Create base plot
        fig, ax1 = plt.subplots()

        self.set_correlations(use_julia = False, python_normalize = True)
        res1 = np.copy(self.correlations)
        ax1.plot(self.t,self.correlations, lw = 3, color = "k", label = "Julia WINDOWED")
        ax1.legend()

        self.set_correlations(use_julia = True, python_normalize = True)
        res2 = np.copy(self.correlations)
        ax1.plot(self.t, self.correlations , color = 'red', lw = 2.0, label = "Python FFT")
        ax1.legend(loc = 'upper left')
        plt.show()

        if np.abs(res1[1:-1] -res2[1:-1]).max() > 10:
            raise ValueError("The correlation function is not implemented correctly")

