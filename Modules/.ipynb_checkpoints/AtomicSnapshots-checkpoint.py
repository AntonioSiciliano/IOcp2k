import numpy as np

import ase
from ase import Atoms
import ase.geometry
import ase.neighborlist
import ase.calculators.singlepoint
import ase.data

import copy
import os, sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from   matplotlib.ticker import MaxNLocator

import AtomicSnap
from AtomicSnap import AtomicSnapshots

# import Conductivity
from AtomicSnap import Conductivity

from AtomicSnap import Vibrational

import scipy

import MDAnalysis
import MDAnalysis.analysis
import MDAnalysis.analysis.rdf
import MDAnalysis.analysis.msd

import subprocess

import json

__JULIA__ = False
try:
    from julia import Julia
    # Avoid precompile issues
    jl = Julia(compiled_modules=False)  
    
    # Import a Julia module
    from julia import Main
    
    # Main.include("/home/antonio/IOcp2k/Modules/time_correlation.jl")
    __JULIA__ = True
except:
    __JULIA__ = False
    print("It seems that Julia is not available. Try to run with python-jl\n")

try:
    print("Test if Julia works...\n")
    julia.Main.eval('println("Hello from Julia!")')
    __JULIA__ = True
except:
    print("No Julia found!\n")
    __JULIA__ = False

# Conversions
BOHR_TO_ANGSTROM = 0.529177249 
ANGSTROM_TO_BOHR = 1/BOHR_TO_ANGSTROM
HA_TO_EV = 27.2114079527
HA_TO_KELVIN = 315775.326864009
HA_BOHR_TO_EV_ANGSTROM   = HA_TO_EV / BOHR_TO_ANGSTROM
HA_BOHR3_TO_EV_ANGSTROM3 = HA_TO_EV / BOHR_TO_ANGSTROM**3
BAR_TO_GPA = 0.0001
HABOHR3_TO_GPA = 29421.015697000003  #7355.256621097397
AU_TIME_TO_PICOSEC = 2.4188843265864003e-05

BAR_TO_HA_BOHR3 = BAR_TO_GPA * HABOHR3_TO_GPA**-1  

# Velocities
ANG_FEMPTOSEC_TO_HA = 0.04571028907825843

# Dipoles
DEBEYE_TO_eANG = 0.20819433622621247
DEBEYE_TO_AU = DEBEYE_TO_eANG * ANGSTROM_TO_BOHR

class AtomicSnapshots:
    """
    GET ATOMIC SNAPSHOTS OF CP2K 
    ============================


   
    """
    def __init__(self, **kwargs):
        """
        INITIALIZE THE ATOMIC SNAPSHOTS
        ================================

        Get atomic snapshots from cp2k to do some io
        
        Positions are in ANGSTROM

        Forces are in HARTREE/BOHR
        
        Stresses are in HARTREE/BOHR3 (the output of cp2k are BAR)
        
        Velocities are in BOHR/AU TIME

        Time step in FEMPTOSECOND

        Parameters
        ----------
            -**kwargs : any other attribute of the ensemble

        """
        # Atomic position ANGSTROM
        self.positions = None
        # Atomic forces HARTREE/BOHR
        self.forces = None
        # Energies HARTREE
        self.energies = None
        # stress HARTREE/BOHR3
        self.stresses = None
        # Atomic types
        self.types = None
        # The number of atoms
        self.N_atoms = -1
        # The number of snapshots
        self.snapshots = 0
        # The cell NVT
        self.unit_cell = np.eye(3) * 10
        # The unit cell NPT
        self.unit_cells = None
        # Potential and kientic energy
        self.kinetic_energies = None
        self.potential_energies = None
        self.temperatures = None
        self.cons_quant = None
        self.velocities = None
        # Dipole moments with berry phase in ATOMIC UNITS so BOHR
        self.dipoles = None
        self.dipoles_quantum = None
        self.dipoles_origin = None
        # the time step in FEMPTOSEC
        self.dt = -1
        # The type of the calculation
        self.calc_type = None
        
        
        # Setup the attribute control
        self.__total_attributes__ = [item for item in self.__dict__.keys()]
        # This must be the last attribute to be setted
        self.fixed_attributes = True 

        # Setup any other keyword given in input (raising the error if not already defined)
        for key in kwargs:
            self.__setattr__(key, kwargs[key])
        
        

    def init(self, file_name_head, unit_cell, debug = False, verbose = True, calc_type = 'GEO_OPT',
                   ext_pos = None, ext_force = None,
                   ext_stress = None, ext_vel = None,
                   ext_ener = None, ext_cell = None, ext_dipoles = None):
        """
        READ XYZ FILE AS CREATED BY CP2K

        Positions-Unit cells are in ANGSTROM

        Energies are in HARTREE

        Forces are in HARTREE/BOHR
        
        Stresses are in HARTREE/BOHR^3
        
        Velocities are in BOHR/AU TIME
        
        Paramters:
        ----------
            -file_name_head: the path to the extension of the file
            -unit_cell: np.array with the cell shape in ANGSTROM, lattice vectors are the rows.
            -debug: bool,
            -verbose: bool,
            -calc_type: strg, it is needed to understand which calculation was done
            -ext_pos: strg, the extension of the position file
            -ext_force: strg, the extension of the force file
            -ext_stress: strg, the extension of the stress file
            -ext_velocities: strg, the extension of the velocities file
            -ext_ener: strg, the extension of the energ file
            -ext_cell: strg, the extension of the cell file
            -ext_dipoles: strg, the extension of the dipoles file
        """ 
        # Set up the unit cell in Angstrom
        self.unit_cell = np.copy(unit_cell)
        
        avail_calc_type = ['GEO_OPT', 'MD']
        if not(calc_type in avail_calc_type):
            raise ValueError('Choose calc_type among {}'.format(avail_calc_type))
        self.calc_type = calc_type

        ####### READ THE POSITIONS FILE ########
        if ext_pos is None:
            raise ValueError('Initalize the class with file.ext_pos')
        # Open the target file
        # Note that at least the postion file should exists otherwise we can not initialize the class
        file_xyz = open(file_name_head + ext_pos, 'r')
        lines = file_xyz.readlines()

        # Get the number of atoms
        self.N_atoms = int(lines[0])
        # Get the number of MD, GEO_OPT steps
        for index, line in enumerate(lines):
            if line.startswith(' i ='):
                self.snapshots += 1
            if calc_type == 'MD':
                if line.startswith(' i =        1'):
                    self.dt = float(lines[index].split()[5][:-1])
                
        if verbose:
            print('\n====> CP2K SNAPSHOTS <====')
            print('CP2k reading from files beginning with {}'.format(file_name_head))               
            print('N_atoms   = {}'.format(self.N_atoms)) 
            print('Snapshots = {}'.format(self.snapshots))
        
        # Initialize everything
        # Positions ANGSTROM 
        self.positions = np.zeros((self.snapshots, self.N_atoms, 3))
        # Atomic types
        self.types = [''] * self.N_atoms
        # Forces HARTREE/BOHR
        self.forces = np.zeros((self.snapshots, self.N_atoms, 3))
        # Unit cells ANGSTROM  each row has a unit cell vector
        self.unit_cells = np.zeros((self.snapshots, 3, 3))
        # Conserved quantity in HARTREE
        self.cons_quant = np.zeros(self.snapshots)
        # Energies HARTREE
        self.energies = np.zeros(self.snapshots)
        # Stresses HARTREE/BOHR3
        self.stresses = np.zeros((self.snapshots, 3, 3))
        # Velocities BOHR/AU TIME
        self.velocities = np.zeros((self.snapshots, self.N_atoms, 3))
        
        # Kinetic and potential energy in HARTREE
        self.kinetic_energies   = np.zeros(self.snapshots)
        self.potential_energies = np.zeros(self.snapshots)
        # Temperature in KELVIN
        self.temperatures = np.zeros(self.snapshots)

        # Dipoles are in ATOMIC UNITS units
        self.dipoles    = np.zeros((self.snapshots, 3))
        self.dipoles_quantum    = np.zeros((self.snapshots, 3, 3))
        # Dipole orgin in ANGSTROM
        self.dipoles_origin = np.zeros((self.snapshots, 3))
        
        
        # Read the atomic positions of each snapshots
        for isnap in range(self.snapshots):
            # The index where the atomic positions start
            file_index = isnap * (self.N_atoms + 2) + 2
            
            self.energies[isnap] = float(lines[file_index - 1].split()[-1])
            for iatom in range(self.N_atoms):
                coords = np.asarray(lines[file_index + iatom].split()[1:])
                self.positions[isnap, iatom, :] = coords
                # Get the types
                self.types[iatom] = lines[file_index + iatom].split()[0].capitalize()
                
            if debug:
                print('COORDS SNAP={}'.format(isnap))
                print(self.positions[isnap,:,:])
                print(self.types)
                print()

        # close the file
        file_xyz.close()
        
        if verbose:
            if debug:
                print('Types of atoms {}'.format(self.types))
            print('Unique Types of atoms {}'.format(set(self.types)))
        ####### END READ THE POSITIONS FILE ########
        

        ####### READ THE FORCES FILE ########
        if not(ext_force is None):
            if not os.path.exists(file_name_head + ext_force):
                raise ValueError('I do not find the force file {}'.format(file_name_head + ext_force))
            # Open the target file
            file_force = open(file_name_head + ext_force, 'r')
            lines = file_force.readlines()

            # Read the atomic positions of each snapshots
            for isnap in range(self.snapshots):
                # The index where the forces start
                file_index = isnap * (self.N_atoms + 2) + 2

                # Read the force for the atoms
                for iatom in range(self.N_atoms):
                    atom_force = np.asarray(lines[file_index + iatom].split()[1:])
                    self.forces[isnap, iatom, :] = atom_force

                if debug:
                    print('FORCES SNAP={}'.format(isnap))
                    print(self.forces[isnap,:,:])
                    print()
            # Close the file
            file_force.close()
        
        ####### END READ THE FORCES FILE ########
        
        
        
        
        ####### READ THE STRESS FILE ########
        units_bar = False
        if not(ext_stress is None):
            if not os.path.exists(file_name_head + ext_stress):
                raise ValueError('I do not find the stress file {}'.format(file_name_head + ext_stress))
            file_stress = open(file_name_head + ext_stress, 'r')
            lines = file_stress.readlines()
            
            if '[bar]' in lines[0].split():
                units_bar = True
            else:
                raise ValueError('I do not know the units for pressure')
                
            # Read the atomic positions of each snapshots
            for isnap in range(self.snapshots):
                #print(np.asarray(lines[isnap + 1].split()[2:], dtype = float))
                #xx = float(lines[isnap + 1].split()[2])
                #xy = float(lines[isnap + 1].split()[3])
                #xz = float(lines[isnap + 1].split()[4])
                #yx = float(lines[isnap + 1].split()[5])
                #yy = float(lines[isnap + 1].split()[6])
                #yz = float(lines[isnap + 1].split()[7])
                #zx = float(lines[isnap + 1].split()[8])
                #zy = float(lines[isnap + 1].split()[9])
                #zz = float(lines[isnap + 1].split()[10])
                #self.stresses[isnap,:,:] = np.asarray([[xx, xy, xz],
                #                                       [yx, yy, yz],
                #                                       [zx, zy, zz]])
                self.stresses[isnap,:,:] = np.asarray(lines[isnap + 1].split()[2:], dtype = float).reshape((3,3))
                #print(self.stresses[isnap,:,:] - np.asarray(lines[isnap + 1].split()[2:], dtype = float).reshape((3,3)))
                
            if units_bar:
                print('Stress tensor in HA BOHR3')
                # print(BAR_TO_HA_BOHR3)
                self.stresses[:, :, :] *= BAR_TO_HA_BOHR3
                
            # Close the file
            file_stress.close()
        ####### END READ THE STRESS FILE ########
        
        
        
        
        ####### READ THE VELOCITY FILE ########
        if not(ext_vel is None):
            if not os.path.exists(file_name_head + ext_vel):
                raise ValueError('I do not find the velocity file {}'.format(file_name_head + ext_vel))
            # Open the target file
            file_vel = open(file_name_head + ext_vel, 'r')
            lines = file_vel.readlines()

            # Read the atomic velocties of each snapshots
            for isnap in range(self.snapshots):
                file_index = isnap * (self.N_atoms + 2) + 2

                for iatom in range(self.N_atoms):
                    coords = np.asarray(lines[file_index + iatom].split()[1:])
                    self.velocities[isnap, iatom, :] = coords
                if debug:
                    print('VEL SNAP={}'.format(isnap))
                    print(self.velocities[isnap,:,:])
                    print()

            # close the file
            file_vel.close()
        ####### END READ THE VELOCITY FILE ########
        
        
        
        ####### READ THE ENERG FILE ########
        if not(ext_ener is None):
            
            if not os.path.exists(file_name_head + ext_ener):
                raise ValueError('I do not find the energ file {}'.format(file_name_head + ext_ener))
            # Open the target file
            file_ener = open(file_name_head + ext_ener, 'r')
            lines = file_ener.readlines()

            for isnap in range(self.snapshots):
                self.kinetic_energies[isnap]   = float(lines[isnap + 1].split()[2])
                self.temperatures[isnap]       = float(lines[isnap + 1].split()[3])
                self.potential_energies[isnap] = float(lines[isnap + 1].split()[4])
                self.cons_quant[isnap]         = float(lines[isnap + 1].split()[5])
            if self.calc_type == 'MD':
                try:
                    self.dt = float(lines[2].split()[1]) - float(lines[1].split()[1])
                except:
                    print("Set anually the time step. The file is too short. Probably only one step was done.")
                    self.dt = float(input("Please enter the dt: "))
                print('Update the dt to {} fs'.format(self.dt))
                self.dt = float(self.dt)
            # close the file
            file_ener.close()
        ####### END READ THE ENER FILE ########


        ####### READ THE DIPOLE FILE ########
        if not(ext_dipoles is None):
            
            if not os.path.exists(file_name_head + ext_dipoles):
                raise ValueError('I do not find the dipoles file {}'.format(file_name_head + ext_dipoles))
            # Open the target file
            file_dipoles = open(file_name_head + ext_dipoles, 'r')
            lines = file_dipoles.readlines()

            for isnap in range(self.snapshots):
                index = (isnap + 1) * 10 + 9

                # DEBYE TO ATOMIC UNITS
                self.dipoles[isnap, :] = np.array([float(lines[index].split()[1]),
                                                   float(lines[index].split()[3]),
                                                   float(lines[index].split()[5])]) * DEBEYE_TO_AU
                # IN ANGSTROM as all the positions
                self.dipoles_origin[isnap, :] = np.array([float(lines[index - 9].split()[-3]),
                                                          float(lines[index - 9].split()[-2]),
                                                          float(lines[index - 9].split()[-1])]) * BOHR_TO_ANGSTROM

                # DEBYE TO ATOMIC UNITS x, y, z along eahc row
                self.dipoles_quantum[isnap, :, :] = np.array([[float(lines[index - 4].split()[2]), float(lines[index - 4].split()[3]), float(lines[index - 4].split()[4])],
                                                              [float(lines[index - 3].split()[1]), float(lines[index - 3].split()[2]), float(lines[index - 3].split()[3])],
                                                              [float(lines[index - 2].split()[2]), float(lines[index - 2].split()[3]), float(lines[index - 2].split()[4])]]) * DEBEYE_TO_AU
                                                
                
            # close the file
            file_dipoles.close()

            # self.refold_all_dipoles(Nmax = 10, tol = 1.0, debug = False, debug_visualize = False)

                    
        ####### END READ THE ENER FILE ########



        ####### READ THE CELL FILE ########
        if not(ext_cell is None):
            
            if not os.path.exists(file_name_head + ext_cell):
                raise ValueError('I do not find the cell file {}'.format(file_name_head + ext_cell))
            # Open the target file
            file_cells = open(file_name_head + ext_cell, 'r')
            lines = file_cells.readlines()

            for isnap in range(self.snapshots):
                for i in range(3):
                    self.unit_cells[isnap, 0, i]   = float(lines[isnap + 1].split()[2 + i])

                for i in range(3):
                    self.unit_cells[isnap, 1, i]   = float(lines[isnap + 1].split()[5 + i])

                for i in range(3):
                    self.unit_cells[isnap, 2, i]   = float(lines[isnap + 1].split()[8 + i])
                

            # close the file
            file_cells.close()
        ####### END READ THE CELL FILE ########
        else:
            print("No unit cells files was used\nwe are copying")
            print(self.unit_cell)
            print('\nin self.unit_cells')
            self.unit_cells[:,:,:] = np.copy(self.unit_cell)

        return 



            


    def create_ase_snapshots(self, wrap_positions, subtract_com, pbc = True):
        """
        CREATE ASE SNAPSHOTS 
        ====================
        
        Rember that ase use EV, ANGSTROM, EV/ANGSTROM3 ANGSTROM/PICOSECOND

        It correctly prints energy forces stresses and position

        In cp2k the coordinates are saved not wrapped so we wrap them back in the cell

        In addition, the center of mass can drift. So, after wrapping  the positions
        
        Paramters:
        ----------
            -wrap_positions: bool, use False to compute Mean Square Displacement
            -subtract_com: bool, if True we subract the position of the center of mass (useful for MSD analysis)
            -pbc: bool, the PBC conditions along x y z (def True)
        Returns:
        --------
            -all_atoms: a list of ase objects
        """
        # Set up all the ase atoms
        all_atoms = []

        print("\nCREATING ASE ATOMS SNAPSHOTS...")
        print("Wrapping the positions? {}".format(wrap_positions))
        print("Subtracting the center of mass positions? {}".format(subtract_com))
        print("Setting PBC? {}\n".format(pbc))

        if subtract_com:
            # Get the center of mass positions (self.snaphsots, 3)
            R_com = self.get_com_positions()
            
        # Range in the snapshots ans use ASE units
        # ev and angstrom, ev/angstrom3, angstrom/picosecond
        for isnap in range(self.snapshots):

            energy     = self.energies[isnap] * HA_TO_EV
            forces     = self.forces[isnap,:,:] * HA_BOHR_TO_EV_ANGSTROM
            stress     = -transform_voigt(self.stresses[isnap,:,:]) * HA_BOHR3_TO_EV_ANGSTROM3
            velocities = self.velocities[isnap,:,:] * BOHR_TO_ANGSTROM /AU_TIME_TO_PICOSEC

            if wrap_positions:
                positions = ase.geometry.wrap_positions(self.positions[isnap,:,:], self.unit_cells[isnap,:,:], pbc = True)
            
            if subtract_com:
                positions = np.copy(self.positions[isnap,:,:] - R_com[isnap,:])
            else:
                positions = np.copy(self.positions[isnap,:,:])
            
            # Create the ase atoms object
            structure = Atoms(self.types, positions)
            structure.set_cell(self.unit_cells[isnap,:,:])
            structure.pbc[:] = pbc
            # Set the velocties
            structure.set_velocities(velocities)

            # Now set the calculato so that we have energies and forces
            calculator = ase.calculators.singlepoint.SinglePointCalculator(structure, energy = energy, forces = forces, stress = stress)
            # Attach the calculator
            structure.calc = calculator

            # Append to the list
            all_atoms.append(structure)
            
        return all_atoms

    def copy_snapshots(self):
        """
        RETURN A COPY OF THE CURRENT CLASS
        """
        snapcopy = AtomicSnapshots()

        # Iterate over instance attributes
        for attr in self.__dict__:  
            # Get the attribute's value
            value = getattr(self, attr) 
            
            # Use np.copy() for numpy arrays, deep copy for others
            if isinstance(value, np.ndarray):
                setattr(snapcopy, attr, np.copy(value))
            else:
                # General deep copy
                setattr(snapcopy, attr, copy.deepcopy(value))  
            
        return snapcopy  

    def merge_multiple_snapshots(self, list_of_snapshots):
        """
        MERGE MULTIPLE SNAPSHOTS 

        Parameters:
        -----------
            -atomic_snapshot: a list of AtomicSnapshots() objects to merge with them selves

        Retrurns:
        ---------
            -snap_merge: the final AtomicSnapshots() object
        """
        snap_merge = AtomicSnapshots()
        # snap_merge
        
        for i, snap in enumerate(list_of_snapshots):

            if i == 0:
                snap_merge = snap.copy_snapshots()
            else:
                snap_merge = snap_merge.merge_snapshots(snap)

        return snap_merge
        
    def merge_snapshots(self, atomic_snapshots):
        """
        MERGE TWO ATOMIC SNAPSHOTS CLASS

        Check that the two are compatible

        Parameters:
        -----------
            -atomic_snapshot: the AtomicSnapshots() object to merge with self
        
        Returns:
        --------
            -snap_merge: the merged AtomicSnapshots() object
        """
        
        if self.N_atoms != atomic_snapshots.N_atoms:
            raise ValueError("The number of atoms should be the same")

        if self.types != atomic_snapshots.types:
            print(self.types, atomic_snapshots.types)
            raise ValueError("The atomic types are different")

        if self.dt != atomic_snapshots.dt:
            raise ValueError("The time step dt should be the same")

        if np.max(np.abs(self.unit_cell - atomic_snapshots.unit_cell)) > 1e-10:
            raise ValueError("The unit cell should be the same")

        if self.calc_type != atomic_snapshots.calc_type:
            raise ValueError("The type of calculation should be the same")

        snap_merge = AtomicSnapshots()

        # Iterate over instance attributes
        for attr in self.__dict__:  
            if not(attr in ['unit_cell', 'types', 'calc_type', 'snapshots', 'dt']):
                # Get the attribute's value
                value1 =             getattr(self, attr) 
                value2 = getattr(atomic_snapshots, attr)
                
                # Use np.copy() for np arrays, copy.deepcopy for others
                if isinstance(value1, np.ndarray):
                    setattr(snap_merge, attr, np.concatenate((value1, value2), axis=0))
                # Use deepcopy for lists, string, dictionary etc
                else:
                    setattr(snap_merge, attr, copy.deepcopy(value1))  

        snap_merge.snapshots = self.snapshots + atomic_snapshots.snapshots
        
        snap_merge.unit_cell = np.copy(self.unit_cell)

        snap_merge.calc_type = copy.deepcopy(self.calc_type)

        snap_merge.types = copy.deepcopy(self.types)

        snap_merge.dt = copy.deepcopy(self.dt)
        
        return snap_merge  
    
    def plot_energy_force(self, img_name = None):
        """
        PLOT ENERGY FORCE FROM ASE SNAPSHOTS 
        
        Rember that ase use EV, ANGSTROM
        
        To plot the force at each time step 
        
        .. math::  \sum_{I=1}^{N_at} F_I \cdot F_I /N_at
        
         Paramters:
        ----------
            -uc: a np.array with the cell BOHR
            -img_name: imag name if you want to save
        """
        matplotlib.use('tkagg')

        energies = self.energies * HA_TO_EV /self.N_atoms
        forces = np.einsum('iab, iab -> i', self.forces * HA_BOHR_TO_EV_ANGSTROM, self.forces * HA_BOHR_TO_EV_ANGSTROM) /self.N_atoms
        
        x = np.arange(self.snapshots, dtype = int)
        
        # Width and height
        fig = plt.figure(figsize=(8, 5))
        gs = gridspec.GridSpec(2, 1, figure=fig)
        ax = fig.add_subplot(gs[0,0])
        ax.plot(x, energies, 's', color = 'k')
        ax.set_ylabel('Energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax = fig.add_subplot(gs[1,0])
        ax.plot(x, forces, 'd', color = 'red')
        ax.set_xlabel('Steps', size = 15)
        ax.set_ylabel('Force [eV/Ang/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        
        plt.tight_layout()

        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        plt.show()
        
        return


    def plot_temperature_evolution(self, average_window = [0,-1], img_name = None):
        """
        PLOT TEMPERATURE EVOLUTION FROM AND MD
        ======================================

        The average temperature is 

        .. math:: T_{av} = \frac{1}{N} \sum_{n=1}^{N} T_{n}
        
        Paramters:
        ----------
            -average_window: a list of two numbers, so we will average self.temperatures[average_window[0]:average_window[1]]
            -img_name: imag name if you want to save
        """
        matplotlib.use('tkagg')
        # Width and height
        fig = plt.figure(figsize=(7, 4))
        gs = gridspec.GridSpec(1,1, figure=fig)

        x = np.arange(self.snapshots) * self.dt

        T_av = np.average(self.temperatures[average_window[0]: average_window[1]])
        # Get the standard error
        N_samples = len(self.temperatures[average_window[0]: average_window[1]])
        T_err = np.sqrt(np.sum((self.temperatures[average_window[0]: average_window[1]] - T_av)**2))/N_samples
        
        ax = fig.add_subplot(gs[0,0])
        xmin, xmax = np.sort(x[average_window])
        ax.plot(x, self.temperatures,  color = 'purple', lw = 3, label = 'T = {:.1f} {:.3f} K from {:.2f} to {:.2f} ps'.format(T_av, T_err,
                                                                                                                              xmin * 1e-3, xmax * 1e-3))
        ax.fill_between(x, self.temperatures, np.min(self.temperatures), where=(x >= xmin) & (x <= xmax), color='purple', alpha=0.3)
        ax.axhline(y = T_av, color = 'k', linestyle=':')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.set_ylabel('Temperature [Kelvin]', size = 12)
        ax.set_xlabel('Time [fs]', size = 15)
        plt.legend(fontsize = 12)
        plt.tight_layout()
        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        plt.show()


    def plot_pressure_evolution(self, average_window = [0,-1], img_name = None):
        """
        PLOT PRESSURE EVOLUTION FROM AND MD
        ======================================

        The average temperature is 

        .. math:: P_{av} = \frac{1}{N} \sum_{n=1}^{N} P_{n}
        
        Paramters:
        ----------
            -average_window: a list of two numbers, so we will average self.temperatures[average_window[0]:average_window[1]]
            -img_name: imag name if you want to save
        """
        matplotlib.use('tkagg')
        # Width and height
        fig = plt.figure(figsize=(7, 4))
        gs = gridspec.GridSpec(1,1, figure=fig)

        x = np.arange(self.snapshots) * self.dt

        # The pressure is the trace of the stress tensor
        pressures = np.einsum("iaa -> i", self.stresses) * BAR_TO_HA_BOHR3**-1 * BAR_TO_GPA /3 

        # Get the average with the standard error
        N_samples = len(pressures[average_window[0]: average_window[1]])
        P_av  = np.average(pressures[average_window[0]: average_window[1]])
        P_err = np.sqrt(np.sum((pressures[average_window[0]: average_window[1]] - P_av)**2))/N_samples
        
        ax = fig.add_subplot(gs[0,0])
        xmin, xmax = np.sort(x[average_window])
        ax.plot(x, pressures,  color = 'green', lw = 3, label = 'P={:.3f} {:.3f} GPa from {:.2f} to {:.2f} ps'.format(P_av, P_err, xmin * 1e-3, xmax * 1e-3))
        ax.fill_between(x, pressures, np.min(pressures), where=(x >= xmin) & (x <= xmax), color='green', alpha=0.3)
        ax.axhline(y = P_av, color = 'k', linestyle=':')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.set_ylabel('Pressure [GPa]', size = 12)
        ax.set_xlabel('Time [fs]', size = 15)
        plt.legend(fontsize = 12)
        plt.tight_layout()
        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        plt.show()


    def plot_unit_cell_evolution(self, average_window = [0,-1], img_name = None):
        """
        PLOT ISOTROPIC UNIT CELL and VOLUME EVOLUTION FROM MD
        =====================================================

        The unit cell is plotted in ANGSTROM and the volume in ANGSTROM^3
        
        Paramters:
        ----------
            -average_window: a list of two numbers, so we will average self.unit_cell[average_window[0]:average_window[1],:,:]
            -img_name: imag name if you want to save
        """
        matplotlib.use('tkagg')
        # Width and height
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1,2, figure=fig)

        x = np.arange(self.snapshots) * self.dt
        # Get the number of samples
        N_samples = len(self.unit_cells[average_window[0]: average_window[1], 0, 0])
        
        # Get the average and the standard error in ANGSTROM
        UC_av = np.einsum('iab -> ab', self.unit_cells[average_window[0]: average_window[1],:,:]) /N_samples
        UC_err = np.sqrt(np.einsum('iab -> ab', (self.unit_cells[average_window[0]: average_window[1],:,:] - UC_av[:,:])**2))/N_samples

        # Get the volumes in ANGSTROM^3
        volumes = np.zeros(self.snapshots, dtype = float)
        for i in range(self.snapshots):
            volumes[i] = np.linalg.det(self.unit_cells[i,:,:]) 
        # Get the average and the standard error in ANGSTROM^3
        V_average = np.sum(volumes[average_window[0]: average_window[1]]) /N_samples
        V_err = np.sqrt(np.sum((volumes[average_window[0]: average_window[1]] - V_average)**2)) /N_samples

        for i in range(2):
                ax = fig.add_subplot(gs[0,i])
                xmin, xmax = np.sort(x[average_window])
                if i == 0:
                    # Plot only the x component
                    ax.plot(x, self.unit_cells[:,i,i],  color = 'green', lw = 3,
                            label = 'L={:.2f} {:.2f} Ang\nfrom {:.2f} to {:.2f} ps'.format(UC_av[i,i], UC_err[i,i], xmin * 1e-3, xmax * 1e-3))
                    ax.fill_between(x,  self.unit_cells[:,i,i], np.min(self.unit_cells[:,i,i]), where=(x >= xmin) & (x <= xmax), color='green', alpha=0.3)
                    ax.axhline(y = UC_av[i,i], color = 'k', linestyle=':')
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax.tick_params(axis = 'both', labelsize = 12)
                    ax.set_ylabel('Cell [Angstrom]', size = 12)
                else:
                    ax.plot(x, volumes,  color = 'blue', lw = 3,
                            label = 'V={:.2f} {:.2f} Ang3\nfrom {:.2f} to {:.2f} ps'.format(V_average, V_err, xmin * 1e-3, xmax * 1e-3))
                    ax.fill_between(x,  np.asarray(volumes), np.min(volumes), where=(x >= xmin) & (x <= xmax), color='blue', alpha=0.3)
                    ax.axhline(y = V_average, color = 'k', linestyle=':')
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax.tick_params(axis = 'both', labelsize = 12)
                    ax.set_ylabel('Volume [Angstrom$^{3}$]', size = 12)
                ax.set_xlabel('Time [fs]', size = 15)
                plt.legend(fontsize = 12)
                plt.tight_layout()
        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        plt.show()


    def plot_density_evolution(self, average_window = [0,-1], img_name = None):
        """
        PLOT DENISTY EVOLUTION FROM MD
        ==============================

        The density in g/cm3. Use ANGSTROM for length and g/mol for the masses
        
        Paramters:
        ----------
            -average_window: a list of two int numbers, so we will average self.unit_cell[average_window[0]:average_window[1],:,:]
            -img_name: image name if you want to save
        """
        matplotlib.use('tkagg')
        # Width and height
        fig = plt.figure(figsize=(7, 4))
        gs = gridspec.GridSpec(1,1, figure=fig)

        # Time steps in PICOSECOND
        x = np.arange(self.snapshots) * self.dt

        # Get the total molar mass in g/mol
        total_mass = np.sum(self.get_masses_from_types())

        # Get the volumes in ANGSTROM^3
        volumes = np.zeros(self.snapshots, dtype = float)
        for i in range(self.snapshots):
            volumes[i] = np.linalg.det(self.unit_cells[i,:,:]) 
            
        # Get the average and error of the volume in the time window in ANGSTROM^3
        N_samples = len(volumes[average_window[0]: average_window[1]])
        V_average = np.sum(volumes[average_window[0]: average_window[1]]) /N_samples
        V_err = np.sqrt(np.sum((volumes[average_window[0]: average_window[1]] - V_average)**2)) /N_samples
        
        # Get the density in g/cm^3
        rho = total_mass /(0.602214076 * volumes)
        
        rho_av  = np.sum(rho[average_window[0]: average_window[1]]) /N_samples
        rho_err = np.sqrt(np.sum((rho[average_window[0]: average_window[1]] - rho_av)**2))/N_samples

        # Plot everything
        ax = fig.add_subplot(gs[0,0])
        xmin, xmax = np.sort(x[average_window])
        ax.plot(x, rho,  color = 'purple', lw = 3,
                label = '$\\rho$' + '={:.4f} {:.4f} g/cm3\nfrom {:.2f} to {:.2f} ps'.format(rho_av, rho_err, xmin * 1e-3, xmax * 1e-3))
        ax.fill_between(x,  rho, np.min(rho), where=(x >= xmin) & (x <= xmax), color='purple', alpha=0.3)
        ax.axhline(y = rho_av, color = 'k', linestyle=':')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.set_ylabel('Density [g/cm$^{3}$]', size = 12)
        ax.set_xlabel('Time [fs]', size = 15)
        plt.legend(fontsize = 12)
        plt.tight_layout()
        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        plt.show()
        

    def plot_md_nvt(self, img_name = None):
        """
        PLOT ENERGY FORCE KINETIC ENERGY TEMERATURE FROM ASE SNAPSHOTS FOR NVT SIMULATION
        ==================================================================================
        
        Rember that ase use EV, ANGSTROM
        
        To plot the force at each time step 
        
        .. math::   \sum_{I=1}^{N_at} F_I \cdot F_I /N_at
        
         Paramters:
        ----------
            -img_name: imag name if you want to save
        """
        matplotlib.use('tkagg')
        
        energies = self.energies * HA_TO_EV /self.N_atoms
        forces = np.einsum('iab, iab -> i', self.forces * HA_BOHR_TO_EV_ANGSTROM, self.forces * HA_BOHR_TO_EV_ANGSTROM) /self.N_atoms
        
        x = np.arange(self.snapshots, dtype = int) * self.dt
        
        # Width and height
        fig = plt.figure(figsize=(10, 12))
        gs = gridspec.GridSpec(3,2, figure=fig)
        
        ax = fig.add_subplot(gs[0,:])
        ax.plot(x, energies,  color = 'k', lw = 3)
        ax.set_ylabel('Energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        


        
        ax = fig.add_subplot(gs[1,0])
        ax.plot(x, self.kinetic_energies * HA_TO_EV/self.N_atoms,  color = 'darkorange', lw = 3)
        ax.set_ylabel('Kin energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        ax = fig.add_subplot(gs[1,1])
        ax.plot(x, forces,  color = 'red', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('Force [eV/Ang/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        

        ax = fig.add_subplot(gs[2,0])
        ax.plot(x, self.temperatures,  color = 'purple', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('Temperature [Kelvin]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        ax = fig.add_subplot(gs[2,1])
        ax.plot(x, self.cons_quant * HA_TO_EV * 1000/self.N_atoms,  color = 'darkblue', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('Cons qunt [meV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        
        plt.tight_layout()

        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        plt.show()
        
        return

    def plot_md_npt(self, img_name = None):
        """
        PLOT ENERGY FORCE FROM ASE SNAPSHOTS FOR NVT SIMULATION
        
        Rember that ase use EV, ANGSTROM
        
        To plot the force at each time step 
        
        .. math::   \sum_{I=1}^{N_at} F_I \cdot F_I /N_at
        
         Paramters:
        ----------
            -img_name: imag name if you want to save
        """
        matplotlib.use('tkagg')

        energies = self.energies * HA_TO_EV /self.N_atoms
        forces = np.einsum('iab, iab -> i', self.forces * HA_BOHR_TO_EV_ANGSTROM, self.forces * HA_BOHR_TO_EV_ANGSTROM) /self.N_atoms

        
        x = np.arange(self.snapshots, dtype = int) * self.dt
        
        # Width and height
        fig = plt.figure(figsize=(10, 12))
        gs = gridspec.GridSpec(4,2, figure=fig)
        
        ax = fig.add_subplot(gs[0,:])
        ax.plot(x, energies,  color = 'k', lw = 3)
        ax.set_ylabel('Energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        

        ax = fig.add_subplot(gs[1,0])
        ax.plot(x, self.kinetic_energies * HA_TO_EV/self.N_atoms,  color = 'darkorange', lw = 3)
        ax.set_ylabel('Kin energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


        ax = fig.add_subplot(gs[1,1])
        ax.plot(x, forces,  color = 'red', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('Force [eV/Ang/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


        
        ax = fig.add_subplot(gs[2,0])
        ax.plot(x, self.temperatures,  color = 'purple', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('Temperature [Kelvin]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


        ax = fig.add_subplot(gs[2,1])
        ax.plot(x, self.cons_quant * HA_TO_EV * 1000/self.N_atoms,  color = 'darkblue', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('Cons qunt [meV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


        ax = fig.add_subplot(gs[3,:])
        pressure = np.einsum("iaa -> i", self.stresses) * BAR_TO_HA_BOHR3**-1 * BAR_TO_GPA /3
        ax.plot(x, pressure,  color = 'green', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('P [GPa]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        
        plt.tight_layout()

        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        plt.show()
        
        return

    def get_masses_from_types(self):
        """
        GET THE MASSES FOR ALL THE ATOMS IN THE SNAPSHOTS
        =================================================

        Use units of ASE
        """
        # Get the masses
        masses_array = np.zeros(self.N_atoms)
        for i, at_type in enumerate(self.types):
            masses_array[i] = ase.data.atomic_masses[ase.data.atomic_numbers[at_type]]

        return masses_array
        
    def get_com_positions(self):
        """
        GET THE CENTER OF MASS POSITION
        ===============================

        In Angstrom

        Returns:
        --------
            -R_com: np.array with shape (snapshots, 3), the center of mass positions
        """
        masses_array = self.get_masses_from_types()

        # Get the center of mass position
        R_com = np.einsum('a, iab -> ib', masses_array, self.positions) /np.sum(masses_array)

        return R_com


    def get_com_velocities(self):
        """
        GET THE CENTER OF MASS VELOCITY
        ===============================

        In BOHR/AU_TIME

        Returns:
        --------
            -V_com: np.array with shape (snapshots, 3), the center of mass velocities
        """
        # Get the masses
        masses_array = np.zeros(self.N_atoms)
        for i, at_type in enumerate(self.types):
            masses_array[i] = ase.data.atomic_masses[ase.data.atomic_numbers[at_type]]

        # Get the center of mass velocities
        V_com = np.einsum('a, iab -> ib', masses_array, self.velocities) /np.sum(masses_array)

        return V_com



    def show_com_motion(self, img_name = None):
        """
        PLOT THE MOTION OF THE CENTER OF MASS
        =====================================

        Checks the drift of the com

        Use units of AMU for the mass from ase
        """
        matplotlib.use('tkagg')
       
        # Get the center of mass positions (self.snaphsots, 3)
        R_com = self.get_com_positions()

        # Time in femptosecond
        x = np.arange(self.snapshots, dtype = int) * self.dt

        
        # Width and height
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(2,3, figure=fig)
        colors = ["red", "green", "purple"]
        for i in range(3):
            ax = fig.add_subplot(gs[0,i])
            ax.plot(x, R_com[:,i],  color = colors[i], lw = 3)
            ax.set_xlabel('Time [fs]', size = 15)
            ax.set_ylabel('R$_{com}$ [Angstrom]', size = 12)
        for i in range(3):
            ax = fig.add_subplot(gs[1,i])
            ax.plot(x, np.gradient(R_com[:,i], x[1] - x[0]) * ANG_FEMPTOSEC_TO_HA,  color = colors[i], lw = 3)
            ax.set_xlabel('Time [fs]', size = 15)
            ax.set_ylabel('V$_{com}$ [Bohr/t$_{au}$]', size = 12)
        plt.tight_layout()
        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        plt.show()


        # Width and height
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(1,1, figure=fig)
        colors = ["red", "green", "purple"]
        ax = fig.add_subplot(gs[0,0])
        ax.plot(x, np.einsum('abc -> a',self.forces) * HA_BOHR_TO_EV_ANGSTROM,  color = colors[i], lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('F$_{tot}$ [eV/Angstrom]', size = 12)
        plt.tight_layout()
        if not(img_name is None):
            plt.savefig('forces_' + img_name , dpi = 500)
        plt.show()
        
        return


    def get_pair_correlation_functions(self, selected_atoms, t_ini = 2.5, my_r_range = (0.01, 6.0), bins = 500,
                                       ase_atoms_file = "atoms_gr.xyz", save_ase_atoms_file = False, wrap_positions = True, use_pbc = True,
                                       json_file_result = "pair_corr_function.json" , show_results = True, save_plot = False):
        """
        GET THE PAIR CORRELATION FUNCTION
        =================================

        Note MDAnalysis always uses PBC irrespectively of wheater the ase atoms used have PBC.
        Note that to have correct PBC in MDAnalysis we must initialize the dimensions.

        Parameters:
        -----------
            -selected_atoms: list, list of atomic types for which we compute the pair correlation function
            
            -t_ini: float, the equilibration time PIDCOSECOND. After t_ini we will start the sampling
            
            -my_r_range: tuple, the minimum and maximum value for r in ANGSTROM
            -bins: int, the number of r values
            
            -ase_atoms_file: the name of the xyz file containing all the snapshots info
            -save_ase_atoms_file: bool, if True we do not delete the xyz file ase_atoms_file
            
            -wrap_positions: bool, if True the positions are wrapped in the ase atoms objects
            -use_pbc: bool, if True PBC are imposed in the ase atoms objects
            
            -json_file_result: the name of the json file containing the dictionary with the results
            
            -show_results: bool, if True we print the diffusion constants and the MSD
            -save_plot: bool, if True we save the plot

        Returns:
        --------
            -g_results: a dictionary containing as items the atomic pair types with r and g(r)
        """
        matplotlib.use('tkagg')
        
        # If there are no selected atoms we compute the MSD and D for all the atomic types
        print("\n\n========PAIR CORRELATION FUNCTION with MDanalysis========")
        
        # Get the atomic types for which we want the diffusion constant
        if selected_atoms is None:
            raise ValueError("Provide a list of atom pairs for which you want to compute the g(r)")

        # if my_r_range is None:
        #     my_r_range = (0.0, 6.0)

        # Prepare the ase atoms to be read
        ase_atoms = self.create_ase_snapshots(wrap_positions = wrap_positions, subtract_com = False, pbc = use_pbc)
        index_ini = int(t_ini * 1e+3/self.dt)
        ase.io.write(ase_atoms_file, ase_atoms[index_ini:])

        # A dictionary with atom types and the corresponding diffusion constant and error
        g_results = {}
        # Read the xyz file
        MD_atoms = MDAnalysis.Universe(ase_atoms_file)

        # Prepare a plot
        if show_results:
            # Width and height
            fig = plt.figure(figsize=(8, 8))
            if len(selected_atoms) == 1:
                gs = gridspec.GridSpec(1, 1, figure = fig)
            else:
                gs = gridspec.GridSpec(len(selected_atoms)//2, 2, figure = fig)

        print("Setting the cell dimension and the time step...")
        for snapshot, MD_atoms_snapshot in enumerate(MD_atoms.trajectory):
            # Manually set the unit cell dimensions in ANGSTROM
            MD_atoms_snapshot.dimensions = [self.unit_cells[snapshot,0,0], self.unit_cells[snapshot,1,1], self.unit_cells[snapshot,2,2], 90.0, 90.0, 90.0]  
            # Apply the time step to all frames in PICOSECONDS
            MD_atoms_snapshot.dt = self.dt* 1e-3

        for index, atomic_pair in enumerate(selected_atoms):
            print('\nRDF for {} {} after equilibration of {} ps'.format(atomic_pair[0], atomic_pair[1], t_ini))
            # Select only the atomic type we want
            MD_atoms_selected1 = MD_atoms.select_atoms('name {}'.format(atomic_pair[0]))
            # Select only the atomic type we want
            MD_atoms_selected2 = MD_atoms.select_atoms('name {}'.format(atomic_pair[1]))
            # #  Wrap again the positions
            # MD_atoms_selected1.wrap()
            # MD_atoms_selected2.wrap()

            # Compute RDF
            rdf_calc = MDAnalysis.analysis.rdf.InterRDF(MD_atoms_selected1, MD_atoms_selected2,
                                                        range = my_r_range, nbins = bins, norm = 'rdf')
            rdf_calc.run(verbose = True)

            r, gr = rdf_calc.results.bins, rdf_calc.results.rdf
            # Save the results
            g_results.update({"{}{}".format(atomic_pair[0], atomic_pair[1]) : [list(r), list(gr)]})
            
            if show_results:
                ax = fig.add_subplot(gs[index // 2, index %2])
                ax.plot(r, gr, lw = 3, color = "purple", label = "g(r) for {} {}".format(atomic_pair[0], atomic_pair[1]))
                if gr.max() > 10 * gr[-1]:
                    ax.set_ylim(0, 2 * gr[-1])
                ax.set_xlabel('r [Angstrom]', size = 15)
                ax.set_ylabel('g(r)', size = 12)
                ax.tick_params(axis = 'both', labelsize = 12)
                plt.legend(fontsize = 15)
        plt.tight_layout()
        if save_plot:
            plt.savefig("gr.png", dpi = 500)
        if show_results:
            plt.show()

        save_dict_to_json(json_file_result, g_results)
        
        if not save_ase_atoms_file:
            subprocess.run("rm {}".format(ase_atoms_file), shell = True)

        return g_results


    def get_diffusion_constant(self, t_ini = 2.5, time_windows = None, subtract_com = True, selected_atoms = None,
                               ase_atoms_file = "atoms_msd.xyz", save_ase_atoms_file = False,
                               json_file_result = "self_diffusion.json" , show_results = True, save_plot = False):
        """
        GET THE SELF-DIFFUSION CONSTANT FROM THE FIT OF THE MEAN SQUARE DISPLACEMENT
        ============================================================================

        Remeber that for the MSD the coordinates should not be wrapped otherwise all the distances are fucked up

        Also remember to subtract the COM position during the MD

        Parameters:
        -----------
            -t_ini: float, the initial equilibration time, i.e. after t_ini we start sampling

            -time_windows: list
            
            -subtract_com: bool, if true we subtract the center of mass positions to the all atomic positions in the snapshots
            -selected_atoms: list, list of atomic types for which we compute the self diffusion constant
            
            -ase_atoms_file: the name of the xyz file containing all the snapshots info
            -save_ase_atoms_file: bool, if True we do not delete the xyz file ase_atoms_file
            
            -json_file_result: the name of the json file containing the dictionary with the results
            
            -show_results: bool, if True we print the diffusion constants and the MSD
            -save_plot: bool, if True we save the plot

        Returns:
        --------
            -D_results: a dictionary containing as items the atomic types with diffusion constants and its error
        """
        matplotlib.use('tkagg')
        
        # If there are no selected atoms we compute the MSD and D for all the atomic types
        print("\n\n========MSD ANALYSIS========")
        
        # Get the atomic types for which we want the diffusion constant
        if selected_atoms is None:
            selected_atoms = set(self.types)
            print("MSD will be computed for ", selected_atoms)
        else:
            if np.sum(np.isin(selected_atoms, self.types), dtype = int) != len(selected_atoms):
                raise ValueError("{} not found in the snapshots!")

        # Prepare the ase atoms to be read
        ase_atoms = self.create_ase_snapshots(wrap_positions = False, subtract_com = True, pbc = False)
        index_ini = int(t_ini * 1e+3/self.dt)
        ase.io.write(ase_atoms_file, ase_atoms[index_ini:])

        # A dictionary with atom types and the corresponding diffusion constant and error
        D_results = {}
        # Read the xyz file
        MD_atoms = MDAnalysis.Universe(ase_atoms_file)
        
        # Prepare a plot
        if show_results:
            # Width and height
            fig = plt.figure(figsize=(8, 8))
            if len(selected_atoms) == 1:
                 gs = gridspec.GridSpec(1, 1, figure = fig)
            else:
                gs = gridspec.GridSpec(len(selected_atoms)//2, 2, figure = fig)

        print("Setting the cell dimension and the time step...")
        for snapshot, MD_atoms_snapshot in enumerate(MD_atoms.trajectory):
            # Manually set the unit cell dimensions in ANGSTROM
            MD_atoms_snapshot.dimensions = [self.unit_cells[snapshot,0,0], self.unit_cells[snapshot,1,1], self.unit_cells[snapshot,2,2], 90.0, 90.0, 90.0]  
            # Apply the time step to all frames in PICOSECONDS
            MD_atoms_snapshot.dt = self.dt* 1e-3
            # if snapshot % 100 == 0:
            #     print(f"Frame {MD_atoms_snapshot.frame}: Unit cell = {MD_atoms_snapshot.dimensions}")
        

        for index, atomic_type in enumerate(selected_atoms):
            # Select only the atomic type we want
            MD_atoms_selected = MD_atoms.select_atoms('name {}'.format(atomic_type))
            # Prepare the MSD analysis
            MSD_tool = MDAnalysis.analysis.msd.EinsteinMSD(MD_atoms_selected, select = 'all', msd_type = 'xyz',  fft = True)
            print("\nGetting the Mean Square Displacement for {}".format(atomic_type))
            # Run the calculation
            MSD_tool.run()
            print("The number of atoms is {} and the number of frames {}".format(MSD_tool.n_particles, MSD_tool.n_frames))
            # PICOSCOND and ANGSTROM^2
            

            # print(MSD.results.msds_by_particle.shape )
            my_msd =  MSD_tool.results.timeseries
            # prepare the lagtimes in PICOSECOND
            timestep = MD_atoms.trajectory[0].dt
            lagtimes = np.arange(len(MD_atoms.trajectory)) * timestep

            if time_windows is None:
                start_time,  end_time  = lagtimes[len(lagtimes)//2] - 0.3 * lagtimes[-1], lagtimes[len(lagtimes)//2] + 0.3 * lagtimes[-1]
            else:
                start_time,  end_time  = time_windows
            print("The fit of MSD is done from {:.2f} to {:.2f} ps with a total time {:.2f} ps".format(start_time, end_time, lagtimes[-1]))
            mask = (start_time < lagtimes) & (lagtimes < end_time)
            
            linear_model = scipy.stats.linregress(lagtimes[mask], my_msd[mask])
            slope, slope_error = linear_model.slope, linear_model.stderr
            # dim_fac is 3 as we computed a 3D msd with 'xyz', ANGSTROM/PICOSECOND
            D, D_error = slope * 1/(2 * MSD_tool.dim_fac), slope_error * 1/(2 * MSD_tool.dim_fac)
            # Now in METER^2/SECOND
            D *= 1e-8
            D_error *= 1e-8

            D_results.update({atomic_type : [D, D_error]})
            print("=>Diffusion constant for atom {}".format(atomic_type))
            print("==>D {:.3e} +- {:.3e} ".format(D, D_error) + "m$^2$/s")
            # plt.plot(lagtimes, res)
        
            if show_results:
                ax = fig.add_subplot(gs[index // 2, index %2])
                ax.set_title("MSD for {}".format(atomic_type))
                # PICOSCOND and ANGSTROM^2
                ax.plot(lagtimes, my_msd, lw = 3, color = "k")
                ax.plot(lagtimes[mask], lagtimes[mask] * slope + linear_model.intercept, lw = 3, ls = ":", color = "red",
                        label = 'Fit D={:.2e} '.format(D) + "m$^2$/s")
                ax.set_xlabel('Time [ps]', size = 15)
                ax.set_ylabel('MSD [Angstrom$^2$]', size = 12)
                ax.tick_params(axis = 'both', labelsize = 12)
                plt.legend()
        plt.tight_layout()
        if save_plot:
            plt.savefig("diffusion.png", dpi = 500)
        if show_results:
            plt.show()

        # Save the results to a json file
        save_dict_to_json(json_file_result, D_results)
        # Remove the ase atoms object
        if not save_ase_atoms_file:
            subprocess.run("rm {}".format(ase_atoms_file), shell = True)

        return D_results


    def get_conductivity(self, types_q = {"Na" : +1, "Cl" : -1}, 
                         time_window = None,
                         use_julia = False, python_normalize = True,
                         pad = True, omega_ir = [0, 5000], smearing = None, save_jj = False,
                         test = True,
                         ase_atoms_file = 'ase_cond.xyz'):
        """
        GET THE CONDUCTIVITY
        =====================

        First we  compute the current current correlation function both in time and frequency

        then we compute the integral, i.e. the DC conductivity from

        ..math:: \sigma = \frac{1}{3 V k T} \int dt \left\langle J(t) J(0) \right\rangle

        This should coincide with

        ..math:: \sigma = \sigma(\omega=0)


        Parameters:
        -----------
            -types_q: dict with the atomic types and the ox charges
            
            -time_window: list, the initial and final time for sampling in PICOSECONDS
            
            -use_julia: bool, if True the dipole dipole correlation fucntion is computed using the Windowed average in JULIA
            -python_normalize: bool, if True the FFT is normalized so it coicides with the windowed average of Julia

            -pad: bool
            -omega_ir: list, the range to plot in cm-1
            -smearing: float the smaering in cm-1 for plotting the IR spectra
            -save_ir: bool, if True we save the IR spectra as a json file

            -test: bool: if True we compare the julia and python correlation functions
        """
        if time_window is None:
            index1 = 0
            index2 = self.snapshots + 1
            t_min = 0
            t_max = self.snapshots * self.dt * 1e-3
        else:
            t_min, t_max = np.sort(np.asarray(time_window))
            # Convert in FEMPTOSCEON ONLY TO GET THE INDICES
            index1 = int(t_min * 1e+3/self.dt)
            index2 = int(t_max * 1e+3/self.dt)

        print("\n\n================= CONDUCTIVITY ANALYSIS from {:.2f} to {:.2f} ps =================".format(t_min, t_max))
            
        # Prepare the ase atoms to be read
        ase_atoms = self.create_ase_snapshots(wrap_positions = False, subtract_com = False, pbc = False)

        # Set up the class
        cond = Conductivity.Conductivity()
        
        # Give the dipoles and the time step
        cond.init(ase_atoms = ase_atoms[index1: index2], dt = self.dt,
                  temperatures = self.temperatures[index1: index2],
                  types_q = types_q)

        # Set the time correlations
        cond.set_correlations(use_julia = use_julia, python_normalize = python_normalize)

        # x1, res1 = cond.get_spectra(cond.correlations, delta = None, pad = False)

        # x2, res2 = cond.get_spectra(cond.correlations, delta = None, pad = True)

        # plt.plot(x1, res1)
        # plt.plot(x2, res2)
        # plt.show()

        # Plot everyhting
        cond.plot_correlation_function(omega_min_max = omega_ir, smearing = smearing, pad = pad, save_data = save_jj)

        if test:
            cond.test_implementation()

        

        

    

    def get_ir_spectra(self, time_window = None,
                       Nmax = 5, tol_refold = 1.0, debug_refold = False,
                       use_julia = False, python_normalize = False,
                       omega_ir = [0, 5000], smearing = None, save_ir = False,
                       test = False):
        """
        GET THE IR SPECTRA FROM DIPOLE-DIPOLE CORRELATION FUNCTION
        ==========================================================

        Parameters:
        -----------
            -time_window: list, the initial and final time for sampling in PICOSECONDS

            -Nmax: int, the max integer to add/subtract to the dipoles to refold them
            -tol_refold: float, the tolerance to consider the dipoles snapshots continuous
            -debug_refold: bool, if True we print the refolding details
            
            -use_julia: bool, if True the dipole dipole correlation fucntion is computed using the Windowed average in JULIA
            -python_normalize: bool, if True the FFT is normalized so it coicides with the windowed average of Julia

            -omega_ir: list, the range to plot in cm-1
            -smearing: float the smaering in cm-1 for plotting the IR spectra
            -save_ir: bool, if True we save the IR spectra as a json file

            -test: bool: if True we compare the julia and python correlation functions
        """
        if time_window is None:
            index1 = 0
            index2 = self.snapshots + 1
            t_min = 0
            t_max = self.snapshots * self.dt * 1e-3
        else:
            t_min, t_max = np.sort(np.asarray(time_window))
            # Convert in FEMPTOSCEON ONLY TO GET THE INDICES
            index1 = int(t_min * 1e+3/self.dt)
            index2 = int(t_max * 1e+3/self.dt)

        
        print("\n\n================= VIBRATIONAL ANALYSIS from {:.2f} to {:.2f} ps =================".format(t_min, t_max))
        
        # First we refold the dipoles
        self.refold_all_dipoles(Nmax = Nmax, tol = tol_refold, debug = debug_refold)

        # Set up the class
        vibrations = Vibrational.Vibrational()
        
        # Give the dipoles and the time step
        vibrations.init(self.dipoles[index1:index2], self.dt)
        
        # Get the dipole-dipole time correlation function
        vibrations.set_dipole_dipole_correlation_function(use_julia = use_julia, python_normalize = python_normalize)
        
        # Plot the results
        vibrations.plot_results(omega_min_max = omega_ir, delta = smearing, save_data = save_ir)

        if test:
            vibrations.test_implementation()
        

    
    def refold_all_dipoles(self, Nmax = 10, tol = 1.0, debug = False):
        """
        REFOLD ALL THE DIPOLES USING MULTIPLES OF BERRY QUANTUM
        =======================================================

        If the component i of the dipoles is discontinous we add (or subtract) a multiple integer of the trivial phase
        """
        cmps = ["X", "Y", "Z"]
        # Refold the dipoles 
        for cart in range(3):
            print("\n\n\n   ========== REFOLDING {} ==========".format(cmps[cart]))
            new_P = refold_dipole(self.dipoles[:,cart], self.dipoles_quantum[:,cart,cart],
                                  Nmax = Nmax, tol = tol, debug = debug)
            self.dipoles[:,cart] = np.copy(new_P)
        
        





def save_dict_to_json(json_file_name, my_dict):
    """
    SAVE A DICTIONARY TO JSON FILE
    ==============================
    """

    # Save dictionary to a JSON file
    with open(json_file_name, "w") as file:
         # 'indent=4' makes the JSON human-readable
        json.dump(my_dict, file, indent = 4) 
        

def load_dict_from_json(json_file_name):
    """
    LOAD A DICTIONARY FROM A JSON FILE
    ==================================
    """
    # Load dictionary from a JSON file
    with open(json_file_name, "r") as file:
        # Converts JSON into a Python dictionary
        dictionary = json.load(file)   
    
    return dictionary


def refold_dipole(P, cell, Nmax = 1, tol = 1.0, debug = True):
    """
    REFOLD THE DIPOLES USING INTEGER MULTIPLES OF THE BERRY QUANTUM OF POLARIZATION
    ===============================================================================

    For every P[i], very simply we check if the difference |P[i] - P[i-1]| is to large. 
    If so, we subtract/add multiple integers of the quantum of polarization to P[i] 
    
    Parameters:
    -----------
        -P:    np.array with shape N, the dipoles along a given axis x, y, z for a trajectory of size N
        -cell: np.array with shape N, the Berry quantum along the given axis for a trajectory of size N
        -Nmax: int, the integer number we multiply the quantum of polarization to have a continuous function
        -tol: float, the tolerance which we use to tell if the dipoles are continuous or not
        -debug: bool

    Returns:
    --------
        -Pini: np.array with shape N, the CONTINUOUS dipoles along a given axis x, y, z for a trajectory of size N
        
    """
    # The dipoles we manipulate
    Pini = np.copy(P)
    # Get the size of the trajectory
    trajectory_size = len(P)
    
    all_failure = []
    for i in range(1, trajectory_size):
        
        delta = np.abs(Pini[i] - Pini[i-1])
        if delta > tol:
            all_failure.append(i)
            Pold = np.copy(Pini[i])

            for integer in range(-Nmax, +Nmax + 1):
                Pnew = Pold + integer * cell[i]
                new_delta = np.abs(Pnew - Pini[i-1])
                if new_delta < tol:
                    if debug:
                        print("     INDEX {} delta {:.1f} solved with delta {:.1f} @ step {}".format(i, delta, new_delta, integer))
                        print(Pini[i], Pnew)
                    Pini[i] = np.copy(Pnew)
                    # Get out from this loop
                    break
                    
    if debug:
        plt.plot(P, label = "BEFORE")
        plt.plot(Pini, ls = ":", label = "AFTER")
        plt.legend()
        plt.show()

    return Pini
                


# def OLDrefold_dipole(P, cell, Nmax = 1, tol = 1.0, debug = True, debug_visualize = False):
#     """
#     REFOLD THE DIPOLES USING INTEGER MULTIPLES OF THE BERRY QUANTUM OF POLARIZATION
#     ===============================================================================

#     1) We identify all the intervals where the dipole is discontinuous
#     2) For each of these ranges, we try to add the quantum time -Nmax, -(Nmax-1), .., 0, 1, .. +Nmax

#     Note: array[i1:i2] select the indices from i1 to i2-1
    
#     Parameters:
#     -----------
#         -P:    np.array with shape N, the dipoles along a given axis x, y, z for a trajectory of size N
#         -cell: np.array with shape N, the Berry quantum along the given axis for a trajectory of size N
#         -Nmax: int, the integer number we multiply the quantum of polarization to have a continuous function
#         -tol: float, the tolerance which we use to tell if the dipoles are continuous or not
#         -debug: bool
#         -debug_visualize: bool, if True the code outputs some plots

#     Returns:
#     --------
#         -Pini: np.array with shape N, the CONTINUOUS dipoles along a given axis x, y, z for a trajectory of size N
        
#     """
#     # The dipoles we manipulate
#     Pini = np.copy(P)
#     # Get the size of the trajectory
#     trajectory_size = len(P)

#     # First check all the differences
#     #differences = np.zeros(trajectory_size, dtype = float)
#     differences = np.abs(Pini[:-1] - Pini[1:])
#     # If they are all smaller than a given treshold we assume no jumps
#     if np.all(differences < tol):
#         print("Nothing to do. Everthing is continuous")
#         return P
#     else:
#         # Check where the differences are larger than tol
#         mask = np.where(differences > tol)[0]
            
#         # Reshape the indices and check if len(mask) is an integer multiple of 2
#         even = True
#         try:
#             # The number of discontinuous parts 
#             N_disc = len(mask)//2
#             mask = np.asarray(mask).reshape((N_disc, 2))
#         except:
#             # Maybe the very last bit is discontinous,
#             # so we append to the mask the very last index of the trajectory
#             mask = np.append(mask, trajectory_size - 1)
#             N_disc = len(mask)//2
#             mask = np.asarray(mask).reshape((N_disc, 2))
#             even = False

#         if debug:
#             print("The inidices where the differences were large")
#             print(mask)
#             print("{} discontinous regiond found. The tol is {}. Even {}\n".format(N_disc, tol, even))
#         for i in range(N_disc):
#             if debug:
#                 print("Range {} the min-max indices are {}".format(i, mask[i]))
#         # We will make continuous each range where the dipoles are discontinuous
#         continuous = [False] * N_disc

#         for i in range(N_disc):
#             # Get all the indices from min to max, 
#             # np.arange create mask[i,0], mask[i,0] + 1, ... mask[i,1] -1
#             # WITH THE +1 WE CONSIDER FROM mask[i,01 + 1 to mask[i,1] INCLUDED
#             indices = np.arange(mask[i,0], mask[i,1])  #+ 1
#             if debug:
#                 print("\nTrying to fix the interval #{} from {} to {}".format(i, indices[0], indices[-1]))
#             # Loop over integer numbers
#             for integer in range(-Nmax, Nmax + 1):
#                 if debug:
#                     print("Trying with {}".format(integer))
#                 # Store the value of the dipoles before the shift
#                 Pini0 = np.copy(Pini)
#                 # Now shift the values of the dipoles by the Berry quantum
#                 Pini[indices] += integer * cell[indices]
#                 if debug_visualize:
#                     plt.plot(np.arange(trajectory_size), Pini)
#                     plt.plot(np.arange(trajectory_size)[indices], Pini[indices], color = "red")
#                     plt.show()
#                 # Check again the differences IN THE CURRENT RANGE 
#                 if not even and i == N_disc - 1:
#                     if debug:
#                         print('ANALYZING {} {}'.format(indices[0], indices[-1]))
#                     differences = np.abs(Pini[:-1] - Pini[1:])[indices[0]-1:indices[-1]]
#                 else:
#                     if debug:
#                         print('ANALYZING {} {}'.format(indices[0], indices[-1]))
#                     differences = np.abs(Pini[:-1] - Pini[1:])[indices]
                
#                 # If it is continuous IN THE CURRENT RANGE then we break the integer loop
#                 if np.all(differences < tol):
#                     if debug:
#                         print("The dipole is continous with {} max diff {:.2e}\n".format(integer,differences.max()))
#                     continuous[i] = True
                    
#                     break
#                 # Otherwise we restore the initial value of the dipole and we keep going
#                 else:
#                     if debug:
#                         print("The dipole is NOT continous with {} max diff {:.2e}".format(integer, differences.max()))
#                     # restore the original values
#                     Pini = np.copy(Pini0)
#                     # This range i is not continous
#                     continuous[i] = False

#         if debug:
#             print("\n\nFINAL RESULTS")
#             print(continuous)
#         if not np.all(np.asarray(continuous) == True):
#             raise ValueError("The dipoles are not continous try to increase Nmax")

#         return Pini


def transform_voigt(tensor, voigt_to_mat = False):
    """
    TRANSFORM TO VOIGT NOTATION
    ===========================
    
    Copied from Cellconstructor useful to convert everything in ase format.

    The voigt notation transforms a symmetric tensor

    ..math: \sigma_{00} \sigma_{01} \sigma_{02}
            \sigma_{10} \sigma_{11} \sigma_{12}
            \sigma_{20} \sigma_{21} \sigma_{22}

    to a 6 compoentn vector

    ..math: \sigma_{00} \sigma_{11} \sigma_{22} \sigma_{12} \sigma_{02} \sigma_{01}
     
    Parameters:
    -----------
        -voit_to_mat. bool default is False, Otherwise it is assumed as a 3x3 symmetric matrix (upper triangle will be read).
                      If True the tensor is assumed to be in voigt format.
    Returns:
    --------
        -new_tensor: np.array
    """

    if voigt_to_mat:
        assert len(tensor) == 6
        new_tensor = np.zeros((3,3), dtype = type(tensor[0]))
        for i in range(3):
            new_tensor[i,i] = tensor[i]

        new_tensor[1,2] = new_tensor[2,1] = tensor[3]
        new_tensor[0,2] = new_tensor[2,0] = tensor[4]
        new_tensor[1,0] = new_tensor[0,1] = tensor[5]
    else:
        assert tensor.shape == (3,3)
        new_tensor = np.zeros(6, dtype =  type(tensor[0,0]))
        # First 
        for i in range(3):
            new_tensor[i] = tensor[i,i]
        new_tensor[3] = tensor[1,2]
        new_tensor[4] = tensor[0,2]
        new_tensor[5] = tensor[0,1]
    
    return new_tensor
