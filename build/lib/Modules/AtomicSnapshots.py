import numpy as np
import ase
from matplotlib.ticker import MaxNLocator
from ase import Atoms
import ase.geometry
import os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import AtomicSnap
from AtomicSnap import AtomicSnapshots
import copy

__JULIA__ = False
try:
    print("Try to import Julia...\n")
    import julia, julia.Main
    # Load the Julia file
    julia.Main.include("/home/antonio/IOcp2k/Modules/pair_correlation.jl")
    # print(os.path.join(os.path.dirname(__file__), "pair_correlation.jl"))
    # julia.Main.include(os.path.join(os.path.dirname(__file__), "pair_correlation.jl"))
except:
    print("Probably a warning...\n")

try:
    print("Test if Julia works...\n")
    julia.Main.eval('println("Hello from Julia!")')
    __JULIA__ = True
except:
    print("No Julia found!")
    __JULIA__ = False

BOHR_TO_ANGSTROM = 0.529177249 
HA_TO_EV = 27.2114079527
HA_TO_KELVIN = 315775.326864009
HA_BOHR_TO_EV_ANGSTROM   = HA_TO_EV / BOHR_TO_ANGSTROM
HA_BOHR3_TO_EV_ANGSTROM3 = HA_TO_EV / BOHR_TO_ANGSTROM**3
BAR_TO_HA_BOHR3 = 6.89475728e-10 
# Velocities
ANG_FEMPTOSEC_TO_HA = 0.04571028907825843

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
        
        Stresses are in BAT
        
        Velocities are in BOHR/AU TIME

        Time step in fs

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
        # stress BAR
        self.stresses = None
        # Atomic types
        self.types = None
        # The number of atoms
        self.N_atoms = -1
        # The number of snapshots
        self.snapshots = 0
        # The cell
        self.unit_cell = np.eye(3) * 10
        # Potential and kientic energy
        self.kinetic_energies = None
        self.potential_energies = None
        self.temperatures = None
        self.cons_quant = None
        self.velocities = None
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
                   ext_ener = None):
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
            -calc_type: it is needed to understand which calculation was done
            -ext_pos: the extension of the position file
            -ext_force: the extension of the force file
            -ext_stress: the extension of the stress file
            -ext_velocities: the extension of the velocities file
            -ext_ener: the extension of the energ file
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
        # Conserved quantity in HARTREE
        self.cons_quant = np.zeros(self.snapshots)
        # Energies HARTREE
        self.energies = np.zeros(self.snapshots)
        # Stresses HARTREE/BOHR3
        self.stresses = np.zeros((self.snapshots, 3, 3))
        # Velocities BOHR/AU TIME
        self.velocities = np.zeros((self.snapshots, self.N_atoms, 3))
        # Kinetic and potential energy in HARTREE
        self.kinetic_energies = np.zeros(self.snapshots)
        self.potential_energies = np.zeros(self.snapshots)
        # Temperature in KELVIN
        self.temperatures = np.zeros(self.snapshots)
        
        
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
                    self.stresses[isnap, :, :] *= BAR_TO_HA_BOHR3
                
            # Close the file
            file_stress.close()
        ####### END READ THE STRESS FILE ########
        
        
        
        
        ####### READ THE VELOCITY FILE ########
        if not(ext_vel is None):
            if not os.path.exists(file_name_head + ext_vel):
                raise ValueError('I do not find the stress file {}'.format(file_name_head + ext_vel))
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
                self.dt = float(lines[2].split()[1]) - float(lines[1].split()[1])
                print('Update the dt to {}'.format(self.dt))
            # close the file
            file_ener.close()
        ####### END READ THE VELOCITY FILE ########

        return 


    def create_ase_snapshots(self, pbc = [1,1,1], wrap_positions = True):
        """
        CREATE ASE SNAPSHOTS 
        
        Rember that ase use EV, ANGSTROM
        
        Paramters:
        ----------
            -uc: a np.array with the cell BOHR
            -pbc: the PBC conditions along x y z (1=True)
            
        Returns:
        --------
            -all_atoms: a list of ase objects
        """
        # Set up all the ase atoms
        all_atoms = []


        if wrap_positions:
            print("Wrapping the positons")
            
        # Range in the snapshots ans use ASE units
        # ev and angstrom
        for isnap in range(self.snapshots):
            if wrap_positions:
                wrap_coords = ase.geometry.wrap_positions(self.positions[isnap,:,:], self.unit_cell, pbc=True)
                atoms = Atoms(self.types, positions = wrap_coords, pbc = pbc)
            else:
                atoms = Atoms(self.types, positions = self.positions[isnap,:,:], pbc = pbc)
            # print(atoms.unit_cell.reshape)
            atoms.set_cell(self.unit_cell)
            atoms.forces = self.forces[isnap,:,:] * HA_BOHR_TO_EV_ANGSTROM
            atoms.energy = self.energies[isnap] * HA_BOHR_TO_EV_ANGSTROM
            atoms.stress = transform_voigt(self.stresses[isnap,:,:]) * HA_BOHR3_TO_EV_ANGSTROM3

            all_atoms.append(atoms)
            
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
            
            # Use np.copy() for NumPy arrays, deep copy for others
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
            if not(attr in ['uc', 'types', 'calc_type', 'snapshots', 'dt']):
                # Get the attribute's value
                value1 =             getattr(self, attr) 
                value2 = getattr(atomic_snapshots, attr)
                
                # Use np.copy() for np arrays, copy.deepcopy for others
                if isinstance(value1, np.ndarray):
                    setattr(snap_merge, attr, np.concatenate((value1, value2), axis=0))
                else:
                    setattr(snap_merge, attr, copy.deepcopy(value1))  

        snap_merge.snapshots = self.snapshots + atomic_snapshots.snapshots
        
        snap_merge.unit_cell = np.copy(self.unit_cell)

        snap_merge.calc_type = copy.deepcopy(self.calc_type)

        snap_merge.types = copy.deepcopy(self.types)

        snap_merge.dt = np.copy(self.dt)
        
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
        # Get the ase atoms
        ase_atoms = self.create_ase_snapshots()
        ase_energies = [ase_atoms[i].energy for i in range(self.snapshots)]
        ase_energies = np.asarray(ase_energies) /self.N_atoms
        
        forces = [ase_atoms[i].forces[:,:] for i in range(self.snapshots)]
        forces = np.asarray(forces).reshape((self.snapshots, self.N_atoms, 3))
        ase_forces = np.einsum('iab, iab -> i', forces, forces) /self.N_atoms
        
        x = np.arange(self.snapshots, dtype = int)
        
        # Width and height
        fig = plt.figure(figsize=(8, 5))
        gs = gridspec.GridSpec(2, 1, figure=fig)
        ax = fig.add_subplot(gs[0,0])
        ax.plot(x, ase_energies, 's', color = 'k')
        ax.set_ylabel('Energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax = fig.add_subplot(gs[1,0])
        ax.plot(x, ase_forces, 'd', color = 'red')
        ax.set_xlabel('Steps', size = 15)
        ax.set_ylabel('Force [eV/Ang/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        
        plt.tight_layout()

        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        plt.show()
        
        return


    def plot_temperature_evolution(self, average_window = [0,1], img_name = None):
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
        # Width and height
        fig = plt.figure(figsize=(7, 4))
        gs = gridspec.GridSpec(1,1, figure=fig)

        x = np.arange(self.snapshots) * self.dt

        T_av = np.average(self.temperatures[average_window[0]: average_window[1]])
        
        ax = fig.add_subplot(gs[0,0])
        xmin, xmax = np.sort(x[average_window])
        ax.plot(x, self.temperatures,  color = 'green', lw = 3, label = 'T = {:.1f} K from {:.0f} to {:.0f} fs'.format(T_av, xmin, xmax))
        ax.fill_between(x, self.temperatures, np.min(self.temperatures), where=(x >= xmin) & (x <= xmax), color='green', alpha=0.3)
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
        

    def plot_md_nvt(self, img_name = None):
        """
        PLOT ENERGY FORCE FROM ASE SNAPSHOTS FOR NVT SIMULATION
        
        Rember that ase use EV, ANGSTROM
        
        To plot the force at each time step 
        
        .. math::   \sum_{I=1}^{N_at} F_I \cdot F_I /N_at
        
         Paramters:
        ----------
            -img_name: imag name if you want to save
        """
        # Get the ase atoms
        ase_atoms = self.create_ase_snapshots()
        ase_energies = [ase_atoms[i].energy for i in range(self.snapshots)]
        ase_energies = np.asarray(ase_energies) /self.N_atoms
        
        forces = [ase_atoms[i].forces[:,:] for i in range(self.snapshots)]
        forces = np.asarray(forces).reshape((self.snapshots, self.N_atoms, 3))
        ase_forces = np.einsum('iab, iab -> i', forces, forces) /self.N_atoms
        
        x = np.arange(self.snapshots, dtype = int) * self.dt
        
        # Width and height
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(2,3, figure=fig)
        
        ax = fig.add_subplot(gs[0,0])
        ax.plot(x, ase_energies,  color = 'k', lw = 3)
        ax.set_ylabel('Energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        
        ax = fig.add_subplot(gs[0,1])
        ax.plot(x, self.potential_energies * HA_TO_EV/self.N_atoms,  color = 'purple', lw = 3)
        ax.set_ylabel('Pot energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        
        ax = fig.add_subplot(gs[0,2])
        ax.plot(x, self.kinetic_energies * HA_TO_EV/self.N_atoms,  color = 'darkorange', lw = 3)
        ax.set_ylabel('Kin energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        

        
        ax = fig.add_subplot(gs[1,0])
        ax.plot(x, ase_forces,  color = 'red', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('Force [eV/Ang/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


        ax = fig.add_subplot(gs[1,1])
        ax.plot(x, self.temperatures,  color = 'green', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('Temperature [Kelvin]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


        ax = fig.add_subplot(gs[1,2])
        ax.plot(x, self.cons_quant * HA_TO_EV * 1000/self.N_atoms,  color = 'k', lw = 3)
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



    def show_com_motion(self, dictionary, img_name = None):
        """
        PLOT THE MOTION OF THE CENTER OF MASS
        =====================================

        Checks the drift of the com

        Use units of AMU for the mass

        Paramters:
        ----------
            -dictionary: a dictionary with the atomic symbols and corresponing masses dictionary = {"H" : 1.007825, "O" : 15.99491 , "Na" : 22.98976928, "Cl" : 35.446}
        """
        
        masses_array = np.zeros(self.N_atoms)
        for i , at_type in enumerate(self.types):
            masses_array[i] = dictionary[at_type] 
        
        R_com = np.einsum('a, iab -> ib', masses_array, self.positions) /np.sum(masses_array)

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
            ax.set_ylabel('V$_{com}$ [Bohr/t_${au}$]', size = 12)
        plt.tight_layout()
        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        plt.show()


        # Width and height
        fig = plt.figure(figsize=(10, 5))
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

    def get_pair_correlation_function(self, type1, type2, dr, Nr):
        """
        RETURNS THE PAIR CORRELATION FUNCTION
        =====================================
        
        Rember that ase use EV, ANGSTROM
        
        It computes the pair correlation function
        
        .. math::  g(r) = \\frac{V}{4 \pi r^2 N^2} \left\langle \delta(r - R_{IJ}) \right\rangle
        
         Paramters:
        ----------
            -type1, type2: the pair of atoms you want to consider in the g(r)
            -dr: the radial spacing in ANGSTROM
            -Nr: the number of points in the grid

        Returns:
        --------
            -r: np.array, with the radial grid
            -g: np.array, the pair correlation function
        """
        
        if not __JULIA__:
            raise ValueError('You need Julia')

        print("Get the pair correlation function for {} {}".format(type1, type2))
        
        # Get the ase atoms
        ase_atoms = self.create_ase_snapshots()

        # Exclude the zero ANGTROM
        r_grid = dr * np.arange(1, Nr + 1)

        # RDF
        g_r = np.zeros(len(r_grid), dtype = type(dr))

        # Check that the maxium is compatible with the PBC
        for i in range(3):
            if r_grid[-1] > self.unit_cell[i,i]:
                raise ValueError('Reduce the number of points Nr or reduce the dr')

        # def get_mask_array_string(string_list, mytype):

        # mask_types = np.where(self.types == type1) or np.where(self.types == type2)

        def generate_mask(input_list, target_elements):
            mask = [elem in target_elements for elem in input_list]
            return np.asarray(mask, dtype = bool)


        for ir, r in enumerate(r_grid):
            for isnap in range(1):
                # Total number of atoms for the current snapshots
                N_at        = ase_atoms[isnap].get_number_of_atoms()
                # Get the distances ANGSTROM
                distances   = ase_atoms[isnap].get_all_distances(mic = True)
                # Get the atomic types
                atoms_types = ase_atoms[isnap].get_chemical_symbols()
                # Get the mask to select only type1 and type2 atoms
                mask = generate_mask(atoms_types, [type1, type2])

                # In this way the julia code runs only on a subset of atoms
                N_at_mask   = np.arange(N_at)[mask]
                g_r[ir]     = julia.Main.pair_correlation_function(type1, type2, r, dr, N_at_mask, atoms_types, distances, np.linalg.det(ase_atoms[isnap].cell), N_at)
        # for ir, r in enumerate(r_grid):
        #     for isnap in range(self.snapshots):
        #         N_at = ase_atoms[isnap].get_number_of_atoms()
        #         atoms_types = ase_atoms[isnap].get_chemical_symbols()

        #         for atom1 in range(N_at):
        #             for atom2 in range(N_at):
        #                 if atom1 != atom2:
        #                     if (atoms_types[atom1] == type1 and atoms_types[atom2] == type2) or (atoms_types[atom2] == type1 and atoms_types[atom1] == type2):
        #                         d = ase_atoms[isnap].get_distance(atom1, atom2,  mic = True)
    
        #                         if (r - dr) < d and d < (r + dr):
        #                             g_r[ir] += dr * np.linalg.det(ase_atoms[isnap].cell)/(4 * np.pi * r**2 * N_at**2) 
        #     g_r[ir] /= self.snapshots
        
        return r_grid, g_r






def transform_voigt(tensor, voigt_to_mat = False):
    """
    TRANSFORM TO VOIGT NOTATION
    ===========================
    
    Copied from Cellconstructor useful to ocnvert everything in ase format
    
    Parameters:
    -----------
        -voit_to_mat. bool is True, the tensor is assumed to be in voigt format.
        Otherwise it assumed as a 3x3 symmetric matrix (upper triangle will be read).
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

        for i in range(3):
            new_tensor[i] = tensor[i,i]
        new_tensor[3] = tensor[1,2]
        new_tensor[4] = tensor[0,2]
        new_tensor[5] = tensor[0,1]
    
    return new_tensor
