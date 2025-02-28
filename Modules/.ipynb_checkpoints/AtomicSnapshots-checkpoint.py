import numpy as np
import ase
from ase import Atoms
import os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BOHR_TO_ANGSTROM = 0.529177249 
HA_TO_EV = 27.2114079527
HA_BOHR_TO_EV_ANGSTROM   = HA_TO_EV / BOHR_TO_ANGSTROM
HA_BOHR3_TO_EV_ANGSTROM3 = HA_TO_EV / BOHR_TO_ANGSTROM**3
BAR_TO_HA_BOHR3 = 6.89475728e-10 

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
        
        Positions are in BOHR

        Forces are in HARTREE/BOHR
        
        Stresses are in HARTREE/BOHR^3
        
        Velcoties are in BOHR/AU TIME

        Parameters
        ----------
            -**kwargs : any other attribute of the ensemble

        """
        # Atomic position BOHR
        self.positions = None
        # Atomic forces HARTREE/BOHR
        self.forces = None
        # Energies
        self.energies = None
        # stress
        self.stresses = None
        # Atomic types
        self.types = None
        # The number of atoms
        self.N_atoms = -1
        # The number of snapshots
        self.snapshots = 0
        # The cell
        self.uc = np.eye(3) * 10
        # Potential and kientic energy
        self.kinetic_energies = None
        self.potential_energies = None
        self.temperatures = None
        # the time step
        self.dt = None
        
        
        # Setup the attribute control
        self.__total_attributes__ = [item for item in self.__dict__.keys()]
        # This must be the last attribute to be setted
        self.fixed_attributes = True 

        # Setup any other keyword given in input (raising the error if not already defined)
        for key in kwargs:
            self.__setattr__(key, kwargs[key])
        
        

    def init(self, file_name_head, debug = False, verbose = True,
                      ext_pos = '.xyz', ext_force = '.force',
                      ext_stress = '.stress', ext_vel = '.vel',
                      ext_ener = '.ener'):
        """
        READ XYZ FILE AS CREATED BY CP2K

        Positions are in BOHR

        Forces are in HARTREE/BOHR
        
        Stresses are in HARTREE/BOHR^3
        
        Velcoties are in BOHR/AU TIME
        
        Paramters:
        ----------
            -file_name_head: the path to the extension of the file
            -debug: bool,
            -verbose: bool,
            -ext_pos: the extension of the position file
            -ext_force: the extension of the force file
            -ext_stress: the extension of the stress file
            -ext_velocities: the extension of the velocities file
            -ext_ener: the extension of the energ file
        """ 
        # Open the target file
        # Note that at least the postion file should exists
        file_xyz = open(file_name_head + ext_pos, 'r')
        lines = file_xyz.readlines()

        # Get the number of atoms
        self.N_atoms = int(lines[0])
        # Get the number of MD, GEO_OPT steps
        for index, line in enumerate(lines):
            if line.startswith(' i ='):
                self.snapshots += 1
            if line.startswith(' i =        1'):
                self.dt = float(lines[index].split()[5][:-1])
                
        if verbose:
            print('\n====> CP2K SNAPSHOTS <====')
            print('CP2k reading from files beginning with {}'.format(file_name_head))               
            print('N_atoms   = {}'.format(self.N_atoms)) 
            print('Snapshots = {}'.format(self.snapshots))
        
        # Initialize everything
        # Positions BOHR and atomic types
        self.positions = np.zeros((self.snapshots, self.N_atoms, 3))
        # Atomic types
        self.types = [''] * self.N_atoms
        # Forces HARTREE/BOHR
        self.forces = np.zeros((self.snapshots, self.N_atoms, 3))
        # Energies HARTREE
        self.energies = np.zeros(self.snapshots)
        # Stresses HARTREE/BOHR3
        self.stresses = np.zeros((self.snapshots, 3, 3))
        # Velocities BOHR/AU TIME
        self.velocities = np.zeros((self.snapshots, self.N_atoms, 3))
        # Kinetic and potential energy in HARTREE
        self.kinetic_energies = np.zeros(self.snapshots)
        self.potential_energies = np.zeros(self.snapshots)
        # Temperature in Kelvin
        self.temperatures = np.zeros(self.snapshots)
        
        
        # Read the atomic positions of each snapshots
        for isnap in range(self.snapshots):
            file_index = isnap * (self.N_atoms + 2) + 2
            
            self.energies[isnap] = float(lines[file_index - 1].split()[-1])
            for iatom in range(self.N_atoms):
                coords = np.asarray(lines[file_index + iatom].split()[1:])
                self.positions[isnap, iatom, :] = coords
                self.types[iatom] = lines[file_index + iatom].split()[0]
            if debug:
                print('COORDS SNAP={}'.format(isnap))
                print(self.positions[isnap,:,:])
                print(self.types)
                print()

        # close the file
        file_xyz.close()
        
        if verbose:
            print('Types of atoms {}'.format(self.types))
            print('Unique Types of atoms {}'.format(set(self.types)))
        
        

        ####### READ THE FORCES FILE ########
        if not(ext_force is None):
            if not os.path.exists(file_name_head + ext_force):
                raise ValueError('I do not find the force file {}'.format(file_name_head + ext_force))
            # Open the target file
            file_force = open(file_name_head + ext_force, 'r')
            lines = file_force.readlines()

            # Read the atomic positions of each snapshots
            for isnap in range(self.snapshots):
                # The index after which we start reading
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
            # close the file
            file_ener.close()
        ####### END READ THE VELOCITY FILE ########

        return 


    def create_ase_snapshots(self, uc = None, pbc = [1,1,1]):
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
        # Set up the unit cell
        if uc is None:
            uc = self.uc.copy()
        # Check the shape of the uc
        if uc.shape != (3,3):
            raise ValueError('The unit cell is not correct')
            
        # Range in the snapshots ans use ASE units
        # ev and angstrom
        for isnap in range(self.snapshots):
            atoms = Atoms(self.types,
                          positions = self.positions[isnap,:,:] * BOHR_TO_ANGSTROM,
                          cell = uc * BOHR_TO_ANGSTROM, pbc = pbc)
            atoms.forces = self.forces[isnap,:,:] * HA_BOHR_TO_EV_ANGSTROM
            atoms.energy = self.energies[isnap] * HA_BOHR_TO_EV_ANGSTROM
            atoms.stress = transform_voigt(self.stresses[isnap,:,:]) * HA_BOHR3_TO_EV_ANGSTROM3

            all_atoms.append(atoms)
            
        return all_atoms
    

    
    def plot_energy_force(self, uc = None, img_name = None):
        """
        PLOT ENERGY FORCE FROM ASE SNAPSHOTS 
        
        Rember that ase use EV, ANGSTROM
        
        To plot the force at each time step 
        
            \sum_{I=1}^{N_at} F_I \cdot F_I /N_at
        
         Paramters:
        ----------
            -uc: a np.array with the cell BOHR
            -img_name: imag name if you want to save
        """
        # Get the ase atoms
        ase_atoms = self.create_ase_snapshots(uc = uc)
        ase_energies = [ase_atoms[i].energy for i in range(self.snapshots)]
        ase_energies = np.asarray(ase_energies) /self.N_atoms
        
        forces = [ase_atoms[i].forces[:,:] for i in range(self.snapshots)]
        forces = np.asarray(forces).reshape((self.snapshots, self.N_atoms, 3))
        ase_forces = np.einsum('iab, iab -> i', forces, forces) /self.N_atoms
        
        x = np.arange(self.snapshots)
        
        # Width and height
        fig = plt.figure(figsize=(8, 5))
        gs = gridspec.GridSpec(2, 1, figure=fig)
        ax = fig.add_subplot(gs[0,0])
        ax.plot(x, ase_energies, 's', color = 'k')
        ax.set_ylabel('Energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        
        ax = fig.add_subplot(gs[1,0])
        ax.plot(x, ase_forces, 'd', color = 'red')
        ax.set_xlabel('Steps', size = 15)
        ax.set_ylabel('Force [eV/Ang/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        
        plt.tight_layout()

        if not(img_name is None):
            plt.savefig(img_name, dpi = '500')
        plt.show()
        
        return
 






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