import ase, ase.io, ase.atoms, ase.visualize
import os, sys
import numpy as np

def read_custom_lammps_file(filename, types_to_atoms):
    """
    READ CUSTOM LAMMPS FILE
    =======================

    Read LAMMPS file using METAL UNITS
    """
    # The number of snapshots
    Nsnap = 0
    # All the ase atoms objects
    all_atoms_objects = []

    # Read the Lammps custom file
    file = open(filename, 'r')
    # Read the lines
    lines = file.readlines()
    
    for index, line in enumerate(lines):
        if line.startswith('ITEM: TIMESTEP'):
            Nsnap += 1
            # Get the number of atoms
            Nat = int(lines[index + 3])

            # Get the cell in 
            cell_a, cell_b, cell_c = float(lines[index + 5].split()[-1]), float(lines[index + 6].split()[-1]), float(lines[index + 7].split()[-1])
            if cell_a != cell_b or cell_a != cell_c or cell_b != cell_c:
                raise ValueError("the cell is not isotropic")
            # Set the cell
            cell = np.eye(3) * cell_a

            # Get all the other atomic properties
            arrays = np.zeros((Nat, 2 + 3 + 3 + 3), dtype = float)
            for at in range(Nat):
                # Each line has
                # id type xu yu zu vx vy vz fx fy fz
                arrays[at,:] = np.asarray(lines[index + 9 + at].split())

            if Nsnap%500 == 0:
                print("\nSTEP {}".format(Nsnap))
            # print("POSITIONS")
            # print(arrays[:, 0])
            # print("VELOCITIES")
            # print(arrays[0,2:5])
            # print("FORCES")
            # print(arrays[0,5:8])
            # print("CELL")
            # print(cell)
            # print("TYPES")
            # print(types)
            # print()


            
            
            # Order according to the ids
            arrays = arrays[arrays[:, 0].argsort()]
            # Get the atomic types
            types = [types_to_atoms[item] for item in arrays[:,1]]
            # The ids of the
            ids = arrays[:,0]

        
            # Get positions, velocities and forces
            positions   = arrays[:, 2:5] # Angstrom
            velocities  = arrays[:, 5:8] # Angstrom/ps
            forces      = arrays[:, 8:]  # eV/Angstrom

            # Create the ase atoms
            structure = ase.atoms.Atoms(types, positions, cell = cell, pbc = [True, True, True])
            structure.set_cell(cell)
            structure.pbc = True
            # Set the velocities
            structure.set_velocities(velocities)

            # Now set the calculato so that we have energies and forces
            calculator = ase.calculators.singlepoint.SinglePointCalculator(structure, energy = None, forces = forces, stress = None)
            # Attach the calculator
            structure.calc = calculator


    all_atoms_objects.append(structure)


    # ase.io.write("output.xyz", all_atoms_objects)
    # ase.visualize.view(all_atoms_objects)

    return all_atoms_objects

if __name__ == '__main__':
    """
    SCRIPT TO READ OUTPUT FILES OF LAMMPS
    =====================================

    Call this script in the direcot
    """
    # Put this to go in the directory where the file is
    # total_path = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(total_path)

    filename = sys.argv[1] #"output.dump'
    atomic_types = list(sys.argv[2:]) # ["O", "H", "Na", "Cl"]

    # print(os.getcwd())

    atomic_types = sorted(atomic_types)
    types_to_atoms = {}
    for i, item in enumerate(atomic_types):
        types_to_atoms.update({int(i+1) : item})

    # print(types_to_atoms)
    
    # Get the ase atoms
    ase_atoms = read_custom_lammps_file(filename, types_to_atoms)

    # Define the name of the file
    name_xyz = filename.split(".")[0] + "_nowrap" + ".xyz"

    ase.io.write(name_xyz, ase_atoms)