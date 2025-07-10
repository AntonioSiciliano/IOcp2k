import ase, ase.io
import sys
import os
filename = sys.argv[1]

atoms = ase.io.read(filename, index = ":")
print("Size of the traj", len(atoms))
print("Saving .traj file in {}".format(os.getcwd()))
ase.io.write((filename.split("/")[-1]).split(".")[0] + '.traj', atoms, format = 'traj')