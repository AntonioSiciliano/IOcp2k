import pytest

import ase, ase.io, ase.visualize
from ase import Atoms

import MDAnalysis
import MDAnalysis.analysis
import MDAnalysis.analysis.rdf
import MDAnalysis.analysis.msd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os, sys

def get_ase_atoms(pos1, pos2, L, pbc):
    """
    GENERATE ASE ATOMS WITH PBC
    ===========================
    """
    all_atoms = []

    av_d = 0
    
    for i in range(100):
        # Define atom positions
        positions = [(0, 0, pos1), (0, 0, pos2 + np.random.uniform(low = -0.01, high = 0.01))]
        # Create the Atoms object
        atoms = Atoms('C2', positions = positions, cell = [L, L, L], pbc = pbc)
        all_atoms.append(atoms)

        av_d += atoms.get_distances(0, 1, mic = pbc)
    
    ase.io.write('all_ase_atoms.xyz', all_atoms)
    
    return av_d /len(all_atoms)


def test_gr_md_analysis(pos1 = 0, pos2 = 8 , L = 10, pbc = True):
    """
    GET THE GR ANALYSIS
    ===================
    """
    matplotlib.use('tkagg')
    
    # Create the ase atoms and get the expected distance with pbc or without
    exp_d = get_ase_atoms(pos1, pos2, L, pbc)
    
    MD_atoms = MDAnalysis.Universe('all_ase_atoms.xyz')

    for snapshot, MD_atoms_snapshot in enumerate(MD_atoms.trajectory):
        # Manually set the unit cell dimensions in ANGSTROM
        MD_atoms_snapshot.dimensions = [L, L, L, 90.0, 90.0, 90.0]  
        # Apply the time step to all frames in PICOSECONDS
        MD_atoms_snapshot.dt = 0.5 * 1e-3
        
    # Select only the atomic type we want
    MD_atoms_selected1 = MD_atoms.select_atoms('name C')
    # Compute RDF
    rdf_calc = MDAnalysis.analysis.rdf.InterRDF(MD_atoms_selected1, MD_atoms_selected1, range = (0.1, 10.0), nbins = 1000, norm = 'rdf')

    rdf_calc.run()
    
    r, gr = rdf_calc.results.bins, rdf_calc.results.rdf
    
    # plt.plot(r, gr)
    # plt.show()

    mask = np.where(gr != 0)
    # If there are PBC in the ase atoms then MD Analysis and ase distances with PBC must give the same result
    if pbc:
        if np.abs(exp_d - np.average(r[mask])) > 1e-2:
            raise ValueError('PBC distances are not correctly captured {} {}'.format(exp_d , np.average(r[mask])))
    # If there are NO PBC in the ase atoms then MD Analysis and ase distances of ase MUST BE DIFFERENT
    else:
        if np.abs(exp_d - np.average(r[mask])) < 1:
            raise ValueError('PBC distances are not correctly captured {} {}'.format(exp_d , np.average(r[mask])))
        
    
if __name__ == '__main__':
    """
    SCRIPT TO TEST THE AWARNESS OF PBC IN MD Analysis
    =================================================

    MD Analysis always takes into account PBC even if the ase atoms we set pbc = False.

    Here we create ase atoms object without PBC and with just two atoms.

    The first atoms in placed in the origin while the second one is close to the edge of the unit cell.

    We check that MD Analysis gives a peak in the g(r) that is compatible with PBC
    """
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    test_gr_md_analysis(pos1 = 0, pos2 = 8, L = 10, pbc = True)

    test_gr_md_analysis(pos1 = 0, pos2 = 8, L = 10, pbc = False)

    # test_gr_md_analysis(pos1 = 0, pos2 = 8, L = 20, pbc = True)