import ase, ase.io
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import AtomicSnap
import AtomicSnap.Conductivity

def test():
    atoms = ase.io.read("atoms.xyz", index= ":")
    cond = AtomicSnap.Conductivity.Conductivity()

    # Initialize the class
    cond.init(atoms, dt = 0.5, T = 673)

    # 
    cond.plot_current_current_correlation_function()
    # 
    cond.test_implementation()

    # cond.test_implementation2()
    cond.plot_sigma()

    # print(cond.V)
    # print(np.sqrt(np.sum((cond.volumes - cond.V)**2))/np.sqrt(cond.N))
    # print(np.std(cond.volumes)/np.sqrt(cond.N))
    # print(np.sqrt(np.sum((cond.volumes - cond.V)**2))/cond.N)
    # plt.plot(cond.volumes)
    # plt.show()


if __name__ == '__main__':
    """
    TEST CONDUCTIVITY
    =================
    """
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    test()