
#!/bin/bash
#MSUB -r brines_11
#MSUB -q rome   #rome has 128 prc per node, skylake has 48
#MSUB -N 1
#MSUB -n 128
#MSUB -m scratch
#MSUB -x
#MSUB -E '--no-requeue'
#MSUB -Q normal    #normal or long
#MSUB -T 80000   
#MSUB -A gen2309
set -x
        
module switch feature/openmpi/net/ib/ucx-noxpmem

module load gnu/11 mpi/openmpi/4 dbcsr/2.7.0

module load cp2k/2024.3


        

cd /ccc/scratch/cont003/gen2309/sicilana/NPT/128_H2O_4_NaCl_10_wt_673K_1000bar/BATCH_11_MD_T_673_P_1000_steps_1000_dt_0.5_NH2O_128_NaCl_4_abc_17.19_17.19_17.19_wt_9.20
ccc_mprun  cp2k.psmp  -i input.inp -o output.out
    
cp ./brines-1.restart /ccc/scratch/cont003/gen2309/sicilana/NPT/128_H2O_4_NaCl_10_wt_673K_1000bar/BATCH_12_MD_T_673_P_1000_steps_1000_dt_0.5_NH2O_128_NaCl_4_abc_17.19_17.19_17.19_wt_9.20/initial.restart

cd /ccc/scratch/cont003/gen2309/sicilana/NPT/128_H2O_4_NaCl_10_wt_673K_1000bar/BATCH_12_MD_T_673_P_1000_steps_1000_dt_0.5_NH2O_128_NaCl_4_abc_17.19_17.19_17.19_wt_9.20
chmod g+s ./*
ccc_msub  run.sh
