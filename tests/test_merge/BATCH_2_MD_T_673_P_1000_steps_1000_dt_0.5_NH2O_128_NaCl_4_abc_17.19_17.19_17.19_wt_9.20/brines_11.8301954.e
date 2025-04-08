+ SCRIPT_PID=3066465
+ /bin/ksh -x /tmp/tmp.Lw3xGEaBwm
+ set +x
+ set -x
+ module switch feature/openmpi/net/ib/ucx-noxpmem
+ unset _mlshdbg
+ [ 1 '=' 1 ]
+ unset _mlshdbg
+ _mlredir=0
+ typeset _mlredir
+ [ -n '' ]
+ [ 0 -eq 0 ]
+ _module_raw switch feature/openmpi/net/ib/ucx-noxpmem
+ unset _mlshdbg
+ [ 1 '=' 1 ]
load module feature/openmpi/net/ib/ucx-noxpmem
+ unset _mlshdbg
+ return 0
+ module load gnu/11 mpi/openmpi/4 dbcsr/2.7.0
+ _mlredir=0
+ typeset _mlredir
+ [ -n '' ]
+ [ 0 -eq 0 ]
+ _module_raw load gnu/11 mpi/openmpi/4 dbcsr/2.7.0
+ unset _mlshdbg
+ [ 1 '=' 1 ]
load module flavor/buildcompiler/gcc/11
load module flavor/gnu/standard
load module c++/gnu/11.2.0
load module c/gnu/11.2.0
load module fortran/gnu/11.2.0
load module gnu/11.2.0
load module flavor/buildmpi/openmpi/4
load module feature/openmpi/mpi_compiler/gcc
load module feature/mkl/single_node
load module feature/openmpi/io/standard
load module flavor/cuda/nvhpc-222
load module cuda/11.6
load module flavor/libccc_user/hwloc2
load module hwloc/2.5.0
load module pmix/4.2.2
load module mpi/openmpi/4.1.4
load module fypp/3.2
load module dbcsr/2.7.0
+ unset _mlshdbg
+ return 0
+ module load cp2k/2024.3
+ _mlredir=0
+ typeset _mlredir
+ [ -n '' ]
+ [ 0 -eq 0 ]
+ _module_raw load cp2k/2024.3
+ unset _mlshdbg
+ [ 1 '=' 1 ]
load module flavor/cp2k/xc
load module flavor/libint/cp2k
load module libint/2.6.0
load module libvori/220621
load module libxc/6.1.0
load module feature/mkl/lp64
load module feature/mkl/sequential
load module feature/mkl/vector/amd
load module mkl/20.0.0
load module plumed/2.8.2
[36mWARNING: the loaded flavor/cp2k/xc is not found, fallback to flavor/cp2k/standard[0m
load module cp2k/2024.3
+ unset _mlshdbg
+ return 0
+ cd /ccc/scratch/cont003/gen2309/sicilana/NPT/128_H2O_4_NaCl_10_wt_673K_1000bar/BATCH_11_MD_T_673_P_1000_steps_1000_dt_0.5_NH2O_128_NaCl_4_abc_17.19_17.19_17.19_wt_9.20
+ ccc_mprun cp2k.psmp -i input.inp -o output.out
+ cp ./brines-1.restart /ccc/scratch/cont003/gen2309/sicilana/NPT/128_H2O_4_NaCl_10_wt_673K_1000bar/BATCH_12_MD_T_673_P_1000_steps_1000_dt_0.5_NH2O_128_NaCl_4_abc_17.19_17.19_17.19_wt_9.20/initial.restart
+ cd /ccc/scratch/cont003/gen2309/sicilana/NPT/128_H2O_4_NaCl_10_wt_673K_1000bar/BATCH_12_MD_T_673_P_1000_steps_1000_dt_0.5_NH2O_128_NaCl_4_abc_17.19_17.19_17.19_wt_9.20
+ chmod g+s ./initial.restart ./input.inp ./param_cluster.json ./param_cp2k.json ./run.sh ./structure.xyz ./when_created.json
+ ccc_msub run.sh
+ exit 0
