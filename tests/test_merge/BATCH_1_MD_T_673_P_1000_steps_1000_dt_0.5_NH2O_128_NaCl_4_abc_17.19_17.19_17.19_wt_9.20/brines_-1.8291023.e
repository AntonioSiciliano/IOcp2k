+ SCRIPT_PID=2519662
+ /bin/ksh -x /tmp/tmp.ylImkJGKIS
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
unload module cp2k/2024.3
unload module plumed/2.8.2
unload module libvori/220621
unload module dbcsr/2.7.0
unload module mpi/openmpi/4.1.4
unload module tuning/openmpi/4.1.4
unload module feature/openmpi/net/ib/ucx-noxpmem
load module feature/openmpi/net/ib/ucx-noxpmem
load module tuning/openmpi/4.1.4
load module mpi/openmpi/4.1.4
load module dbcsr/2.7.0
load module libvori/220621
load module plumed/2.8.2
[36mWARNING: the loaded flavor/cp2k/xc is not found, fallback to flavor/cp2k/standard[0m
load module cp2k/2024.3
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
+ unset _mlshdbg
+ return 0
+ cd /ccc/scratch/cont003/gen2309/sicilana/NPT/128_H2O_4_NaCl_10_wt_673K_1000bar/BATCH_10_MD_T_673_P_1000_steps_1000_dt_0.5_NH2O_128_NaCl_4_abc_17.19_17.19_17.19_wt_9.20
+ ccc_mprun cp2k.psmp -i input.inp -o output.out
+ exit 0
