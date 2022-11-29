Lecture 8: Distributed computing

- Code exercise 8.1

                   -  `lecture8/scripts/diffusion_1D_2procs.jl`
                   -  `lecture8/scripts/diffusion_1D_nprocs.jl`
                   -  `lecture8/scripts/diffusion_2D_mpi.jl`
                   -  `lecture8/scripts/diffusion_2D_mpi_gpu.jl`


- Code exercise 8.2

                   -  `lecture8/scripts/diffusion_2D_perf_multixpu.jl`



## Code Exercise 8.1: Towards distributed memory computing on GPUs

### Task 1:  Finalise the scripts in class

TODO: two figures with short description
of final distribution of the concentration C

### Task 2:  Finalise the 2D MPI script in class

- `update_halo` allows for correct internal boundary exchange among the distributed parallel MPI processes

- The .gif animation shows the diffusion of the quantity C, running on 4 MPI processes 

The command used to launch the script was FIXME:


### Task 2:  Finalise the 2D MPI script in class


- The following .gif animation shows the diffusion of the quantity C, running on 4 GPUs (MPI processes)

FIXME: add the animation

- The changes are annotated as comments in the script `l8_diffusion_2D_mpi_gpu.jl` as required in the task description

> Note what changes were needed to go from CPU to GPU in this distributed solver.



## Code Exercise 8.2: Multi-xPU computing

