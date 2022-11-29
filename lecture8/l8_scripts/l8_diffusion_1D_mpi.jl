# 1D linear diffusion Julia MPI solver
# run: ~/.julia/bin/mpiexecjl -n 4 julia --project diffusion_1D_mpi.jl
using Plots, Printf, MAT
import MPI

# enable plotting by default
if !@isdefined do_save; do_save = true end

# MPI functions
@views function update_halo(A, neighbors_x, comm)

    
    # Send to / receive from neighbor 1 ("left neighbor")
    if neighbors_x[1] != MPI.MPI_PROC_NULL

        sendbuf = A[2]
        recvbuf = zeros(1)   # used later for copy back

        # send local [2] to neighbor [end] 
        MPI.Send!(sendbuf, neighbors_x[1], 0, comm)

        # receive local [1] from neighbor [end-1] 
        MPI.Recv!(recvbuf, neighbors_x[1], 1, comm)
        A[1] = recvbuf

    end
    # Send to / receive from neighbor 2 ("right neighbor")
    if neighbors_x[2] != MPI.MPI_PROC_NULL
        
        sendbuf = A[end-1]
        recvbuf = zeros(1)
        
        # send local [end-1] to neighbor[1]
        MPI.Send!(sendbuf, neighbors_x[2], 1, comm)

        # receive local [end] from neighbor [2]
        MPI.Recv!(recvbuf, neighbors_x[2], 0, comm)
        A[end] = recvbuf
    end
    return
end

@views function diffusion_1D_mpi(; do_save=false)
    # MPI
    MPI.Init()
    dims        = [0]               # cartesian dim of mpi processes
    comm        = MPI.COMM_WORLD
    nprocs      = MPI.Comm_size(comm)
    MPI.Dims_create!(nprocs, dims)
    comm_cart   = MPI.Cart_create(comm, dims, [0], 1)
    me          = MPI.Comm_rank(comm_cart)
    coords      = MPI.Cart_coords(comm_cart)
    neighbors_x = MPI.Cart_shift(comm_cart, 0, 1)

    if (me==0) println("nprocs=$(nprocs), dims[1]=$(dims[1])") end
   
    # Physics
    lx    = 10.0
    D     = 1.0
    nt    = 100
   
    # Numerics
    nx    = 32                 # local number of grid points
    nx_g  = dims[1]*(nx-2) + 2 # global number of grid points
   
    # Derived numerics
    dx    = lx/nx_g            # global
    dt    = dx^2/D/2.1
   
    # Array allocation
    qx    = zeros(nx-1)
   
    # Initial condition
    x0    = coords[1]*(nx-2)*dx
    xc    = [x0 + ix*dx - dx/2 - 0.5*lx  for ix=1:nx]
    C     = exp.(.-xc.^2)
    t_tic = 0.0
   
    # Time loop
    for it = 1:nt
        if (it==11) t_tic = Base.time() end
        qx         .= .-D*diff(C)/dx
        C[2:end-1] .= C[2:end-1] .- dt*diff(qx)/dx
        update_halo(C, neighbors_x, comm_cart)
    end

    t_toc = Base.time()-t_tic

    if (me==0) @printf("Time = %1.4e s, T_eff = %1.2f GB/s \n", t_toc, round((2/1e9*nx*sizeof(lx))/(t_toc/(nt-10)), sigdigits=2)) end
    if do_save file = matopen("$(@__DIR__)/mpi1D_out_C_$(me).mat", "w"); write(file, "C", Array(C)); close(file) end

    MPI.Finalize()
    return
end

diffusion_1D_mpi(; do_save=true)
