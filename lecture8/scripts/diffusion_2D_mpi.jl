# # 2D linear diffusion Julia MPI solver
# run: ~/.julia/bin/mpiexecjl -n 4 julia --project diffusion_2D_mpi.jl
using Plots, Printf, MAT
import MPI

# enable plotting by default
if !@isdefined do_save; do_save = false end
if !@isdefined do_gif;   do_gif = true end

# MPI functions
@views function update_halo(A, neighbors_x, neighbors_y, comm)

    lx = size(A)[1]
    ly = size(A)[2]

    # Send to / receive from neighbor 1 in dimension x ("left neighbor")
    if neighbors_x[1] != MPI.PROC_NULL
        sendbuf = A[:,2]
        recvbuf = zeros(ly)
        
        # send local [:,2] to neighbour [:,end]
        MPI.Send(sendbuf, neighbors_x[1], 0, comm)
    
        # receive from neighbour [:,end-1] to local [:,1]
        MPI.Recv!(recvbuf, neighbors_x[1], 1, comm)
        A[:, 1] .= recvbuf
        
    end
    
    # Send to / receive from neighbor 2 in dimension x ("right neighbor")
    if neighbors_x[2] != MPI.PROC_NULL
        sendbuf = A[:,end-1]
        recvbuf = zeros(ly)
        
        # send local [:,end-1] to neighbour [:,1]
        MPI.Send(sendbuf, neighbors_x[2], 1, comm)
    
        # receive from neighbour [:,2] to local [:,end]
        MPI.Recv!(recvbuf, neighbors_x[2], 0, comm)
        A[:,end] .= recvbuf
        
    end

    # Send to / receive from neighbor 1 in dimension y ("top neighbor")
    if neighbors_y[1] != MPI.PROC_NULL
        sendbuf = A[2,:]        
        recvbuf = zeros(lx)
        
        # send local [2,:] to neighbour [end,:]
        MPI.Send(sendbuf, neighbors_y[1], 2, comm)
        
        # receive from neighbour [end-1,:] to local [1,:]
        MPI.Recv!(recvbuf, neighbors_y[1],3, comm)
        A[1,:]  .= recvbuf
    end
    
    # Send to / receive from neighbor 2 in dimension y ("bottom neighbor")
    if neighbors_y[2] != MPI.PROC_NULL
        sendbuf = A[end-1,:]
        recvbuf = zeros(lx)
        
        # send local [end-1,:] to neighbour [1,:]
        MPI.Send(sendbuf, neighbors_y[2], 3, comm)
    
        # receive from neighbour [2,:] to local [end,:]
        MPI.Recv!(recvbuf, neighbors_y[2], 2, comm)
        A[end,:] .= recvbuf
    end
    return
end


@views function diffusion_2D_mpi(; do_save=false, do_gif=false)
    # MPI
    MPI.Init()
    dims        = [0,0]
    comm        = MPI.COMM_WORLD
    nprocs      = MPI.Comm_size(comm)
    MPI.Dims_create!(nprocs, dims)
    comm_cart   = MPI.Cart_create(comm, dims, [0,0], 1)
    me          = MPI.Comm_rank(comm_cart)
    coords      = MPI.Cart_coords(comm_cart)
    neighbors_x = MPI.Cart_shift(comm_cart, 1, 1)   # left and right neighbours
    neighbors_y = MPI.Cart_shift(comm_cart, 0, 1)   # upper and bottom neighbours

    if (me==0) println("nprocs=$(nprocs), dims[1]=$(dims[1]), dims[2]=$(dims[2])") end

    # for storing data for gif 
    outdir = string(@__DIR__, "/out2D"); if (me==0) if isdir("$outdir")==false mkdir("$outdir") end end


    # Physics
    lx, ly     = 10.0, 10.0
    D          = 1.0
    nt         = 100
    
    # Numerics
    nx, ny     = 32, 32                             # local number of grid points
    nx_g, ny_g = dims[1]*(nx-2)+2, dims[2]*(ny-2)+2 # global number of grid points
    
    # Derived numerics
    dx, dy     = lx/nx_g, ly/ny_g                   # global
    dt         = min(dx,dy)^2/D/4.1
    
    # Array allocation
    qx         = zeros(nx-1,ny-2)
    qy         = zeros(nx-2,ny-1)
    
    # Initial condition
    x0, y0     = coords[1]*(nx-2)*dx, coords[2]*(ny-2)*dy
    xc         = [x0 + ix*dx - dx/2 - 0.5*lx  for ix=1:nx]
    yc         = [y0 + iy*dy - dy/2 - 0.5*ly  for iy=1:ny]
    C          = exp.(.-xc.^2 .-yc'.^2)
    t_tic = 0.0
    
    # Time loop
    for it = 1:nt
        if (it==11) t_tic = Base.time() end
        qx  .= .-D*diff(C[:,2:end-1], dims=1)/dx
        qy  .= .-D*diff(C[2:end-1,:], dims=2)/dy
        C[2:end-1,2:end-1] .= C[2:end-1,2:end-1] .- dt*(diff(qx, dims=1)/dx .+ diff(qy, dims=2)/dy)
        update_halo(C, neighbors_x, neighbors_y, comm_cart)
        
        if do_gif file = matopen("$(outdir)/C_$(me)_$(it).mat", "w"); write(file, "C", Array(C)); close(file) end
    end
    t_toc = (Base.time()-t_tic)
    if (me==0) @printf("Time = %1.4e s, T_eff = %1.2f GB/s \n", t_toc, round((2/1e9*nx*ny*sizeof(lx))/(t_toc/(nt-10)), sigdigits=2)) end
   
    # Save to visualise
    if do_save file = matopen("$(@__DIR__)/mpi2D_out_C_$(me).mat", "w"); write(file, "C", Array(C)); close(file) end
    MPI.Finalize()
    return
end

diffusion_2D_mpi(; do_save=do_save, do_gif=do_gif)
