# 2D linear diffusion Julia MPI solver
# with CUDA programming
# run: ~/.julia/bin/mpiexecjl -n 4 julia --project diffusion_2D_mpi.jl
using Plots, Printf, MAT
using CUDA    # GPU programming
import MPI

# enable plotting by default
if !@isdefined do_save; do_save = false end
if !@isdefined do_gif;  do_gif = false end

# # GPU programming
# # CUDA Kernel functions
# macro d_xa(A)  esc(:( $A[ix+1,iy]-$A[ix,iy] )) end
# macro d_ya(A)  esc(:( $A[ix,iy+1]-$A[ix,iy] )) end


# # compute flux update
# function compute_flux!(qx, qy, C, D, dx, dy)
#     nx, ny = size(C)
#     ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

#     # manual bound checking
#     if (ix <= nx-1 && 2<= iy <= ny-1)  qx[ix, iy-1] = -D * @d_xa(C) / dx end
#     if (2 <= ix <= nx-1 && iy <= ny-1) qy[ix-1, iy] = -D * @d_ya(C) / dy end

#     return nothing
# end


# # compute temperature update
# function update_C!(C, qx, qy, dt, dx, dy)
#     nx, ny = size(C)
#     ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

#     if (2 <= ix <= nx-1 && 2 <= iy <= ny-1)
#         C[ix,iy]  -= dt * (@d_xa(qx) /dx + @d_ya(qy) /dy)
#     end        

#     # qx  .= .-D*diff(C[:,2:end-1], dims=1)/dx
#     # qy  .= .-D*diff(C[2:end-1,:], dims=2)/dy
#     # C[2:end-1,2:end-1] .= C[2:end-1,2:end-1]

#     return nothing
# end


# MPI functions
# Modify the update_halo function; use copyto! to copy device data 
# to the host into the send buffer or to copy host data to the device 
# from the receive buffer
@views function update_halo(A, neighbors_x, neighbors_y, comm)

    lx = size(A)[1]
    ly = size(A)[2]

    # Send to / receive from neighbor 1 in dimension x ("left neighbor")
    if neighbors_x[1] != MPI.PROC_NULL
        sendbuf = CUDA.zeros(ly)
        copyto!(sendbuf, A[:,2])

        recvbuf = CUDA.zeros(ly)

        # send local [:,2] to neighbour [:,end]
        MPI.Send(sendbuf, neighbors_x[1], 0, comm)
    
        # receive from neighbour [:,end-1] to local [:,1]
        MPI.Recv!(recvbuf, neighbors_x[1], 1, comm)
        copyto!(A[:,1],recvbuf)
    end
    
    # Send to / receive from neighbor 2 in dimension x ("right neighbor")
    if neighbors_x[2] != MPI.PROC_NULL
        sendbuf = CUDA.zeros(ly)
        copyto!(A[:,end-1], sendbuf)
        
        recvbuf = CUDA.zeros(ly)
        
        # send local [:,end-1] to neighbour [:,1]
        MPI.Send(sendbuf, neighbors_x[2], 1, comm)
    
        # receive from neighbour [:,2] to local [:,end]
        MPI.Recv!(recvbuf, neighbors_x[2], 0, comm)
        copyto!(A[:,end], recvbuf)
        
    end

    # Send to / receive from neighbor 1 in dimension y ("top neighbor")
    if neighbors_y[1] != MPI.PROC_NULL
        sendbuf = CUDA.zeros(lx)
        copyto!(sendbuf, A[2,:])
        
        recvbuf = CUDA.zeros(lx)
        
        # send local [2,:] to neighbour [end,:]
        MPI.Send(sendbuf, neighbors_y[1], 2, comm)
        
        # receive from neighbour [end-1,:] to local [1,:]
        MPI.Recv!(recvbuf, neighbors_y[1],3, comm)
        copyto!(A[1,:], recvbuf)
    end
    
    # Send to / receive from neighbor 2 in dimension y ("bottom neighbor")
    if neighbors_y[2] != MPI.PROC_NULL
        sendbuf = CUDA.zeros(lx)
        copyto!(sendbuf, A[end-1,:])
        
        recvbuf = CUDA.zeros(lx)
        
        # send local [end-1,:] to neighbour [1,:]
        MPI.Send(sendbuf, neighbors_y[2], 3, comm)
    
        # receive from neighbour [2,:] to local [end,:]
        MPI.Recv!(recvbuf, neighbors_y[2], 2, comm)
        copyto!(A[end,:], recvbuf)
    end
    return
end


@views function diffusion_2D_mpi_gpu(; do_save=false, do_gif=false)
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

    # GPU
    # Select the GPU based on node-local MPI infos 
    comm_l = MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED, me)
    me_l   = MPI.Comm_rank(comm_l) # per node numbering to obtain the GPU ID
    GPU_ID = CUDA.device!(me_l)

    sleep(0.1me)
    println("Hello world, I am $(me) of $(MPI.Comm_size(comm)) using $(GPU_ID)")


    if (me==0) println("nprocs=$(nprocs), dims[1]=$(dims[1]), dims[2]=$(dims[2])") end

    # for storing data for gif 
    # outdir = string(@__DIR__, "/out2D"); if (me==0) if isdir("outdir")==false mkdir("outdir") end end


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
    

    # GPU launch data
    # threads = (32,16)
    # blocks  = (Int(ceil(nx/threads[1])), Int(ceil(ny/threads[2])))


    # Array allocation
    # Use GPU array initialisation (CUDA.zeros, CuArray(), ...)
    qx         = CUDA.zeros(nx-1,ny-2)
    qy         = CUDA.zeros(nx-2,ny-1)
    
    # Initial condition
    x0, y0     = coords[1]*(nx-2)*dx, coords[2]*(ny-2)*dy
    xc         = [x0 + ix*dx - dx/2 - 0.5*lx  for ix=1:nx]
    yc         = [y0 + iy*dy - dy/2 - 0.5*ly  for iy=1:ny]
    C_cpu      = exp.(.-xc.^2 .-yc'.^2)
    C          = CuArray(C_cpu)
    t_tic = 0.0
    
    # Time loop
    for it = 1:nt
        if (it==11) t_tic = Base.time() end

        # using GPU array programming
        qx  .= .-D*diff(C[:,2:end-1], dims=1)/dx
        qy  .= .-D*diff(C[2:end-1,:], dims=2)/dy
        C[2:end-1,2:end-1] .= C[2:end-1,2:end-1] .- dt*(diff(qx, dims=1)/dx .+ diff(qy, dims=2)/dy)

        # GPU programming
        # @cuda blocks=blocks threads=threads compute_flux!(qx, qy, C, D, dx, dy); synchronize()
        # @cuda blocks=blocks threads=threads update_C!(C, qx, qy, dt, dx, dy); synchronize()

        # MPI exchange
        update_halo(C, neighbors_x, neighbors_y, comm_cart)
        
        # Gather the GPU arrays back on the host memory for visualisation or saving (using Array())
        if do_gif file = matopen("(outdir)/C_$(me)_$(it).mat", "w"); write(file, "C", Array(C)); close(file) end
    end
    t_toc = (Base.time()-t_tic)
    if (me==0) @printf("Time = %1.4e s, T_eff = %1.2f GB/s \n", t_toc, round((2/1e9*nx*ny*sizeof(lx))/(t_toc/(nt-10)), sigdigits=2)) end
   
    # Save to visualise
    # Gather the GPU array back
    if do_save file = matopen("(@__DIR__)/mpi2D_out_C_$(me).mat", "w"); write(file, "C", Array(C)); close(file) end
    MPI.Finalize()
    return
end

diffusion_2D_mpi_gpu(; do_save=do_save, do_gif=do_gif)
