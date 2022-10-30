using BenchmarkTools, Printf, JLD
using CUDA
collect(devices())   # see avaliable GPUs
device!(0)           # assign to one GPU

macro d_xa(A)  esc(:( $A[ix+1,iy]-$A[ix,iy] )) end
macro d_ya(A)  esc(:( $A[ix,iy+1]-$A[ix,iy] )) end

# compute flux update
function compute_flux!(qDx,qDy,Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ)
    nx,ny=size(Pf)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    # manual bound checking
    if(ix <= nx-1 && iy <= ny)
        qDx[ix+1,iy] -= (qDx[ix+1,iy] + k_ηf_dx * @d_xa(Pf))*_1_θ_dτ
    end

    if(ix <= nx && iy <= ny-1)
        qDy[ix,iy+1] -= (qDy[ix,iy+1] + k_ηf_dy * @d_ya(Pf))*_1_θ_dτ
    end

    return nothing
end

# compute pressure update
function update_Pf!(Pf,qDx,qDy,_dx,_dy,_β_dτ)
    nx,ny=size(Pf)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (ix <= nx && iy <= ny)
        Pf[ix,iy]  -= (@d_xa(qDx)*_dx + @d_ya(qDy)*_dy)*_β_dτ
    end

    return nothing
end


# computation function that gets called
function compute!(Pf,qDx,qDy,k_ηf_dx,k_ηf_dy,_1_θ_dτ,_dx,_dy,_β_dτ)
    @cuda compute_flux!(qDx,qDy,Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ); synchronize()
    @cuda update_Pf!(Pf,qDx,qDy,_dx,_dy,_β_dτ); synchronize()
    return nothing
end

function Pf_diffusion_2D_gpu(nx_, ny_ ;do_check=false)
    # physics
    lx,ly   = 20.0,20.0
    k_ηf    = 1.0
  
    # numerics
    threads = (32,16)
    
    # perform weak scaling test
    nx, ny  = nx_, ny_
    blocks  = (Int(ceil(nx/threads[1])), Int(ceil(ny/threads[2])))
    print(blocks)
    maxiter = max(nx, ny)

    ϵtol    = 1e-8
    ncheck  = ceil(Int,0.25max(nx,ny))
    cfl     = 1.0/sqrt(2.1)
    re      = 2π
  
    # derived numerics
    dx,dy   = lx/nx,ly/ny
    xc,yc   = LinRange(dx/2,lx-dx/2,nx),LinRange(dy/2,ly-dy/2,ny)
    θ_dτ    = max(lx,ly)/re/cfl/min(dx,dy)
    β_dτ    = (re*k_ηf)/(cfl*min(dx,dy)*max(lx,ly))
    _1_θ_dτ = 1.0/(1.0 + θ_dτ)
    _β_dτ   = 1.0/(β_dτ)
    _dx,_dy = 1.0/dx,1.0/dy
    k_ηf_dx,k_ηf_dy = k_ηf/dx,k_ηf/dy
    
    # array initialisation
    Pf      = CuArray(@. exp(-(xc-lx/2)^2 -(yc'-ly/2)^2))
    qDx,qDy = CuArray(zeros(Float64, nx+1,ny)), CuArray(zeros(Float64, nx,ny+1))
    r_Pf    = CuArray(zeros(Float64, nx,ny))
    
    # iteration loop
    iter = 1; err_Pf = 2ϵtol
    t_tic = 0.0; niter = 0
    while err_Pf >= ϵtol && iter <= maxiter
        if (iter==11) t_tic = Base.time(); niter = 0 end
        
        compute!(Pf,qDx,qDy,k_ηf_dx,k_ηf_dy,_1_θ_dτ,_dx,_dy,_β_dτ)
              
        if do_check && (iter%ncheck == 0)
            r_Pf  .= diff(qDx, dims=1)./dx .+ diff(qDy, dims=2)./dy
            err_Pf = maximum(abs.(r_Pf))
        end
        iter += 1; niter += 1
    end
        
    t_toc = Base.time() - t_tic
    A_eff = (3 * 2) / 1e9 * nx * ny * sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc / niter                              # Execution time per iteration [s]
    T_eff = A_eff / t_it                               # Effective memory throughput [GB/s]

    # free the memory
    CUDA.unsafe_free!(Pf)
    CUDA.unsafe_free!(qDx)
    CUDA.unsafe_free!(qDy)
    CUDA.unsafe_free!(r_Pf)
    
   return T_eff
end





# function for plotting the memory throughput
function mem_throughput(; start_at_two = false)

    # optional parameters to have test size starting at nx == 2
    if start_at_two
        nx = ny = 2 * 2 .^ (0:11)
    else
        nx = ny = 32 .* 2 .^ (0:8) .- 1
    end

    T_eff_list = []
    temp = []
 
    for i in ny
            temp = Pf_diffusion_2D_gpu(i, i;do_check=false)
            append!(T_eff_list, temp)
    end

    # store the data for later plotting    
    save("nx.jld", "data", nx)
    save("T_eff_list.jld", "data", T_eff_list)

end

mem_throughput()

