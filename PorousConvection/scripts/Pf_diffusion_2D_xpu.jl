# approach 1: using @parallel
const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using CUDA, Printf, Test, JLD

# (NEW) add support for plots
using Plots,Plots.Measures,Printf

collect(devices())   # see avaliable GPUs
device!(0)           # assign to one GPU


# compute flux update
@parallel function compute_flux!(qDx,qDy,Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ)

    # without manual bound checking
    @inn_x(qDx) = (@inn_x(qDx) - @inn_x(qDx) + k_ηf_dx * @d_xa(Pf)) * _1_θ_dτ
    @inn_y(qDy) = (@inn_y(qDy) - @inn_y(qDy) + k_ηf_dy * @d_ya(Pf)) * _1_θ_dτ

    return nothing
end

# compute pressure update
@parallel function update_Pf!(Pf,qDx,qDy,_dx,_dy,_β_dτ)
    
    # not using the self-implemented macros
    @all(Pf) = @all(Pf) - (@d_xa(qDx)*_dx + @d_ya(qDy)*_dy)*_β_dτ

    return nothing
end


# computation function that gets called
function compute!(Pf,qDx,qDy,k_ηf_dx,k_ηf_dy,_1_θ_dτ,_dx,_dy,_β_dτ)
    
    # no cuda launch needed here using @cuda
    @parallel compute_flux!(qDx,qDy,Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ)
    @parallel update_Pf!(Pf,qDx,qDy,_dx,_dy,_β_dτ)
    return nothing
end

function Pf_diffusion_2D_xpu(nx_, ny_ ;do_check=true, test=true)
    # physics
    lx,ly   = 20.0,20.0
    k_ηf    = 1.0
  
    # numerics
    # (NEW) no launch parameters needed
    nx = nx_ 
    ny = ny_
    maxiter = 500

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
    
    # array initialisation - not using CuArray
    Pf      = Data.Array(@. exp(-(xc-lx/2)^2 -(yc'-ly/2)^2))
    qDx,qDy = @zeros(nx+1,ny), @zeros(nx,ny+1)
    r_Pf    = @zeros(nx,ny)
    

    # new visu


    # iteration loop
    iter = 1; err_Pf = 2ϵtol
    t_tic = 0.0; niter = 0
    while err_Pf >= ϵtol && iter <= maxiter
        if (iter==11) t_tic = Base.time(); niter = 0 end
        
        compute!(Pf,qDx,qDy,k_ηf_dx,k_ηf_dy,_1_θ_dτ,_dx,_dy,_β_dτ)
      
        if do_check && (iter%ncheck == 0)
            r_Pf  .= diff(qDx, dims=1)./dx .+ diff(qDy, dims=2)./dy
            err_Pf = maximum(abs.(r_Pf))
            
            # new visualization
            # png((), @sprintf(""))
            # TODO:
        end
        iter += 1; niter += 1
    end
    
    t_toc = Base.time() - t_tic
    A_eff = (3 * 2) / 1e9 * nx * ny * sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc / niter                              # Execution time per iteration [s]
    T_eff = A_eff / t_it                               # Effective memory throughput [GB/s]
    
    @printf("Time = %1.3f sec, T_eff = %1.3f GB/s \n", t_toc, T_eff)

    
    if test == true
        save("../test/Pf_127.jld", "data", Pf)
    end
    # no memory free needed
    
   return Pf
end

if isinteractive()
    Pf_diffusion_2D_xpu(511, 511; do_check=true, test=false)
end

