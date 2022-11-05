# approach 2: using @parallel_indices
const USE_GPU = true
using ParallelStencil

# NOTE: Not using the finite difference module "ParallelStencil.FiniteDifferences2D" here!
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using CUDA, Printf, Test, JLD

# (NEW) add support for plots
using Plots,Plots.Measures,Printf

# (OLD) self-implemented macros
macro d_xa(A)  esc(:( $A[ix+1,iy]-$A[ix,iy] )) end
macro d_ya(A)  esc(:( $A[ix,iy+1]-$A[ix,iy] )) end

# compute flux update
@parallel_indices (ix,iy) function compute_flux!(qDx,qDy,Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ)
    
    # still need to obtain the size but no cuda-like ix,it definitions needed
    nx,ny=size(Pf)

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
@parallel_indices (ix,iy) function update_Pf!(Pf,qDx,qDy,_dx,_dy,_β_dτ)
    nx,ny=size(Pf)

    if (ix <= nx && iy <= ny)
        Pf[ix,iy]  -= (@d_xa(qDx)*_dx + @d_ya(qDy)*_dy)*_β_dτ
    end

    return nothing
end


# computation function that gets called
function compute!(Pf,qDx,qDy,k_ηf_dx,k_ηf_dy,_1_θ_dτ,_dx,_dy,_β_dτ)

    # no cuda-liked launch needed
    @parallel compute_flux!(qDx,qDy,Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ)
    @parallel update_Pf!(Pf,qDx,qDy,_dx,_dy,_β_dτ)
    return nothing
end

function Pf_diffusion_2D_gpu(nx_, ny_;do_check=true, do_visu=false, test=false)
    # physics
    lx,ly   = 20.0,20.0
    k_ηf    = 1.0
  
    # numerics
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
    
    # array initialisation
    Pf      = Data.Array(@. exp(-(xc-lx/2)^2 -(yc'-ly/2)^2))
    qDx,qDy = @zeros(nx+1,ny), @zeros(nx,ny+1)
    r_Pf    = @zeros(nx,ny)
    
    # visu
    if do_visu
        ENV["GKSwstype"]="nul"
        if isdir("viz_out")==false mkdir("viz_out") end
        loadpath = "viz_out/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        iframe = 0
    end

    # iteration loop
    iter = 1; err_Pf = 2ϵtol
    t_tic = 0.0; niter = 0
    while err_Pf >= ϵtol && iter <= maxiter
        if (iter==11) t_tic = Base.time(); niter = 0 end
        
        compute!(Pf,qDx,qDy,k_ηf_dx,k_ηf_dy,_1_θ_dτ,_dx,_dy,_β_dτ)
      
        if do_check && (iter%ncheck == 0)
            r_Pf  .= diff(qDx, dims=1)./dx .+ diff(qDy, dims=2)./dy
            err_Pf = maximum(abs.(r_Pf))

            # visu
            if do_visu
                @printf("  iter/nx=%.1f, err_Pf=%1.3e\n",iter/nx,err_Pf)
                png((heatmap(xc,yc,Array(Pf)';xlims=(xc[1],xc[end]),ylims=(yc[1],yc[end]),aspect_ratio=1,c=:turbo)),@sprintf("viz_out/xpu_perf_%04d.png",iframe+=1))
            end
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


   return Pf
end

if isinteractive()
    Pf_diffusion_2D_gpu(511, 511; do_check=true, do_visu=false, test=false)
end

