# approach 1: using @parallel
# no collect(devices()) needed
const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using CUDA, Printf, Test, JLD

# (NEW) add support for plots
using Plots,Plots.Measures,Printf


# compute flux update
@parallel function compute_flux!(qDx,qDy,qDz,Pf,k_ηf_dx,k_ηf_dy,k_ηf_dz,_1_θ_dτ)

    # without manual bound checking
    @inn_x(qDx) = @inn_x(qDx) - (@inn_x(qDx) + k_ηf_dx * @d_xa(Pf)) * _1_θ_dτ
    @inn_y(qDy) = @inn_y(qDy) - (@inn_y(qDy) + k_ηf_dy * @d_ya(Pf)) * _1_θ_dτ
    @inn_z(qDz) = @inn_z(qDz) - (@inn_z(qDz) + k_ηf_dz * @d_za(Pf)) * _1_θ_dτ

    return nothing
end

# compute pressure update
@parallel function update_Pf!(Pf,qDx,qDy,qDz,_dx,_dy,_dz,_β_dτ)
    
    # not using the self-implemented macros
    @all(Pf) = @all(Pf) - (@d_xa(qDx)*_dx + @d_ya(qDy)*_dy + @d_za(qDz)*_dz)*_β_dτ

    return nothing
end



# computation function that gets called
function compute!(Pf,qDx,qDy,qDz,k_ηf_dx,k_ηf_dy,k_ηf_dz,_1_θ_dτ,_dx,_dy,_dz,_β_dτ)
    
    # no cuda launch needed here using @cuda
    @parallel compute_flux!(qDx,qDy,qDz,Pf,k_ηf_dx,k_ηf_dy,k_ηf_dz,_1_θ_dτ)
    @parallel update_Pf!(Pf,qDx,qDy,qDz,_dx,_dy,_dz,_β_dτ)
    return nothing
end


@parallel function calc_r_Pf!(r_Pf, qDx, qDy, qDz, _dx, _dy, _dz)
    @all(r_Pf)  = @d_xa(qDx) * _dx + @d_ya(qDy) * _dy + @d_za(qDz) * _dz
    return nothing
end


function Pf_diffusion_3D_xpu(nx_, ny_, nz_ ;do_check=true, do_visu=false, test=false)
    # physics
    lx,ly,lz   = 20.0,20.0, 20.0
    k_ηf    = 1.0
  
    # numerics
    # (NEW) no launch parameters needed
    nx = nx_ 
    ny = ny_
    nz = nz_
    maxiter = 500

    ϵtol    = 1e-8
    ncheck  = ceil(Int,0.25max(nx,ny))
    cfl     = 1.0/sqrt(3.1)    # was 2.1 for 2D
    re      = 2π
  
    # derived numerics
    dx,dy,dz   = lx/nx,ly/ny, lz/nz
    xc,yc,zc   = LinRange(dx/2,lx-dx/2,nx),LinRange(dy/2,ly-dy/2,ny),LinRange(dz/2,lz-dz/2,nz)
    θ_dτ       = max(lx,ly,lz)/re/cfl/min(dx,dy,dz)
    β_dτ       = (re*k_ηf)/(cfl*min(dx,dy,dz)*max(lx,ly,lz))
    _1_θ_dτ    = 1.0/(1.0 + θ_dτ)
    _β_dτ      = 1.0/(β_dτ)
    _dx,_dy,_dz = 1.0/dx,1.0/dy,1.0/dz
    k_ηf_dx,k_ηf_dy,k_ηf_dz = k_ηf/dx,k_ηf/dy,k_ηf/dz
    
    # array initialisation - not using CuArray
    # Pf          = Data.Array(@. exp(-(xc-lx/2)^2 -(yc'-ly/2)^2)) 2D
    Pf = Data.Array([exp(-(xc[ix]-lx/2)^2 -(yc[iy]-ly/2)^2 -(zc[iz]-lz/2)^2) for ix=1:nx,iy=1:ny,iz=1:nz])

    qDx,qDy,qDz = @zeros(nx+1,ny, nz), @zeros(nx,ny+1, nz), @zeros(nx, ny, nz+1)
    r_Pf        = @zeros(nx,ny,nz)
    

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
        
        compute!(Pf,qDx,qDy,qDz,k_ηf_dx,k_ηf_dy,k_ηf_dz,_1_θ_dτ,_dx,_dy,_dz,_β_dτ)
      
        if do_check && (iter%ncheck == 0)
            r_Pf  .= diff(qDx, dims=1) .* _dx .+ diff(qDy, dims=2).* _dy .+ diff(qDz, dims=3).* _dz
            err_Pf = maximum(abs.(r_Pf))
            
            # visu
            if do_visu
                # FIXME: add it for 3D!
                # @printf("  iter/nx=%.1f, err_Pf=%1.3e\n",iter/nx,err_Pf)
                # png((heatmap(xc,yc,Array(Pf)';xlims=(xc[1],xc[end]),ylims=(yc[1],yc[end]),aspect_ratio=1,c=:turbo)),@sprintf("viz_out/xpu_%04d.png",iframe+=1))
            end
        end
        iter += 1; niter += 1
    end
    
    t_toc = Base.time() - t_tic
    A_eff = (4 * 2) / 1e9 * nx * ny * nz * sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc / niter                              # Execution time per iteration [s]
    T_eff = A_eff / t_it                               # Effective memory throughput [GB/s]
    
    @printf("Time = %1.3f sec, T_eff = %1.3f GB/s \n", t_toc, T_eff)

    
    if test == true
        save("../test/Pf_511_3D.jld", "data", Pf)
    end
    # no memory free needed
    
   return Pf
end

if isinteractive()
    Pf_diffusion_3D_xpu(511, 511, 511; do_check=true, do_visu=false, test=false)
end

