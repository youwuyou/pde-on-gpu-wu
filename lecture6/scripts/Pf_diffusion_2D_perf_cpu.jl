# using Plots,Plots.Measures,Printf
# default(size=(600,500),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=11,tickfontsize=11,titlefontsize=11)

macro d_xa(A)  esc(:( $A[ix+1,iy]-$A[ix,iy] )) end
macro d_ya(A)  esc(:( $A[ix,iy+1]-$A[ix,iy] )) end

function compute_flux!(qDx,qDy,Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ)
    nx,ny=size(Pf)
    Threads.@threads for iy=1:ny
    # for iy=1:ny
        for ix=1:nx-1
            qDx[ix+1,iy] -= (qDx[ix+1,iy] + k_ηf_dx*@d_xa(Pf))*_1_θ_dτ
        end
    end
    Threads.@threads for iy=1:ny-1
    # for iy=1:ny-1
        for ix=1:nx
            qDy[ix,iy+1] -= (qDy[ix,iy+1] + k_ηf_dy*@d_ya(Pf))*_1_θ_dτ
        end
    end
    return nothing
end

function update_Pf!(Pf,qDx,qDy,_dx,_dy,_β_dτ)
    nx,ny=size(Pf)
    Threads.@threads for iy=1:ny
    # for iy=1:ny
        for ix=1:nx
            Pf[ix,iy]  -= (@d_xa(qDx)*_dx + @d_ya(qDy)*_dy)*_β_dτ
        end
    end
    return nothing
end


function compute!(Pf,qDx,qDy,k_ηf_dx,k_ηf_dy,_1_θ_dτ,_dx,_dy,_β_dτ)
    compute_flux!(qDx,qDy,Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ)
    update_Pf!(Pf,qDx,qDy,_dx,_dy,_β_dτ)
    return nothing
end

function Pf_diffusion_2D_cpu(;do_check=true, test=true)
    # physics
    lx,ly   = 20.0,20.0
    k_ηf    = 1.0
  
    # numerics
    nx,ny   = 127, 127
    ϵtol    = 1e-8
    maxiter = 50
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
    Pf      = @. exp(-(xc-lx/2)^2 -(yc'-ly/2)^2)
    qDx,qDy = zeros(Float64, nx+1,ny),zeros(Float64, nx,ny+1)
    r_Pf    = zeros(nx,ny)
    
    # iteration loop
    iter = 1; err_Pf = 2ϵtol
    t_tic = 0.0; niter = 0
    while err_Pf >= ϵtol && iter <= maxiter
        if (iter==11) t_tic = Base.time(); niter = 0 end
      
        compute!(Pf,qDx,qDy,k_ηf_dx,k_ηf_dy,_1_θ_dτ,_dx,_dy,_β_dτ)
      
        if do_check && (iter%ncheck == 0)
            r_Pf  .= diff(qDx,dims=1)./dx .+ diff(qDy,dims=2)./dy
            err_Pf = maximum(abs.(r_Pf))
  #          @printf("  iter/nx=%.1f, err_Pf=%1.3e\n",iter/nx,err_Pf)
   #         display(heatmap(xc,yc,Pf';xlims=(xc[1],xc[end]),ylims=(yc[1],yc[end]),aspect_ratio=1,c=:turbo))
        end
        iter += 1; niter += 1
    end
    
    t_toc = Base.time() - t_tic
    A_eff = (3 * 2) / 1e9 * nx * ny * sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc / niter                              # Execution time per iteration [s]
    T_eff = A_eff / t_it                               # Effective memory throughput [GB/s]
    
 #   @printf("Time = %1.3f sec, T_eff = %1.3f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits=3), niter)
    if test == true
        @printf("niter = %1.3d", niter) 
        save("../test/Pf_ref_127.jld", "data", Pf)
    end 
 
    return Pf
end

if isinteractive()
    Pf_diffusion_2D_cpu(do_check=true, test=true)
end