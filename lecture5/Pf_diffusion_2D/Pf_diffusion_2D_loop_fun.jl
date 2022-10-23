using Plots,Plots.Measures,Printf, BenchmarkTools
default(size=(600,500),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=11,tickfontsize=11,titlefontsize=11)

# use macro as syntactic sugar for derivatives
macro d_xa(A) esc(:($A[ix+1,iy] - $A[ix,iy]))end
macro d_ya(A) esc(:($A[ix, iy+1]- $A[ix,iy]))end


# compute flux update
function compute_flux!(Pf, qDx, qDy, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
    nx,ny = size(Pf)
    
    Threads.@threads for iy = 1:ny
        for ix = 1:nx-1
            qDx[ix+1,iy] -= (qDx[ix,iy] + k_ηf_dx * @d_xa(Pf)) * _1_θ_dτ
        end
    end
    
    Threads.@threads for iy = 1:ny-1
        for ix = 1:nx
            qDy[ix,iy+1] -= (qDy[ix,iy+1] + k_ηf_dy * @d_ya(Pf)) * _1_θ_dτ
        end
    end
    return nothing
end

# compute pressure update
function compute_Pf!(Pf, qDx, qDy, _dx_β_dτ, _dy_β_dτ)
    nx,ny = size(Pf)
    Threads.@threads for iy = ny
        for ix = nx
            Pf[ix,iy]    -= @d_xa(qDx) * _dx_β_dτ + @d_ya(qDy) * _dy_β_dτ
        end
    end
    return nothing
end

# computation function that gets called
function compute!(Pf, qDx, qDy, _dx_β_dτ, _dy_β_dτ, k_ηf_dx, k_ηf_dy, _1_θ_dτ)

    compute_flux!(Pf, qDx, qDy, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
    compute_Pf!(Pf, qDx, qDy, _dx_β_dτ, _dy_β_dτ)

    return nothing
end


function Pf_diffusion_2D()
    # physics
    lx,ly   = 20.0,20.0
    k_ηf    = 1.0
   
    # numerics
    nx,ny   = 1*511, 1*511
    cfl     = 1.0/sqrt(2.1)
    re      = 2π
   
    # derived numerics
    dx,dy   = lx/nx,ly/ny
    xc,yc   = LinRange(dx/2,lx-dx/2,nx),LinRange(dy/2,ly-dy/2,ny)
    θ_dτ    = max(lx,ly)/re/cfl/min(dx,dy)
    β_dτ    = (re*k_ηf)/(cfl*min(dx,dy)*max(lx,ly))
   
    # array initialisation
    Pf      = @. exp(-(xc-lx/2)^2 -(yc'-ly/2)^2)
    qDx,qDy = zeros(Float64, nx+1,ny),zeros(Float64, nx,ny+1)
    k_ηf_dx = k_ηf / dx
    k_ηf_dy = k_ηf / dy

    _1_θ_dτ  =  1.0 ./(1.0 + θ_dτ)
    _dx_β_dτ = 1.0 / dx / β_dτ
    _dy_β_dτ = 1.0 / dy / β_dτ
  
    # iteration loop
    niter = 1
    t_toc = @belapsed compute!($Pf, $qDx, $qDy, $_dx_β_dτ, $_dy_β_dτ, $k_ηf_dx, $k_ηf_dy, $_1_θ_dτ)

    A_eff = 3 * 2 * nx * ny * sizeof(eltype(Pf)) /  1e9   # effective memory access per second
    t_it  =  t_toc/niter                                  # execution time per iterations [s]
    T_eff = A_eff / t_it

    @printf("Time = %1.3f sec, Teff = %1.3f, niter = %d \n", t_toc, T_eff, niter)

    return
end

Pf_diffusion_2D()
