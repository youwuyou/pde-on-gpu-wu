using Plots,Plots.Measures,Printf
default(size=(600,500),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=11,tickfontsize=11,titlefontsize=11)

# use macro as syntactic sugar for derivatives
macro d_xa(A) esc(:($A[ix+1,iy] - $A[ix,iy]))end
macro d_ya(A) esc(:($A[ix, iy+1]- $A[ix,iy]))end


# compute flux update
function compute_flux!(Pf, qDx, qDy, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
    nx,ny = size(Pf)

    Threads.@threads for iy = 1:ny
        for ix = 1:nx-1
            qDx[ix+1,iy] -= (qDx[ix+1,iy] + k_ηf_dx * @d_xa(Pf))* _1_θ_dτ
        end
    end

    Threads.@threads for iy = 1:ny-1
        for ix = 1:nx
            qDy[ix,iy+1] -= (qDy[ix,iy+1] + k_ηf_dy * @d_ya(Pf))* _1_θ_dτ
        end
    end

    return nothing

end

# compute pressure update
function compute_Pf!(Pf, qDx, qDy, _dx_β_dτ, _dy_β_dτ)
    nx,ny = size(Pf)

    Threads.@threads for iy = 1:ny
        for ix = 1:nx
            Pf[ix,iy]     -= (@d_xa(qDx)* _dx_β_dτ + @d_ya(qDy) * _dy_β_dτ )
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



function Pf_diffusion_2D(nx_, ny_;do_check=false)
    # physics
    lx,ly   = 20.0,20.0
    k_ηf    = 1.0
 
    # numerics
    nx,ny   = nx_, ny_
    ϵtol    = 1e-8
    maxiter = 500
    ncheck  = ceil(Int,0.25max(nx,ny))
    cfl     = 1.0/sqrt(2.1)
    re      = 2π
 
    # derived numerics
    dx,dy   = lx/nx,ly/ny
    xc,yc   = LinRange(dx/2,lx-dx/2,nx),LinRange(dy/2,ly-dy/2,ny)
    _dx, _dy = 1. /dx, 1. /dy
    θ_dτ    = max(lx,ly)/re/cfl/min(dx,dy)
    β_dτ    = (re*k_ηf)/(cfl*min(dx,dy)*max(lx,ly))
    _dx_β_dτ, _dy_β_dτ = _dx / β_dτ, _dy / β_dτ
    k_ηf_dx, k_ηf_dy = k_ηf * _dx, k_ηf * _dy


    _1_θ_dτ = 1. / (1. + θ_dτ)
 
    # array initialisation
    Pf      = @. exp(-(xc-lx/2)^2 -(yc'-ly/2)^2)
    qDx,qDy = zeros(Float64, nx+1,ny),zeros(Float64, nx,ny+1)
    r_Pf    = zeros(nx,ny)
 
    # iteration loop
    iter = 1; err_Pf = 2ϵtol

    # testing
    xtest = [5, Int(cld(0.6*lx, dx)), nx-10]
    ytest = Int(cld(0.5*ly, dy))

    while err_Pf >= ϵtol && iter <= maxiter
       
        compute!(Pf, qDx, qDy, _dx_β_dτ, _dy_β_dτ, k_ηf_dx, k_ηf_dy, _1_θ_dτ)

        if do_check && iter%ncheck == 0
            r_Pf  .= diff(qDx,dims=1).* _dx .+ diff(qDy,dims=2) .* _dy
            err_Pf = maximum(abs.(r_Pf))
            display(heatmap(xc,yc,Pf';xlims=(xc[1],xc[end]),ylims=(yc[1],yc[end]),aspect_ratio=1,c=:turbo))
        end
        iter += 1
    end

    return Pf[xtest, ytest]
end

@testset "Diffusion Acoustic 2D" begin
    nx = ny = 16 * 2 .^ (2:5) .- 1

    @test reshape(Pf_diffusion_2D(nx[1], ny[1]; do_check=false), 1, 3) ≈  [0.00785398056115133 0.007853980637555755 0.007853978592411982] atol = 1e-9
    @test reshape(Pf_diffusion_2D(nx[2], ny[2]; do_check=false), 1, 3) ≈  [0.00787296974549236 0.007849556884184108 0.007847181374079883] atol = 1e-9
    @test reshape(Pf_diffusion_2D(nx[3], ny[3]; do_check=false), 1, 3) ≈  [0.00740912103848251 0.009143711648167267 0.007419533048751209] atol = 1e-9
    @test reshape(Pf_diffusion_2D(nx[4], ny[4]; do_check=false), 1, 3) ≈  [0.00566813765849919 0.004348785338575644 0.005618691590498087] atol = 1e-9

end;
