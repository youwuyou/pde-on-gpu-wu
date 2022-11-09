# Preferring @parallel approach
# using false when pushing to github <= no github action support for gpu
const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end


using Printf, Plots
using JLD  # for storing testing data

@views av1(A) = 0.5.*(A[1:end-1].+A[2:end])
@views avx(A) = 0.5.*(A[1:end-1,:].+A[2:end,:])
@views avy(A) = 0.5.*(A[:,1:end-1].+A[:,2:end])


# Darcy's flux update in x, y directions
@parallel function compute_flux_darcy!(Pf, T, qDx, qDy, _dx, _dy, k_ηf, αρgx, αρgy, _1_θ_dτ_D)

    @inn_x(qDx) = @inn_x(qDx) - (@inn_x(qDx) + k_ηf * (@d_xa(Pf) * _dx - αρgx *  @av_xa(T))) * _1_θ_dτ_D
    @inn_y(qDy) = @inn_y(qDy) - (@inn_y(qDy) + k_ηf * (@d_ya(Pf) * _dy - αρgy *  @av_ya(T))) * _1_θ_dτ_D

    return nothing
end


# pressure update
@parallel function compute_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ_D)

    @all(Pf) = @all(Pf) - (@d_xa(qDx) * _dx + @d_ya(qDy)* _dy) * _β_dτ_D

    return nothing
end


# Temperature flux update in x, y directions
@parallel function compute_flux_temp!(Pf, T, qTx, qTy, _dx, _dy, λ_ρCp, _1_θ_dτ_T, T_inn_x, T_inn_y)

    # qTx[ix,iy]  -= (qTx[ix,iy] + λ_ρCp*(@d_xa(T[:,2:end-1])* _dx)) * _1_θ_dτ_T                    
    # qTy[ix,iy]  -= (qTy[ix,iy] + λ_ρCp*(@d_ya(T[2:end-1,:])* _dy)) * _1_θ_dτ_T

    @all(T_inn_y) = @inn_y(T)
    @all(T_inn_x) = @inn_x(T)

    @all(qTx)  = @all(qTx) - (@all(qTx) + λ_ρCp* @d_xa(T_inn_y) * _dx) * _1_θ_dτ_T                    
    @all(qTy)  = @all(qTy) - (@all(qTy) + λ_ρCp* @d_ya(T_inn_x) * _dy) * _1_θ_dτ_T

    return nothing
end



# update the temperature
@parallel function compute_T!(T, dTdt, qTx, qTy, _dx, _dy, _dt_β_dτ_T)

    @inn(T) = @inn(T) - (@all(dTdt) + @d_xa(qTx)* _dx + @d_ya(qTy)* _dy)* _dt_β_dτ_T                    

    return nothing
end


# update boundary condition
@parallel_indices (iy) function bc_x!(A)

    A[1  ,iy] = A[2    ,iy]
    A[end,iy] = A[end-1,iy]
    return
end


function compute!(Pf, T, T_old, qDx, qDy,  qTx, qTy, dTdt, _dx, _dy, _dt, k_ηf, αρgx, αρgy,  _ϕ, _1_θ_dτ_D, _β_dτ_D, λ_ρCp, _1_θ_dτ_T, _dt_β_dτ_T, T_inn_x, T_inn_y)

    # hydro
    @parallel compute_flux_darcy!(Pf, T, qDx, qDy, _dx, _dy, k_ηf, αρgx, αρgy, _1_θ_dτ_D)
    @parallel compute_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ_D)

    # thermo
    @parallel compute_flux_temp!(Pf, T, qTx, qTy, _dx, _dy, λ_ρCp, _1_θ_dτ_T, T_inn_x, T_inn_y)

    dTdt           .= (T[2:end-1,2:end-1] .- T_old[2:end-1,2:end-1]).* _dt .+
                           (max.(qDx[2:end-2,2:end-1],0.0).*diff(T[1:end-1,2:end-1],dims=1).* _dx .+
                            min.(qDx[3:end-1,2:end-1],0.0).*diff(T[2:end  ,2:end-1],dims=1).* _dx .+
                            max.(qDy[2:end-1,2:end-2],0.0).*diff(T[2:end-1,1:end-1],dims=2).* _dy .+
                            min.(qDy[2:end-1,3:end-1],0.0).*diff(T[2:end-1,2:end  ],dims=2).* _dy).* _ϕ

    @parallel compute_T!(T, dTdt, qTx, qTy, _dx, _dy, _dt_β_dτ_T)

    # Boundary condition
    @parallel (1:size(T,2)) bc_x!(T)


    return nothing
end





@views function porous_convection_2D_xpu(ny_, nt_, nvis_; do_visu=false, do_check=true, test=true)
    # physics
    lx,ly       = 40., 20.
    k_ηf        = 1.0
    αρgx,αρgy   = 0.0,1.0
    αρg         = sqrt(αρgx^2+αρgy^2)
    ΔT          = 200.0
    ϕ           = 0.1
    Ra          = 1000                    # changed from 100
    λ_ρCp       = 1/Ra*(αρg*k_ηf*ΔT*ly/ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
  
    # numerics
    ny          = ny_                     # ceil(Int,nx*ly/lx)
    nx          = 2 * (ny+1) - 1          # 127
    nt          = nt_                     # 500
    re_D        = 4π
    cfl         = 1.0/sqrt(2.1)
    maxiter     = 10max(nx,ny)
    ϵtol        = 1e-6
    nvis        = nvis_
    ncheck      = ceil(max(nx,ny)) # ceil(0.25max(nx,ny))
  
    # preprocessing
    dx,dy       = lx/nx,ly/ny
    xn,yn       = LinRange(-lx/2,lx/2,nx+1),LinRange(-ly,0,ny+1)
    xc,yc       = av1(xn),av1(yn)
    θ_dτ_D      = max(lx,ly)/re_D/cfl/min(dx,dy)
    β_dτ_D      = (re_D*k_ηf)/(cfl*min(dx,dy)*max(lx,ly))

    # hpc value precomputation
    _dx, _dy    = 1. /dx, 1. /dy
    _ϕ          = 1. / ϕ
    _1_θ_dτ_D   = 1 ./(1.0 + θ_dτ_D)
    _β_dτ_D     = 1. /β_dτ_D
   
    # array initialization
    Pf          = @zeros(nx,ny)
    r_Pf        = @zeros(nx,ny)
    qDx,qDy     = @zeros(nx+1,ny), @zeros(nx,ny+1)
    qDx_c,qDy_c = zeros(nx,ny),   zeros(nx,ny)
    qDmag       = zeros(nx,ny)    

    T_cpu       = @. ΔT*exp(-xc^2 - (yc'+ly/2)^2); T_cpu[:,1] .= ΔT/2; T_cpu[:,end] .= -ΔT/2
    T           = Data.Array(T_cpu)
    T_old       = Data.Array(copy(T_cpu))
    # FIXME: check if it is correct to assign it like this


    # (NEW) added for the temperature update to select @inn_y(T) and @inn_x(T)
    #   since the nested macro does not work
    T_inn_y     = @zeros(nx, ny-2)
    T_inn_x     = @zeros(nx-2, ny)


    dTdt        = @zeros(nx-2,ny-2)
    r_T         = @zeros(nx-2,ny-2)
    qTx         = @zeros(nx-1,ny-2)
    qTy         = @zeros(nx-2,ny-1)
   
    # vis
    st          = ceil(Int,nx/25)
    Xc, Yc      = [x for x=xc, y=yc], [y for x=xc,y=yc]
    Xp, Yp      = Xc[1:st:end,1:st:end], Yc[1:st:end,1:st:end]
    iframe = 0
   

    # visu - needed parameters for plotting
    if do_visu
        # plotting environment
        ENV["GKSwstype"]="nul"
        if isdir("viz_out")==false mkdir("viz_out") end
        loadpath = "viz_out/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        iframe = 0
    end


    # action
    t_tic = 0.0; niter = 0
    for it = 1:nt
        T_old .= T

        # time step
        dt = if it == 1 
            0.1*min(dx,dy)/(αρg*ΔT*k_ηf)
        else
            min(5.0*min(dx,dy)/(αρg*ΔT*k_ηf),ϕ*min(dx/maximum(abs.(qDx)), dy/maximum(abs.(qDy)))/2.1)
        end
        
        _dt = 1. /dt   # precomputation

        re_T    = π + sqrt(π^2 + ly^2/λ_ρCp/dt)
        θ_dτ_T  = max(lx,ly)/re_T/cfl/min(dx,dy)
        β_dτ_T  = (re_T*λ_ρCp)/(cfl*min(dx,dy)*max(lx,ly))
        
        _1_θ_dτ_T   = 1 ./ (1.0 + θ_dτ_T)
        _dt_β_dτ_T  = 1 ./(_dt + β_dτ_T) # precomputation

        # iteration loop
        iter = 1; err_D = 2ϵtol; err_T = 2ϵtol
        while max(err_D,err_T) >= ϵtol && iter <= maxiter

            if (it==1 && iter == 11) t_tic = Base.time(); niter=0 end

            compute!(Pf, T, T_old, qDx, qDy,  qTx, qTy, dTdt, _dx, _dy, _dt, k_ηf, αρgx, αρgy,  _ϕ, _1_θ_dτ_D, _β_dτ_D, λ_ρCp, _1_θ_dτ_T, _dt_β_dτ_T, T_inn_x, T_inn_y)

            
            if do_check && iter % ncheck == 0
                r_Pf  .= diff(qDx,dims=1).* _dx .+ diff(qDy,dims=2).* _dy
                r_T   .= dTdt .+ diff(qTx,dims=1).* _dx .+ diff(qTy,dims=2).* _dy
                err_D  = maximum(abs.(r_Pf))
                err_T  = maximum(abs.(r_T))
            end
            iter += 1; niter += 1
        end


        if it % nvis == 0
            qDx_c .= avx(Array(qDx))
            qDy_c .= avy(Array(qDy))
            qDmag .= sqrt.(qDx_c.^2 .+ qDy_c.^2)
            qDx_c ./= qDmag
            qDy_c ./= qDmag
            qDx_p = qDx_c[1:st:end,1:st:end]
            qDy_p = qDy_c[1:st:end,1:st:end]


            # visualisation
            if do_visu
                heatmap(xc,yc,Array(T');xlims=(xc[1],xc[end]),ylims=(yc[1],yc[end]),aspect_ratio=1,c=:turbo)
                png(quiver!(Xp[:], Yp[:], quiver=(qDx_p[:], qDy_p[:]), lw=0.5, c=:black),
                    @sprintf("viz_out/porous2D_%04d.png",iframe+=1))
            end
        end
    end

    t_toc = Base.time() - t_tic
    A_eff = (6 * 2 + 2) / 1e9 * nx * ny * sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc / niter                              # Execution time per iteration [s]
    T_eff = A_eff / t_it                               # Effective memory throughput [GB/s]
    
    @printf("Time = %1.3f sec, T_eff = %1.3f GB/s \n", t_toc, T_eff)

    
    
    if test == true
        save("../test/qDx_p_ref_30_2D_xpu.jld", "data", qDx_c[1:st:end,1:st:end])  # store case for reference testing
        save("../test/qDy_p_ref_30_2D_xpu.jld", "data", qDy_c[1:st:end,1:st:end])
    end

    
    # Return qDx_p and qDy_p at final time
    return [qDx_c[1:st:end,1:st:end], qDy_c[1:st:end,1:st:end]]   
end



if isinteractive()
    # porous_convection_2D_xpu(63, 500, 20; do_visu=true, do_check=true,test=false)    # RUN IT FOR EX01, TASK 3 (WEEK7)! ny = 63, nt = 500, nvis = 20
    porous_convection_2D_xpu(511, 4000, 50; do_visu=true, do_check=true,test=false)  # RUN IT FOR EX01, TASK 4 (WEEK7)! ny = 511, nt = 4000, nvis = 50

end