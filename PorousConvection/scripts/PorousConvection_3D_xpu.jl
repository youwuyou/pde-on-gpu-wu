# Preferring @parallel approach
const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end


using Printf, Plots
using JLD  # for storing testing data

"""
I/O helper method

Stores the final state of the temperature property array as 'out_T.bin' for later plotting as Float32.
"""
function save_array(Aname,A)
    fname = string(Aname,".bin")
    out = open(fname,"w"); write(out,A); close(out)
end



@views av1(A) = 0.5.*(A[1:end-1].+A[2:end])

"""
Darcy's flux update in x, y directions
"""
@parallel function compute_flux_darcy!(Pf, T, qDx, qDy, qDz, _dx, _dy, _dz, k_ηf, αρg, _1_θ_dτ_D)

    # αρg acting only in z-direction
    @inn_x(qDx) = @inn_x(qDx) - (@inn_x(qDx) + k_ηf * (@d_xa(Pf) * _dx)) * _1_θ_dτ_D
    @inn_y(qDy) = @inn_y(qDy) - (@inn_y(qDy) + k_ηf * (@d_ya(Pf) * _dy)) * _1_θ_dτ_D
    @inn_z(qDz) = @inn_z(qDz) - (@inn_z(qDz) + k_ηf * (@d_za(Pf) * _dz - αρg *  @av_za(T))) * _1_θ_dτ_D

    return nothing
end


"""
Pressure update using Darcy flux
"""
@parallel function compute_Pf!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ_D)

    @all(Pf) = @all(Pf) - (@d_xa(qDx) * _dx + @d_ya(qDy)* _dy + @d_za(qDz) * _dz) * _β_dτ_D

    return nothing
end


"""
Temperature flux update in x, y, z directions
"""
@parallel function compute_flux_temp!(T, qTx, qTy, qTz, _dx, _dy, _dz, λ_ρCp, _1_θ_dτ_T)
    # select inner elements using @d_xi etc.
    @all(qTx)  = @all(qTx) - (@all(qTx) + @d_xi(T) * λ_ρCp * _dx) * _1_θ_dτ_T                    
    @all(qTy)  = @all(qTy) - (@all(qTy) + @d_yi(T) * λ_ρCp * _dy) * _1_θ_dτ_T
    @all(qTz)  = @all(qTz) - (@all(qTz) + @d_zi(T) * λ_ρCp * _dz) * _1_θ_dτ_T

    return nothing
end


"""
Temperature derivative update
"""
@parallel_indices (ix, iy, iz) function compute_dTdt!(dTdt, T, T_old, qDx, qDy, qDz, _dx, _dy, _dz, _dt, _ϕ)

    if (ix<=size(dTdt,1) && iy<=size(dTdt,2) && iz<=size(dTdt,3))
        dTdt[ix,iy,iz] = (T[ix+1,iy+1,iz+1] - T_old[ix+1,iy+1,iz+1])*_dt +
                         (max(qDx[ix+1,iy+1,iz+1],0.0)*(T[ix+1,iy+1,iz+1]-T[ix  ,iy+1,iz+1])*_dx +
                          min(qDx[ix+2,iy+1,iz+1],0.0)*(T[ix+2,iy+1,iz+1]-T[ix+1,iy+1,iz+1])*_dx +
                          max(qDy[ix+1,iy+1,iz+1],0.0)*(T[ix+1,iy+1,iz+1]-T[ix+1,iy  ,iz+1])*_dy +
                          min(qDy[ix+1,iy+2,iz+1],0.0)*(T[ix+1,iy+2,iz+1]-T[ix+1,iy+1,iz+1])*_dy +
                          max(qDz[ix+1,iy+1,iz+1],0.0)*(T[ix+1,iy+1,iz+1]-T[ix+1,iy+1,iz  ])*_dz +
                          min(qDz[ix+1,iy+1,iz+2],0.0)*(T[ix+1,iy+1,iz+2]-T[ix+1,iy+1,iz+1])*_dz)*_ϕ
    end

    return nothing

end


"""
update the temperature
"""
@parallel function compute_T!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _dt_β_dτ_T)

    @inn(T) = @inn(T) - (@all(dTdt) + @d_xa(qTx)* _dx + @d_ya(qTy)* _dy + @d_za(qTz) * _dz)* _dt_β_dτ_T                    

    return nothing
end


"""
update boundary condition along x
"""
@parallel_indices (iy,iz) function bc_x!(A)
    A[1  ,iy,iz] = A[2    ,iy,iz]
    A[end,iy,iz] = A[end-1,iy,iz]
    return
end

"""
update boundary condition along y
"""
@parallel_indices (ix,iz) function bc_y!(A)
    A[ix, 1  ,iz] = A[ix, 2    ,iz]
    A[ix, end,iz] = A[ix, end-1,iz]
    return
end

"""
update the residual for convergence monitoring
"""
@parallel function compute_r!(r_Pf, r_T, qDx, qDy, qDz, qTx, qTy, qTz, dTdt, _dx, _dy, _dz)
    @all(r_Pf)  = @d_xa(qDx)* _dx + @d_ya(qDy)* _dy + @d_za(qDz)* _dz
    @all(r_T)   = @all(dTdt) + @d_xa(qTx)* _dx + @d_ya(qTy)* _dy + @d_za(qTz)* _dz

    return nothing
end


"""
compute kernel that gets launched
"""
function compute!(Pf, T, T_old, qDx, qDy, qDz, qTx, qTy, qTz, dTdt, _dx, _dy, _dz, _dt, k_ηf, αρg, _ϕ, _1_θ_dτ_D, _β_dτ_D, λ_ρCp, λ_ρCp_dx, λ_ρCp_dy, λ_ρCp_dz, _1_θ_dτ_T, _dt_β_dτ_T)

    # hydro
    @parallel compute_flux_darcy!(Pf, T, qDx, qDy, qDz, _dx, _dy, _dz, k_ηf, αρg, _1_θ_dτ_D)
    @parallel compute_Pf!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ_D)

    # thermo
    @parallel compute_flux_temp!(T, qTx, qTy, qTz, _dx, _dy, _dz, λ_ρCp, _1_θ_dτ_T)
    @parallel compute_dTdt!(dTdt, T, T_old, qDx, qDy, qDz, _dx, _dy, _dz, _dt, _ϕ)
    @parallel compute_T!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _dt_β_dτ_T)

    # Boundary condition
    @parallel (1:size(T,2),1:size(T,3)) bc_x!(T)
    @parallel (1:size(T,1),1:size(T,3)) bc_y!(T)

    return nothing
end


@doc raw"""
    PorousConvection3D
The incompressible darcy flow equations with thermal coupling. Solver in 3D with multi-xpu switch enabled.

```math
\vec{q_D} = -\frac{k}{\eta}(\nabla p - \rho \vec{g})
```

```math
\nabla \cdot \vec{q_D} = 0
```

```math
\vec{q_F} = -k \nabla T
```

```math
\frac{\partial T}{\partial t}+ \frac{1}{\varphi} \vec{q_D} \cdot \nabla T + \nabla \cdot \vec{q_T} = 0
```

"""
@views function porous_convection_3D_xpu(nz_, nt_, nvis_; do_visu=false, do_check=true, test=true)
    # physics
    lx,ly,lz    = 40., 20.,20.
    k_ηf        = 1.0
    αρg         = 1.0
    ΔT          = 200.0
    ϕ           = 0.1
    Ra          = 1000                    # changed from 100
    λ_ρCp       = 1/Ra*(αρg*k_ηf*ΔT*lz/ϕ) # Ra = αρg*k_ηf*ΔT*lz/λ_ρCp/ϕ
  
    # numerics
    nz          = nz_
    ny          = nz
    nx          = 2*(nz+1)-1

    nt          = nt_                     # 500
    re_D        = 4π
    cfl         = 1.0/sqrt(3.1)           # was 2.1 for 2D
    maxiter     = 10max(nx,ny,nz)
    ϵtol        = 1e-6
    nvis        = nvis_
    ncheck      = ceil(2max(nx,ny,nz))    # ceil(0.25max(nx,ny))
  
    # preprocessing
    dx,dy,dz       = lx/nx,ly/ny,lz/nz
    xn,yn,zn       = LinRange(-lx/2,lx/2,nx+1),LinRange(-ly/2,ly/2,ny+1),LinRange(-lz,0,nz+1)
    xc,yc,zc       = av1(xn),av1(yn),av1(zn)

    θ_dτ_D         = max(lx,ly,lz)/re_D/cfl/min(dx,dy,dz)
    β_dτ_D         = (re_D*k_ηf)/(cfl*min(dx,dy,dz)*max(lx,ly,lz))

    # hpc value precomputation
    _dx, _dy, _dz  = 1. /dx, 1. /dy, 1. /dz
    _ϕ             = 1. / ϕ
    _1_θ_dτ_D      = 1 ./(1.0 + θ_dτ_D)
    _β_dτ_D        = 1. /β_dτ_D
    λ_ρCp_dx, λ_ρCp_dy, λ_ρCp_dz = λ_ρCp/dx, λ_ρCp/dy, λ_ρCp/dz

    # array initialization
    Pf             = @zeros(nx,ny,nz)
    r_Pf           = @zeros(nx,ny,nz)
    qDx,qDy,qDz    = @zeros(nx+1,ny,nz), @zeros(nx,ny+1,nz), @zeros(nx,ny,nz+1)

    dTdt           = @zeros(nx-2,ny-2,nz-2)
    r_T            = @zeros(nx-2,ny-2,nz-2)
    qTx            = @zeros(nx-1,ny-2,nz-2)
    qTy            = @zeros(nx-2,ny-1,nz-2)
    qTz            = @zeros(nx-2,ny-2,nz-1)
    
    T_cpu          = [ΔT*exp(-xc[ix]^2 -yc[iy]^2 -(zc[iz]+lz/2)^2) for ix=1:nx,iy=1:ny,iz=1:nz]
    T_cpu[:,:,1]  .= ΔT/2; T_cpu[:,:,end].=-ΔT/2    # B.C.

    T              = Data.Array(T_cpu)   # Type: Float64
    T_old          = Data.Array(copy(T_cpu))
    
    # vis
    st          = ceil(Int,nx/25)
   
    # visu - needed parameters for plotting
    if do_visu
        # plotting environment
        ENV["GKSwstype"]="nul"
        if isdir("viz3D_out")==false mkdir("viz3D_out") end
        loadpath = "viz3D_out/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        iframe = 0
    end

    # action
    t_tic = 0.0; niter = 0
    for it = 1:nt
        T_old .= T

        # time step
        dt = if it == 1
            0.1*min(dx,dy,dz)/(αρg*ΔT*k_ηf)
        else
            min(5.0*min(dx,dy,dz)/(αρg*ΔT*k_ηf),ϕ*min(dx/maximum(abs.(qDx)), dy/maximum(abs.(qDy)), dz/maximum(abs.(qDz)))/3.1)
        end
        
        _dt = 1. /dt   # precomputation

        re_T    = π + sqrt(π^2 + ly^2/λ_ρCp/dt)
        θ_dτ_T  = max(lx,ly,lz)/re_T/cfl/min(dx,dy,dz)
        β_dτ_T  = (re_T*λ_ρCp)/(cfl*min(dx,dy,dz)*max(lx,ly,lz))
        
        _1_θ_dτ_T   = 1 ./ (1.0 + θ_dτ_T)
        _dt_β_dτ_T  = 1 ./ (_dt + β_dτ_T) # precomputation

        # iteration loop
        iter = 1; err_D = 2ϵtol; err_T = 2ϵtol
        while max(err_D,err_T) >= ϵtol && iter <= maxiter

            if (it==1 && iter == 11) t_tic = Base.time(); niter=0 end

            compute!(Pf, T, T_old, qDx, qDy, qDz, qTx, qTy, qTz, dTdt, _dx, _dy, _dz, _dt, k_ηf, αρg, _ϕ, _1_θ_dτ_D, _β_dτ_D, λ_ρCp, λ_ρCp_dx, λ_ρCp_dy, λ_ρCp_dz, _1_θ_dτ_T, _dt_β_dτ_T)
            
            if do_check && iter % ncheck == 0
                @parallel compute_r!(r_Pf, r_T, qDx, qDy, qDz, qTx, qTy, qTz, dTdt, _dx, _dy, _dz)
                err_D  = maximum(abs.(r_Pf))
                err_T  = maximum(abs.(r_T))
            end
            iter += 1; niter += 1
        end

        
        if do_visu && it % nvis == 0            
            # visualisation
            if do_visu
                p1=heatmap(xc,zc,Array(T)[:,ceil(Int,ny/2),:]';xlims=(xc[1],xc[end]),ylims=(zc[1],zc[end]),aspect_ratio=1,c=:turbo)
                png(p1,@sprintf("viz3D_out/%04d.png",iframe+=1))
            end

        end
    end
    
    # timing
    t_toc = Base.time() - t_tic
    A_eff = (9 * 2 + 2) / 1e9 * nx * ny * nz * sizeof(eltype(T))  # Effective main memory access per iteration [GB]
    t_it  = t_toc / niter                              # Execution time per iteration [s]
    T_eff = A_eff / t_it                               # Effective memory throughput [GB/s]
    
    @printf("Time = %1.3f sec, T_eff = %1.3f GB/s \n", t_toc, T_eff)
    
    # visualize the last state
    save_array("out_T",convert.(Float32,Array(T)))

    
    # store data in case further testing needed
    if test == true
        save("../test/temp_ref_5_3D_xpu.jld", "data", T)  # store case for reference testing
    end


    # Return temperature T at final time
    return Array(T);
end


if isinteractive()
    # porous_convection_3D_xpu(63, 3, 1; do_visu=true, do_check=true,test=false)          # DEBUG CASE
    # porous_convection_3D_xpu(63, 500, 20; do_visu=true, do_check=true,test=false)       # RUN IT FOR EX02, TASK 2 (WEEK7)! nz = 63, nt = 500, nvis = 20
    porous_convection_3D_xpu(127, 2000, 50; do_visu=false, do_check=true,test=false)      # RUN IT FOR EX02, TASK 3 (WEEK7)! nz = 127, nt = 2000, nvis = 50

end
