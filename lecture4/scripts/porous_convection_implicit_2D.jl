using Plots,Plots.Measures,Printf
default(size=(1300,1000),framestyle=:box,label=false,grid=false,margin=15mm, top_margin=5mm, right_margin=40mm, lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)


@views function implicit_porous_convection_2D()
    # physics
    lx, ly        = 40.0, 20.0
    k_ηf          = 1.0
    re_D          = 4π                                      # adjusted from 2π -> now fully coupled
    αρgx,αρgy     = 0.0,1.0
    αρg           = sqrt(αρgx^2+αρgy^2)
    ΔT            = 200.0
    ϕ             = 0.1
    Ra            = 1000.0
    λ_ρCp         = 1/Ra*(αρg*k_ηf*ΔT*ly/ϕ)                 # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
    
    # numerics
    nx            = 127
    ny            = ceil(Int, ly * nx / lx)
    dx, dy        = lx/nx, ly/ny
    nt            = 500

    # extrema
    max_num       = max(nx, ny)
    min_step      = min(dx, dy)
    max_len       = max(lx, ly)
    ϵtol          = 1e-8
    maxiter       = 100max_num
    ncheck        = ceil(Int,0.25max_num)
    cfl           = 1.0 / sqrt(2.1)

    # derived numerics
    xc            = LinRange(-lx/2+dx/2,lx/2-dx/2,nx)      # 100
    yc            = LinRange(-ly + dy/2, -dy/2,ny)         # 50
    θ_dτ_D        = max_len /re_D /cfl /min_step
    β_dτ_D        = (re_D * k_ηf) / (cfl * min_step * max_len)
    
    # array initialisation
    Pf            = @. exp(-(xc-lx/4)^2 -(yc'-ly/4)^2)     # 100 x 50
    r_Pf          = zeros(nx,ny)
    qDx           = zeros(Float64, nx + 1, ny)             # 101 x 50
    qDy           = zeros(Float64, nx, ny + 1)             # 100 x 51
    qDxc          = zeros(Float64, nx, ny)                 # 100 x 50
    qDyc          = zeros(Float64, nx, ny)                 # 100 x 50
    qDmag         = zeros(Float64, nx, ny)

    T             = @. ΔT*exp(-xc^2 - (yc'+ly/2)^2); T_old = copy(T)
    dTdt          = zeros(nx-2,ny-2)
    r_T           = zeros(nx-2,ny-2)
    
    # different boundary conditions for temp -> different size
    qTx           = zeros(nx-1,ny-2)                       # 99 x 48
    qTy           = zeros(nx-2,ny-1)                       # 98 x 49   

    # boundary conditions
    T[:,1]       .=  ΔT/2                         # upper boundary - heating
    T[:,end]     .= -ΔT/2                         # lower boundary - cooling
    T[[1,end],:] .= T[[2,end-1],:]                # horizontal - adiabatic
     
    # visualisation init
    nvis      = 5
    st        = ceil(Int,nx/25)
    Xc, Yc    = [x for x=xc, y=yc], [y for x=xc,y=yc]
    Xp, Yp    = Xc[1:st:end,1:st:end], Yc[1:st:end,1:st:end]

    # for estimating no.iterations needed
    iter_tot = 0

    # physical time loop
    anim = @animate for it = 1:nt
        T_old .= T
        
        # time step
        dt = if it == 1
            0.1*min(dx,dy)/(αρg*ΔT*k_ηf)    # avoid division by 0
        else
            min(5.0*min(dx,dy)/(αρg*ΔT*k_ηf),ϕ*min(dx/maximum(abs.(qDx)), dy/maximum(abs.(qDy)))/2.1)
        end

        re_T    = π + sqrt(π^2 + ly^2/λ_ρCp/dt)
        θ_dτ_T  = max(lx,ly)/re_T/cfl/min(dx,dy)
        β_dτ_T  = (re_T*λ_ρCp)/(cfl*min(dx,dy)*max(lx,ly))


        # iteration loop
        iter = 1; err_D = 2ϵtol; err_T = 2ϵtol; iter_evo = Float64[]; err_D_evo = Float64[]

        while err_D >= ϵtol && iter <= maxiter
            # fluid pressure update
            qDx[2:end-1, :]  .-= (qDx[2:end-1, :] .+ k_ηf.* (diff(Pf, dims=1) ./ dx - αρgx .* T[1:end-1,:])) ./ (1.0 + θ_dτ_D)
            qDy[:, 2:end-1]  .-= (qDy[:, 2:end-1] .+ k_ηf.* (diff(Pf, dims=2) ./ dy - αρgy .* T[:,1:end-1])) ./ (1.0 + θ_dτ_D)
            Pf               .-= ( diff(qDx, dims=1)./dx  +  diff(qDy, dims=2) ./ dy) ./ β_dτ_D 

            # temperature update
            qTx    .-= (qTx .+ λ_ρCp .* (diff(T, dims=1)[:,2:end-1] ./ dx )) ./ (1.0 + θ_dτ_T)
            qTy    .-= (qTy .+ λ_ρCp .* (diff(T, dims=2)[2:end-1,:] ./ dy )) ./ (1.0 + θ_dτ_T)

        
            # Material physical time derivative  ↔  DT/Dt = ∂T/∂t + q_D /ϕ · grad(T) 
            # reaction-like term
            dTdt .= (T[2:end-1,2:end-1] .- T_old[2:end-1,2:end-1])./dt
           
            # upwind advection term
            dTdt[2:end, :]  .+= (max.(qDx[2:end-3,2:end-1],0.0) .* diff(T, dims=1)[2:end-1, 2:end-1] ./dx) ./ ϕ
            dTdt[1:end-1,:] .+= (min.(qDx[2:end-3,2:end-1],0.0) .* diff(T, dims=1)[2:end-1, 2:end-1] ./dx) ./ ϕ

            dTdt[:,2:end]   .+= (max.(qDy[2:end-1,2:end-3],0.0) .* diff(T, dims=2)[2:end-1, 2:end-1] ./dy) ./ ϕ
            dTdt[:,1:end-1] .+= (min.(qDy[2:end-1,2:end-3],0.0) .* diff(T, dims=2)[2:end-1, 2:end-1] ./dy) ./ ϕ


            # compute temperature update
            T[2:end-1,2:end-1] .-= (dTdt .+ diff(qTx, dims=1)./dx  +  diff(qTy, dims=2) ./ dy)./(1.0/dt + β_dτ_T)

            
            # boundary conditions
            T[:,1:2]            .=  ΔT/2                             # upper boundary - heating
            T[:,end-1:end]      .= -ΔT/2                             # lower boundary - cooling
            T[[1,end],:]        .= T[[2,end-1],:]                    # horizontal     - adiabatic
    

            if iter % ncheck == 0

                r_Pf  .= diff(qDx, dims=1)./dx  +  diff(qDy, dims=2) ./ dy
                r_T   .= dTdt .+ (diff(qTx,dims=1) ./ dx .+ diff(qTy, dims=2) ./ dy)  

                err_D  = maximum(abs.(r_Pf))
                err_T  = maximum(abs.(r_T))
                @printf("  iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n",iter/nx,err_D,err_T)
            end

            iter += 1
            iter_tot += 1
        end

        # visualisation
        if it % nvis == 0
            qDxc            .= qDx[1:end-1, :] .+ diff(qDx, dims=1) ./ 2    # average qDx in x
            qDyc            .= qDy[:, 1:end-1] .+ diff(qDy, dims=2) ./ 2    # average qDx in y
            qDmag           .= sqrt.(qDxc.^2 .+ qDyc.^2)
            qDxc           ./= qDmag
            qDyc           ./= qDmag
            qDx_p            = qDxc[1:st:end,1:st:end]
            qDy_p            = qDyc[1:st:end,1:st:end]
            
            heatmap(xc,yc,T';xlims=(xc[1],xc[end]),ylims=(yc[1],yc[end]),aspect_ratio=1,c=:turbo, clims=(-100, 100))
            display(quiver!(Xp[:], Yp[:], quiver=(qDx_p[:], qDy_p[:]), lw=0.5, c=:black))
        end

    end

    print(iter_tot)    
    # file I/O
    gif(anim, "implicit_pressure_temperature_2D_quiver.gif", fps = 50)
end


implicit_porous_convection_2D()
