using Plots,Plots.Measures,Printf
default(size=(1300,1000),framestyle=:box,label=false,grid=false,margin=15mm, top_margin=5mm, right_margin=40mm, lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)


@views function porous_convection_2D_heatmap()
    # physics
    lx, ly        = 40.0, 20.0
    k_ηf          = 1.0
    re            = 2π
    αρgx,αρgy     = 0.0,1.0
    αρg           = sqrt(αρgx^2+αρgy^2)
    ΔT            = 200.0
    ϕ             = 0.1
    Ra            = 1000.0
    λ_ρCp         = 1/Ra*(αρg*k_ηf*ΔT*ly/ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
    
    # numerics
    nx            = 127
    ny            = ceil(Int, ly * nx / lx)
    dx, dy        = lx/nx, ly/ny
    nt            = 10
    dt_diff       = min(dx,dy)^2/λ_ρCp/4.1

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
    θ_dτ          = max_len /re /cfl /min_step
    β_dτ          = (re * k_ηf) / (cfl * min_step * max_len)
    
    # array initialisation
    Pf            = @. exp(-(xc-lx/4)^2 -(yc'-ly/4)^2)   # 100 x 50
    r_Pf          = zeros(nx,ny)
    qDx           = zeros(Float64, nx + 1, ny)           # 101 x 50
    qDy           = zeros(Float64, nx, ny + 1)           # 100 x 51

    T             = @. ΔT*exp(-xc^2 - (yc'+ly/2)^2); T_old = copy(T)
    T[:,1]       .= ΔT/2
    T[:,end]     .= -ΔT/2
    T[[1,end],:] .= T[[2,end-1],:]                   # horizontal - adiabatic

    # visualisation init


    # physical time loop
    anim = @animate for it = 1:nt
        T_old .= T
        
        # iteration loop
        iter = 1; err_Pf = 2ϵtol; iter_evo = Float64[]; err_Pf_evo = Float64[]
        while err_Pf >= ϵtol && iter <= maxiter
            
            # darcy flux
            qDx[2:end-1, :]  .-= (qDx[2:end-1, :] .+ k_ηf.* (diff(Pf, dims=1) ./ dx - αρgx .* T[1:end-1,:])) ./ (1.0 + θ_dτ)
            qDy[:, 2:end-1]  .-= (qDy[:, 2:end-1] .+ k_ηf.* (diff(Pf, dims=2) ./ dy - αρgy .* T[:,1:end-1])) ./ (1.0 + θ_dτ)
            
            Pf               .-= ( diff(qDx, dims=1)./dx  +  diff(qDy, dims=2) ./ dy) ./ β_dτ 
            
            if iter%ncheck == 0
                err_Pf = maximum(abs.(diff(qDx, dims=1)./dx  .+  diff(qDy, dims=2) ./ dy))
                push!(iter_evo, iter/nx)
                push!(err_Pf_evo,err_Pf)
            end
            
            iter += 1
            
        end
        
        # Preparation: for temperature update
        dt_adv              = ϕ*min(dx/maximum(abs.(qDx)), dy/maximum(abs.(qDy)))/2.1
        dt                  = min(dt_diff,dt_adv)
        

        # PART I:   
        # diffusive update  ↔  ∂T/∂τ + ∇ · qT = 0
        T[2:end-1, 2:end-1] .+= dt.*λ_ρCp.*((diff(diff(T ,dims=1), dims=1)[:, 2:end-1]) .+ (diff(diff(T, dims=2), dims=2)[2:end-1, :]))


        # PART II:  
        # advection term - using upwinding scheme
        T[2:end,   :] .-= dt .* max.(qDx[2:end-1,:] ./ ϕ, 0.0) .* (diff(T, dims=1)./dx)
        T[1:end-1, :] .-= dt .* min.(qDx[1:end-2,:] ./ ϕ, 0.0) .* (diff(T, dims=1)./dx)

        T[:,   2:end] .-= dt .* max.(qDy[:, 2:end-1] ./ ϕ, 0.0) .* (diff(T, dims=2)./dy)
        T[:, 1:end-1] .-= dt .* min.(qDy[:, 1:end-2] ./ ϕ, 0.0) .* (diff(T, dims=2)./dy)

        # boundary conditions
        T[:,1:2]            .=  ΔT/2                       # upper boundary - heating
        T[:,end-1:end]      .= -ΔT/2                       # lower boundary - cooling
        T[[1,end],:]        .= T[[2,end-1],:]              # horizontal - adiabatic

        # visualisation
        r_Pf               .= diff(qDx,dims=1)./dx .+ diff(qDy,dims=2)./dy
        err_Pf              = maximum(abs.(r_Pf))
        
        p1 = heatmap(xc,yc,Pf';xlims=(xc[1],xc[end]),ylims=(yc[1],yc[end]),aspect_ratio=1, interpolation=true, c=:turbo)
        p2 = heatmap(xc,yc, T';xlims=(xc[1],xc[end]),ylims=(yc[1],yc[end]),aspect_ratio=1, interpolation=true, c=:thermal)
        display(plot(p1,p2; layout=(2,1)))


        @printf("it = %d, iter/nx=%.1f, err_Pf=%1.3e\n",it,iter/nx,err_Pf)
    end
    # file I/O
    gif(anim, "pressure_temperature_2D.gif", fps = 2)
end


porous_convection_2D_heatmap()