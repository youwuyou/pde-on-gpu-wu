using Plots,Plots.Measures,Printf
default(size=(1200,800),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)

@views function implicit_advection_diffusion_1D()
    
    # physics
    dc      = 1.0
    vx      = 1.0                     # advection velocity
    
    # numerics
    lx      = 20.0
    nx      = 100
    nt      = 40                     # no. physical time steps -> adjusted from 10
    tol     = 1e-8
    maxiter = 50nx
    ncheck  = ceil(Int,0.05nx)       # freq -> adjusted from 0.25nx
    dx      = lx/nx
    dt      = dx/abs(vx)             # renamed from ξ
    
    # derived physics
    da      = lx^2/dc/dt             # before as constant
    re      = π + sqrt(π^2 + da)
    ρ       = (lx/(dc*re))^2         # ρ ↔ parameter for inertia term
    
    # derived numerics
    xc      = LinRange(dx/2,lx-dx/2,nx)
    dτ      = dx/sqrt(1/ρ)
    
    # derived physics
    
    # array initialisation
    C       = @. 1.0 + exp(-(xc-lx/4)^2) - xc/lx; C_i = copy(C)
    C_old   = copy(C)               # structurally same as C_eq
    qx      = zeros(Float64, nx-1)
    
    # iteration loop
    anim = @animate for it = 1:nt
        C_old    = copy(C)          # keep a copy
        iter     = 1                # pseudo-time stepping
        iter_evo = Float64[]        # evolution tracking
        err_evo  = Float64[]        # error tracking
        err      = 2 * tol       
        
        while err >= tol && iter <= maxiter
            
            # updating using implicit timestepping
            qx          .-= dτ./(ρ   .+ dτ/dc ).*(                         qx./dc     .+ diff(C) ./dx)    # remains unchanged
            C[2:end-1]  .-= dτ./(1.0 .+ dτ/dt ).*((C[2:end-1] .- C_old[2:end-1])./ dt .+ diff(qx)./dx)    # transient term

            # calculate residual norm and store
            if iter % ncheck == 0
                err = maximum(abs.(diff(dc.*diff(C)./dx)./dx .- (C[2:end-1] .- C_old[2:end-1])./ dt))
                push!(iter_evo,iter/nx)
                push!(err_evo,err)
            end

            iter += 1 # pseudo time increment
        end
        
        # advection term using upwinding scheme
        (vx > 0) && (C[2:end]   .-= dt .* vx .* diff(C)./dx)
        (vx < 0) && (C[1:end-1] .-= dt .* vx .* diff(C)./dx)
        
        (it % (nt÷2) == 0) && (vx = - vx) # change the direction of propagation

        # boundary conditon (optional)
        # C[1] = 1
        # C[end] = 0

        # visualisation
        # the evolution plot
        p1 = plot(xc,[C_i,C];xlims=(0,lx), ylims=(-0.1,2.0),
                    xlabel="lx",ylabel="Concentration",title="iter/nx=$(round(iter/nx,sigdigits=3))")
        
        # the convergence plot
        p2 = plot(iter_evo,err_evo;xlabel="iter/nx",ylabel="err",
                    yscale=:log10,grid=true,markershape=:circle,markersize=10)
        display(plot(p1,p2;layout=(2,1)))

    end

    # file I/O
    gif(anim, "implicit_advection_diffusion_1D.gif", fps = 10)
end

implicit_advection_diffusion_1D()

