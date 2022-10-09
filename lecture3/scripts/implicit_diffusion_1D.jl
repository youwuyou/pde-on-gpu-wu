using Plots,Plots.Measures,Printf
default(size=(1200,800),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)

@views function implicit_diffusion_1D()
    # physics
    lx      = 20.0
    dc      = 1.0
    da      = 1000.0                # before 10.0
    re      = π + sqrt(π^2 + da)
    ρ       = (lx/(dc*re))^2        # ρ ↔ parameter for inertia term
    
    # numerics
    nx      = 100
    nt      = 40                    # no. physical time steps -> adjusted from 10
    tol     = 1e-8
    maxiter = 50nx
    ncheck  = ceil(Int,0.05nx)      # freq -> adjusted from 0.25nx
    
    # derived numerics
    dx      = lx/nx
    xc      = LinRange(dx/2,lx-dx/2,nx)
    dτ      = dx/sqrt(1/ρ)
    dt       = lx^2/dc/da           # renamed from ξ
    
    # array initialisation
    C       = @. 1.0 + exp(-(xc-lx/4)^2) - xc/lx; C_i = copy(C)
    C_old   = copy(C)              # structurally same as C_eq
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
    gif(anim, "experiment01.gif", fps = 10)
end

implicit_diffusion_1D()

