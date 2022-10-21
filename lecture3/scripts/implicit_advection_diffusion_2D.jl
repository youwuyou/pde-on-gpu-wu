using Plots,Plots.Measures,Printf
using LinearAlgebra, MAT  # for matrix norm
default(size=(1200,800),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)

@views function implicit_advection_diffusion_2D()
    
    # physics
    dc      = 1.0

    # CORRECTION! Intial speed wrongly set to 1.0 somehow
    vx, vy  = 10.0, -10.0                   # advection velocity
    
    # numerics
    lx, ly  = 10.0, 10.0
    nx, ny  = 200, 201
    dx, dy  = lx/nx, ly/ny
    xc, yc  = LinRange(dx/2,lx-dx/2,nx), LinRange(dy/2,ly-dy/2,ny)
    nt      = 50                         # no. physical time steps -> 10 for task 2
    tol     = 1e-8
    maxiter = 10nx
    ncheck  = ceil(Int, 0.02nx)        
    dt = min(dx/abs(vx),dy/abs(vy))/2

    # derived physics
    da      = lx^2/dc/dt             
    re      = π + sqrt(π^2 + da)
    ρ       = (lx/(dc*re))^2            # ρ ↔ parameter for inertia term
    dτ      = min(dx,dy)/sqrt(1/ρ)/sqrt(2)
    
    # array initialisation
    # using yc' ↔ transpose of yc to create matrix
    C       = @. exp(-(xc-lx/4)^2 -(yc'-3ly/4)^2)
    C_old   = copy(C)
    qx, qy  = zeros(nx-1, ny), zeros(nx, ny-1)
    

    anim = @animate for it = 1:nt
        C_old   .= C                   # keep a copy
        iter     = 1                   # pseudo-time stepping
        iter_evo = Float64[]           # evolution
        err_evo  = Float64[]           # evolution of error
        err      = 2 * tol       
        

        while err >= tol && iter <= maxiter
            
            # updating using implicit timestepping
            qx                 .-= dτ./(ρ   .+ dτ/dc ).*( qx./dc .+ diff(C, dims=1) ./dx)
            qy                 .-= dτ./(ρ   .+ dτ/dc ).*( qy./dc .+ diff(C, dims=2) ./dy)

            # dimension => 98 x 98
            C[2:end-1,2:end-1] .-= dτ./(1.0 .+ dτ/dt ) .* ((C[2:end-1,2:end-1] .- C_old[2:end-1,2:end-1]) ./ dt
                                                            .+  ((diff(qx, dims=1)[:, 2:end-1] ) ./ dx .+
                                                                 (diff(qy, dims=2)[2:end-1, :] ) ./ dy ) 
                                                            )

            # calculate residual norm and store
            if iter % ncheck == 0

                # using definition of error norm in 'diff_2D_lin.jl' script of Pseudo-transient solver
                ResC = -((C[2:end-1,2:end-1] .- C_old[2:end-1,2:end-1]) ./ dt
                    .+  ((diff(qx, dims=1)[:, 2:end-1] ) ./ dx .+
                        (diff(qy, dims=2)[2:end-1, :] ) ./ dy ))

                err = norm(ResC) / sqrt(length(ResC))

                push!(iter_evo,iter/nx)
                push!(err_evo,err)

            end

            iter += 1 # pseudo time increment
        end # of pseudo time loop
        
        # advection term using upwinding scheme
        (vx > 0) && (C[2:end, 1:end]   .-= dt .* vx .* diff(C, dims = 1)./dx)
        (vx < 0) && (C[1:end-1, 1:end] .-= dt .* vx .* diff(C, dims = 1)./dx)        
        
        (vy > 0) && (C[1:end, 2:end]   .-= dt .* vy .* diff(C, dims = 2)./dy)
        (vy < 0) && (C[1:end, 1:end-1] .-= dt .* vy .* diff(C, dims = 2)./dy)


        # visualisation - the evolution of concentration
        p1 = heatmap(xc,yc,C', xlims=(0,lx), ylims=(0,ly), clims=(0,1), aspect_ratio=1,
                    xlabel="lx",ylabel="ly",title="it=$(it)")
        
        p2 = plot(iter_evo,err_evo;xlabel="iter/nx",ylabel="err",
                    yscale=:log10,grid=true,markershape=:circle,markersize=5)
        
        display(plot(p1,p2;layout=(2,1)))

    end  # of physical loop

    # file I/O
    gif(anim, "implicit_advection_diffusion_2D.gif", fps = 2)
end

implicit_advection_diffusion_2D()

