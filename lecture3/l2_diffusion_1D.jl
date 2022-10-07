using Plots,Plots.Measures,Printf
default(size=(1200,400),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)

@views function diffusion_1D()
    # physics
    lx   = 20.0
    dc   = 1.0
    ρ    = (lx/(dc*2π))^2


    # numerics
    nx   = 200
    # nvis = 50
    # nvis = 5
    
    # derived numerics
    dx   = lx/nx
    # dt = dx/sqrt(1/ρ)/2
    dτ = dx/sqrt(1/ρ)/2 # not physical time but pseudo time for transient
    # dt   = dx^2/dc/2
    # nt   = nx^2 ÷ 5
    # nt = 5nx
    
    
    # correction-based
    ϵtol = 1e-8
    maxiter = 100nx
    ncheck = ceil(Int, 0.25nx)
    xc   = LinRange(dx/2,lx-dx/2,nx)


    # array initialisation
    C    = @. 1.0 + exp(-(xc-lx/4)^2) - xc/lx; C_i = copy(C)
    qx   = zeros(Float64, nx-1)
    
    # new arrays to monitor iterations
    iter_evo = Float64[]  # creating empty arrays
    err_evo = Float64[]

    # iteration loop -> not a time loop anymore
    err = 2ϵtol
    iter = 1

    while err >= ϵtol && iter <= maxiter

        #        qx          .= .-dc.*diff(C )./dx
        #        qx .-= (dt / ρ) .* (diff(C)./ dx)
        #qx .= (ρ .* qx ./ dτ - diff(C)./ dx) ./ (ρ/dτ + 1/dc)  # implicit method
        # qx .-= (dτ ./ ρ) .* (diff(C)./ dx .+ qx ./ dc)
        qx .-= (dτ ./ (ρ + dτ ./ dc))  .* ( diff(C) ./ dx + qx ./ dc)
        C[2:end-1] .-=   dτ.*diff(qx)./dx
        
            if iter%ncheck == 0

                # monitor residual
                err = maximum(abs.(dc * diff(diff(C)./dx)./dx))   # maxnorm using abs.()

                push!(iter_evo, iter/nx)
                push!(err_evo, err)

                p1 = plot(xc,[C_i,C];xlims=(0,lx),ylims=(-0.1,2.0),
                xlabel="lx",ylabel="Concentration",title="iter/nx=$(round(iter/nx,sigdigits=3))")
                
                p2 = plot(iter_evo,err_evo;xlabel="iter/nx",ylabel="err",
                yscale=:log10,grid=true,markershape=:circle,markersize=10)
                
                display(plot(p1,p2;layout=(2,1)))
            end



        # increase the iteration counter
        iter += 1

    end





    end

diffusion_1D()
