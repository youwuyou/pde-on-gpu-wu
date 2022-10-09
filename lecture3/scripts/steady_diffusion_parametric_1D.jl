using Plots,Plots.Measures,Printf
default(size=(1200,800),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)

@views function steady_diffusion_1D()
    # physics
    lx      = 20.0
    dc      = 1.0
   
    # numerics
    nx      = 100
    ϵtol    = 1e-8
    maxiter = 100nx
    ncheck  = ceil(Int,0.25nx)
 
    # numerics => convergence
    fact = 0.5:0.1:1.5         # range of factors to mult with re
    conv = zeros(size(fact))   # stores number of iterations per grid block
 
    # derived numerics
    dx      = lx/nx
    xc      = LinRange(dx/2,lx-dx/2,nx)
    

    for ifact in eachindex(fact)
        # array initialisation
        C       = @. 1.0 + exp(-(xc-lx/4)^2) - xc/lx; C_i = copy(C)
        qx      = zeros(Float64, nx-1)
        
        # for convergence test
        re = 2π*fact[ifact]
        ρ  = (lx/(dc*re))^2
        dτ      = dx/sqrt(1/ρ)
        
        # iteration loop
        iter = 1; err = 2ϵtol

        while err >= ϵtol && iter <= maxiter
            qx         .-= dτ./(ρ*dc .+ dτ).*(qx .+ dc.*diff(C)./dx)
            C[2:end-1] .-= dτ.*diff(qx)./dx
            err = maximum(abs.(diff(dc.*diff(C)./dx)./dx))
            iter += 1
        end

        conv[ifact] = iter / nx
    end
    
    plot(fact, conv, xlabel="fact",ylabel="iter/nx",
        yscale=:log10,grid=true,markershape=:circle,markersize=10)

    savefig("convergence_study.png")

end



steady_diffusion_1D()

