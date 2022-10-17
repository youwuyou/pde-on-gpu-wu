using Plots,Plots.Measures,Printf
default(size=(1200,800),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)

@views function porous_convection_2D()
    # physics
    lx, ly    = 40.0, 20.0
    k_ηf      = 1.0
    
    # numerics
    nx      = 100
    ny      = ceil(Int, nx * ly/lx)    # making sure dx, dy cells are equal
    ϵtol    = 1e-8
    cfl     = 1.0/sqrt(2.1)
    re      = 2π
    
    # derived numerics
    dx, dy  = lx/nx, ly/ny
    max_length   = max(lx, ly)
    min_gridsize = min(dx, dy)
    max_number   = max(nx, ny)
    maxiter      = 100max_number
    
    ncheck  = ceil(Int,0.25max_number)
    
    xc      = LinRange(dx/2,lx-dx/2,nx)
    yc      = LinRange(-ly+dy/2,-dy/2,ny)

    θ_dτ    = max_length/re/cfl/min_gridsize
    β_dτ    = (re*k_ηf)/(cfl * min_gridsize * max_length)

    # array initialisation
    Pf       = @. exp(-(xc-lx/4)^2 - (yc'+ly/2)^2)
    qDx      = zeros(nx+1, ny)
    qDy      = zeros(nx, ny+1)

    # iteration loop
    iter = 1; err_Pf = 2ϵtol; iter_evo = Float64[]; err_evo = Float64[]


    # plotting
    #  clims=(0.0, 1.0) for adjusting the range of the color bar
    opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]),
            c=:turbo, xlabel="Lx", ylabel="Ly", title="iter/nx=$(round(iter/nx,sigdigits=3))")
	

    
    while err_Pf >= ϵtol && iter <= maxiter
        # qDx[2:end-1] .-= 1.0 / (1.0 + θ_dτ).*(qDx[2:end-1] .+ k_ηf.*diff(Pf)./dx)
        qDx[2:end-1, :] .-= (qDx[2:end-1, :] .+ k_ηf.*diff(Pf, dims=1)./dx) ./ (1.0 .+ θ_dτ)
        qDy[:, 2:end-1] .-= (qDy[:, 2:end-1] .+ k_ηf.*diff(Pf, dims=2)./dy .+ 1e-20) ./ (1.0 .+ θ_dτ)
        
        Pf              .-= (diff(qDx, dims=1)./ dx .+ diff(qDy, dims=2)./ dy) ./ β_dτ

        if iter % ncheck == 0
            err_Pf = maximum(abs.(diff(qDx, dims=1)./ dx .+ diff(qDy, dims=2)./ dy))
            
            push!(iter_evo,iter/nx)
            push!(err_evo,err_Pf)
            
            # p1 = heatmap(xc, yc, Pf'; xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), aspect_ratio=1,
            #          xlabel="lx",ylabel="ly",title="iter/nx=$(round(iter/nx,sigdigits=3))", c=:turbo)
            
	        p1 = heatmap(xc, yc, Pf'; opts...)

            p2 = plot(iter_evo,err_evo;xlabel="iter/nx",ylabel="err",
                      yscale=:log10,grid=true,markershape=:circle,markersize=10)
      
            display(plot(p1,p2;layout=(2,1)))
        end
        iter += 1
    end
end

porous_convection_2D()
