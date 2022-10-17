using Plots,Plots.Measures,Printf
default(size=(1200,800),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)

@views function porous_convection_2D()
    # physics
    lx, ly  = 40.0, 20.0
    k_ηf    = 1.0
    re      = 2π


    # numerics
    nx      = 100
    ny      = ceil(Int, ly * nx / lx)
    dx, dy  = lx/nx, ly/ny
    nt      = 10
   
    # extrema
    max_num  = max(nx, ny)
    min_step = min(dx, dy)
    max_len  = max(lx, ly)


    ϵtol    = 1e-8
    maxiter = 100max_num
    ncheck  = ceil(Int,0.25max_num)
    cfl     = 1.0 / sqrt(2.1)

    # derived numerics
    xc      = LinRange(dx/2,lx-dx/2,nx) # 100
    yc      = LinRange(dy/2,ly-dy/2,ny) # 50
    θ_dτ    = max_len /re /cfl /min_step
    β_dτ    = (re * k_ηf) / (cfl * min_step * max_len)


    # array initialisation
    Pf       = @. exp(-(xc-lx/4)^2 -(yc'-ly/4)^2) # 100 x 50
    qDx      = zeros(Float64, nx + 1, ny) # 101 x 50
    qDy      = zeros(Float64, nx, ny + 1) # 100 x 51
    
    
    # plotting option
    opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), 
    c=:turbo, xlabel="Lx", ylabel="Ly")
    
    
    # physical time loop
    anim = @animate for it = 1:nt
        
        # iteration loop
        iter = 1; err_Pf = 2ϵtol; iter_evo = Float64[]; err_Pf_evo = Float64[]
        
        while err_Pf >= ϵtol && iter <= maxiter
            qDx[2:end-1, :]  .-= (qDx[2:end-1, :] .+ k_ηf.*diff(Pf, dims=1)./dx) ./ (1.0 + θ_dτ)
            qDy[:, 2:end-1]  .-= (qDy[:, 2:end-1] .+ k_ηf.*diff(Pf, dims=2)./dy) ./ (1.0 + θ_dτ)

            Pf   .-= ( diff(qDx, dims=1)./dx  +  diff(qDy, dims=2) ./ dy) ./ β_dτ 

            if iter%ncheck == 0
                err_Pf = maximum(abs.(diff(qDx, dims=1)./dx  .+  diff(qDy, dims=2) ./ dy))
                
                push!(iter_evo, iter/nx)
                push!(err_Pf_evo,err_Pf)
            end
            
            iter += 1

        end
        
        p1 = heatmap(xc, yc, Pf'; title="time = $(round(it,sigdigits=3))", opts...)
        
        display(p1)
        
        @printf("it = %d, iter/nx=%.1f, err_Pf=%1.3e\n",it,iter/nx,err_Pf)
    end
    # file I/O
    gif(anim, "transient_momentum_eq_pressure_2D.gif", fps = 2)
end

porous_convection_2D()
