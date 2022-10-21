using Plots,Plots.Measures,Printf
default(size=(1200,400),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)

@views function advection_diffusion_1D()
    # physics
    Pe   = 100.0 # the Peclet number Pe = lx*vx/dc
    lx   = 20.0
    vx   = 1.0
    ttot = 20.0
    # Derived physics
    dc   = lx*vx/Pe
    # numerics
    nx   = 200
    nvis = 2
    # derived numerics
    dx   = lx/nx
    dta  = dx/abs(vx)
    dtd  = dx^2/dc/2
    dt   = min(dtd, dta)
    nt   = cld(ttot, dt)
    xc   = LinRange(dx/2,lx-dx/2,nx)
    # array initialisation
    C    = @. exp(-(xc-lx/4)^2); C_i = copy(C)
    qx   = zeros(Float64, nx-1)
    # time loop
    for it = 1:nt
        qx          .= .-dc.*diff(C )./dx
        C[2:end-1] .-=   dt.*diff(qx)./dx
        C[2:end  ] .-=   dt.*max(vx,0.0).*diff(C)./dx
        C[1:end-1] .-=   dt.*min(vx,0.0).*diff(C)./dx
        (it % (nt÷2) == 0) && (vx = -vx)
        if it%nvis == 0
            display( plot(xc,[C_i,C];xlims=(0,lx), ylims=(-0.1,1.1), 
                          xlabel="lx", ylabel="Concentration",
                          title="time = $(round(it*dt,digits=1))") )
        end
    end
end

advection_diffusion_1D()
