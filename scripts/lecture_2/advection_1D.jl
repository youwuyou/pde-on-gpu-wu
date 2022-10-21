using Plots,Plots.Measures,Printf
default(size=(1200,400),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)

@views function advection_1D()
    # physics
    lx   = 20.0
    vx   = 1.0
    # numerics
    nx   = 200
    nvis = 2
    # derived numerics
    dx   = lx/nx
    dt   = dx/abs(vx)
    nt   = nx
    xc   = LinRange(dx/2,lx-dx/2,nx)
    # array initialisation
    C    = @. exp(-(xc-lx/4)^2); C_i = copy(C)
    # time loop
    ispath("anim")&&rm("anim",recursive=true);mkdir("anim");iframe = -1
    for it = 1:nt
        C[2:end  ] .-= dt.*max(vx,0.0).*diff(C)./dx
        C[1:end-1] .-= dt.*min(vx,0.0).*diff(C)./dx
        (it % (nt÷2) == 0) && (vx = -vx)
        ((it%nvis) == 0) && png(
        plot(xc,[C_i,C];xlims=(0,lx), ylims=(-0.1,1.1), 
                        xlabel="lx", ylabel="Concentration",
                        title="time = $(round(it*dt,digits=1))")
        ,@sprintf("anim/%04d.png",iframe+=1))
    end
end

advection_1D()

run(`ffmpeg -framerate 30 -i anim/%04d.png -c libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white" -y advection_1D.mp4`)
