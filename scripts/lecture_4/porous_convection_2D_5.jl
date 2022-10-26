using Plots,Plots.Measures,Printf
default(size=(1200,800),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)

@views avx(A) = 0.5.*(A[1:end-1,:].+A[2:end,:])
@views avy(A) = 0.5.*(A[:,1:end-1].+A[:,2:end])

@views function porous_convection_2D()
    # physics
    lx,ly     = 40.0,20.0
    k_ηf      = 1.0
    αρgx,αρgy = 0.0,1.0
    αρg       = sqrt(αρgx^2+αρgy^2)
    ΔT        = 200.0
    ϕ         = 0.1
    Ra        = 1000.0
    λ_ρCp     = 1/Ra*(αρg*k_ηf*ΔT*ly/ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
    # numerics
    nx        = 127
    ny        = ceil(Int,nx*ly/lx)
    nt        = 500
    ϵtol      = 1e-8
    maxiter   = 10max(nx,ny)
    cfl       = 1.0/sqrt(2.1)
    re        = 2π
    ncheck    = ceil(Int,0.25max(nx,ny))
    nvis      = 5
    # derived numerics
    dx,dy     = lx/nx,ly/ny
    xc,yc     = LinRange(-lx/2+dx/2,lx/2-dx/2,nx),LinRange(-ly+dy/2,-dy/2,ny)
    θ_dτ      = max(lx,ly)/re/cfl/min(dx,dy)
    β_dτ      = (re*k_ηf)/(cfl*min(dx,dy)*max(lx,ly))
    dt_diff   = min(dx,dy)^2/λ_ρCp/4.1
    # array initialisation
    Pf        = zeros(nx,ny)
    r_Pf      = zeros(nx,ny)
    qDx,qDy   = zeros(nx+1,ny),zeros(nx,ny+1)
    qDxc,qDyc = zeros(nx,ny),zeros(nx,ny)
    qDmag     = zeros(nx,ny)  
    T         = @. ΔT*exp(-xc^2 - (yc'+ly/2)^2); T[:,1] .= ΔT/2; T[:,end] .= -ΔT/2
    # vis
    st        = ceil(Int,nx/25)
    Xc, Yc    = [x for x=xc, y=yc], [y for x=xc,y=yc]
    Xp, Yp    = Xc[1:st:end,1:st:end], Yc[1:st:end,1:st:end]
    # ispath("anim")&&rm("anim",recursive=true);mkdir("anim");iframe = 0
    # action
    for it = 1:nt
        # hydro
        iter = 1; err_Pf = 2ϵtol
        while err_Pf >= ϵtol && iter <= maxiter
            qDx[2:end-1,:] .-= (qDx[2:end-1,:] .+ k_ηf.*(diff(Pf,dims=1)./dx .- αρgx.*avx(T)))./(1.0 + θ_dτ)
            qDy[:,2:end-1] .-= (qDy[:,2:end-1] .+ k_ηf.*(diff(Pf,dims=2)./dy .- αρgy.*avy(T)))./(1.0 + θ_dτ)
            Pf             .-= (diff(qDx,dims=1)./dx .+ diff(qDy,dims=2)./dy)./β_dτ
            if iter%ncheck == 0
                r_Pf  .= diff(qDx,dims=1)./dx .+ diff(qDy,dims=2)./dy
                err_Pf = maximum(abs.(r_Pf))
                @printf("  iter/nx=%.1f, err_Pf=%1.3e\n",iter/nx,err_Pf)
            end
            iter += 1
        end
        @printf("it = %d, iter/nx=%.1f, err_Pf=%1.3e\n",it,iter/nx,err_Pf)
        # time step
        dt_adv = ϕ*min(dx/maximum(abs.(qDx)), dy/maximum(abs.(qDy)))/2.1
        dt     = min(dt_diff,dt_adv)
        # thermo
        T[2:end-1,2:end-1] .+= dt.*λ_ρCp.*(diff(diff(T[:,2:end-1],dims=1)./dx,dims=1)./dx .+
                                           diff(diff(T[2:end-1,:],dims=2)./dy,dims=2)./dy)
        T[2:end-1,2:end-1] .-= dt./ϕ.*(max.(qDx[2:end-2,2:end-1],0.0).*diff(T[1:end-1,2:end-1],dims=1)./dx .+
                                       min.(qDx[3:end-1,2:end-1],0.0).*diff(T[2:end  ,2:end-1],dims=1)./dx .+
                                       max.(qDy[2:end-1,2:end-2],0.0).*diff(T[2:end-1,1:end-1],dims=2)./dy .+
                                       min.(qDy[2:end-1,3:end-1],0.0).*diff(T[2:end-1,2:end  ],dims=2)./dy)
        T[[1,end],:] .= T[[2,end-1],:]
        # visualisation
        if it % nvis == 0
            qDxc  .= avx(qDx)
            qDyc  .= avy(qDy)
            qDmag .= sqrt.(qDxc.^2 .+ qDyc.^2)
            qDxc  ./= qDmag
            qDyc  ./= qDmag
            qDx_p = qDxc[1:st:end,1:st:end]
            qDy_p = qDyc[1:st:end,1:st:end]
            p1=heatmap(xc,yc,T';xlims=(xc[1],xc[end]),ylims=(yc[1],yc[end]),aspect_ratio=1,c=:turbo)
            display(quiver!(Xp[:], Yp[:], quiver=(qDx_p[:], qDy_p[:]), lw=0.5, c=:black))
            # png(p1,@sprintf("anim/%04d.png",iframe)); iframe += 1
        end
    end
end

porous_convection_2D()

# run(`ffmpeg -framerate 2 -i anim/%04d.png -c libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white" -y porous_convection_2D_ex1.mp4`)
