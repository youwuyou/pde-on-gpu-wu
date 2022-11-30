# Linear 1D diffusion with n fake mpi processes
using Plots, Test

@views function diffusion_1D_nprocs(; do_visu=true)
    # Physics
    lx  = 10.0
    D   = 1.0
    nt  = 200

    # Numerics
    np  = 4             # number of procs
    nx  = 32            # local number of grid points
    
    # Derived numerics
    nxg = (nx-2)*np+2   # global number of grid points
    dxg = lx/nxg        # dx for global grid
    dt  = dxg^2/D/2.1
    
    # Array allocation
    x   = zeros(nx,np)  # local coord array
    C   = zeros(nx,np)  # local C array
    xt  = zeros(nxg)    # global coord array
    Ct  = zeros(nxg)    # global initial C array
    Cg  = zeros(nxg)    # global C array
    
    # Initial condition
    # shall spann over the local domain
    # xc      = LinRange(dx/2,lx-dx/2,nx)

    # for each copy of the domain that gets assigned to a process
    for ip = 1:np

        # initial condition for each copy
        for ix = 1:nx
            x[ix,ip] =  dxg / 2. + dxg * ((ix-1) + (ip -1) * (nx-2)) - lx/2 # 
            C[ix,ip] = exp(-x[ix,ip]^2)
        end
        i1 = 1 + (ip-1)*(nx-2)
        xt[i1:i1+nx-2] .= x[1:end-1,ip]; if (ip==np) xt[i1+nx-1] = x[end,ip] end
        Ct[i1:i1+nx-2] .= C[1:end-1,ip]; if (ip==np) Ct[i1+nx-1] = C[end,ip] end
    end
    
    # Time loop
    anim = @animate for it = 1:nt

        # compute for each process
        for ip = 1:np # compute physics locally
            C[2:end-1,ip] .= C[2:end-1,ip] .+ dt*D*diff(diff(C[:,ip])/dxg)/dxg
        end

        # update local boundaries
        for ip = 1:np-1 
            # the first entry corresponds to the 2 procs case's halo update
            # the second entry corresponds to the choice of CL, CR
            C[end, ip] = C[2, ip+1] 
            C[1, ip+1] = C[end-1, ip]
        end
        
        # put local pics back into the global array
        for ip = 1:np
            i1 = 1 + (ip-1)*(nx-2)
            Cg[i1:i1+nx-2] .= C[1:end-1,ip]
        end
        
        # Visualise
        if do_visu && it % 5 == 0
            fontsize = 12
            # plotting the initial condition
            plot(xt, Ct, legend=false, linewidth=1, markershape=:circle, markersize=3, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"), titlefontsize=fontsize, titlefont="Courier")
           
           # plotting the global updated array
            display(plot!(xt, Cg, legend=false, linewidth=3, framestyle=:box, xlabel="Lx", ylabel="H", title="diffusion (it=$(it))"))
            
            # 
            for ip = 1:np
                display(plot!(x[:,ip], C[:,ip], legend=false, linewidth=5, framestyle=:box, xlabel="Lx", ylabel="H", title="diffusion (it=$(it))"))
            end
        end
        
    end

    if do_visu
        gif(anim, "fake_parallelisation_gaussian.gif", fps = 50)
    end
    

    return
end


if isinteractive()
    diffusion_1D_nprocs(; do_visu=true)
end