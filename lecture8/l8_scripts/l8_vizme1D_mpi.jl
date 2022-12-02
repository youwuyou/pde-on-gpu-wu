# Visualisation script for the 1D MPI solver
using Plots, MAT

# basic information
nprocs = 8
nx    = 32                 # local number of grid points

@views function vizme1D_mpi(nprocs, nx)

    # local
    xc   = zeros(nx-2,nprocs)  # local coord array
    C    = zeros(nx-2,nprocs)  # local C array

    for ip = 1:nprocs
        file = matopen("mpi1D_out_C_$(ip-1).mat"); C_loc = read(file, "C"); close(file)
        file = matopen("mpi1D_out_xc_$(ip-1).mat"); xc_loc = read(file, "xc"); close(file)

        # local
        C[:,ip]  .= C_loc[2:end-1]
        xc[:,ip] .= xc_loc[2:end-1]

    end
    fontsize = 12

    # display(plot(C, legend=false, framestyle=:box, linewidth=3, xlims=(1, length(C)), ylims=(0, 1), xlabel="nx", title="diffusion 1D MPI", yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"), titlefontsize=fontsize, titlefont="Courier"))
   
    fontsize = 12
    plot(;legend=false, framestyle=:box, linewidth=3, xlims=(xc[1,1], xc[end,end]), ylims=(min.(xc), max.(C)), xlabel="nx", title="diffusion 1D MPI", yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"), titlefontsize=fontsize, titlefont="Courier")

    for ip = 1:nprocs
        plot!(xc[:,ip], C[:,ip], legend=false, linewidth=5, framestyle=:box, xlabel="Lx", ylabel="H")
    end
   
   
    savefig("diffusion1D_$(nprocs)_procs.png")
    return
end

vizme1D_mpi(nprocs, nx)
