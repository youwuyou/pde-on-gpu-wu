using BenchmarkTools, Printf, Plots

include("Pf_diffusion_2D_Teff.jl")
include("Pf_diffusion_2D_Perf.jl")
include("Pf_diffusion_2D_loop_fun.jl")



function Pf_diffusion_2D(nx_, ny_; bench= :btool, print= :false)
    # Numerics
    nx, ny  = nx_, ny_
    nt      = min(nx * ny,1e4)
    
    # array initialisation
    T_eff_1 = []     # Teff
    T_eff_2 = []     # Perf
    T_eff_3 = []     # loop_fun


    # compute function 1:  Pf_diffusion_2D_Teff.jl
    if bench == :loop
        # iteration loop
        t_tic = 0.
        for iter=1:nt
            if iter == 11 t_tic = Base.time(); end
            
            Pf_diffusion_2D_Teff(nx, ny)
        end
        t_toc = Base.time() - t_tic
        niter = nt

    elseif bench == :btool
        # array initialisation
        t_toc = @belapsed Pf_diffusion_2D_Teff($nx, $ny)
        niter = 1
    end

    A_eff = 2 * 3 * nx * ny * sizeof(Float64) /  1e9     # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                                    # Execution time per iteration [s]
    T_eff_1 = A_eff/t_it                                   # Effective memory throughput [GB/s]

    if print != false
        @printf("Array based:   Time = %1.3f sec, Teff = %1.3f, niter = %d \n", t_toc, T_eff_1, niter)
    end



    # compute function 2:  Pf_diffusion_2D_perf.jl
    if bench == :loop
        # iteration loop
        t_tic = 0.
        for iter=1:nt
            if iter == 11 t_tic = Base.time(); end
            
            Pf_diffusion_2D_Perf(nx, ny)
        end
        t_toc = Base.time() - t_tic
        niter = nt

    elseif bench == :btool
        # array initialisation
        t_toc = @belapsed Pf_diffusion_2D_Perf($nx, $ny)
        niter = 1
    end

    A_eff = 2 * 3 * nx * ny * sizeof(Float64) /  1e9     # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                                    # Execution time per iteration [s]
    T_eff_2 = A_eff/t_it                                   # Effective memory throughput [GB/s]

    if print != false
        @printf("Array based:   Time = %1.3f sec, Teff = %1.3f, niter = %d \n", t_toc, T_eff_1, niter)
    end


    # compute function 3:  Pf_diffusion_2D_loop_fun.jl
    if bench == :loop
        # iteration loop
        t_tic = 0.
        for iter=1:nt
            if iter == 11 t_tic = Base.time(); end
            
            Pf_diffusion_2D_loop_fun(nx, ny)
        end
        t_toc = Base.time() - t_tic
        niter = nt

    elseif bench == :btool
        # array initialisation
        t_toc = @belapsed Pf_diffusion_2D_loop_fun($nx, $ny)
        niter = 1
    end

    A_eff = 2 * 3 * nx * ny * sizeof(Float64) /  1e9     # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                                    # Execution time per iteration [s]
    T_eff_3 = A_eff/t_it                                   # Effective memory throughput [GB/s]

    if print != false
        @printf("Array based:   Time = %1.3f sec, Teff = %1.3f, niter = %d \n", t_toc, T_eff_1, niter)
    end


    return [T_eff_1, T_eff_2, T_eff_3]

end


# function for plotting the memory throughput
function mem_throughput(; start_at_two = false, bench= :btool)

    # optional parameters to have test size starting at nx == 2
    if start_at_two
        nx = ny = 2 * 2 .^ (0:11)
    else
        nx = ny = 16 * 2 .^ (1:8)
    end

    T_eff_1_list = []
    T_eff_2_list = []
    T_eff_3_list = []
    temp = []

    # general plotting feature
    plot(xlims=(nx[1], nx[end]), xscale= :log10, xlabel="nx", ylabel="Teff  [GB/s]" )
    
    if bench == :btool
        # using BenchmarkTools.jl
        for i in ny
            temp = Pf_diffusion_2D(i, i; bench= :btool, print= :false)
            append!(T_eff_1_list, temp[1])
            append!(T_eff_2_list, temp[2])
            append!(T_eff_3_list, temp[3])
        end
        
        plot(nx, [T_eff_1_list, T_eff_2_list, T_eff_3_list]; xscale= :log10, label=["T_eff" "scalar precomputations" "using compute functions"], title="Pf_diffusion_2D() using BenchmarkTools.jl")
    
    elseif bench == :loop
        # using self-implemented loop
        for i in ny
            temp = Pf_diffusion_2D(i, i; bench= :loop, print= :false)
            append!(T_eff_1_list, temp[1])
            append!(T_eff_2_list, temp[2])
            append!(T_eff_3_list, temp[3])
         end
         plot(nx, [T_eff_1_list, T_eff_2_list, T_eff_3_list]; clims=(10, nx[end]), xscale= :log10, label=["T_eff" "scalar precomputations" "using compute functions"], title="Pf_diffusion_2D() using loop")
    end

    # plot reference line for the peak memory
    tuple = findmax(T_eff_3_list)
    hline!([64],       color=:green,  label="Peak memory (vendor)"  )
    hline!([tuple[1]], color=:orange, label="Peak memory (measured)")
    

    # save figure
    savefig("Pf_diffusion_2D.png")

end

mem_throughput()

