using BenchmarkTools, Printf, Plots


# C2 = C + A
# compute function 1:  array programming based
function compute_ap!(C, C2, A)
    C2 .= C .+ A
    return C2
end


# compute function 2: kernel programming based
function compute_kp!(C, C2, A)
    
    nx, ny = size(A)

    for iy = 1:ny
        for ix = 1:nx
            C2[ix,iy] = C[ix,iy] + A[ix,iy]
        end
    end 

    return C2
end


function memcopy(nx_, ny_; bench= :btool, print= :false)
    # Numerics
    nx, ny  = nx_, ny_
    nt      = min(nx * ny,1e4)

    # array initialisation
    C       = rand(Float64, nx, ny)
    C2      = copy(C)
    A       = copy(C)

    # call compute function 1
    if bench == :loop
        # iteration loop
        t_tic = 0.
        for iter=1:nt
            if iter == 11 t_tic = Base.time(); end
            
            compute_ap!(C, C2, A);
        end
        t_toc = Base.time() - t_tic
        niter = nt

    elseif bench == :btool
        # array initialisation
        t_toc = @belapsed compute_ap!($C, $C2, $A);
        niter = 1
    end

    # not using 3 * nx * ny due to the hardware optimization performed
    A_eff = nx * ny * sizeof(eltype(A)) /  1e9         # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                                    # Execution time per iteration [s]
    T_eff_1 = A_eff/t_it                                   # Effective memory throughput [GB/s]

    if print != false
        @printf("Array based:   Time = %1.3f sec, Teff = %1.3f, niter = %d \n", t_toc, T_eff_1, niter)
    end

    # call compute function 2
    if bench == :loop
        # iteration loop
        t_tic = 0.
        for iter=1:nt
            if iter == 11 t_tic = Base.time(); end
            compute_kp!(C, C2, A);
        end
        t_toc = Base.time() - t_tic

    elseif bench == :btool
        t_toc = @belapsed compute_kp!($C, $C2, $A);
    end

    # not using 3 * nx * ny due to the hardware optimization performed
    A_eff =  nx * ny * sizeof(eltype(C)) /  1e9         # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter           # Execution time per iteration [s]
    T_eff_2 = A_eff/t_it         # Effective memory throughput [GB/s]

    if print != false
        @printf("Kernel based:  Time = %1.3f sec, Teff = %1.3f, niter = %d \n", t_toc, T_eff_2, niter)
    end

    return [T_eff_1, T_eff_2]

end


# function for plotting the memory throughput
function mem_throughput(; start_at_two = true, bench= :loop)

    # optional parameters to have test size starting at nx == 2
    if start_at_two
            nx = ny = 2 * 2 .^ (0:11)
    else
        nx = ny = 16 * 2 .^ (1:8)
    end


    T_eff_1_list = []
    T_eff_2_list = []
    temp = []

    # general plotting feature
    plot(xlims=(nx[1], nx[end]), xscale= :log10, xlabel="nx", ylabel="Teff  [GB/s]" )
    
    if bench == :btool
        for i in ny
                temp = memcopy(i, i; bench= :btool, print= :false)
                append!(T_eff_1_list, temp[1])
                append!(T_eff_2_list, temp[2])
        end
        
        plot(nx, [T_eff_1_list, T_eff_2_list]; xscale= :log10, label=["Array-based" "Kernel-based"], title="memcopy() using BenchmarkTools.jl")
    elseif bench == :loop
        for i in ny
            temp = memcopy(i, i; bench= :loop, print= :false)
            append!(T_eff_1_list, temp[1])
            append!(T_eff_2_list, temp[2])
         end
         plot(nx, [T_eff_1_list, T_eff_2_list]; xscale= :log10, label=["Array-based" "Kernel-based"], title="memcopy() using loop")
    end

    # plot reference line for the peak memory
    tuple = findmax(T_eff_1_list)
    hline!([64], color=:green, label="Peak memory (vendor)")
    hline!([tuple[1]], color=:orange, label="Peak memory (measured)")
    
    # save figure
    savefig("memcopy.png")

end

mem_throughput()

