using Plots, JLD



# obtain the data
nx = load("nx.jld")["data"]
T_eff_list = load("T_eff_list.jld")["data"]


# plot
plot(nx, T_eff_list;  label=["Diffusion 2D"], title="Pf_diffusion_2D_gpu() using loop")


# plot reference line for the peak memory
tuple = findmax(T_eff_list)
hline!([765.2], color=:green, label="Peak memory GPU (Task 3)")
hline!([tuple[1]], color=:orange, label="Peak memory GPU(Task 4)")

# save figure
savefig("plot.png")