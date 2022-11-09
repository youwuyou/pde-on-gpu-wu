# PorousConvection

[![Build Status](https://github.com/youwuyou/pde-on-gpu-wu/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/youwuyou/pde-on-gpu-wu/actions/workflows/CI.yml?query=branch%3Amain)


Documentation of the first course project of course 101-0250-00 from HS22 week 07 - 09.

## Structure

```bash
PorousConvection
├── LICENSE
├── Manifest.toml
├── Project.toml
├── README.md
├── scripts       # contains scripts for 2D & 3D porous convection applications
├── src
└── test
```


## Porous convection 2D

- `Pf_diffusion_2D_xpu.jl`      <-> complete the script from class using `@parallel` approach
                                <-> Performance on V100 GPU `Time = 0.019 sec, T_eff = 320.776 GB/s`

- `Pf_diffusion_2D_perf_xpu.jl` <-> complete the script from class using `@parallel_indices` approach
                                <-> Performance on V100 GPU `Time = 0.019 sec, T_eff = 325.022 GB/s`

- `PorousConvection_2D_xpu.jl`  <-> edited from previous script with `@parallel` approach preferred



The `PorousConvection_2D_xpu.jl` is modified from the previous exercise `porous_convection_implicit_2D.jl` of week 04. The difference between it and the this week's script is that we slightly changed the parameters (eg. Raynolds number). The differences are commented within the script.

In our implementation from the serial code to the parallelized code, we preferred the approach of `@parallel` whenever possible.


NOTE: The code is largely parallelized on the GPU but with the exception for the computation of `dTdt`, I am not so sure how to put it into a kernel function yet, any help would be appreciated! :)



---

### Result


#### Lecture 7 - Task 1.3: Small case

Using `ny = 63` , `nt = 500`, we obtain the identical plot for the final state as given in the task description

 <img src="./docs/porous2D_0025.png" width="60%">



####  Lecture 7 - Task 1.4: Large case


With the following parameters, we run our code on Piz Daint using one Tesla P100 GPU, where we expected the run to take 1-2 hours of time.

```bash
Ra      = 1000
# [...]
nx,ny   = 511,1023
nt      = 4000
ϵtol    = 1e-6
nvis    = 50
ncheck  = ceil(2max(nx,ny))
```

*Memory throughput:*  

```bash
julia> include("PorousConvection_2D_xpu.jl")
Animation directory: viz_out/
Time = 7774.123 sec, T_eff = 115.558 GB/s 
```


TODO: add final 2D animation showing evolution of temperature with velocity quiver







## Porous convection 3D

- `Pf_diffusion_3D_xpu.jl`      <-> complete the script from class using `@parallel` approach
                                <-> using ParallelStencil.FiniteDifferences3D submodule
                                <-> `Time = 14.278 sec, T_eff = 293.071 GB/s` on Tesla V100 GPU

- `PorousConvection_3D_xpu.jl`  <-> edited from `PorousConvection_2D_xpu.jl` script with `@parallel` approach preferred


With the following parameters, we run our code on Piz Daint using one Tesla P100 GPU, where we expected the run to take 2 hours of time.

```bash
Ra       = 1000
# [...]
nx,ny,nz = 255,127,127
nt       = 2000
ϵtol     = 1e-6
nvis     = 50
ncheck   = ceil(2max(nx,ny,nz))
```

TODO: add parameters on the function signature for reproducing ex03/04 differently

TODO: add final 3D animation showing evolution of temperature with GLMakie



#   Troubleshooting

- for the computation of the diffusion equation using 1 GPU, it seems like the performance is not improved by defining a kernel function for the error tracking of `r_Pf`

The following function definition was added

```julia

@parallel function calc_r_Pf!(r_Pf, qDx, qDy, _dx, _dy)
    @all(r_Pf)  = @d_xa(qDx) * _dx + @d_ya(qDy) * _dy
    return nothing
end

```

And it is called within do_check 

```julia
            @parallel calc_r_Pf!(r_Pf, qDx, qDy, _dx, _dy)
```

Which caused the performance to drop from `Time = 0.019 sec, T_eff = 327.480 GB/s ` to `Time = 0.103 sec, T_eff = 59.444 GB/s` on a single Nvidia Tesla V100 GPU. 