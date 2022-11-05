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

- `Pf_diffusion_2D_perf_xpu.jl` <-> complete the script from class using `@parallel_indices` approach

- `PorousConvection_2D_xpu.jl`  <-> edited from previous script with `@parallel` approach preferred



The `PorousConvection_2D_xpu.jl` is modified from the previous exercise `porous_convection_implicit_2D.jl` of week 04. The difference between it and the this week's script is that we slightly changed the parameters (eg. Raynolds number). The differences are commented within the script.

In our implementation from the serial code to the parallelized code, we preferred the approach of `@parallel` whenever possible.


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

TODO: add parameters on the function signature for reproducing ex03/04 differently

TODO: add final 2D animation showing evolution of temperature with velocity quiver







## Porous convection 3D

- `Pf_diffusion_3D_xpu.jl`      <-> complete the script from class using `@parallel` approach
                                <-> using ParallelStencil.FiniteDifferences3D submodule

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