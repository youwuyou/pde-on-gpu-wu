Lecture 6: GPU computing

- Code exercise 6.1
                    `l6_1-gpu-memcopy.ipynb`


- Code exercise 6.2
                    Task 1 - 4 ↔ `Pf_diffusion_2D_perf_gpu.jl`


- Code exercise 6.3
                    Task 1 -  ↔ `L6TestingExercise`


## Code Exercise 6.1: Data transfer optimisations

### Task 1: 

### Task 2: 

### Task 3: 

### Task 4: 

### Task 5: 

### Task 6: 

### Task 7: 



## Code Exercise 6.2: Solving PDEs on GPUs 

### Task 1: 

### Task 2: 

### Task 3: 

### Task 4: 


## Code Exercise 6.3: Unit and reference tests 

```bash
 ╰─λ tree -a
.
├── Manifest.toml
├── Project.toml
├── scripts
│   ├── diffusion_1D_test.jl
│   ├── .diffusion_1D_test.jl.swn
│   ├── .diffusion_1D_test.jl.swo
│   └── .diffusion_1D_test.jl.swp
├── src
│   └── L6TestingExercise.jl
└── test
    ├── C_ref.jld
    ├── qx_ref.jld
    └── runtests.jl

```

Above we have the structure of the tests, where within the `/test` folder we have `C_ref.jld` and `qx_ref.jld`, which contain the data of the `C_ref` and `qx_ref` for the reference test.

In order to obtain these data, we used the package `JLD.jl` to preserve the data type using
```bash
julia> using JLD
julia> save("C_ref.jld", "data", C)
julia> save("qx_ref.jld", "data", qx)
```

Then we can load them in the tests as reference using
```bash
     C_ref = load("C_ref.jld")["data"]
     qx_ref = load("qx_ref.jld")["data"]
```


For the random sampling of the indices for the reference test, we utilized the `StatsBase.jl` package in order to obtain non-repetitive indices within a certaian range. 