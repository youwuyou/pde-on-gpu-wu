Lecture 6: GPU computing

- Code exercise 6.1
                    `l6_1-gpu-memcopy.ipynb`


- Code exercise 6.2
                    Task 1,2,4 ↔ `Pf_diffusion_2D_perf_gpu.jl`
                    Task 3     ↔ `readwrite_triad_KP_2D.jl`


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

### Task 1: GPU version of `Pf_diffusion_2D`

We used the CUDA.jl package and redefined the arrays using CUDA array as required in the task.


### Task 2: Testing for CPU and GPU

In this task we add an optional variable `test` which can either be set as `true` or `false` in the funciton signature. We enabled the testing using the `@testset` macro of the `Test.jl` and perform a similar test for 2D result `Pf` that we obtained as the function return value similarly as to the 1D case in the exercise 03.

- using `nx=ny=127`, `maxiter=50`

- indices to be tested are randomly sampled



### Task 3: assess the peak memory of Tesla P100 GPU

On Piz Daint we used the following parameters to find out the peak memory

```bash
nx = ny  = 16384
xthreads = 32 
```

We obtained the following result using `readwrite_triad_KP_2D.jl`

```bash
# add result

```


The found peak memory is ... GB/s, where the theoretical peak memory is 732 GB/s.






*NOTE:* for completeness the testing for the peak memory was also performed on the Tesla V100 GPU of the racklette clusters with the following output. The parameters used for testing (i.e `nx, ny, xthreads`) were obtained using previous testing.

```bash
# parameters used
nx = ny  = 32768
xthreads = 128
```
The found peak memory is $T_\text{peak} \approx 817 \, GB/s$ where the theoretical peak memory shall be 900GB/s according to the [vendor](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf)

```bash
# results
julia> include("readwrite_triad_KP_2D.jl")
Max thread number = 1024
(threads=(128, 1), blocks=(256, 32768)) T_tot = 816.2288879064365
(threads=(128, 2), blocks=(256, 16384)) T_tot = 817.0392300711534
(threads=(128, 4), blocks=(256, 8192))  T_tot = 811.8032292436367
(threads=(128, 8), blocks=(256, 4096))  T_tot = 806.3929573101956
```


### Task 4: Plotting memory throughput







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