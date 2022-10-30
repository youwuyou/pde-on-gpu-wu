Lecture 6: GPU computing

- Code exercise 6.1
                    `l6_1-gpu-memcopy.ipynb`


- Code exercise 6.2
                    Task 1,2   ↔ `Pf_diffusion_2D_perf_gpu.jl`
                    Task 3     ↔ `readwrite_triad_KP_2D.jl`
                    Task 4     ↔ `benchmark.jl`


- Code exercise 6.3
                    Task 1 -  ↔ `L6TestingExercise`


## Code Exercise 6.1: Data transfer optimisations

### Task 1:  unoptimized array programming function

- benchmarked the array programming based 2D diffusion

### Task 2 & 3: single array programming statement

- mathematically transformed the formulation of the temperature update

- removed redundant arrays both in the main as well as in the stepping function

- changed the stepping to a function with only a single statement

- benchmarked the single-statement function

*Expected speedup:* 2

*obtained speedup:* 0.65


=> somehow the speedup measured here is wrong, we expected a speedup of 2 but the measured time in task 1 seems to be pretty good. Assumption is that the array size we tried is probably not large enough and the synchronization time needed for task 2's function brings the overhead.

### Task 4 & 5: CUDA kernel based programming

- changed the fused kernel function by using CUDA arrays

- performed manual range checking for the kernel function

*Expected speedup:* 5

*obtained speedup:* 4.75

=> The implementation shall be correct.


### Task 6: Using new metrics

- used the new metrics "Effective memory throughput" for the measurement of the speedup in task 7

=> obtained speedup for both task 3 and 5 are not as expected (proportional)



### Task 7: Report the measured $T_\text{eff}$ and $T_\text{Peak}

Given the information from the vendor that $T_\text{Peak} = 732 GB/s$, we obtained $T_\text{eff} = 704.8 GB/s$

The resulting quotient is:

```bash
T_eff / T_peak = 0.9628638014096739
0.9628638014096739
```

*Result:*  As seen in the result, we are already pretty close to the peak memory but there is still a 8%-off difference. The peak memory is rather theoretical and cannot be reached that closely in the reality. 


## Code Exercise 6.2: Solving PDEs on GPUs 

### Task 1: GPU version of `Pf_diffusion_2D`

We used the CUDA.jl package and redefined the arrays using CUDA array as required in the task.


### Task 2: Testing for CPU and GPU

In this task we add an optional variable `test` which can either be set as `true` or `false` in the funciton signature. We enabled the testing using the `@testset` macro of the `Test.jl` and perform a similar test for 2D result `Pf` that we obtained as the function return value similarly as to the 1D case in the exercise 03.

- using `nx=ny=127`, `maxiter=50`

- indices to be tested are randomly sampled

#### Running the test:

- before running the `runtests.jl`, we need to run the Pf_diffusion_2D scripts using CPU and GPU with `test=true` enabled, at least once in order to have files for the test to compare with

- run the test using `include("runtests.jl")` within the test folder

#### Result:

```
julia> include("runtests.jl")
Test Summary:                | Pass  Total
reference test: 2D diffusion |   20     20
```


#### Debugging:

For the CUDA programming, another interesting obeservation when choosing the block size leads to the conclusion that the blocksize should be carefully chosen.

At the beginning the block size was chosen as followed using the similar definition as in the lecture `blocks  = (nx÷threads[1], ny÷threads[2])`, this leads to results where certain regular blocks of the resulting matrix have significant.

```
julia> include("Pf_diffusion_2D_perf_gpu.jl")
(3, 7)127×127 CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}:
 2.30637e-33  1.32725e-32  8.4023e-32   5.15047e-31  3.04815e-30  1.74167e-29  9.60854e-29  5.11852e-28  …  1.81281e-78  1.04651e-79  5.74898e-81  3.00537e-82  1.49508e-83  7.07767e-85  3.18842e-86
 1.3274e-32   7.50885e-32  4.68058e-31  2.82547e-30  1.64677e-29  9.26671e-29  5.03504e-28  2.64015e-27     4.0241e-77   2.32305e-78  1.27616e-79  6.67136e-81  3.3188e-82   1.57111e-83  7.07767e-85
 8.40594e-32  4.68236e-31  2.87853e-30  1.71401e-29  9.85444e-29  5.47053e-28  2.92984e-27  1.51584e-26     8.50047e-76  4.90719e-77  2.69576e-78  1.40925e-79  7.0106e-81   3.3188e-82   1.49508e-83
 5.15663e-31  2.82922e-30  1.71583e-29  1.00811e-28  5.71944e-28  3.12939e-27  1.65419e-26  8.44465e-26     1.70874e-74  9.86429e-76  5.41894e-77  2.83284e-78  1.40925e-79  6.67136e-81  3.00537e-82
 3.05671e-30  1.65225e-29  9.88754e-29  5.73356e-28  3.20527e-27  1.7312e-26   9.03178e-26  4.56237e-25     3.26865e-73  1.88694e-74  1.03659e-75  5.41894e-77  2.69576e-78  1.27616e-79  5.74898e-81
 1.7519e-29   9.33203e-29  5.51213e-28  3.14854e-27  1.73772e-26  9.26631e-26  4.78715e-25  2.38308e-24  …  5.95004e-72  3.43487e-73  1.88694e-74  9.86429e-76  4.90719e-77  2.32305e-78  1.04651e-79
 9.71568e-29  5.10184e-28  2.96689e-27  1.67323e-26  9.12114e-26  4.81991e-25  2.45194e-24  1.20306e-23     1.0307e-70   5.95004e-72  3.26865e-73  1.70874e-74  8.50047e-76  4.0241e-77   1.81281e-78
 5.21784e-28  2.69267e-27  1.54585e-26  8.61466e-26  4.65676e-25  2.42067e-24  1.21274e-23  5.89706e-23     1.69903e-69  9.8082e-71   5.38813e-72  2.81673e-73  1.40124e-74  6.63342e-76  2.98829e-77
 ⋮                                                                ⋮                                      ⋱  ⋮                                                                ⋮            
 2.98829e-77  6.63342e-76  1.40124e-74  2.81673e-73  5.38813e-72  9.8082e-71   1.69903e-69  2.80072e-68     1.69903e-69  9.8082e-71   5.38813e-72  2.81673e-73  1.40124e-74  6.63342e-76  2.98829e-77
 1.81281e-78  4.0241e-77   8.50047e-76  1.70874e-74  3.26865e-73  5.95004e-72  1.0307e-70   1.69903e-69  …  1.0307e-70   5.95004e-72  3.26865e-73  1.70874e-74  8.50047e-76  4.0241e-77   1.81281e-78
 1.04651e-79  2.32305e-78  4.90719e-77  9.86429e-76  1.88694e-74  3.43487e-73  5.95004e-72  9.8082e-71      5.95004e-72  3.43487e-73  1.88694e-74  9.86429e-76  4.90719e-77  2.32305e-78  1.04651e-79
 5.74898e-81  1.27616e-79  2.69576e-78  5.41894e-77  1.03659e-75  1.88694e-74  3.26865e-73  5.38813e-72     3.26865e-73  1.88694e-74  1.03659e-75  5.41894e-77  2.69576e-78  1.27616e-79  5.74898e-81
 3.00537e-82  6.67136e-81  1.40925e-79  2.83284e-78  5.41894e-77  9.86429e-76  1.70874e-74  2.81673e-73     1.70874e-74  9.86429e-76  5.41894e-77  2.83284e-78  1.40925e-79  6.67136e-81  3.00537e-82
 1.49508e-83  3.3188e-82   7.0106e-81   1.40925e-79  2.69576e-78  4.90719e-77  8.50047e-76  1.40124e-74     8.50047e-76  4.90719e-77  2.69576e-78  1.40925e-79  7.0106e-81   3.3188e-82   1.49508e-83
 7.07767e-85  1.57111e-83  3.3188e-82   6.67136e-81  1.27616e-79  2.32305e-78  4.0241e-77   6.63342e-76  …  4.0241e-77   2.32305e-78  1.27616e-79  6.67136e-81  3.3188e-82   1.57111e-83  7.07767e-85
 3.18842e-86  7.07767e-85  1.49508e-83  3.00537e-82  5.74898e-81  1.04651e-79  1.81281e-78  2.98829e-77     1.81281e-78  1.04651e-79  5.74898e-81  3.00537e-82  1.49508e-83  7.07767e-85  3.18842e-86
```

Only using the following definition, we were able to bring the correct result back.

`blocks  = (Int(ceil(nx/threads[1])), Int(ceil(ny/threads[2])))`

```
julia> include("Pf_diffusion_2D_perf_gpu.jl")
(4, 8)127×127 CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}:
 2.30637e-33  1.32725e-32  8.40207e-32  5.15026e-31  3.04776e-30  1.74111e-29  9.60576e-29  5.1172e-28   …  9.06952e-29  1.5571e-29   2.74016e-30  4.90961e-31  8.33049e-32  1.3156e-32   1.95687e-33
 1.3274e-32   7.50842e-32  4.68018e-31  2.82476e-30  1.64578e-29  9.26183e-29  5.03263e-28  2.64064e-27     4.76236e-28  8.29198e-29  1.47668e-29  2.6867e-30   4.63753e-31  7.44102e-32  1.13103e-32
 8.40519e-32  4.68165e-31  2.87734e-30  1.71234e-29  9.84575e-29  5.4661e-28   2.9303e-27   1.5172e-26      2.77877e-27  4.89963e-28  8.81766e-29  1.62615e-29  2.85028e-30  4.63896e-31  7.18972e-32
 5.15531e-31  2.82739e-30  1.71325e-29  1.00669e-28  5.71197e-28  3.12947e-27  1.65608e-26  8.4664e-26      1.57369e-26  2.8093e-27   5.1074e-28   9.54276e-29  1.69791e-29  2.80234e-30  4.42705e-31
 3.05392e-30  1.64857e-29  9.86618e-29  5.72195e-28  3.20443e-27  1.7335e-26   9.06094e-26  4.57271e-25     8.62505e-26  1.55862e-26  2.86183e-27  5.41515e-28  9.77781e-29  1.63616e-29  2.63391e-30
 1.74674e-29  9.30244e-29  5.49547e-28  3.14609e-27  1.74013e-26  9.30165e-26  4.79928e-25  2.38264e-24  …  4.57783e-25  8.37316e-26  1.55241e-26  2.97371e-27  5.44729e-28  9.239e-29    1.51502e-29
 9.67212e-29  5.07976e-28  2.96216e-27  1.67526e-26  9.15954e-26  4.8322e-25   2.45121e-24  1.20118e-23     2.35383e-24  4.35785e-25  8.15756e-26  1.58149e-26  2.93822e-27  5.04981e-28  8.4315e-29
 5.18645e-28  2.68518e-27  1.54694e-26  8.6515e-26   4.6669e-25   2.41913e-24  1.21109e-23  5.88222e-23     1.17221e-23  2.19732e-24  4.15327e-25  8.14874e-26  1.53536e-26  2.67353e-27  4.54371e-28
 ⋮                                                                ⋮                                      ⋱  ⋮                                                                ⋮            
 5.08437e-28  2.62296e-27  1.50899e-26  8.38715e-26  4.48201e-25  2.32219e-24  1.16195e-23  5.60557e-23     1.15075e-23  2.29188e-24  4.43074e-25  8.19422e-26  1.48735e-26  2.54982e-27  1.09534e-28
 9.60156e-29  5.00272e-28  2.91158e-27  1.64375e-26  8.94043e-26  4.68274e-25  2.37404e-24  1.16209e-23  …  2.34838e-24  4.63688e-25  8.77991e-26  1.62777e-26  2.90293e-27  4.90407e-28  2.00707e-29
 1.74077e-29  9.25405e-29  5.43735e-28  3.10779e-27  1.71576e-26  9.13277e-26  4.68664e-25  2.32541e-24     4.64988e-25  9.01013e-26  1.70455e-26  3.10251e-27  5.45053e-28  9.03528e-29  3.55034e-30
 3.04705e-30  1.64514e-29  9.832e-29    5.68132e-28  3.1775e-27   1.71599e-26  8.94017e-26  4.4937e-25      8.85427e-26  1.70884e-26  3.17384e-27  5.69203e-28  9.82629e-29  1.60714e-29  6.06249e-31
 5.14918e-31  2.82347e-30  1.71113e-29  1.00444e-28  5.68572e-28  3.11189e-27  1.64441e-26  8.38644e-26     1.64028e-26  3.10958e-27  5.69211e-28  1.00443e-28  1.71019e-29  2.75976e-30  9.99281e-32
 8.40086e-32  4.67823e-31  2.875e-30    1.71113e-29  9.832e-29    5.45043e-28  2.91966e-27  1.51001e-26     2.91835e-27  5.45386e-28  9.832e-29    1.71111e-29  2.87349e-30  4.57513e-31  1.59058e-32
 1.32709e-32  7.50624e-32  4.67823e-31  2.82347e-30  1.64514e-29  9.25405e-29  5.024e-28    2.63469e-27  …  5.02565e-28  9.25405e-29  1.64514e-29  2.82343e-30  4.67592e-31  7.34578e-32  2.52763e-33
 2.3062e-33   1.32709e-32  8.40086e-32  5.14918e-31  3.04705e-30  1.74077e-29  9.60156e-29  5.11262e-28     9.60156e-29  1.74077e-29  3.04705e-30  5.14913e-31  8.3975e-32   1.30674e-32  1.34927e-33
```



### Task 3: assess the peak memory of Tesla P100 GPU

On Piz Daint we used the following parameters to find out the peak memory

```bash
nx = ny  = 16384
xthreads = 32 
```

We obtained the following result using `readwrite_triad_KP_2D.jl`

```bash
julia> include("readwrite_triad_KP_2D.jl")
Max thread number = 1024
(threads=(32, 1), blocks=(512, 16384)) T_tot = 440.1775197357152
(threads=(32, 2), blocks=(512, 8192)) T_tot = 763.3952981601233
(threads=(32, 4), blocks=(512, 4096)) T_tot = 765.1535032539313
(threads=(32, 8), blocks=(512, 2048)) T_tot = 763.4057914531317
(threads=(32, 16), blocks=(512, 1024)) T_tot = 764.2789202142488
(threads=(32, 32), blocks=(512, 512)) T_tot = 746.3432890165582
```


The found peak memory is 765.2 GB/s, where the theoretical peak memory is 732 GB/s. We slightly go over the $T_\text{peak}$, possibly the multiplication with the scalar and the addition of array elements are fused into one operation, thus there shall be less than 3 array access in each iteration performed, we thus overcounted it and arrive at an impossibly high $T_\text{tot}$



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


#### todo: for the 2D fluid pressure diffusion GPU code 

*Peak memory throughput used:*

$T_\text{peak} = 765.2 GB/s$

```
Report in a figure the effective memory throughput $T_\text{eff}$ as function of number of grid points nx = ny. 


Realise a weak scaling benchmark varying nx = ny = 32 .* 2 .^ (0:8) .- 1 (or until you run out of device memory). 

Peak memory is reported as a horizontal line

Small comment about the finding

```




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
