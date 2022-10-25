Lecture 5: Parallel computing

- Code exercise 5.1
                    `Pf_diffusion_2D_Teff.jl`
                    `Pf_diffusion_2D_Perf.jl`
                    `Pf_diffusion_2D_loop_fun.jl`

- Code exercise 5.2
                    Task 1 - ↔ `.jl`


- Code exercise 5.3
                    Task 1 -  ↔ `Pf_diffusion_2D_Test.jl`
                   

## Code Exercise 5.1: Performance implementation: Diffusion 2D

We run the following scripts using the command `julia -O3 --check-bounds=no -t 20` on the local laptop.


```bash
 ╰─λ lscpu
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         39 bits physical, 48 bits virtual
  Byte Order:            Little Endian
CPU(s):                  20
  On-line CPU(s) list:   0-19
Vendor ID:               GenuineIntel
  Model name:            12th Gen Intel(R) Core(TM) i7-12700H

```


i). `Pf_diffusion_2D_Teff.jl`

- `T_eff` implementation


```
Time = 5.482 sec, Teff = 3.302, niter = 90 
```


ii). `Pf_diffusion_2D_Perf.jl`

- added scalar precomputations, avoided divisions

- added `do_check=false` to disable ncheck


```
Time = 3.827 sec, Teff = 4.730, niter = 90 
```


iii). `Pf_diffusion_2D_loop_fun.jl`

- added `compute!()` functions

- added macro for derivatives

- added multithreading using `Threads.@threads` infront of the outer for-loop


```
Time = 0.000 sec, Teff = 113.205, niter = 1 
```




*Misc: tested on 10 threads regarding row-/col-major*

```
julia> include("Pf_diffusion_2D_Perf_loop_fun.jl")
Time = 0.000 sec, Teff = 112.565, niter = 1 

julia> include("Pf_diffusion_2D_Perf_loop_fun.jl")
Time = 0.000 sec, Teff = 28.966, niter = 1 

```



---

## Code Exercise 5.2: Performance evaluation: Diffusion 2D (strong scaling test)




---

## Code Exercise 5.3: Unit test

In the following exercise we changed the function signature to `function Pf_diffusion_2D(nx_, ny_;do_check=false)` in order to perform unit tests more practically.

Besides of that we added the testing parameters as followed right above the iteration loop

```
    # testing
    xtest = [5, Int(cld(0.6*lx, dx)), nx-10]
    ytest = Int(cld(0.5*ly, dy))
```

At the end we want the function to have selected return values depending on the entries given by `xtest`, `ytest` arrays.

```
    return Pf[xtest, ytest]

```


The definition of the testset is straight forward. The only thing we need to pay attention to is the type matching between the return type and the values to be compared. Thus we reshaped the return values of the function, after this we are able to do direct comparison between the values using '≈'.

 Our testset has the following output with `do_check = false` and `atol = 1e-9` set.

```bash
julia> include("Pf_diffusion_2D_Test.jl")
Test Summary:         | Pass  Total  Time
Diffusion Acoustic 2D |    4      4  0.7s
Test.DefaultTestSet("Diffusion Acoustic 2D", Any[], 4, false, false, true, 1.666726258251579e9, 1.666726258966965e9)
```


---