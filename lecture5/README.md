Lecture 5: Parallel computing

- Code exercise 5.1
                    `Pf_diffusion_2D_Teff.jl`
                    `Pf_diffusion_2D_Perf.jl`
                    `Pf_diffusion_2D_loop_fun.jl`
                    

- Code exercise 5.2
                    Task 1 - ↔ `.jl`


- Code exercise 5.3
                    Task 1 -  ↔ `.jl`
                   

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



---