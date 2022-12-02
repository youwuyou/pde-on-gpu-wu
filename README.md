---
Current maintainer: youwuyou (youwuyou@ethz.ch)
In relation to lecture by: @luraess, @utkinis, @mauro3, @omlins
---

## **Exercise Overview**

## Part 1 - Introduction


### Lecture 1: Why Julia GPU

| Number | Topic | Finished |
| --- | --- | --- |
| Code Exercise 1.1 | Car travel | :heavy_check_mark: |
| Code Exercise 1.2 | Car travel in 2 dimensions | :heavy_check_mark: |
| Code Exercise 1.3 | Volcanic bomb | :heavy_check_mark: |
| Code Exercise 1.4 | (optional) - Orbital around a centre of mass |  |
| Code Exercise 1.5 | (optional) - Many volcanic bombs |  |



### Lecture 2: PDEs & physical processes

| Number | Topic | Finished |
| --- | --- | --- |
| Code Exercise 2.1 | Advection-Diffusion | :heavy_check_mark: |
| Code Exercise 2.2 | Reaction-Diffusion | :heavy_check_mark: |
| Code Exercise 2.3 | Nonlinear problems | :heavy_check_mark: |
| Code Exercise 2.4 | Julia install and Git repo | :heavy_check_mark: |


### Lecture 3: Solving elliptic PDEs

| Number | Topic | Finished |
| --- | --- | --- |
| Code Exercise 3.1 | Implicit transient diffusion using dual timestepping | :heavy_check_mark: |
| Code Exercise 3.2 | Operator-splitting for advection-diffusion | :heavy_check_mark: |
| Code Exercise 3.3 | Advection-diffusion in 2D | :heavy_check_mark: |
| Code Exercise 3.4 | Optimal iteration parameters for pseudo-transient method | :heavy_check_mark: |




## Part 2 - solving PDEs on GPUs


### Lecture 4: Porous convection

| Number | Topic | Finished |
| --- | --- | --- |
| Code Exercise 4.1 | Thermal porous convection in 2D | :heavy_check_mark: |
| Code Exercise 4.2 | Thermal porous convection with implicit temperature update | :heavy_check_mark: |


### Lecture 5: Parallel computing

| Number | Topic | Finished |
| --- | --- | --- |
| Code Exercise 5.1 | Performance implementation: Diffusion 2D | :heavy_check_mark: |
| Code Exercise 5.2 | Performance evaluation: Diffusion 2D (strong scaling test)  | :heavy_check_mark: |
| Code Exercise 5.3 | Unit tests | :heavy_check_mark: |

### Lecture 6: GPU computing & perf

| Number | Topic | Finished |
| --- | --- | --- |
| Code Exercise 6.1 | Data transfer optimisations | :heavy_check_mark: |
| Code Exercise 6.2 | Solving PDEs on GPUs | :heavy_check_mark: |
| Code Exercise 6.3 | Unit and reference tests | :heavy_check_mark: |







## Part 3 - Multi-GPU computing (projects)

### Lecture 7: XPU computing

| Number | Topic | Finished |
| --- | --- | --- |
| Code Exercise 7.1 | 2D Thermal porous convection xPU implementation | :heavy_check_mark: |
| Code Exercise 7.2 | 3D Thermal porous convection xPU implementation | :heavy_check_mark: |
| Code Exercise 7.3 | CI and GitHub Actions | :heavy_check_mark: |


### Lecture 8: Julia MPI & multi-XPU

| Number | Topic | Finished |
| --- | --- | --- |
| Code Exercise 8.1 | Towards distributed memory computing on GPUs | :heavy_exclamation_mark: |
| Code Exercise 8.2 | Multi-XPU computing |  |

NOTE: code exercise 8.1 has one last problem remained

### Lecture 9: Multi-xPU & Projects

| Number | Topic | Finished |
| --- | --- | --- |
| Code Exercise 9.1 | Multi-xPU computing projects |  |
| Code Exercise 9.2 | Automatic documentation in Julia | :heavy_check_mark: |



### Lecture 10: Advanced optimisations

| Number | Topic | Finished |
| --- | --- | --- |
| Code Exercise 10.1 | Push-ups with memory copy |  |
| Code Exercise 10.2 | Advanced data transfer optimisations (part 1) |  |
| Code Exercise 10.3 | Advanced data transfer optimisations (part 2) |  |


### Lecture 11 - 14: Final Projects

**Topic:** Hydro-mechanical solver for (in)compressible two-phase flow equations

**Repository:**  [HydroMech.jl](https://github.com/youwuyou/HydroMech.jl)

**Description:** 

The project aims to implement a collection of hydro-mechanical solvers that solve the (in)compressible two-phase flow equations in 2-/3D. As a starting point, we extend and use the existing code of the 2D incompressible solver to reproduce the porosity wave benchmark as in ([Räss 2019](https://doi.org/10.1093/gji/ggz239)). The second step of the project consists of implementing the 3D version of it and to reproduce the 3D benchmark in the same paper.

After verifying the reproducibility of the developed incompressible solver, we will implement the compressible solver and reproduce the fluid injection 2D benchmark as in ([Dal Zilio 2022](https://doi.org/10.1016/j.tecto.2022.229516))
