# Lecture 3: Solving elliptic PDEs

## Code exercise 3.1:  Implicit transient diffusion using dual timestepping

**Idea:** Using structurally similar diffusion-reaction 1D solver and change it to solve the implicite transient diffusion problem.

### Task 1: Implementation of the solver

- renaming & resetting of the variables

- introduce dual time-stepping using nested loops

=> Outer loop: `nt = 10` as physical time steps

=> Inner loop: pseudo transient iterations



### Task 2: Numerical experiment using the solver

- Plot the gif-animations of:

a) the spatial distribution of concentration C after nt=10 time steps, on top of the plot of the initial concentration distribution

b) the error as function of iteration/nx.

=>  visualization realized in the 1st loop (physical loop)

Case 1: `nt = 10` & `ncheck  = ceil(Int,0.25nx)`

 <img src="./docs/implicit_diffuson_1D.gif" width="60%">


Case 2: `nt = 40` & `ncheck  = ceil(Int,0.05nx)`

 <img src="./docs/experiment01.gif" width="60%">