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


 - Description:

Using the structural similarity of the code, we implemented the transient diffusion solver using the steady reaction-diffusion solver as provided from the class and plotted the evolution of the solution and the error evolution of the solver.

 We can clearly see the accelerated method converges very fast to the steady state.


---

## Code Exercise 3.2:  Operator-splitting for advection-diffusion

**Idea:** Modify code from exercise 3.2 and add the advection term in the outer loop (physical timestepping)


### Task 1: implement solver

- using stability criteria for advection `dt = dx/abs(vx)`

- Dämkohler's number as derived property `da = lx^2/dc/dt`


_Case 1: With boundary condition_

- `C[1] = 1`, `C[end] = 0`

 <img src="./docs/ex02-withBC.gif" width="60%">


_Case 2: Without boundary condition_

 <img src="./docs/ex02-withBC.gif" width="60%">


- Description:  

The solution with the boundary condition seems more reasonable to me. 

I am not sure if the boundary conditions should be simply manually set like this or if there are better ways to incorporate them.

---

## Code Exercise 3.3: Advection-diffusion in 2D

### Task 1:



---

## Code Exercise 3.4: Optimal iteration parameters for pseudo-transient method

### Task 1:


### Task 2:

