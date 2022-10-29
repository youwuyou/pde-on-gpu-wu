using Test

# reference test: using stored outputs
#     parameters: nx=ny=127, maxiter = 50
@testset "reference test: 2D diffusion" begin
     Pf_ref = CuArray(load("Pf_ref_127.jld")["data"])           # cpu version
     Pf     = load("Pf_127.jld")["data"]           # gpu version
 
     # choose indices to test
     using StatsBase
     tuple = size(Pf)
     I = sample(1:tuple[1], 20, replace=false)
     J = sample(1:tuple[2], 20, replace=false)
 
     entries = []
     for i = 1:20
         push!(entries, (I[i], J[i]))
     end
 
     @testset "randomly chosen entries $e" for e in entries
        @test Pf[e[1],e[2]] ≈ Pf_ref[e[1],e[2]] atol = 1e-9
     end
 end;
