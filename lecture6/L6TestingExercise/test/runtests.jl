include("../scripts/diffusion_1D_test.jl")

using Test

# unit tests for our own Diff function
@testset "unit test: diffusion" begin
    C = rand(4)
    D = ones(3)
    @test Diff(C) == diff(C)
    @test Diff(D) == [0., 0.]
end;


# reference test
@testset "reference test: diffusion" begin
     using JLD
     C_ref = load("../test/C_ref.jld")
     qx_ref = load("../test/qx_ref.jld")

     C, qx = diffusion_1D()

     # choose indices to test
     using StatsBase
     I = sample(1:length(qx), 20, replace=false)

     @testset "randomly chosen entries $i" for i in I
        @test C[i]  == C_ref[i]
        @test qx[i] == qx_ref[i]
     end
end;

