using PorousConvection
using Test


include("../scripts/PorousConvection_2D_xpu.jl")

# TODO: 2D thermal porous PorousConvection
@testset "Unit test: PorousConvection_2D_xpu.jl" begin
    # unit test





end;


@testset "Reference test: PorousConvection_2D_xpu.jl" begin
    # reference test
    using JLD

    qDx_p_ref    = load("qDx_p_ref_30_2D.jld")["data"]
    qDy_p_ref    = load("qDy_p_ref_30_2D.jld")["data"]

    # FIXME: change the signatures when any modifications exist
    qDx_p, qDy_p = porous_convection_2D_xpu(63, 30; do_visu=false, do_check=true, test=false)  # ny = 63, nt = 30


    using StatsBase
    I = sample(1:length(qDx_p_ref), 20, replace=false)

    @testset "randomly chosen entries $i" for i in I
        @test qDx_p[i] ≈ qDx_p_ref[i]
        @test qDy_p[i] ≈ qDy_p_ref[i]
    end

end;
    
    

# include("../scripts/PorousConvection_3D_xpu.jl")
# TODO: 3D thermal porous PorousConvection
# @testset "Unit test: PorousConvection_3D_xpu.jl" begin

#     # add one unit test    

# end;

# @testset "Reference test: PorousConvection_3D_xpu.jl" begin
    
#     # add one reference test
#     # => run on a small grid, shall not take longer than 5 s

# end;

    
    
