using Test

include("../scripts/PorousConvection_2D_xpu.jl")

# TODO: 2D thermal porous PorousConvection
@testset "Unit test: PorousConvection_2D_xpu.jl" begin
    # unit test for one pressure update

    lx,ly   = 20.0,20.0
    nx,ny   = 5, 5
    dx,dy   = lx/nx,ly/ny

    xc,yc   = LinRange(dx/2,lx-dx/2,nx),LinRange(dy/2,ly-dy/2,ny)
    Pf      = Data.Array(@. exp(-(xc-lx/2)^2 -(yc'-ly/2)^2))
    Pf_     = copy(Pf)

    qDx,qDy  = @zeros(nx+1,ny), @zeros(nx,ny+1)
    _dx, _dy = 1.0/dx, 1.0/dy
    _β_dτ    = 0.114

    # compute pressure update
    @parallel compute_Pf!(Pf,qDx,qDy,_dx,_dy,_β_dτ)
    
    # not using the self-implemented macros
    Pf_   .-= (diff(qDx,dims=1).* _dx .+ diff(qDy,dims=2).* _dy).* _β_dτ

    @test Pf ≈ Pf_

end;



@testset "Reference test: PorousConvection_2D_xpu.jl" begin
    # reference test
    using JLD

    qDx_p_ref    = load("qDx_p_ref_5_2D.jld")["data"]
    qDy_p_ref    = load("qDy_p_ref_5_2D.jld")["data"]

    # SMALLER CASE: ny = 10, nt = 5, nvis=1 but dummy value for visualization
    qDx_p, qDy_p = porous_convection_2D_xpu(10, 5, 1; do_visu=false, do_check=true, test=false)  
    
    # choosing 5 non-repeative sample
    using StatsBase
    I = sample(1:length(qDx_p_ref), 5, replace=false)

    @testset "randomly chosen entries $i" for i in I
        @test qDx_p[i] ≈ qDx_p_ref[i]
        @test qDy_p[i] ≈ qDy_p_ref[i]
    end

end;