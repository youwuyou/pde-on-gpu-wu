using Test

include("../scripts/PorousConvection_3D_xpu.jl")
# TODO: 3D thermal porous PorousConvection
@testset "Unit test: PorousConvection_3D_xpu.jl" begin

    # add one unit test
    lx,ly,lz   = 20.0,20.0,20.0
    nx,ny,nz   = 5, 5, 5
    dx,dy,dz   = lx/nx,ly/ny, lz/nz

    xc,yc,zc   = LinRange(dx/2,lx-dx/2,nx),LinRange(dy/2,ly-dy/2,ny), LinRange(dz/2,lz-dz/2,nz)

    Pf      = Data.Array([exp(-(xc[ix]-lx/2)^2 -(yc[iy]-ly/2)^2 -(zc[iz]-lz/2)^2) for ix=1:nx,iy=1:ny,iz=1:nz])
    Pf_     = copy(Pf)

    qDx,qDy,qDz = @zeros(nx+1,ny, nz), @zeros(nx,ny+1, nz), @zeros(nx, ny, nz+1)
    
    _dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz
    _β_dτ         = 0.114
    
    # compute pressure update
    @parallel compute_Pf!(Pf,qDx,qDy,qDz,_dx,_dy,_dz,_β_dτ)

    # not using the self-implemented macros
    Pf_   .-= (diff(qDx,dims=1).* _dx .+ diff(qDy,dims=2).* _dy .+ diff(qDz, dims=3).* _dz).* _β_dτ

    @test Pf ≈ Pf_

end;

# @testset "Reference test: PorousConvection_3D_xpu.jl" begin
    
#     # add one reference test
#     # => run on a small grid, shall not take longer than 5 s

# end;

    
