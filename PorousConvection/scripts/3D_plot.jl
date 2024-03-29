using GLMakie

"""
I/O helper method

Load the final state of the temperature property array 'out_T.bin', which was previously stored in Float32 for plotting.
"""
function load_array(Aname,A)
    fname = string(Aname,".bin")
    fid=open(fname,"r"); read!(fid,A); close(fid)
end

function visualise()
    lx,ly,lz = 40.0,20.0,20.0

    # debug case
    nz          = 127             # DEBUG 63
    ny          = nz
    nx          = 2*(nz+1)-1

    T  = zeros(Float32,nx,ny,nz)

    # load data
    load_array("out_T",T)


    xc,yc,zc = LinRange(0,lx,nx),LinRange(0,ly,ny),LinRange(0,lz,nz)
    fig      = Figure(resolution=(1600,1000),fontsize=24)
    ax       = Axis3(fig[1,1];aspect=(1,1,0.5),title="Temperature",xlabel="lx",ylabel="ly",zlabel="lz")
    surf_T   = contour!(ax,xc,yc,zc,T;alpha=0.05,colormap=:turbo)
    save("T_3D.png",fig)
    return fig
end

visualise()