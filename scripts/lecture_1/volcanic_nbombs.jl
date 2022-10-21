using Plots

@views function volcanic_bombs()
    # Physics
    nb         = 5          # number of volcanic bombs
    Xb_ini     = 0.0        # x location
    Yb_ini     = 480.0      # y location
    V          = 120.0      # velocity of ejection
    α          = 30.0*randn(nb) .+ 90.0 # vertical angle of ejection
    g          = 9.81       # gravity accel.
    ttot       = 60.0       # total time, s
    # Numerics
    dt         = 0.5        # time step, s
    nt         = Int(cld(ttot, dt)) # number of timesteps
    # Initialisation
    Vx         = cosd.(α)*V # horizontal velocity
    Vy_i       = sind.(α)*V # vertical velocity
    Xbomb      = zeros(nb,nt)
    Ybomb      = zeros(nb,nt)
    Vy         = zeros(nb)
    Xbomb[:,1].= Xb_ini
    Ybomb[:,1].= Yb_ini
    Vy         = Vy_i
    scatter([Xb_ini], [Yb_ini])
    # Time loop
    for it = 2:nt
        for ib = 1:nb
            Vy[ib]       =    Vy[ib]      - dt*g
            Xbomb[ib,it] = Xbomb[ib,it-1] + dt*Vx[ib]
            Ybomb[ib,it] = Ybomb[ib,it-1] + dt*Vy[ib]
            if Ybomb[ib,it] <= 0.0
                Ybomb[ib,it] = 0.0
                Xbomb[ib,it] = Xbomb[ib,it-1]
            end
        end
        # Visualisation
        display(scatter!([Xbomb[:,it]], [Ybomb[:,it]], markersize=5, markercolor=:blue, xlabel="horizontal distance, m", ylabel="elevation, m", framestyle=:box, legend=:none))
    end
    return
end

volcanic_bombs()
