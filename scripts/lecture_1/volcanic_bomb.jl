using Plots

@views function volcanic_bomb()
    # Physics
    Xb_ini   = 0.0       # x location
    Yb_ini   = 480.0     # y location
    V        = 120.0     # velocity of ejection
    α        = 60.0      # vertical angle of ejection
    g        = 9.81      # gravity accel.
    ttot     = 30.0      # total time, s
    # Numerics
    dt       = 0.5       # time step, s
    nt       = Int(cld(ttot, dt)) # number of timesteps
    # Initialisation
    Vx       = cosd(α)*V # horizontal velocity
    Vy_i     = sind(α)*V # vertical velocity
    Xbomb    = zeros(nt)
    Ybomb    = zeros(nt)
    Vy       = zeros(nt)
    Time     = zeros(nt)
    Xbomb[1] = Xb_ini
    Ybomb[1] = Yb_ini
    Vy[1]    = Vy_i
    flag     = 0
    Ypos     = 0.0
    # Time loop
    for it = 2:nt
        Time[it]  =  Time[it-1] + dt
        Vy[it]    =    Vy[it-1] - dt*g
        Xbomb[it] = Xbomb[it-1] + dt*Vx
        Ybomb[it] = Ybomb[it-1] + dt*Vy[it]
        if Xbomb[it] >= 900 && flag == 0
            Ypos = Ybomb[it]
            flag = 1
        end
        if (Ybomb[it] <= 0) break end
        # Visualisation
        display(scatter(Xbomb, Ybomb, markersize=5, xlabel="horizontal distance, m", ylabel="elevation, m", framestyle=:box, legend=:none))
    end
    println("Y-pos for X=900: $(round(Ypos, sigdigits=5))")
    return
end

volcanic_bomb()
