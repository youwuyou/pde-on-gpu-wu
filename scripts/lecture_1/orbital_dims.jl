using Plots

@views function orbital()
    # Physics
    G    = 6.67e-11
    m    = 5.97 * 1e24 # Attention, 10^24 returns an Int64 which gets truncated...
    M    = 1.9891 * 1e30
    x0   = 149597870 * 1e3
    y0   = 0.0
    Vx0  = 0.0
    Vy0  = 29783.0
    # tt   = 365.0
    # Numerics
    dt   = 60*60*24
    nt   = 365
    # Initial conditions
    xpos = x0
    ypos = y0
    Vx   = Vx0
    Vy   = Vy0
    t    = 0.
    scatter([xpos], [ypos])
    # Time loop
    for it = 1:nt
        xpos = xpos + Vx*dt
        ypos = ypos + Vy*dt
        R    = sqrt(xpos^2 + ypos^2)
        Fx   = -G * m * M / R^2 * xpos / R
        Fy   = -G * m * M / R^2 * ypos / R
        Vx   = Vx + dt * Fx / m
        Vy   = Vy + dt * Fy / m
        # Visualisation
        display(scatter!([xpos], [ypos], title="$it", aspect_ratio=1, markersize=5, markercolor=:blue, framestyle=:box, legend=:none#=, xlims=(-1.1, 1.1), ylims=(-1.1, 1.1)=#))
    end
    return
end

orbital()
