using Plots

@views function orbital()
    # Physics
    G    = 1.0
    m    = 1.0
    M    = 1.0
    x0   = 0.0
    y0   = 1.0
    Vx0  = 1.0
    Vy0  = 0.0
    tt   = 6.0
    # Numerics
    dt   = 0.05
    nt   = cld(tt, dt)
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
        display(scatter!([xpos], [ypos], title="$it", aspect_ratio=1, markersize=5, markercolor=:blue, framestyle=:box, legend=:none, xlims=(-1.1, 1.1), ylims=(-1.1, 1.1)))
    end
    return
end

orbital()
