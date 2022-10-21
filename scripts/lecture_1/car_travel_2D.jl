using Plots

@views function car_travel_2D()
    # Physical parameters
    V     = 113.0          # speed, km/h
    L     = 200.0          # length of segment, km
    dir   = 1              # switch 1 = go right, -1 = go left
    ttot  = 12.0           # total time, h
    # Numerical parameters
    dt    = 0.1                # time step, h
    nt    = Int(cld(ttot, dt)) # number of time steps
    # Initialisation
    Vx    = cosd(45)*V
    Vy    = Vx
    T     = zeros(nt)
    X     = zeros(nt)
    Y     = zeros(nt)
    # Time loop
    for it = 2:nt
        T[it] = T[it-1] + dt
        X[it] = X[it-1] + dt*dir*Vx  # move the car
        Y[it] = Y[it-1] + dt    *Vy  # move the car
        if X[it] > L
            dir = -1      # if beyond L, go back (left)
        elseif X[it] < 0
            dir = 1       # if beyond 0, go back (right)
        end
    end
    # Visualisation
    display(scatter(X, Y, markersize=5, xlabel="x-dist, km", ylabel="y-dist, km", framestyle=:box, legend=:none, ylims=(0, ttot*Vy)))
    return
end

car_travel_2D()
