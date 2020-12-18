# TODO:
# 1. Get reasonable ball parameters so it can run successfully.
# 2. Using DataFrames and save results as csv and make separate post-processing file.
# 3. Implement RK4.
# 4. Start with how kalman filters work.

# Usings.
using Plots

# Includes.
include("src/ball/Ball.jl")
using .Ball

include("src/solvers/Euler.jl")
using .Euler


function main()

    # Initialize the ball.
    ball_state_k = Ball.state()
    ball_params = Ball.params()

    t0 = 0
    t1 = 5
    dt = 0.05

    # Set up data gathering.
    # ball_states = Array{Ball.State, 1}(undef, Int((t1-t0) / dt))
    # ball_states[1] = ball_state_k
    xy_data = Array{Float64, 2}(undef, Int((t1-t0) / dt + 1), 2)
    xy_data[1, :] = [ball_state_k.pos[1] ball_state_k.pos[2]]

    # During one iteration of this for loop we move the simulation from t_km1 to t_k.
    # TODO: Check how Tucker did this in overdot.
    for (i_km1, (t_km1, t_k)) in enumerate(zip(t0:dt:t1-dt, t0+dt:dt:t1))

        # What used to be the present (k) is now the past (k-1).
        ball_state_km1 = ball_state_k

        # Start by updating the ball from t_km1 to t_k.
        ball_state_dot_km1 = Ball.state_dot(ball_state_km1, ball_params)
        ball_state_k = Euler.step(ball_state_km1, ball_state_dot_km1, dt)

        # Store results at the t_k (now) time, so index i_km1 + 1.
        # ball_states[i_km1+1] = ball_state_k
        xy_data[i_km1+1, :] = [ball_state_k.pos[1] ball_state_k.pos[2]]
    end


    # Plotting.
    plot(xy_data[:, 1], xy_data[:, 2])
end

main()
