# TODO:
# 4. Implement RK4 (and RK4_2 if that could be a thing).
# 5. More structure: World -> Ball, Sensor, Filter
# 6. Start with how Kalman filters work.

# Usings.
using DataFrames
using CSV

# Includes and their usings.
include("src/ball/Ball.jl")
include("src/solvers/Euler.jl")
include("src/solvers/Euler2.jl")
using .Ball
using .Euler
using .Euler2


# The main simulation loop.
function main()

    # Initialize the ball.
    ball_state_k = Ball.state()
    ball_params = Ball.params()

    t0 = 0
    t1 = 1
    Δt = 0.1

    # Set up data gathering.
    ball_data = DataFrame(t = Float64[], x = Float64[], y = Float64[])
    push!(ball_data, [t0 ball_state_k.pos[1] ball_state_k.pos[2]])

    # During one iteration of this for loop we move the simulation from t_km1 to t_k.
    # TODO: Check how Tucker did this in overdot.
    for (i_km1, (t_km1, t_k)) in enumerate(zip(t0:Δt:t1-Δt, t0+Δt:Δt:t1))

        # What used to be the present (k) is now the past (k-1).
        ball_state_km1 = ball_state_k

        # Start by updating the ball from t_km1 to t_k.
        ball_state_dot_km1 = Ball.state_dot(ball_state_km1, ball_params)
        ball_state_k = Euler2.step(ball_state_km1, ball_state_dot_km1, Δt)

        # Store results at the t_k (now) time, so index i_km1 + 1.
        push!(ball_data, [t_k ball_state_k.pos[1] ball_state_k.pos[2]])
    end


    # Saving the data.
    # Not sure why we should also specify the Julia folder here since this file is inside the Julia folder...
    CSV.write("Julia/results/ball_data", ball_data)
end

@time main()
