# TODO:
# 6. Start with how Kalman filters work.

# Usings.
using DataFrames
using CSV

# Includes and their usings.
include("src/ball/Ball.jl")
include("src/sensor/Sensor.jl")
using .Ball
using .Sensor


# The main simulation loop.
function main()

    ##################
    # Initialization #
    ##################

    # Initialize the ball.
    ball_state_k = Ball.state()
    ball_params = Ball.params()

    # And the sensor.
    sensor_params = Sensor.params()
    pos_star = Sensor.measure(ball_state_k.pos, sensor_params)

    # Set the time span of the sim and timestep.
    # TODO: Think about how to do time. Now we have a continuous update and then
    #  all discrete updates simultanuously and instantly (ok assumption for now).
    t0 = 0
    t1 = 5
    Δt = 0.2      # Time between discrete updates.
    dt = Δt / 10  # Time used by the RK4 method to do the continuous time update.

    # Set up data gathering.
    ball_data = DataFrame(t = Float64[], x = Float64[], y = Float64[])
    push!(ball_data, [t0 ball_state_k.pos[1] ball_state_k.pos[2]])

    sensor_data = DataFrame(t = Float64[], x = Float64[], y = Float64[])
    push!(sensor_data, [t0 pos_star[1] pos_star[2]])

    # During one iteration of this for loop we move the simulation from t_km1 to t_k.
    # TODO: Check how Tucker did this in Overdot -> if t == t0 do initialization
    #  instead of the time update and save the data, if t > t0 do the update and saving data.
    for (i_km1, (t_km1, t_k)) in enumerate(zip(t0:Δt:t1-Δt, t0+Δt:Δt:t1))

        # What used to be the present (k) is now the past (k-1).
        ball_state_km1 = ball_state_k


        ###########################
        # Continuous time updates #
        ###########################

        # The only continuous time object is the ball. But we want more datapoints
        # for each continuous time update, so we'll divide Δt up to dt.
        ball_state_t = ball_state_km1
        for t in t_km1+dt : dt : t_km1+Δt
            ball_state_t = Ball.step(ball_state_t, ball_params, dt)
            push!(ball_data, [t ball_state_t.pos[1] ball_state_t.pos[2]])
        end
        ball_state_k = ball_state_t  # TODO: This is kinda awkward, maybe just get rid of the whole k and k-1 thing?

        #########################
        # Discrete time updates #
        #########################

        # The sensor.
        pos_star = Sensor.measure(ball_state_k.pos, sensor_params)
        push!(sensor_data, [t_k pos_star[1] pos_star[2]])

    end


    # Saving the data.
    # Not sure why we should also specify the Julia folder here since this file is inside the Julia folder...
    CSV.write("Julia/results/data/ball_data", ball_data)
    CSV.write("Julia/results/data/sensor_data", sensor_data)
end

@time main()
