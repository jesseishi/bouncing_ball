# TODO:
# Make the sim run in dt and call filters when t%Δt == 0 (or something like that).
#  Also make sure the sim can run for more than 5 secs without doing weird stuff.
# Start with how Kalman filters work.
# The sensor and filter don't have to run on the same Δt.

# Usings.
using HDF5
using CSV

# Includes and their usings.
include("src/ball/Ball.jl")
include("src/sensor/Sensor.jl")
include("src/filter/ParticleFilter.jl")
using .Ball
using .Sensor
using .ParticleFilter


# The main simulation loop.
function main()

    ##################
    # Initialization #
    ##################

    # TODO: Use init functions for everything. And add hierarchy so that
    # state_k.ball = ... where state_k is ::WorldState.

    # Initialize the ball.
    ball_state_k = Ball.state()
    ball_params = Ball.params()

    # And the sensor.
    sensor_params = Sensor.params()
    pos_star = Sensor.measure(ball_state_k.pos, sensor_params)

    # And the particle filter.
    filter_params = ParticleFilter.params()
    filter_state = ParticleFilter.init(pos_star, filter_params)
    pos_hat = filter_state.pos_hat

    # Set the time span of the sim and timestep.
    # TODO: Think about how to do time. Now we have a continuous update and then
    #  all discrete updates simultanuously and instantly (ok assumption for now).
    t0 = 0
    t1 = 10
    Δt = 0.2      # Time between discrete updates.
    dt = 0.01     # Time used by the RK4 method to do the continuous time update.
    N_discrete_steps = round(Int, (t1-t0) / Δt)
    N_continuous_steps = round(Int, (t1-t0) / dt)

    # Set up data logging.
    # The time dimension is N+1 because we also want to store the value before the first step.
    # I do realize in hindsight that Tucker's method (HDF5Logger) is better; don't keep all this
    #  data in your ram, just save it to the file each step. This also ensures that
    #  if the simulation fails, you have the data up until that point.
    ball_data2 = Array{Float64, 2}(undef, N_continuous_steps+1, 2)
    ball_data2[1, :] = ball_state_k.pos

    sensor_data2 = Array{Float64, 2}(undef, N_discrete_steps+1, 2)
    sensor_data2[1, :] = pos_star

    filter_data2 = Array{Float64, 2}(undef, N_discrete_steps+1, 2)
    filter_data2[1, :] = pos_hat

    particles_data = Array{Float64, 3}(undef, N_discrete_steps+1, 3, filter_params.N)
    particles_data[1, :, :] = ParticleFilter.get_particles_data(filter_state.particles_state, filter_state.weights, filter_params)

    # During one iteration of this for loop we move the simulation from t_km1 to t_k.
    # TODO: Check how Tucker did this in Overdot -> if t == t0 do initialization
    #  instead of the time update and save the data, if t > t0 do the update and saving data.
    for (i_km1, (t_km1, t_k)) in enumerate(zip(t0:Δt:t1-Δt, t0+Δt:Δt:t1))
        i_k = i_km1 + 1

        # What used to be the present (k) is now the past (k-1).
        ball_state_km1 = ball_state_k


        ###########################
        # Continuous time updates #
        ###########################

        # The only continuous time object is the ball. But we want more datapoints
        # for each continuous time update, so we'll divide Δt up to dt.
        ball_state_t = ball_state_km1

        # TODO: Just let the sim run in dt and then if t%Δt == 0 do the discrete update.
        for t in t_km1+dt : dt : t_km1+Δt
            ball_state_t = Ball.step(ball_state_t, ball_params, dt)
            ball_data2[round(Int, t/dt)+1, :] = ball_state_t.pos
        end
        ball_state_k = ball_state_t  # TODO: This is kinda awkward, maybe just get rid of the whole k and k-1 thing?

        #########################
        # Discrete time updates #
        #########################

        # The sensor.
        pos_star = Sensor.measure(ball_state_k.pos, sensor_params)
        sensor_data2[i_k, :] = pos_star

        # The particle filter.
        filter_plot_data, filter_state = ParticleFilter.step(filter_state, filter_params, pos_star, Δt)
        pos_hat = filter_state.pos_hat
        filter_data2[i_k, :] = pos_hat

        # Note that this particle data isn't from when the measurement was taken,
        #  but from after the perturbation.
        particles_data[i_k, :, :] = filter_plot_data

    end


    # Saving the data.
    # Not sure why we should also specify the Julia folder here since this file is inside the Julia folder...
    h5open("Julia/results/data/results.h5", "w") do fid
        g_continuous = g_create(fid, "continuous")
        g_continuous["t"] = collect(t0:dt:t1)
        g_continuous["ball_pos"] = ball_data2

        g_discrete = g_create(fid, "discrete")
        g_discrete["t"] = collect(t0:Δt:t1)
        g_discrete["pos_star"] = sensor_data2
        g_discrete["pos_hat"] = filter_data2
        g_discrete["particles"] = particles_data
    end

    return nothing
end

@time main()
