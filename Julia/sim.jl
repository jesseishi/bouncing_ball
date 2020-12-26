# TODO:
# Sigma point filters (use probabilistic robotis _and_ Tucker's article).
# Hierarchy -> make states mutable (actually I don't think this is strictly needed) (and adjust functions accordingly), init functions for everything, make a World module.
# The sensor and filter don't have to run on the same Δt.

# Usings.
using HDF5Logger
using Random
Random.seed!(1)

# Includes and their usings.
include("src/ball/Ball.jl")
include("src/sensor/Sensor.jl")
include("src/filter/ParticleFilter.jl")
using .Ball
using .Sensor
using .ParticleFilter


# The simulate simulation loop.
function simulate()

    ##################
    # Initialization #
    ##################

    # Initialize the ball.
    ball_params = Ball.params()
    ball_state_k = Ball.state()

    # And the sensor.
    sensor_params = Sensor.params()
    pos_star = Sensor.measure(ball_state_k.pos, sensor_params)

    # And the particle filter.
    particle_filter_params = ParticleFilter.params()
    particle_filter_state = ParticleFilter.init(pos_star, particle_filter_params)

    # Set the time span of the sim and timestep.
    t0 = 0.
    t1 = 20.
    Δt = 0.25     # Time between discrete updates.
    dt = 0.01     # Time used by the RK4 method to do the continuous time update.
    N_discrete_steps = round(Int, (t1-t0) / Δt)
    N_continuous_steps = round(Int, (t1-t0) / dt)

    # Data logging using Tucker's HDF5Logger.
    log = Log("Julia/results/data/results.h5")
    add!(log, "/continuous/ball_pos", ball_state_k.pos, N_continuous_steps+1, true)
    add!(log, "/continuous/t", t0, N_continuous_steps+1, true)
    add!(log, "/discrete/pos_star", pos_star, N_discrete_steps+1, true)
    add!(log, "/discrete/pos_hat", particle_filter_state.pos_hat, N_discrete_steps+1, true)
    add!(log, "/discrete/particles", ParticleFilter.get_particles_data(particle_filter_state, particle_filter_params), N_discrete_steps+1, true)
    add!(log, "/discrete/t", t0, N_discrete_steps+1, true)

    # Put the whole simulation in a try - except - finally block so we can always shutdown the simulation gracefully.
    try

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

            # 16.60 didn't appear in t because 16.4+0.2 = 16.59...98, so the for
            #  loop stopped at 16.59 instead of 16.60. Adding the eps() works,
            #  but obviously isn't a very nice solution.
            for t in t_km1+dt : dt : t_km1+Δt + eps(2 * (t_km1 + Δt))
                ball_state_t = Ball.step(ball_state_t, ball_params, dt)
                log!(log, "/continuous/ball_pos", ball_state_t.pos)
                log!(log, "/continuous/t", t)
            end
            ball_state_k = ball_state_t

            #########################
            # Discrete time updates #
            #########################

            # Log the time
            log!(log, "/discrete/t", t_k)

            # The sensor.
            pos_star = Sensor.measure(ball_state_k.pos, sensor_params)
            log!(log, "/discrete/pos_star", pos_star)

            # The particle filter.
            particle_filter_state, particle_filter_state_before_resample = ParticleFilter.step(particle_filter_state, particle_filter_params, pos_star, Δt)
            log!(log, "/discrete/pos_hat", particle_filter_state.pos_hat)
            log!(log, "/discrete/particles", ParticleFilter.get_particles_data(particle_filter_state_before_resample, particle_filter_params))

        end

    catch err
        rethrow(err)

    # This also runs if we catch an error and rethrow it, so we always close the
    #  file that stores the logs.
    finally
        close!(log)
    end

    return nothing
end

@time simulate()
