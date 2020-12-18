import numpy as np
import matplotlib.pyplot as plt

# TODO: add a simconfig.yaml file
# TODO: make 2D.
from ball import Ball
from sim_helpers.solvers import DP54, Euler
from sim_helpers.sensor import GaussianSensor
from sim_helpers.events import Events
from sim_helpers.particle_filter import ParticleFilter
from sim_helpers.data_logger import DataLogger


# Make the ball that we want to bounce.
MyBall = Ball()

# setup a solver for the ball.
MyBallSolver = DP54(MyBall.ode, safety_factor=0.8, min_dt=1e-5, max_dt=0.05, tol_abs_x=0.0001, tol_rel_x=0.0001)
# And feed it back to the ball.
# TODO: this seems a lil weird. Why not make ball an instance of 'continuous-time object' that already has an update_state function.
# Yeah then continuous-time-object just has a self.ode = None that's overwritten by in the Ball object?.
MyBall.set_up_solver(MyBallSolver)

# Set up a sensor.
# We only measure the position, the first element of the state.
MySensor = GaussianSensor(sigma=0.0, mu=0.0, state_index_start=0, state_index_end=1)

# Set up the sampling frequencies and make the events.
sampling_frequencies = {'sensor': 5, 'state_estimator': 20}
# It would be much cooler if the state estimator had a delay w.r.t. the sensor.
MyEvents = Events(sampling_frequencies)

# Set up the particle filter.
# We are cheating a little bit w.r.t. real-life here. That's because in the real-world you're making a particle
# filter to estimate something of which you don't perfectly know how it behaves. In this case, we're using the
# same dynamics to simulate our ball and simulate all balls of the particle filter. This particle filter will
# thus be able to perfectly imitate the simulation and get very good results.
# TODO: make this particle filter more realistic by making the ParticleFilterBall class that's an approximation
#  of Ball.
ball_for_particle_filter = Ball()
# ball_for_particle_filter_solver = Euler(ball_for_particle_filter.ode, dt=0.01)
ball_for_particle_filter_solver = DP54(MyBall.ode, safety_factor=0.8, min_dt=1e-5, max_dt=0.5, tol_abs_x=0.001, tol_rel_x=0.001)  # So we use the same solver as the real ball, just with looser thresholds. (this is a little cheating)
ball_for_particle_filter.set_up_solver(ball_for_particle_filter_solver)

# Alright now we can set up our particle filter.
particle_filter = ParticleFilter(ball_for_particle_filter)


# Set up the loggers.
ball_logger = DataLogger(MyBall.t, MyBall.x)
MySensor.simulate_measurement(MyBall.x)
sensor_logger = DataLogger(MyBall.t, MySensor.x_star)  # TODO: it's weird that we simulate a measurement before the simulation has even started.
measured_state = np.asarray([MySensor.x_star[0], MyBall.x[1]])  # TODO: fix the fact that the sensor doesn't measure the complete state, but that the particle filter needs that.
particle_filter.estimate_state(MyBall.t, MyBall.t, measured_state)
particle_filter_logger = DataLogger(MyBall.t, particle_filter.estimated_state)
particle_filter_all_logger = DataLogger(MyBall.t, particle_filter.estimated_states)
# TODO: logger can't vstack a list of lists.
temp_all_particles_logger = [particle_filter.estimated_states]

# print(measured_state, particle_filter.estimated_state)
# print(particle_filter.estimated_states)


# Run the sim.
running = True
while running:
    # Get the dt when our next discrete thing needs to run.
    # TODO: maybe it's better to just work with time instead of dt's.
    # Then you could also input something like: at t=4.234, I want this to happen.
    dt, events = MyEvents.get_next_event_dt_and_id(MyBall.t)
    t_now = MyBall.t + dt

    # Run the sim until we are at that time.
    # TODO: time of the whole sim should not be an attribute of the Ball, but more a general kind of thing.
    # Hmm this is a bit weird, we could avoid this if logger lived inside ball.
    while MyBall.t < t_now:
        dt_remaining = t_now - MyBall.t
        MyBall.update_state_max_to_dt(dt_remaining)
        ball_logger.log(MyBall.t, MyBall.x)

    # Check that we did that correctly.
    if MyBall.t != t_now:
        # TODO: use the logger module.
        print("Didn't take the right stepsize")

    # Now based on which event needs to happen at this time, we run that component.
    if 'sensor' in events:
        MySensor.simulate_measurement(MyBall.x)
        sensor_logger.log(MyBall.t, MySensor.x_star)
        measured_state = np.asarray([MySensor.x_star[0], MyBall.x[1]])  # TODO: see above, but this is just wrong, the measured velocity thould be None.

    if 'state_estimator' in events:
        particle_filter.estimate_state(MyBall.t, MyBall.t, measured_state)
        particle_filter_logger.log(particle_filter.particles[0].t, particle_filter.estimated_state)
        # particle_filter_all_logger.log(particle_filter.particles[0].t, particle_filter.estimated_states)
        temp_all_particles_logger.append(particle_filter.estimated_states)
        # TODO: currently, we only log the output of the particle_filter. If we were to log its internal state, it might
        #  be a good idea to put the logger inside of the particle filter.

    if MyBall.t > 10:
        running = False


fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(ball_logger.t, ball_logger.data[:, 0], label=f"Number of evaluations: {MyBall.num_evaluations}")
ax.plot(sensor_logger.t, sensor_logger.data, 'x', label="Measurement")

temp = np.asarray(temp_all_particles_logger)
# ax.plot(particle_filter_logger.t, temp[:, :, 0], '.', color="lightgrey", label='')

for i in range(particle_filter.num_particles):
    positions = temp[:, i, 0]
    ax.plot(particle_filter_logger.t, positions, '.', color="k", alpha=particle_filter.weights[i])
ax.plot(particle_filter_logger.t, particle_filter_logger.data[:, 0], '.', label="Estimated state")

ax.legend()
plt.show()
