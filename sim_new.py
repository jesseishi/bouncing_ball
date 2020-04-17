import numpy as np
import matplotlib.pyplot as plt

# TODO: fix these imports. Probably start by giving everything in these folders proper names.
#  (gnc.solvers, gnc.state_estimators, gnc.environments.bouncing_ball, etc...)
# TODO: add a simconfig.yaml file
# TODO: make 2D.
from ball import Ball
from sim_helpers.solvers.solvers import DP54
from sim_helpers.sensor import GaussianSensor
from sim_helpers.events import Events
from sim_helpers.data_logger import DataLogger


# Make the ball that we want to bounce.
MyBall = Ball()

# setup a solver for the ball.
MyBallSolver = DP54(MyBall.ode, safety_factor=0.8, min_dt=1e-5, max_dt=0.05, tol_abs_x=0.0001, tol_rel_x=0.0001)
# And feed it back to the ball.
# TODO: this seems a lil weird. Why not make ball an instance of 'propagatable object' that already has an update_state function.
# Yeah then propagatable_object just has a self.ode = None that's overwritten by in the Ball object?.
MyBall.set_up_solver(MyBallSolver)

# Set up a sensor.
# We only measure the position, the first element of the state.
MySensor = GaussianSensor(sigma=0.0, mu=0.0, state_index_start=0, state_index_end=1)

# Set up the sampling frequencies.
sampling_frequencies = {'sensor': 2, 'state_estimator': 2}
MyEvents = Events(sampling_frequencies)

# Set up the loggers.
ball_logger = DataLogger(MyBall.t, MyBall.x)
MySensor.simulate_measurement(MyBall.x)
sensor_logger = DataLogger(MyBall.t, MySensor.x_star)

running = True
while running:
    # Get the dt when our next discrete thing needs to run.
    # TODO: maybe it's better to just work with time instead of dt's.
    # Then you could also input something like: at t=4.234, I want this to happen.
    dt, events = MyEvents.get_next_event_dt_and_id(MyBall.t)
    t_this_step = MyBall.t + dt

    # Run the sim until we are at that time.
    # TODO: time of the whole sim should not be an attribute of the Ball, but more a general kind of thing.
    while MyBall.t < t_this_step:
        dt_remaining = (t_this_step) - MyBall.t
        MyBall.update_state_max_to_dt(dt_remaining)
        ball_logger.log(MyBall.t, MyBall.x)

    # Check that we did that correctly.
    if MyBall.t != t_this_step:
        # TODO: use the logger module.
        print("Didn't take the right stepsize")

    # Now based on which event needs to happen at this time, we run that component.
    if 'sensor' in events:
        MySensor.simulate_measurement(MyBall.x)
        sensor_logger.log(MyBall.t, MySensor.x_star)

    if 'state_estimator' in events:
        # StateEstimator.estimate_state(Sensor.x)
        # Logger.log('a bunch of things')
        pass

    if MyBall.t > 10:
        running = False


fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(ball_logger.t, ball_logger.data[:, 0], label=f"Number of evaluations: {MyBall.num_evaluations}")
ax.plot(sensor_logger.t, sensor_logger.data, '.', label="Measurement")

ax.legend()
plt.show()
