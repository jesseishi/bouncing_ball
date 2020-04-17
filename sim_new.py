import numpy as np
import matplotlib.pyplot as plt

# TODO: fix these imports. Probably start by giving everything in these folders proper names.
#  (gnc.solvers, gnc.state_estimators, gnc.environments.bouncing_ball, etc...)
# TODO: add a simconfig.yaml file
from ball import Ball
from sim_helpers.solvers.solvers import DP54
from sim_helpers.sensor import GaussianSensor
from sim_helpers.events import Events


# Make the ball that we want to bounce.
MyBall = Ball()

# setup a solver for the ball.
MyBallSolver = DP54(MyBall.ode, safety_factor=0.8, min_dt=1e-5, max_dt=0.1, tol_abs_x=0.0001, tol_rel_x=0.0001)
# And feed it back to the ball.
MyBall.set_up_solver(MyBallSolver)

# Set up a sensor.
MySensor = GaussianSensor(sigma=0.0, mu=0.05)

# Set up the sampling frequencies.
sampling_frequencies = {'sensor': 2, 'state_estimator': 2, 'gravity_onturner': 0.5}
MyEvents = Events(sampling_frequencies)

# Temporary logger.
t_list = np.array(MyBall.t)
x_list = np.array([MyBall.x])
MySensor.simulate_measurement(MyBall.x)
t_star_list = np.array(MyBall.t)
x_star_list = np.array([MySensor.x_star])

# turn off gravity.
MyBall.g = 0

running = True
while running:
    # Get the dt when our next discrete thing needs to run.
    # TODO: maybe it's better to just work with time instead of dt's.
    # Then you could also input something like: at t=4.234, I want this to happen.
    dt, events = MyEvents.get_next_event_dt_and_id(MyBall.t)
    t_this_step = MyBall.t + dt

    # Run the sim until we are at that time.
    while MyBall.t < t_this_step:
        # TODO: Ughl.
        dt_remaining = (t_this_step) - MyBall.t
        MyBall.update_state_max_to_dt(dt_remaining)
        t_list = np.hstack((t_list, MyBall.t))
        x_list = np.vstack((x_list, MyBall.x))
        # Logger.log('a bunch of things')

    # Check that we did that correctly.
    if MyBall.t != t_this_step:
        print("Didn't take the right stepsize")

    # Now based on which event needs to happen at this time, we run that component.
    if 'sensor' in events:
        MySensor.simulate_measurement(MyBall.x)
        t_star_list = np.hstack((t_star_list, MyBall.t))
        x_star_list = np.vstack((x_star_list, MySensor.x_star))
        # Logger.log('a bunch of things')

    if 'state_estimator' in events:
        # StateEstimator.estimate_state(Sensor.x)
        # Logger.log('a bunch of things')
        pass

    if 'gravity_onturner' in events:
        MyBall.g = 9.81

    if MyBall.t > 10:
        running = False


fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(t_list, x_list[:, 0], label="Number of evaluations: {}".format(MyBall.num_evaluations))
ax.plot(t_star_list, x_star_list[:, 0], '.', label="Measurement")

# ax_r = ax.twinx()
# ax_r.plot(t_list[:-1], np.diff(t_list))
# ax_r.plot(t_star_list[:-1], np.diff(t_star_list))


ax.legend()
plt.show()
