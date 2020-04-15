# Let's simulate a bouncing ball.

import numpy as np
import matplotlib.pyplot as plt

# constants
g = 9.81  # m/s^2, downward acceleration due to gravity.
m = 1     # kg, mass of the ball.


# the function that defines the ode
def f(t, x):
    # we actually don't care about t.

    # the derivative of the position is our current velocity
    # the derivative of velocity is our acceleration, due to gravity.
    dxdt = np.array([x[1],
                     -g])

    return dxdt


def update_state(t, x, dt):
    # get the derivative
    dxdt = f(t, x)

    # update the state
    x += dxdt * dt
    t += dt

    return t, x


# Now it is time to run the simulation
# Initial states
t = np.array(0, dtype=float)
x = np.array([10, 0], dtype=float)

t_list = np.copy(t)
x_list = np.copy(x)

running = True
while running:
    dt = 0.05
    t, x = update_state(t, x, dt=dt)

    t_list = np.hstack((t_list, t))
    x_list = np.vstack((x_list, x))

    if x_list[-1, 0] < 0:
        running = False

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(15, 9))

ax.plot(t_list, x_list[:, 0], label="position")
# ax.plot(t_list, x_list[:, 1], label="velocity")
ax.legend()

plt.show()