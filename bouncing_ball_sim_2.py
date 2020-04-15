# Running in bouncing_ball conda env.

# Import stuff.
import numpy as np
import matplotlib.pyplot as plt


class BouncingBall:
    # Initialize this bouncing ball with some physical constants. Don't set any parameters of the simulation yet so we
    # can compare the same physics with different simulation solvers.
    def __init__(self, mass=1., radius=0.2, spring_constant=1000., damper_coef=2.,
                 t0=None, x0=None, gravity=9.81):
        # What is this ball like?
        self.m = mass
        self.r = radius
        self.S = np.pi * self.r * self.r
        self.k = spring_constant
        self.c = damper_coef
        self.Cd = 0.3

        # The initial values of the drop.
        self.t0 = t0
        self.x0 = x0

        # Set up the environment, only gravity in this case.
        self.g = gravity

    # run the whole sim.
    def run_sim(self, setup):
        # For starters, let's initialize to the point specified for this classinstance.
        # We can't adjust the initial state here because then all runs should model the same thing.
        self._initialize_sim(setup)

        # Now let's run the sim.
        running = True
        while running:
            self._update_state()
            self._update_history()

            # Tell the sim when to stop.
            if self.t >= 10:
                break

    # Initialize the sim
    def _initialize_sim(self, setup):
        # Set up the initial values
        # If we didn't specify a start state, start at the default start state.
        if self.t0 is None:
            self.t0 = 0.
        if self.x0 is None:
            self.x0 = [1, 0.]

        # Now initialize the time and state.
        self.t = np.array(self.t0)
        self.x = np.array(self.x0)
        self.t_list = np.copy(self.t)
        self.x_list = np.copy(self.x)

        # Reset the amount of evaluations we have done of the ode.
        self.num_evaluations = 0

        # And initialize the setup of the sim.
        # TODO: constant -> move
        FIXED_STEP_SOLVERS = ["euler", "rk4", "gl2"]
        ADAPTIVE_STEP_SOLVERS = ["dp54", "gl43"]
        self.solver = setup["solver"].lower()
        if self.solver in FIXED_STEP_SOLVERS:
            self.dt = setup["dt"]
        elif self.solver in ADAPTIVE_STEP_SOLVERS:
            # Even though this is an adaptive step solver, we still need a guess for the first dt.
            # If the simsetup provided one, let's use it. Otherwise, we'll use a default. This can be big because our
            # adaptive-step solver will automatically lower it.
            try:
                self.dt = setup["dt"]
            except KeyError:
                self.dt = 1.0
        # If it's not in our fixed-step or adaptive-step solvers list, we can't find it.
        else:
            raise NameError("Couldn't find solver {}".format(self.solver))

    # Here we update the current state.
    # In the future we can do that with different solvers, e.g. Runge-Kutta or adaptive-step solvers.
    def _update_state(self):
        # Here we update the state based on what solver (and dt) we have selected.
        if "euler" in self.solver:
            self.euler()
        elif "rk4" in self.solver:
            self.rk4()
        elif "dp54" in self.solver:
            self.dp54()
        elif "gl2"in self.solver:
            self.gl2()
        elif "gl43"in self.solver:
            self.gl43()

    # The Euler method.
    def euler(self):
        xdot = self._ode(self.t, self.x)
        self.x += xdot * self.dt
        self.t += self.dt

    # The Runge-Kutta method (classic 4th order method).
    def rk4(self):
        # First get the four derivatives at different points.
        k1 = self._ode(self.t, self.x)
        k2 = self._ode(self.t+0.5*self.dt, self.x+0.5*self.dt*k1)
        k3 = self._ode(self.t+0.5*self.dt, self.x+0.5*self.dt*k2)
        k4 = self._ode(self.t+self.dt, self.x+self.dt*k3)

        # Then average them out to get an estimate for xdot and return.
        xdot = (k1 + 2*k2 + 2*k3 + k4) / 6

        # And finally update the state with it.
        self.x += xdot * self.dt
        self.t += self.dt

    # The Dormand-Prince 5(4) method.
    def dp54(self):
        # We have to find the right timestamp, sometimes that'll take a few tries.
        solution_accepted = False
        while not solution_accepted:

            # These are the fractions of our self.dt where we'll evaluate the derivative at.
            # It's also the most left column of the Butcher table.
            c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])

            # We'll initialize our k-vector, that has all the derivatives, with all zeros.
            k = np.zeros((self.x.size, c.size))

            # Here are the coefficients of the Dormand-Prince 5(4) solver, the lower triangle part of the Butcher table.
            a = np.array([[0,           0,          0,           0,         0,          0,     0],
                          [1/5,         0,          0,           0,         0,          0,     0],
                          [3/40,        9/40,       0,           0,         0,          0,     0],
                          [44/45,      -56/15,      32/9,        0,         0,          0,     0],
                          [19372/6561, -25360/2187, 64448/6561, -212/729,   0,          0,     0],
                          [9017/3168,  -355/33,     46732/5247,  49/176,   -5103/18656, 0,     0],
                          [35/384,      0,          500/1113,    125/192,  -2187/6784,  11/84, 0]], dtype=float)

            # Finally, set the lower two rows of the Butcher table, b and b_hat.
            b = a[-1, :]  # The nice thing about this method is that b is the same as the last row of a.
            # TODO: because they're the same, we can use k7 in the next iteration as k1.
            b_hat = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
            # TODO: we don't have to define this every try or every timestep even.
            #  It might be a good idea to make a sim class (with a butcher subclass??).
            #  Then it would be possible to do sim.butcher.a (idk how to do that yet).
            #  And/or make each solver use a butcher table. (Euler would just be a very simple one)

            # Now we're ready to calculate the derivative at each point.
            for i in range(len(c)):
                k[:, i] = self._ode(self.t + self.dt*c[i], self.x + self.dt*np.dot(k, a[i, :].T))

            # Then we can now calculate both the 5th and the 4th order solution.
            x_5th = self.x + self.dt*np.dot(k, b.T)
            x_4th = self.x + self.dt*np.dot(k, b_hat.T)

            # Let's see if it did a good enough job to accept the result.
            # todo: read some stuff about this.
            #  One idea I have is to divide this by self.dt so it's easier to reason about the error
            tol_abs = 0.0051
            # TODO: once the ball comes in an equilibrium position, x[1] ~= 0, so the relative difference is
            #  going to explode. We could protect against that.
            tol_rel = 0.0001
            diff_abs = np.linalg.norm(x_5th - x_4th, ord=np.inf)
            diff_rel = np.linalg.norm((x_5th - x_4th)/x_5th, ord=np.inf)

            if (diff_abs < tol_abs) & (diff_rel < tol_rel):
                # We have found an acceptable solution
                solution_accepted = True

                # Let's store it.
                self.t += self.dt
                self.x = x_5th
            else:
                # We must decrease our timestamp to reduce the error and get an acceptable solution.
                # We use a cool algorithm for that. Take the minimum of the timestamp needed to drive either the max or
                #  relative difference down enough.
                # todo: make updating the dt a function because we also need to in the if statement above.
                sigma = 0.7  # The safety factor fo choosing a new timestep.
                p = 4  # The lower of the two orders of this method.
                new_dt = np.min([sigma * self.dt * (tol_abs / diff_abs) ** (1 / (p + 1)),
                                 sigma * self.dt * (tol_rel / diff_rel) ** (1 / (p + 1))])

                # The new dt can't be smaller than half the previous try.
                self.dt = np.max([new_dt, self.dt/2])
                # min_dt = 1e-5
                # self.dt = np.max([self.dt, min_dt])
                # print("recalculating at ")
                # Can't /0 because diff_abs can't be zero if we're here.

        # Update the timestamp for the next sample.
        sigma = 0.7  # The safety factor fo choosing a new timestep.
        p = 4  # The lower of the two orders of this method.
        if diff_abs != 0 and diff_rel != 0:
            self.dt = np.min([sigma * self.dt * (tol_abs / diff_abs) ** (1 / (p + 1)),
                              sigma * self.dt * (tol_rel / diff_rel) ** (1 / (p + 1))])
        else:
            # TODO: we actually end up in this condition a lot...
            #  That's just because when we don't have drag, and only acceleration due to gravity (not bouncing),
            #  we follow an order 2 polynomial perfectly (constant acceleration).
            self.dt = 1.0

        max_dt = 0.05
        self.dt = np.min([self.dt, max_dt])

    # The second order Gauss-Legendre method. Our first implicit solver.
    def gl2(self):
        # Put the butcher tableau in vectors.
        c = 0.5
        a = 0.5
        b = 1

        # Ok, so this one of the easiest implicit solvers, so this should not be extremely hard.
        # We'll start by guessing k and using that to make a new estimate.
        # TODO: we could make this guess better by using the k of the previous timestamp.
        k_old = np.array([0, 0])
        k_new = self._ode(self.t + self.dt * c, self.x + self.dt * a * k_old)

        # Alright now we run through a loop until we've converged on a solution.
        num_iterations = 0
        while np.linalg.norm(k_new - k_old, ord=np.inf) > 0.01 and num_iterations < 1000:
            # Note: for a stiff bouncing ball this doesn't work well AT ALL.
            #  You can see why: if you would plug in the analytic solution for k after self.dt, and recalculate k_new,
            #  it'll not accept it as the solution. So for a stiff bouncing ball, this 1st order method won't work
            #  (surprise).
            k_old = k_new
            k_new = self._ode(self.t + self.dt*c, self.x + self.dt*a*k_old)
            num_iterations += 1

            # Let us know if we ran into the max number of iterations.
            if num_iterations > 990:
                print("ran into max number of iterations of {}".format(num_iterations))

        # And now we can calculate xdot.
        xdot = b * k_new

        # And finally update the state.
        self.x += xdot * self.dt
        self.t += self.dt

    # Now a 4th order, implicit, adaptive-step solver. We'll use the 4th order Gauss-Legendre method.
    def gl43(self):
        # Let's start by putting the butcher tableau in vectors.
        c = np.array([1/2 - 1/6*np.sqrt(3), 1/2 + 1/6*np.sqrt(3)])
        a = np.array([[1/4,                  1/4 - 1/6*np.sqrt(3)],
                      [1/4 + 1/6*np.sqrt(3), 1/4]])
        b = np.array([1/2, 1/2])
        b_hat = np.array([1/2 + 1/2*np.sqrt(3), 1/2 - 1/2*np.sqrt(3)])

        # Let's find a step size that's small enough to not build up too much error.
        # Until then, we don't accept the solution.
        solution_accepted = False
        while not solution_accepted:
            # Start with a guess for xdot.
            # TODO: we could use the latest k_old as a better guess.
            k_old = np.zeros((self.x.size, c.size))

            # And then our first estimate of k_new. Let's loop through all rows of k and fill them.
            k_new = np.empty_like(k_old)
            # TODO: there must be a better way to program this, I don't like this. Same in the other adaptive-step
            #  method. In addition, I don't really like the while loop because it means we need to have evaluated k_new
            #  and k_old before entering the while loop.
            for i in range(len(c)):
                k_new[:, i] = self._ode(self.t + c[i] * self.dt, self.x + self.dt * np.dot(k_old, a[i, :].T))

            # Now run a while loop until we're satisfied.
            num_iterations = 0
            while np.linalg.norm(k_new - k_old, ord=np.inf) > 0.1 and num_iterations < 100:
                k_old = k_new
                for i, _ in enumerate(k_new):
                    k_new[:, i] = self._ode(self.t + c[i] * self.dt, self.x + self.dt * np.dot(k_old, a[i, :].T))

                # Make sure we don't spend too much time inside this loop.
                num_iterations += 1
                if num_iterations > 99:
                    print("couldn't converge on k after {} iterations".format(num_iterations))

            # So now that we've found k, we can update our states.
            x_4th = self.x + self.dt * np.dot(k_new, b.T)
            x_3th = self.x + self.dt * np.dot(k_new, b_hat.T)

            # Let's see if it did a good enough job to accept the result.
            # todo: read some stuff about this.
            #  One idea I have is to divide this by self.dt so it's easier to reason about the error.
            tol_abs = 0.001
            # Insight (see DP54), sometimes the relative difference will explode because x[1] will go to zero.
            tol_rel = 0.01
            diff_abs = np.linalg.norm(x_4th - x_3th, ord=np.inf)
            diff_rel = np.linalg.norm((x_4th - x_3th) / x_4th, ord=np.inf)

            if (diff_abs < tol_abs) & (diff_rel < tol_rel):
                # We have found an acceptable solution
                solution_accepted = True

                # Let's store it.
                self.t += self.dt
                self.x = x_4th
            else:
                # We must decrease our timestamp to reduce the error and get an acceptable solution.
                # We use a cool algorithm for that. Take the minimum of the timestamp needed to drive either the max or
                #  relative difference down enough.
                # todo: make updating the dt a function because we also need to in the if statement above.
                sigma = 0.9  # The safety factor fo choosing a new timestep.
                p = 3  # The lower of the two orders of this method.

                new_dt = np.min([sigma * self.dt * (tol_abs / diff_abs) ** (1 / (p + 1)),
                                 sigma * self.dt * (tol_rel / diff_rel) ** (1 / (p + 1))])
                # Can't /0 because diff_abs can't be zero if we're here.

                # The new timestamp that we try can be maximum half the timestamp that we previously tried.
                self.dt = np.max([new_dt, self.dt/2])

        # Last thing to do is update the timestamp for the next sample.
        sigma = 0.7  # The safety factor fo choosing a new timestep.
        p = 3  # The lower of the two orders of this method.
        if diff_abs != 0 and diff_rel != 0:
            self.dt = np.min([sigma * self.dt * (tol_abs / diff_abs) ** (1 / (p + 1)),
                              sigma * self.dt * (tol_rel / diff_rel) ** (1 / (p + 1))])
        else:
            self.dt = 0.1

        max_dt = 0.5
        self.dt = np.min([self.dt, max_dt])

    # Calculate the derivative of the input state.
    # todo: it's a bit silly that we have to input self.t and self.x into this function.
    #  Instead, the input could be Dt and Dx. Or solve it in another smarter way.
    def _ode(self, t, x):
        # We called the function, let's update our number of evaluations.
        self.num_evaluations += 1

        # Now we can calculate xdot.
        f = self._calc_forces(t, x)
        a = f / self.m
        xdot = np.array([x[1],
                         a])
        return xdot

    # Calculate the acceleration, needed to update for the derivative
    def _calc_forces(self, t, x):
        # First, calculate the force due to gravity.
        f_gravity = -self.g * self.m

        # Then the force from drag.
        # It acts in the opposite direction of the velocity so is only defined if our velocity is not zero.
        if x[1] != 0.:
            f_drag = 0.5 * 1.225 * x[1]*x[1] * self.S * self.Cd

            # Turn it in the opposite direction as the velocity.
            f_drag = f_drag * -x[1]/np.abs(x[1])
        else:
            f_drag = 0.

        # Now calculate the force from the bounce.
        # See if we're within our radius from the ground, if not, it's zero.
        if x[0] <= self.r:
            f_bounce = self._calc_bounce_force(t, x)
        else:
            f_bounce = 0.

        # now return the total force
        return f_gravity + f_drag + f_bounce

    def _calc_bounce_force(self, t, x):
        # We'll model this as a mass-spring-damper system
        f_spring = -self.k * (x[0] - self.r)
        f_damper = -self.c * x[1]

        return f_spring + f_damper

    # Used the add the current state to the list where the history is stored.
    def _update_history(self):
        self.t_list = np.hstack((self.t_list, self.t))
        self.x_list = np.vstack((self.x_list, self.x))

    def return_results(self):
        return self.t_list, self.x_list

    # Plot the results from the last run.
    def plot_results(self):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

        ax.plot(self.t_list, self.x_list[:, 0], '.', label="position")
        ax.axhline(self.r)

        plt.show()


# Let's try it out
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax_r = ax.twinx()

sim_setups = [{"solver": "Euler", "dt": 0.001},
              {"solver": "RK4",   "dt": 0.01},
              {"solver": "dp54"},
              {"solver": "gl2", "dt": 0.001},
              {"solver": "gl43"}]

# todo: implement a true state to compare to, solving this problem analytically. We can see how many evaluations each
#  method needs to be ok.

MyBall = BouncingBall()
for i, sim_setup in enumerate(sim_setups):
    print("running sim {}".format(i+1))
    MyBall.run_sim(sim_setup)
    # todo: def get_label(sim_setup), based on whether it is an adaptive/fixed step solver etc...
    ax.plot(MyBall.t_list, MyBall.x_list[:, 0], '.', label="solver: {}, total evaluations: {}".format(
        sim_setup["solver"], MyBall.num_evaluations))
    # ax_r.plot(MyBall.t_list[:-1], np.diff(MyBall.t_list))

ax.set_title("Bouncing ball")
ax.set_ylabel("CG of Ball position above ground [m]")
ax_r.set_ylabel("Timestep [s]")
ax.set_xlabel("Time [s]")
ax.set_ylim(0, 1.5)

ax.legend()

plt.show()


# TODO: list
# - Cool idea: make a sim class, env class and object (ball) class.
# - The sim class should work by calling a bunch of butcher tables.
# - And the ode is an input by the user? As part of the object class.

# - Make this executable from the command line.

# - Make more generic functions for all solvers. They can all be done with butcher tableaus. Sometimes you'd to catch
# that a is empty (Euler method), and sometimes you need to iterate k to find a solution (implicit methods), and
# sometimes certain steps can be taken from the previous iteration (k1=k7_old for dp54 or k_old is the old k_new)

# a sim of a state estimator -> again, follow Tucker's page
# a sim with a controller in the loop
# a complete sim with state estimator and controller.
# maybe a rocket launch, plane flying, drone hovering, drone landing, etc... Probably something where you can use an
# easy PID controller first, then moving on to more difficult controllers. Can also do something that I could then build
# in real life later. Such as: inverted pendulum balancer, flat plate that balances a ball,
# make this a hybrid simulation where continuous dynamics are controlled with a discrete controller:
# if discrete controller is running at 1 Hz. Do continuous to 1-, then update with discrete to 1+, then run continuous
# to 2-, update discrete to 2+ then run continuous again.
# here it would be pretty cool if we could take the time-lag of the controller into account.
# (e.g. the time where the controller got its info is not equal to when its action if performed)

# When we update the state we do: x_k+1 = x_k + x_dot * dt. Why not: x__k+1 = x_k + x_dot * dt + 1/2 * x_2dot * dt^2?

# Could add zero-crossing detection. --> when _a_ signal crosses zero (e.g. height - radius of the ball), we want to
#  decrease the timestep such that we get a timestamp at exactly zero. (it will actually be at exactly zero because we
#  just interpolate using the xdot that we've found for that step)

# It must be easy to plot a new signal.
# Maybe adding a logger object where you could say logger.log('state', sim.x)
# And then it would kind of be like the log_data object, or just a big DataFrame (or nested dict).
# It would be cool if it had both the variables that change over time (state, etc...) but also constant/parameters.
# This logger _could_ also be used to plot stuff: log.plot(). But not per se.

# Adding randomness: don't make the random draw inside f, but do nu = mu + 1/sqrt(dt) * sigma * randn();
# at start of timestep and give that to f -> f(t+dt, x+dx, nu). If: f(t, x) = ... + w where w = N(mu, var).

# A monte carlo simulation.
