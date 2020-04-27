# TODO:
# - write an introduction and go over comments.
# - investigate how you would expand this with a __init__.py file.
# - rename to RungeKuttaAdaptiveStep, etc...
# - make RungeKuttaAdaptiveStepImplicit
# - add num_evalutions, or decide to keep it inside the ode.
# - add classmethod, .from_yaml() and set up any method with a yaml file.
# - add exceptions and catch variables coming in that aren't good (when something is not a numpy array).
# - add zero-crossing detection.
# - break solvers up into smaller functions.


import numpy as np


# Generic RungeKutta solver class out of which the individual solvers will be made by defining the butcher table.
class RungeKutta(object):
    def __init__(self, ode, a, b, c, dt=0.5):
        self.ode = ode
        self.a = a
        self.b = b
        self.c = c
        self.dt = dt
        self.k = None

    def update_state(self, t, x):
        # Find the derivatives.
        self._calc_k(t, x)

        # Now update the time and state.
        t_new = t + self.dt
        x_new = x + self.dt * np.dot(self.k, self.b.T)

        return t_new, x_new

    def update_state_to_t(self, t_old, x_old, t_new):
        # Safe the original dt set up for this solver.
        original_dt = self.dt

        # TODO: t_old is an array for no good reason.
        t_old = t_old[0]
        t_new = t_new[0]

        # Figure out how many steps we should take.
        num_steps = (t_new - t_old) / self.dt
        num_steps = np.ceil(num_steps).astype(int)

        # Then calculate the dt that we need.
        self.dt = (t_new - t_old) / num_steps

        # Now take the right number of steps, updating the state and time in between.
        for _ in range(num_steps):
            t_old, x_old = self.update_state(t_old, x_old)

        # Check that we did that correctly.
        if t_old != t_new:
            # Because of floating point errors, we might need to take an extra step.
            self.dt = t_new - t_old
            t_old, x_old = self.update_state(t_old, x_old)

        # Now check that we did that correctly again.
        if t_old != t_new:
            raise Exception(f"Didn\'t update state to the right time. Target {t_new} but got to {t_old}")

        # Reset the original dt.
        self.dt = original_dt

        # And return our updated time and state.
        return np.asarray([t_old]), x_old


    def _calc_k(self, t, x):
        # Find the change in state for this timestep.
        self.k = np.zeros((len(x), len(self.c)))

        # For each row in the butcher table, find the derivative.
        for i in range(len(self.c)):
            self.k[:, i] = self.ode(t + self.c[i] * self.dt, x + self.dt * np.dot(self.k, self.a[i, :].T))


# The Euler solver is a Runge-Kutta Solver with a certain butcher tableau.
class Euler(RungeKutta):
    def __init__(self, ode, dt):
        a = np.array([[0]])
        b = np.array([1])
        c = np.array([0])
        super().__init__(ode, a, b, c, dt=dt)


# The Runge-Kutte 4 solver is simply defined with a butcher tableau.
class RK4(RungeKutta):
    def __init__(self, ode, dt):
        a = np.array([[0, 0, 0, 0],
                      [0.5, 0, 0, 0],
                      [0, 0.5, 0, 0],
                      [0, 0, 1, 0]])
        b = np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6])
        c = np.array([0, 0.5, 0.5, 1])
        super().__init__(ode, a, b, c)


class AdaptiveStep(RungeKutta):
    def __init__(self, ode, a, b, b_hat, c, p, safety_factor=0.8, min_dt=1e-5, max_dt=1, tol_abs_x=1, tol_rel_x=1):
        super().__init__(ode, a, b, c)

        # b_hat is used to calculate the lower-order solution.
        self.b_hat = b_hat

        # To find the new timestep we use an algorithm that needs the order of the lower-order solution (p) and a
        #  safety factor.
        self.p = p
        self.safety_factor = safety_factor

        # Set a maximum allowable minimum and maximum dt together with the dt that we'll use the first step.
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.dt = max_dt  # We're feeling lucky.

        # Our tolerances for accepting a solution. Both absolute and relative difference.
        self.tol_abs_x = tol_abs_x
        self.tol_rel_x = tol_rel_x

    # Since this is an adaptive-step class, we need to update this function.
    def update_state(self, t, x):

        # Let's keep trying until we find a timestep that gives an acceptable solution.
        found_acceptable_timestep = False
        while not found_acceptable_timestep:
            # Find our derivatives.
            self._calc_k(t, x)

            # Make a higher-order and lower-order update.
            # TODO: We _could_ make this into a function in the RungeKutta class. _calc_new_state(x, 2_outputs=True)
            x_higher_order = x + self.dt * np.dot(self.k, self.b.T)
            x_lower_order = x + self.dt * np.dot(self.k, self.b_hat.T)

            # Now calculate the absolute and relative difference.
            diff_abs_x = np.linalg.norm(x_higher_order - x_lower_order, ord=np.inf)
            # protect against dividing by 0.
            if np.min(x_lower_order) != 0:
                diff_rel_x = np.linalg.norm((x_higher_order - x_lower_order) / x_higher_order, ord=np.inf)
            else:
                # just set it to the same value as abs.
                diff_rel_x = diff_abs_x

            # See if our solution is within our tolerances.
            if (diff_abs_x < self.tol_abs_x) & (diff_rel_x < self.tol_rel_x):
                found_acceptable_timestep = True  # This will break us out of the loop.

            # We didn't find an acceptable solution, but we don't want to lower our timestamp anymore.
            elif self.dt <= self.min_dt:
                print("The minimum allowable dt of {}, didn't result in an acceptable solution.".format(self.dt))
                break

            # Alright, let's try again but with a smaller timestep.
            else:
                self._calc_dt(diff_abs_x, diff_rel_x)

        # Now that we've found the right timestep, we can return the solution
        t_new = t + self.dt
        x_new = x_higher_order

        # Our solution might have been so good that we actually want to increase our timestep.
        self._calc_dt(diff_abs_x, diff_rel_x)

        # Now finally return the new time and state.
        return t_new, x_new

    # Function that calculates the dt to get the desired accuracy.
    def _calc_dt(self, diff_abs, diff_rel):

        # First check if our differences aren't zero, to protect against dividing by zero.
        if (diff_abs != 0) & (diff_rel != 0):
            new_dt = np.min([self.safety_factor * self.dt * (self.tol_abs_x / diff_abs) ** (1 / (self.p + 1)),
                             self.safety_factor * self.dt * (self.tol_rel_x / diff_rel) ** (1 / (self.p + 1))])

            # There are some restrictions on how dt may change.
            # 1. Never >2x higher or lower.
            # 2. Out of the min and max bounds.
            lower_bound = np.max([self.dt / 2, self.min_dt])
            upper_bound = np.min([self.dt * 2, self.max_dt])
            self.dt = np.max([np.min([new_dt, upper_bound]), lower_bound])
        else:
            # We did pretty good, zero error.
            self.dt = self.max_dt

    # Update the state dt into the future.
    def update_state_to_dt(self, t, x, dt):
        # We'll keep updating until the newest update has overshot our target.
        # In that case we can set the dt and run once more from the last update.
        t_new, x_new = t, x  # TODO: make uniform terminology around this, this is confusing.
        while t_new < t+dt:
            t_last, x_last = t_new, x_new
            t_new, x_new = self.update_state(t_last, x_last)

        # So now we set the dt to make it run to t+dt
        self.dt = (t+dt) - t_last
        t_final, x_final = self.update_state(t_last, x_last)
        # todo: hmm weird that sometimes doesn't work, but it does if we just run it one more time...
        if t_final != t+dt:
            self.dt = (t+dt) - t_final
            t_final, x_final = self.update_state(t_final, x_final)
        if t_final != t+dt:
            raise Exception("Couldn't update from {} with timestep {}, "
                            "I got to {}".format(t, dt, t_final))

        # Return it.
        return t_final, x_final

    # Update the state as far as you can, but maximum to dt.
    def update_state_max_to_dt(self, t, x, dt):
        # Just limit the current step size
        self.dt = np.min([self.dt, dt])
        t_new, x_new = self.update_state(t, x)

        # Check if we did ok.
        if t_new - t > dt:
            print("Overshot the maximum dt. Tried from {} to {}, but went to {}".format(t, t+dt, t_new))

        # And return.
        return t_new, x_new

    # TODO: get rid of the other two functions to clean it up.
    def update_state_to_t(self, t_old, x_old, t_target):
        t = t_old
        x = x_old
        while t < t_target:
            t_last = t
            x_last = x
            t, x = self.update_state(t, x)

        dt = t_target - t_last
        t_final, x_final = self.update_state_to_dt(t_last, x_last, dt)  # This isn't great.
        # TODO: rewrite this function the right way. I think that there's a bug somewhere which prevented me from doing it the first time.

        return t_final, x_final


class DP54(AdaptiveStep):
    # TODO: I think that I see now why this way might not be ideal. Currently the default safety factor is defined in
    #  this class, but also in AdaptiveStep. It would be nice if they had one default value.
    # TODO: Find a better name than tol_abs_x.
    def __init__(self, ode, safety_factor=0.8, min_dt=1e-5, max_dt=1, tol_abs_x=1, tol_rel_x=1):
        # First define the butcher table for this method.
        a = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [1 / 5, 0, 0, 0, 0, 0, 0],
                      [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
                      [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
                      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
                      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
                      [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]], dtype=float)
        b = a[-1, :]  # TODO: because b is the last row of a, we can reuse k7_old for k1_new. Can write a new _calc_k().
        b_hat = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
        c = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])

        # Plug that into the standard adaptive step class.
        # p = 4, because the lower-order of DP5(4) is 4.
        super().__init__(ode, a, b, b_hat, c, p=4, safety_factor=safety_factor, min_dt=min_dt, max_dt=max_dt,
                         tol_abs_x=tol_abs_x, tol_rel_x=tol_rel_x)


# A generic class for solvers using an implicit method to calculate the derivatives.
class ImplicitSolver(RungeKutta):
    def __init__(self, ode, a, b, c, max_iterations=100, tol_abs_k=1, tol_rel_k=1):
        super().__init__(ode, a, b, c)

        # Set the maximum number of iterations to converge on the derivatives.
        self.max_iterations = max_iterations

        # Set the tolerance for accepting k.
        self.tol_abs_k = tol_abs_k
        self.tol_rel_k = tol_rel_k

    # The only thing that sets implicit methods apart is the calculation of the set of derivatives (k).
    def _calc_k(self, t, x):
        # Let's start with a guess for k, use the previous value. If that doesn't exist, guess 0.
        if self.k is not None:
            k_old = self.k
        else:
            k_old = np.zeros((len(x), len(self.c)))

        # And initialize the size of k_new
        k_new = np.empty_like(k_old)

        # Now we keep trying until we find an acceptable k.
        found_acceptable_k = False
        num_iterations = 0
        while not found_acceptable_k:
            # Start by updating k.
            for i in range(len(self.c)):
                # TODO: maybe make this update a separate function in the RangeKutta class so we can be sure we don't
                #  make mistakes.
                # We just plug our old guess into the butcher tableau and see what our new estimate it.
                k_new[:, i] = self.ode(t + self.c[i] * self.dt, x + self.dt * np.dot(k_old, self.a[i, :].T))

            # Now we check if the new k is acceptable.
            # TODO: this looks an awful lot like finding an acceptable score for the adaptive step method.
            #  Maybe we can combine this into a general function?
            diff_abs_k = np.linalg.norm(k_new - k_old, ord=np.inf)
            # Protect against dividing by 0.
            if np.min(k_new) != 0:
                diff_rel_k = np.linalg.norm((k_new - k_old) / k_new, ord=np.inf)
            else:
                diff_rel_k = diff_abs_k

            # Let's see if we found an acceptable solution for k.
            if (diff_abs_k < self.tol_abs_k) & (diff_rel_k < self.tol_rel_k):
                found_acceptable_k = True

            # If we didn't find a solution, check if we want to try another time.
            elif num_iterations == self.max_iterations:
                print("Couldn't converge on a set of derivatives (k) after {} iterations".format(num_iterations))
                break

            # Alright let's try again then.
            else:
                k_old = k_new

        # So now we've converged on a certain set of derivatives. Let's store them.
        self.k = k_new


# The second order Gauss-Legendre method. Our first implicit solver.
class GL2(ImplicitSolver):
    def __init__(self, ode, max_iterations=100, tol_abs_k=1, tol_rel_k=1):
        # Define the butcher tableau.
        a = np.array([[0.5]])
        b = np.array([1])
        c = np.array([0.5])
        super().__init__(ode, a, b, c, max_iterations=max_iterations, tol_abs_k=tol_abs_k, tol_rel_k=tol_rel_k)


# TODO: add AdaptiveStepAndImplicitSolver as a subclass.

# And then finally the fourth-order, adaptive step, implicit method solver: Gauss-Legendre 4(3).
class GL43(AdaptiveStep, ImplicitSolver):
    def __init__(self, ode, safety_factor=0.8, min_dt=1e5, max_dt=1, tol_abs_x=1, tol_rel_x=1,
                 max_iterations=100, tol_abs_k=1, tol_rel_k=1):
        # Define the butcher tableau.
        a = np.array([[1/4,                  1/4 - 1/6*np.sqrt(3)],
                      [1/4 + 1/6*np.sqrt(3), 1/4]])
        b = np.array([1/2, 1/2])
        b_hat = np.array([1/2 + 1/2*np.sqrt(3), 1/2 - 1/2*np.sqrt(3)])
        c = np.array([1/2 - 1/6*np.sqrt(3), 1/2 + 1/6*np.sqrt(3)])

        # Initialize
        AdaptiveStep.__init__(self, ode, a, b, b_hat, c, p=3, safety_factor=safety_factor, min_dt=min_dt, max_dt=max_dt,
                         tol_abs_x=tol_abs_x, tol_rel_x=tol_rel_x)
        ImplicitSolver.__init__(self, ode, a, b, c, max_iterations=max_iterations, tol_abs_k=tol_abs_k, tol_rel_k=tol_rel_k)