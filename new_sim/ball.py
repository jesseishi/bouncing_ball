import numpy as np


# TODO: this should probably be the BouncingBall class.
class Ball(object):
    def __init__(self, mass=1., radius=0.2, spring_constant=1000., damper_coef=2., drag_coef=0.0,
                 t0=None, x0=None, gravity=9.81):
        # What is this ball like?
        self.m = mass
        self.r = radius
        self.S = np.pi * self.r * self.r
        self.k = spring_constant
        self.c = damper_coef
        self.C_d = drag_coef

        # Set up the environment, only gravity in this case.
        self.g = gravity

        # And finally, make this ball ready to be simulated. This sets the initial values and sets the number of
        #  ode evaluations to zero.
        self.t, self.x = self._set_initial_values(t0, x0)
        self.num_evaluations = 0

        # Set solver to None. This is always None when initializing the ball because the solver needs a ball's ODE to be
        #  set up. So once the solver is set up, assign it wit set_up_solver.
        self.solver = None

    @staticmethod
    def _set_initial_values(t0, x0):
        # Set the initial values for time and state.
        if t0 is None:
            t0 = np.array([0])
        if x0 is None:
            x0 = np.array([1, 0])

        return t0, x0

    def set_up_solver(self, solver):
        self.solver = solver

    # This is the ODE that we give to the solver.
    def ode(self, t, x):
        # For starters, let's increase the number of evaluations we did.
        self.num_evaluations += 1

        # Now we calculate the acceleration using Newton's second law
        f = self._calc_forces(t, x)
        a = f / self.m

        # Then xdot can be filled in.
        xdot = np.array([x[1],
                         a])

        return xdot

    # Calculate the forces needed for the ODE.
    def _calc_forces(self, t, x):
        # First, gravity.
        f_gravity = -self.g * self.m

        # Then, drag. But only if we have a velocity
        if x[1] != 0:
            # The famous drag formula.
            f_drag = 0.5 * 1.225 * x[1]*x[1] * self.S * self.C_d

            # And turn it into the direction opposite to our velocity.
            f_drag *= -x[1] / np.abs(x[1])
        else:
            f_drag = 0

        # Finally, the force from the bounce. If we're within a radius distance.
        if x[0] < self.r:
            # We'll model this as a mass-spring-damper system.
            f_spring = -self.k * (x[0] - self.r)
            f_damper = -self.c * x[1]
            f_bounce = f_spring + f_damper
        else:
            f_bounce = 0

        return f_gravity + f_drag + f_bounce

    # This function updates the state of the ball, using the solver it has selected.
    def update_state(self):
        if self.solver is None:
            raise Exception("You first have to assign a solver using Ball.set_up_solver")
        self.t, self.x = self.solver.update_state(self.t, self.x)

    def update_state_to_dt(self, dt):
        if self.solver is None:
            raise Exception("You first have to assign a solver using Ball.set_up_solver")
        self.t, self.x = self.solver.update_state_to_dt(self.t, self.x, dt)

    def update_state_max_to_dt(self, dt):
        if self.solver is None:
            raise Exception("You first have to assign a solver using Ball.set_up_solver")
        self.t, self.x = self.solver.update_state_max_to_dt(self.t, self.x, dt)



if __name__ == "__main__":
    my_ball = Ball()
    my_ball.ode(my_ball.t, my_ball.x)
    print(my_ball.x)
    my_ball.update_state()
    print(my_ball.x)