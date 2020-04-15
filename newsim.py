import numpy as np


class Ball:
    def __init__(self, mass=1., radius=0.2, spring_constant=1000., damper_coef=2., drag_coef=0.0,
                 t0=None, x0=None, gravity=9.81):
        # What is this ball like?
        self.m = mass
        self.r = radius
        self.S = np.pi * self.r * self.r
        self.k = spring_constant
        self.c = damper_coef
        self.C_d = drag_coef

        # TODO: This is arguably not a property of the ball, but rather one of the solver/sim.
        # The initial values of the drop.
        self.t0 = t0
        self.x0 = x0

        # Set up the environment, only gravity in this case.
        self.g = gravity


class ODE:
    def __init__(self, ball):
        self.ball = ball
        self.num_evaluations = 0

    def ode(self, t, x):
        # We called the function, let's update our number of evaluations.
        self.num_evaluations += 1

        # Now we can calculate xdot.
        f = self._calc_forces(t, x)
        a = f / self.ball.m
        xdot = np.array([x[1],
                         a])
        return xdot

    # Calculate the acceleration, needed to update for the derivative
    def _calc_forces(self, t, x):
        # First, calculate the force due to gravity.
        f_gravity = -self.ball.g * self.ball.m

        # Then the force from drag.
        # It acts in the opposite direction of the velocity so is only defined if our velocity is not zero.
        if x[1] != 0.:
            f_drag = 0.5 * 1.225 * x[1] * x[1] * self.ball.S * self.ball.Cd

            # Turn it in the opposite direction as the velocity.
            f_drag = f_drag * -x[1] / np.abs(x[1])
        else:
            f_drag = 0.

        # Now calculate the force from the bounce.
        # See if we're within our radius from the ground, if not, it's zero.
        if x[0] <= self.ball.r:
            f_bounce = self._calc_bounce_force(t, x)
        else:
            f_bounce = 0.

        # now return the total force
        return f_gravity + f_drag + f_bounce

    def _calc_bounce_force(self, t, x):
        # We'll model this as a mass-spring-damper system
        f_spring = -self.ball.k * (x[0] - self.ball.r)
        f_damper = -self.ball.c * x[1]

        return f_spring + f_damper


class Solver:
    def __init__(self, ode):
        self.num_iterations = 0
        self.ode = ode

    def update_state(self):
        xdot = self._euler()
        x = self.ode.calc_xdot()
        return x

    def _euler(self):
        xdot = self.ode.ode(self.t, self.x)
        return xdot


class Sim:
    def __init__(self, solver):
        self.solver = solver
        self.x = None

    def run_sim(self):
        # Initialize everything.

        # then Ball.update_state()
        #       - needs ode and solver
        # then Sensor.generate_sensor_data(Ball.state)
        #       - needs ball state and sensor characteristics based on the state
        # then StateEstimator.estimate_state(Sensor.state)
        #       - needs sensor state, but also (a simplified version of) the ode and solver.
        # then Logger.log(everything)
        #       - just needs the numbers.

        self.x = self.solver.update_state()
        print(self.x)


MyBall = Ball()

# todo:
MyODE = ODE(MyBall)
MySolver = Solver(MyODE)
MySim = Sim(MySolver)

MySim.run_sim()






