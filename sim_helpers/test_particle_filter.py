# TODO: learn how to use the python unittest module
# import unittest
#
#
# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)
#
#
# if __name__# We are cheating a little bit w.r.t. real-life here. That's because in the real-world you're making a particle
# # filter to estimate something of which you don't perfectly know how it behaves. In this case, we're using the
# # same dynamics to simulate our ball and simulate all balls of the particle filter. This particle filter will
# # thus be able to perfectly imitate the simulation and get very good results.
# # TODO: make this particle filter more realistic by making the ParticleFilterBall class that's an approximation
# #  of Ball.
# ball_for_particle_filter = Ball()
# ball_for_particle_filter_solver = Euler(ball_for_particle_filter.ode, dt=0.01)
# ball_for_particle_filter.set_up_solver(ball_for_particle_filter_solver)
#
# # Alright now we can set up our particle filter.
# particle_filter = ParticleFilter(ball_for_particle_filter) == '__main__':
#     unittest.main()

# Testing
from solvers import Euler
from ball import Ball
from particle_filter import ParticleFilter
import numpy as np


particle_filter.estimate_state(np.asarray(10), np.asarray(5), np.array([1, 4]))