# A simple particle filter. I say simple, because I have no idea how difficult these can get.
import copy
import numpy as np


class ParticleFilter(object):
    def __init__(self, particle_class, num_particles=5):
        self.num_particles = num_particles
        self.particles = self.init_particles(particle_class)
        self.weights = None
        self.estimated_state = None  # TODO: what's the right name. x_hat, x_star?
        # TODO: maybe a logger should be part of this?

    # Initialize all the particles.
    def init_particles(self, particle_obj):
        # Each particle is a copy of the object that we're propagating.
        # Use deepcopy to avoid that there are any connections between particles.
        # Is it a little weird that we give the filter 1 'particle' and then make copies of it?

        # TODO: We should change the state of each particle a little bit.
        return [copy.deepcopy(particle_obj) for _ in range(self.num_particles)]

    # Based on a (past) measurement, update our estimated state to the current time.
    def estimate_state(self, t_now, t_measurement, measured_state):
        # All our balls have been propagated to some time t. If we have a recent measurement, let's use that to select
        # the best particles. If we don't have a recent measurement, we just propagate the balls without doing any
        # corrections.
        # This construction does mean that we can't get a measurement from before our last propagation. For now this is
        # fine, because there is no delay between taking the measurement and feeding it to the particle filter.
        if t_measurement > self.particles[0].t:
            self._propagate_particles(t_measurement)
            self._update_weights(measured_state)
            self._calc_estimated_state()
            self._resample()
            self._regularize()

        # Check if we still need to propagate a bit to the current time.
        if t_now > self.particles[0].t:
            # Here we can't update the weights, resample, and regularize because we don't have a measurement between
            # t_now and the time where are particles live in.
            self._propagate_particles(t_now)
            self._calc_estimated_state()

    def _propagate_particles(self, t_new):
        for particle in self.particles:
            particle.update_state_to_t(t_new)

    def _update_weights(self, measured_state):
        # For now just an even weight.
        self.weights = 1/self.num_particles * np.ones(self.num_particles)

    def _resample(self):
        pass

    def _regularize(self):
        pass

    def _calc_estimated_state(self):
        # Take a weighted mean.
        states = np.asarray([particle.x for particle in self.particles])
        weighted_sum = np.dot(self.weights, states)
        weighted_mean = weighted_sum / self.num_particles

        # And that's our estimated state.
        self.estimated_state = weighted_mean
