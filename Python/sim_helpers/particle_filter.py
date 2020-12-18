# A simple particle filter. I say simple, because I have no idea how difficult these can get.
import copy
import numpy as np
import random


class ParticleFilter(object):
    def __init__(self, particle_class, num_particles=50):
        self.num_particles = num_particles
        self.particles = self.init_particles(particle_class)
        self.weights = None
        self.estimated_state = self.particles[0].x  # TODO: what's the right name. x_hat, x_star? also make None again.
        self.estimated_states = np.asarray([particle.x for particle in self.particles])  # TODO: make None again in the future, but currently we init with a measurement.
        # TODO: maybe a logger should be part of this?

    # Initialize all the particles.
    def init_particles(self, particle_obj):
        # Use deepcopy to avoid that there are any connections between particles.
        # Is it a little weird that we give the filter 1 already initialized and then make copies of it?
        self.particles = [copy.deepcopy(particle_obj) for _ in range(self.num_particles)]
        # TODO: it's not really fair that these particles have the same starting value as the real ball.
        #  in the real-world, you'd have a ParticleFilterBall that's a model of MyBall, which initializes without a
        #  state. The least we can do here is add some noise.
        self._regularize()

        # Returning a self thingy doesn't make much sense, but this works since I do want to use this _regularize
        # function inside of here.
        return self.particles

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

            # We could also resample and regularize each time we run this thing.
            self._resample()
            self._regularize()

        # Check if we still need to propagate a bit to the current time.
        if t_now > self.particles[0].t:
            # Here we can't update the weights, resample, and regularize because we don't have a measurement between
            # t_now and the time where are particles live in.
            self._propagate_particles(t_now)
            self._calc_estimated_state()

        self.estimated_states = np.asarray([particle.x for particle in self.particles])

    def _propagate_particles(self, t_new):
        for particle in self.particles:
            particle.update_state_to_t(t_new)

    def _update_weights(self, measured_state):
        # Get the difference between the measured state and each particle.
        diff = [particle.x - measured_state for particle in self.particles]

        # For each, calculate the infinity norm.
        dist = np.linalg.norm(diff, np.inf, axis=1)

        # This inverse of that is our weight but protect against 0 division.
        weights = np.asarray([1/d if d != 0 else 1/0.001 for d in dist])

        # Finally, we should normalize the weights such that the sum is 1.
        self.weights = weights / np.sum(weights)

    def _resample(self):
        # Pick random particles using their weights.
        self.particles = np.random.choice(self.particles, size=self.num_particles, p=self.weights)

        # Now we need to make sure that each one is unique again, and not just a pointer to each other.
        self.particles = [copy.deepcopy(particle) for particle in self.particles]

    def _regularize(self):
        # Add some random perturbations to the state of each particle.
        for particle in self.particles:
            sigma_x = [0.1, 0.0]  # TODO: move to init.
            particle.x += np.asarray([random.gauss(0, sigma) for sigma in sigma_x])

    def _calc_estimated_state(self):
        # Take a weighted mean.
        states = np.asarray([particle.x for particle in self.particles])
        weighted_sum = np.dot(self.weights, states)
        weighted_mean = weighted_sum / np.sum(self.weights)  # Not strictly necessary as weights sum to 1.

        # And that's our estimated state.
        self.estimated_state = weighted_mean
