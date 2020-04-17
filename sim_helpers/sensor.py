import numpy as np


# TODO: read the stuff about sensors and noise again in Tuckers article. IIRC, it only applied to continuous noise on
#  the ODE. Noise of a sensor is discrete and doesn't change depending on the frequency at which you call it.
# Also seed the random number generator and maybe make a seed_cycler, that way you don't have to worry when seeding
# multiple sensors. (again, read Tucker's article). Actually, let's play around with this. Let's see if we can make two
# sensors have the same noise.
# A simple sensor that adds noise to a true state according to a gaussian distribution.
class GaussianSensor(object):
    def __init__(self, sigma=1, mu=0):
        # TODO: the input should be a little more real here. It should be encouraged to give it a vector of mu's and
        #  sigma's the size of x.
        self.sigma = sigma
        self.mu = mu
        self.x_star = None  # The measurement of the state. TODO: what's the proper name?

    def simulate_measurement(self, x):
        # Calculate the noise.
        self._calc_noise(x)

        # Add that to the measurement.
        self.x_star = x + self.noise

    def _calc_noise(self, x):
        # Make noise with a gaussian distribution.
        mu_noise = self.mu * np.ones_like(x)
        sigma_noise = self.sigma * np.random.randn(x.size).reshape(x.shape)
        self.noise = mu_noise + sigma_noise
