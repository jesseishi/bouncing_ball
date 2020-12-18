import numpy as np


# TODO: read the stuff about sensors and noise again in Tuckers article. IIRC, it only applied to continuous noise on
#  the ODE. Noise of a sensor is discrete and doesn't change depending on the frequency at which you call it.
# Also seed the random number generator and maybe make a seed_cycler, that way you don't have to worry when seeding
# multiple sensors. (again, read Tucker's article). Actually, let's play around with this. Let's see if we can make two
# sensors have the same noise.
# A simple sensor that adds noise to a true state according to a gaussian distribution.
class GaussianSensor(object):
    def __init__(self, sigma=0, mu=0, state_index_start=None, state_index_end=None):
        # TODO: the input should be a little more real here. It should actuall specify the mu and sigma for each thing we measure
        # together with which part of the state we actually measure. (e.g. state is pos and vel, but the sensor only measures vel)
        # In reality a sensor can be as complicated as the object that you simulate (the ball), heck, it can be one of the main 
        # things that you want to simulate in the first place. This object isn't really reflecting that.
        # This could be a state-measuring sensor though. Maybe that's a good name for it.
        self.sigma = sigma
        self.mu = mu
        self.state_index_start = state_index_start
        self.state_index_end = state_index_end
        self.x_star = None  # The measurement of the state. TODO: what's the proper name?

    def simulate_measurement(self, x):
        x = x[self.state_index_start:self.state_index_end]
        # Calculate the noise.
        self._calc_noise(x)

        # Add that to the measurement.
        self.x_star = x + self.noise

    def _calc_noise(self, x):
        # Make noise with a gaussian distribution.
        mu_bias = self.mu * np.ones_like(x)
        sigma_noise = self.sigma * np.random.randn(x.size).reshape(x.shape)
        self.noise = mu_bias + sigma_noise


# Testing
if __name__ == "__main__":
    my_sensor = GaussianSensor(0, 0, 1, 3)
    x = np.arange(0, 1, 0.1)
    my_sensor.simulate_measurement(x)
    print(my_sensor.x_star)