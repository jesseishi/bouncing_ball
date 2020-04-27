# TODO: this is probably not the most efficient logger but it gets the job done.
#  It'd be nice to have this use pandas and name the columns and stuff. (although that doesn't work as well for vectors)
import numpy as np


class DataLogger(object):
    def __init__(self, t, data):
        t, data = self.asarray(t, data)
        self.t = np.array(t)
        self.data = np.array([data])

    def log(self, t, data):
        t, data = self.asarray(t, data)
        self.t = np.hstack((self.t, t))
        self.data = np.vstack((self.data, data))

    @staticmethod
    def asarray(t, data):
        return np.asarray(t), np.asarray(data)


# Testing
if __name__ == "__main__":
    ball_logger = DataLogger(0, [1, 0])
    ball_logger.log(1, [2, 3])
    ball_logger.log(3, [2, 6])
    print(ball_logger.t, ball_logger.data)
