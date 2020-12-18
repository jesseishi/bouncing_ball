import numpy as np


# Call this TimeBasedEvents
# TODO: also add things that happen once (say, at t = 2.4323 s).
# TODO: don't be dt based but just t based.
class Events(object):
    def __init__(self, events_dict, num_digits=2):
        # Events dict is {'event_1': 50 Hz, 'event2': 20Hz}
        # TODO, could add a custom start time offset.
        self.events_dict = events_dict

        # Set the number of digits with which we calculate, this is done to avoid rounding errors.
        self.num_digits = num_digits

    # Based on the current time, get the dt to and the name of the next event.
    # If there are multiple events with the lowest dt, it returns a list with them.
    def get_next_event_dt_and_id(self, t_now):
        # Get an array with all the dt to the next event and find the lowest one.
        event_dts = np.array([self._get_dt_to_event(t_now, freq) for event, freq in self.events_dict.items()])

        dt = np.min(event_dts[event_dts > 0])

        # Find all the events with the lowest dt.
        events_with_min_dt = [event for event, freq in self.events_dict.items() if (self._get_dt_to_event(t_now, freq) == dt)]

        # Return.
        return np.round(dt, self.num_digits), events_with_min_dt

    # Finds the dt to the next event for each frequency at which the event happens.
    def _get_dt_to_event(self, t, freq):
        dt_to_event = np.round((1/freq) - t % (1/freq), self.num_digits)
        # If an event has a dt of zero, it means that it was executed the previous step.
        # The next time it will happen is the 1/sampling frequency.
        if dt_to_event == 0:
            dt_to_event = 1/freq

        # Return.
        return dt_to_event


# Testing
if __name__ == "__main__":
    my_events = Events({'state_estimator': 1, 'logger': 0.5, 'sensor': 20})
    for i in np.arange(0, 2, 0.01):
        dt, next_event = my_events.get_next_event_dt_and_id(i)
        print(i, dt, next_event)
