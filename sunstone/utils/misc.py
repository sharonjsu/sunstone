import numpy as np
from scipy.ndimage import gaussian_filter


def events_binary(
        series, times, direction='rising', **kwargs_gaussian_filter):
    """
    Extract event times from a binary signal.

    Parameters:
        series (ndarray): Input binary signal.
        times (ndarray): Corresponding time values.
        direction (str): Direction of the event. Can be 'rising', 'falling', or 'cross'.
        **kwargs_gaussian_filter: Keyword arguments for scipy.ndimage.gaussian_filter.

    Returns:
        ndarray: Event times.

    """
    if kwargs_gaussian_filter:
        series = gaussian_filter(series, **kwargs_gaussian_filter)

    thresh = np.ptp(series)/2 + np.min(series)

    return cross(series, times, thresh, direction=direction)

