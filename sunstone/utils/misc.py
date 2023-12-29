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

def cross(series, times, cross_thresh=0, direction='rising'):
    """
    Find index values where the data values cross a threshold.

    Parameters:
        series (ndarray): Input series.
        times (ndarray): Corresponding time values.
        cross_thresh (float): Threshold value.
        direction (str): Direction of the crossing. Can be 'rising', 'falling', or 'cross'.

    Returns:
        ndarray: Index values where the data values cross the threshold.

    """
    # Find if values are above or bellow yvalue crossing:
    above = series > cross_thresh
    below = np.logical_not(above)
    left_shifted_above = above[1:]
    left_shifted_below = below[1:]
    x_crossings = []
    # Find indexes on left side of crossing point
    if direction == 'rising':
        idxs = (left_shifted_above & below[0:-1]).nonzero()[0]
    elif direction == 'falling':
        idxs = (left_shifted_below & above[0:-1]).nonzero()[0]
    else:
        rising = left_shifted_above & below[0:-1]
        falling = left_shifted_below & above[0:-1]
        idxs = (rising | falling).nonzero()[0]

    # Calculate x crossings with interpolation using formula for a line:
    # x_crossings = times[idxs]
    x1 = times[idxs]
    x2 = times[idxs+1]
    y1 = series[idxs]
    y2 = series[idxs+1]
    x_crossings = (cross_thresh-y1)*(x2-x1)/(y2-y1) + x1

    return x_crossings