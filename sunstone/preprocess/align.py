
import warnings
import numpy as np
from scipy.interpolate import interp1d

from sunstone.utils import events_binary

def align_to_trigger(
    unaligned_timestamps, unaligned_trigger,
    aligned_timestamps, aligned_trigger,
    before=None,
    after=None,
    errors='warn', 
    message=True
):
    """
    Align timestamps using a binary signal `trigger`.

    The binary signal `trigger` should end with the OFF value.
    The number of OFF switches must equal for the
    `unaligned_trigger` and `aligned_trigger`.

    Parameters
    ----------
    unaligned_timestamps : numpy.ndarray
        Array of timestamps for `unaligned_trigger` - timestamps from stimulation device that
        sends the trigger, usually starting from zero.
    unaligned_trigger : numpy.ndarray
        Binary signal with same length as `unaligned_timestamps` - the ON-OFF signal (usually 0-5V)
        being sent from the stimulation device to the recording device.
    aligned_timestamps : numpy.ndarray
        Array of timestamps for `aligned_trigger`. The timestamps of the recording device,
        e.g. absolute timestamps of imaging acquisition. 
    aligned_trigger : numpy.ndarray
        Binary signal with same length as `aligned_timestamps`. The ON-OFF signal 
        being recorded by the recording device (e.g. PrairieView). 

    Returns
    -------
    object : callable
        Interpolator for aligning arbitrary "unaligned" timestamps.
    """
    input_message = message

    aligned_falls = events_binary(
        aligned_trigger, aligned_timestamps, 'falling'
    )
    unaligned_falls = events_binary(
        unaligned_trigger, unaligned_timestamps, 'falling'
    )
    if after is not None:
        unaligned_falls = unaligned_falls[:-after]
    if before is not None:
        unaligned_falls = unaligned_falls[before:]

    if len(aligned_falls) == len(unaligned_falls):
        pass
    else:
        triggers_missing = len(unaligned_falls) - len(aligned_falls)

        if triggers_missing < 0:
            raise ValueError(
                "More triggers in aligned trigger "
                "array than unaligned array"
            )

        message = (
            "Trigger alignment is imperfect by "
            f"`{triggers_missing}` "
            "number of triggers"
        )
        if errors not in {'ignore', 'verbose', 'warn'}:
            raise ValueError(message)

        mean_interval = np.diff(aligned_falls).mean()
        last_fall_diff = aligned_timestamps.max() - aligned_falls.max()
        # the first fall is way after the aligned timestamps
        first_fall_diff = aligned_falls.min() - aligned_timestamps.min()
        # assume aligned trigger finished the aligned trigger started later
        if last_fall_diff >= (mean_interval * 2):
            message += (
                '; assuming aligned trigger finished: last fall '
                f'was `{last_fall_diff}` before end of timestamps, '
                f'and average interval between pulses was `{mean_interval}`.'
            )
            unaligned_falls = unaligned_falls[triggers_missing:]
        elif first_fall_diff >= (mean_interval * 2):
            message += (
                '; assuming aligned trigger started late: first fall '
                f'was `{first_fall_diff}` after start of timestamps, '
                f'and average interval between pulses was `{mean_interval}`.'
            )
            unaligned_falls = unaligned_falls[:-triggers_missing]
        elif input_message:
            # first_fall_diff = aligned_falls.min() - aligned_timestamps.min()
            output = input(f"{message}\n At the end (R) OR at the beginning (L) "
                           "OR number with 'L' and 'R' (e.g. 1L) indicating "
                           "how many left or right? ")
            if output == 'L':
                unaligned_falls = unaligned_falls[triggers_missing:]
            elif output == 'R':
                unaligned_falls = unaligned_falls[:-triggers_missing]
            elif 'L' in output:
                number = int(output.replace('L', ''))
                assert number < triggers_missing, "Indicated number must be smaller than missing triggers"
                unaligned_falls = unaligned_falls[number:-(triggers_missing-number)]
            elif 'R' in output:
                number = int(output.replace('R', ''))
                assert number < triggers_missing, "Indicated number must be smaller than missing triggers"
                unaligned_falls = unaligned_falls[triggers_missing-number:-number]
            else:
                message += (
                    '; cannot determine direction of misalignment - '
                    'set the before and after parameters!'
                )            
                raise ValueError(message)
        else:
            message += (
                '; cannot determine direction of misalignment - '
                'set the before and after parameters!'
            )            
            raise ValueError(message)

        if errors == 'ignore':
            pass
        elif errors == 'verbose':
            print(message)
        elif errors == 'warn':
            warnings.warn(message)

    assert len(aligned_falls) == len(unaligned_falls), "STILL MISALIGNED!!!"
    return interp1d(
        unaligned_falls, aligned_falls,
        bounds_error=False, fill_value='extrapolate'
    )