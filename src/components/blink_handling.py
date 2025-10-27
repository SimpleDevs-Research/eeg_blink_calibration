import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.signal import find_peaks


"""
Walk left from peak_idx until the signal drops below a relative threshold of the peak height (peak * (1 - rel_thresh)), OR until the slope indicates we're at valley/flat region.
- `rel_thresh`: fraction of peak height used to define base (0.1 => 10% below peak).
- `min_drop`: tiny positive value to avoid numerical issues on exact equals.
"""
def find_ascent_start(signal, peak_idx, rel_thresh=0.1, min_drop=1e-6):
    # Get the y-value of this signal
    peak_val = signal[peak_idx]
    # baseline threshold: stop once value <= threshold
    threshold = peak_val * (1 - rel_thresh)
    i = peak_idx
    # walk left while signal is >= threshold (and within bounds)
    while i > 0 and signal[i-1] >= threshold - min_drop:
        i -= 1
    return i

"""
Given an entire signal and its expected peaks, return all the re-estimated starts by iteratign through each peak.
- `rel_thresh`: fraction of peak height used to define base (0.1 => 10% below peak).
"""
def peak_starts_by_scan(signal, peaks, rel_thresh=0.1):
    starts = [find_ascent_start(signal, p, rel_thresh=rel_thresh) for p in peaks]
    return np.array(starts)

"""
Given the x-axis data (expected: unix timestamps) and y-axis data (expected: raw eye data of the eye's y-position relative to the screen), find the peaks and valleys associated with this data. We do not make any assumptions about the data itself.
Returns: a LIST of peaks and valleys 
"""
def calculate_peaks(
        _X, 
        _Y, 
        peak_height=0.5, 
        peak_prominence=1, 
        peak_width=0, 
        valley_height=0.5, 
        valley_prominence=1, 
        valley_width=0):
    """
    Given `_X` (expected: unix timestamps) and `_Y` (expected: raw eye data of the eye's y-position relative to the screen)...
    Find the peaks and valleys associated with this data. We do not make any assumptions about the data itself.
    Returns: a LIST of peaks and valleys 
    """

    # Step 1: Using `scipy.stats.zscore()`, calculate the z-scores of this data. Get its inversion too.
    z = zscore(_Y)
    inv_z = [v*-1 for v in z]

    # Step 2: initialize empty arrays for these
    peaks = None
    valleys = None
    results = None

    # Step 2: Find peaks via `scipy.signal.find_peaks()`, then by `peak_starts_by_scan()`, then sort them in order, then aggregate
    # 2a. `find_peaks()
    peak_raws, _ = find_peaks(z, height=peak_height, width=peak_width, prominence=peak_prominence, plateau_size=True)
    # CHECK: do we even have detected peaks?
    if len(peak_raws) > 0:
        peak_indices = peak_starts_by_scan(z, peak_raws)
        peak_indices.sort()
        peaks = [{'type':'peak', 'x':_X[i], 'y':z[i]} for i in peak_indices]

    # Step 3: Find valleys via `scipy.signal.find_peaks()` and inverted z-scores, then by `peak_starts_by_scan()`, then sort them in order, then aggregate
    valley_raws, _ = find_peaks(inv_z, height=valley_height, width=valley_width, prominence=valley_prominence, plateau_size=True)
    # CHECK: do we even have detected valleys
    if len(valley_raws) > 0:
        valley_indices = peak_starts_by_scan(inv_z, valley_raws)
        valley_indices.sort()
        valleys = [{'type':'valley', 'x':_X[i], 'y':z[i]} for i in valley_indices]

    # Step 4: Combine them into a singular list
    combined = []
    if peaks is not None: combined.extend(peaks)
    if valleys is not None: combined.extend(valleys)
    
    # Step 5: Sort `combined` such that the peaks and valleys are ordered respective to both peaks and valleys
    # CHECK: do we even have any? Only proceed if we do have something
    if len(combined) > 0: 
        results = sorted(combined, key=lambda v: v['x']) 

    # Step 6: Return our findings
    return results, peaks, valleys


"""
Given a participant's unique ID and the global directory where data is stored, get the estimated blinks from this participant's VR data. 
Returns: a dataframe of all first peaks and valleys in each trial
"""
def detect_vr_blinks(
        _DATA_DIR:str,
        _PARTICIPANT:str,
        **peak_args):
    
    # Define a global directory for this participant
    pdir = f"./{_DATA_DIR}/{_PARTICIPANT}/"

    # Get trials of this participant
    trials = pd.read_csv(os.path.join(pdir, "trials.csv"))
    
    # list of outputs
    all_results = []

    # iterate through trials
    for _, row in trials.iterrows():
        # Read the eye data for each trial
        trial_id = row['trial_id']

        # Read the necessary datasets
        eye = pd.read_csv(os.path.join(pdir, f"{trial_id}/calibration/eye.csv"))
        start_ms = eye['unix_ms'].iloc[0] + 5000
        end_ms = eye['unix_ms'].iloc[-1] - 500
        eye = eye[eye['unix_ms'].between(start_ms, end_ms)]
        brdf = pd.read_csv(os.path.join(pdir, f"{trial_id}/calibration/blink_ranges.csv"))

        # Iterate through each blink range
        for _, row in brdf.iterrows():

            # Get necessary timestamps
            overlap_counter = row['overlap_counter']
            range_start_ms = row['start_unix_ms']
            range_end_ms = row['end_unix_ms']
            _eye = eye[eye['unix_ms'].between(range_start_ms, range_end_ms)]

            # Get the relevant x and y data
            x = _eye['unix_ms'].to_list()
            y = _eye['gaze_target_screen_pos_y'].to_list()
            
            # Use `find_peaks()` we've created just above to detect combined peaks
            combined, peaks, valleys = calculate_peaks(x, y, **peak_args)

            # Only contribute to `all_results` if `combined` is not None
            if combined is not None: 
                results = [{'trial_id':trial_id, 'overlap_counter':overlap_counter, **c} for c in combined]
                all_results.extend(results)

    # We'll combine all our results into a single dataframe
    df = pd.DataFrame(all_results)
    df = df.groupby(['trial_id', 'overlap_counter'], group_keys=False).apply(lambda g: g.sort_values('x'))

    # We'll extract the first of each trial and overlap
    first_peaks = df.groupby(['trial_id', 'overlap_counter'], as_index=False).first()

    # return the dfs
    return first_peaks, df