# Eeg Blink Calibration

This is a Python-based notebook repository containing Jupyter Notebooks all pertaining to our lab's EEG, Eye-Tracking, and VR experiments.

## File Summary

|File|Status|Description|
|:--|:-:|:--|
|`read_eeg.ipynb`|Archive|Possibly the oldest notebook. Was initially used as a preliminary notebook for debugging EEG files. No longer used.|
|`Preprocess_EEG.ipynb`|Archive|The next iteration after `read_eeg.ipynb`. A short notebook that provides a visualization of the raw EEG signals collected from the [_Muse S_](https://choosemuse.com/pages/muse-s).|
|`Preprocess_Blinks.ipynb`|Archive|An underdeveloped notebook that was MEANT to read and parse blink data from our participants. Is mostly an empty notebook though.|
|`Analyze_EEG.ipynb`|**Active**|A notebook that features the preliminary analysis graphs and such for parsing EEG. Helps to generate graphs for PSD calculations from raw EEG signals.|
|`Participants_Debug.ipynb`|**Active**|A notebook that features participant data handling. Contains a major `Participant` class that handles most operations with participant data.|
|`helpers.py`|**Active**|A simple Python file that contains helper functions used across the various notebooks.|

## Participant Data

* Participant data is actively not hosted in the repo. The data is privy to researchers at the SimSpace lab only.
* To SimSpace lab researchers: The data is hosted online at our Box, under "Pedestrian Encounters/Data".