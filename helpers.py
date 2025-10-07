""" ==============================
IMPORTS 
============================== """
import pandas as pd
import datetime

""" ==============================
TIMESTAMP MANAGEMENT 
============================== """

# Convert a timestamp to unix seconds
def timestamp_to_unix_seconds(x):
    date_format = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
    unix_seconds = datetime.datetime.timestamp(date_format)
    return unix_seconds

# Convert a timestamp to unix milliseconds
def timestamp_to_unix_milliseconds(x):
    unix_seconds = timestamp_to_unix_seconds(x)
    unix_milliseconds = int(unix_seconds * 1000)
    return unix_milliseconds

""" ==============================
EEG INTERPRETATION 
============================== """

# Preprocessing EEG
def read_muse(filename):
    # Read the raw eeg file
    df = pd.read_csv(filename)
    
    # Extract seconds from unix
    df['unix_sec'] = df['TimeStamp'].apply(timestamp_to_unix_seconds)
    df['unix_ms'] = df['TimeStamp'].apply(timestamp_to_unix_milliseconds)
    
    # Calculate relative seconds
    df['rel_sec'] = df['unix_sec'] - df['unix_sec'].iloc[0]
    df['rel_ms'] = df['unix_ms'] - df['unix_ms'].iloc[0]
    
    # separate blinks from raw data
    signals = df[df['Elements'].isna()]
    blinks = df[df['Elements']=='/muse/elements/blink']
    blinks2 = blinks[['TimeStamp', 'unix_sec', 'unix_ms', 'rel_sec', 'rel_ms']]
    
    # Return blinks and signals
    return signals, blinks2

# Separate columns, between raw and processed
def separate_eeg(
        df:pd.DataFrame, 
        time_colnames=['unix_sec','unix_ms', 'rel_sec', 'rel_ms']
):
    raw_colnames = {
        'RAW_TP9':'TP9', 
        'RAW_TP10':'TP10', 
        'RAW_AF7':'AF7', 
        'RAW_AF8':'AF8'
    }
    processed_colnames = [
        'Delta_TP9','Delta_TP10','Delta_AF7','Delta_AF8',
        'Theta_TP9', 'Theta_TP10', 'Theta_AF7', 'Theta_AF8',
        'Alpha_TP9', 'Alpha_TP10', 'Alpha_AF7', 'Alpha_AF8',
        'Beta_TP9', 'Beta_TP10', 'Beta_AF7', 'Beta_AF8',
        'Gamma_TP9', 'Gamma_TP10', 'Gamma_AF7', 'Gamma_AF8'
    ]
    
    raw_df = df[
        [
            *time_colnames, 
            *list(raw_colnames.keys())
        ]
    ]
    
    raw_df.rename(columns=raw_colnames, inplace=True)
    
    processed_df = df[
        [
            *time_colnames,
            *processed_colnames
        ]
    ]
    
    return raw_df, processed_df
