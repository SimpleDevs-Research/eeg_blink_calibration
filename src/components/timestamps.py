import datetime

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