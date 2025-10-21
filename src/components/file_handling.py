import os
import shutil

# Get all files in a provided directory
def find_files(_DIR:str, debug:bool=False):
    # Read all files
    fs = next(os.walk(_DIR), (None, None, []))[2]

    # Remove all `.DS_Store` files
    files =[f for f in fs if f != '.DS_Store']
    
    # If in debug mode, show all files
    if debug: print(files)

    # Return
    return files

# Create directory, delete first if already exists
def mkdirs(_DIR:str, delete_existing:bool=True):
    
    # If the folder already exists, delete it
    if delete_existing and os.path.exists(_DIR): shutil.rmtree(_DIR)

    # Create a new empty directory
    os.makedirs(_DIR, exist_ok=True)

    # Return to indicate completion
    return