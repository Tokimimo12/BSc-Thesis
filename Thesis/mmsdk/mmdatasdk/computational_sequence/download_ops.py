
import h5py
import time
import requests
from tqdm import tqdm
import os
import math
import sys
from mmsdk.mmdatasdk import log

def read_URL(url, destination):

    # Normalize the DESTINATION path

    if destination is None:
        log.error("Destination is not specified when downloading data", error=True)

    normalized_destination = destination.replace("://", "__")
    print(f"Normalized destination path: {normalized_destination}")


    if os.path.isfile(normalized_destination):
        log.error(f"{normalized_destination} file already exists ...", error=True)

    #Normalize the DIRECTORY
    normalized_directory = os.path.dirname(destination).replace("://", "__")

    if normalized_directory and not os.path.isdir(normalized_directory):
        # Print the normalized directory path
        os.makedirs(normalized_directory)
        print(f"Created normalized directory: {normalized_directory}")


    r = requests.get(url, stream=True)
    if r.status_code != 200:
        log.error(f'URL: {url} does not exist', error=True)

    # Total size in bytes
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0

    with open(normalized_destination, 'wb') as f:
        log.status(f"Downloading from {url} to {normalized_destination}...")
        pbar = log.progress_bar(
            total=math.ceil(total_size // block_size),
            data=r.iter_content(block_size),
            postfix="Total in kBs",
            unit='kB',
            leave=False,
        )
        for data in pbar:
            wrote += len(data)
            f.write(data)
    pbar.close()


    if total_size != 0 and wrote != total_size:
        log.error(f"Error downloading the data to {normalized_destination} ...", error=True)

    # Confirm successful download
    log.success(f"Download complete at {normalized_destination}!")
    return True


if __name__=="__main__":
	readURL("http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/acoustic/CMU_MOSEI_COVAREP.csd","./hi.csd")
