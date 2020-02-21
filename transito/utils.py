"""This module provides various helper functions."""
import glob
import os
import logging

import functools
import pandas as pd

from natsort import natsorted


log = logging.getLogger(__name__)

__all__ = ['search_files_across_directories', 'bin_dataframe', 'logged']


def search_files_across_directories(search_directory, search_pattern):
    """
    Search for files with a string pattern across directories.
    Excludes directories with similar string pattern

    Args
    ----------
    search_directory : string
        Directory with fits images,
    search_pattern : string
        string pattern to select files

    Returns
    -------
    List: files that match the search pattern

    """

    # Create list of files with the given filename search pattern
    search_path = os.path.join(search_directory,
                               '**',
                               search_pattern)
    files_list = natsorted(glob.glob(search_path, recursive=True))

    return files_list


def bin_dataframe(data, nbin, bin_dates=False):
    """Bin data into groups by usinf the mean of each group

    Parameters
    ----------
    data : ndarray
        array that contains the data to bin
    nbin : int, optional
        the amount of elements to group
    bin_times : bool, optional
        Flag to bin obervation times

    Returns
    -------
    binned data: numpy array
        Data in bins

    """

    # Makes dataframe of given data
    df = pd.DataFrame({"data_to_bin": data})
    binned_data = df.groupby(df.index // nbin).mean()
    binned_data = binned_data["data_to_bin"]

    if bin_dates:
        binned_data = (df.groupby(df.index // nbin).last()
                       + df.groupby(df.index // nbin).first()) / 2
        binned_data = binned_data["data_to_bin"]
        return binned_data

    return binned_data


def logged(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(func.__name__ + " was called")
        return func(*args, **kwargs)
    return wrapper
