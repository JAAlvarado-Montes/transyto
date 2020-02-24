"""This module provides various helper functions."""
import glob
import os
import logging

import functools

from natsort import natsorted


log = logging.getLogger(__name__)

__all__ = ['search_files_across_directories', 'logged']


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


def logged(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(func.__name__ + " was called")
        return func(*args, **kwargs)
    return wrapper
