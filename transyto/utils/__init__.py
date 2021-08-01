"""This module provides various helper functions."""
import glob
import os
import logging
import shutil
import subprocess

import functools

from natsort import natsorted
from warnings import warn

from astropy.io import fits

log = logging.getLogger(__name__)

__all__ = ['search_files_across_directories', 'logged', 'fpack', 'funpack']


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


def fpack(fits_fname, unpack=False, verbose=False):
    """Compress/Decompress a FITS file

    Uses `fpack` (or `funpack` if `unpack=True`) to compress a FITS file

    Args:
        fits_fname ({str}): Name of a FITS file that contains a WCS.
        unpack ({bool}, optional): file should decompressed instead of compressed, default False.
        verbose ({bool}, optional): Verbose, default False.

    Returns:
        str: Filename of compressed/decompressed file.
    """
    assert os.path.exists(fits_fname), warn(
        "No file exists at: {}".format(fits_fname))

    if unpack:
        fpack = shutil.which('funpack')
        cmd = [fpack, '-D', fits_fname]
        out_file = fits_fname.replace('.fz', '')
    else:
        fpack = shutil.which('fpack')
        cmd = [fpack, '-D', '-Y', fits_fname]
        out_file = fits_fname.replace('.fits', '.fits.fz')

    try:
        assert fpack is not None
    except AssertionError:
        warn("fpack not found (try installing cfitsio). File has not been changed")
        return fits_fname

    if verbose:
        print("fpack command: {}".format(cmd))

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    return out_file


def funpack(*args, **kwargs):
    """Unpack a FITS file.

    Note:
        This is a thin-wrapper around the `fpack` function
        with the `unpack=True` option specified. See `fpack`
        documentation for details.

    Args:
        *args: Arguments passed to `fpack`.
        **kwargs: Keyword arguments passed to `fpack`.

    Returns:
        str: Path to uncompressed FITS file.
    """
    return fpack(*args, unpack=True, **kwargs)


def get_data(fname, *args, **kwargs):
    """Open fits file ensuring it has the right units for ccdproc.

    Make sure the units in BUNIT is what it should be. If BUNIT not in fits,
    then set to data_type input.

    Parameters
    ----------
    fname : string
        Fits file name to load.

    Returns
    -------
        data : data array
    """

    return fits.getdata(fname, *args, header=False, **kwargs)


def get_header(fn, *args, **kwargs):
    """Get the FITS header.

    Small wrapper around `astropy.io.fits.getheader` to auto-determine
    the FITS extension. This will return the header associated with the
    image. If you need the compression header information use the astropy
    module directly.

    Parameters
    ----------
    fn: string
        Path to FITS file.
    *args: Passed to `astropy.io.fits.getheader`.
    **kwargs: Passed to `astropy.io.fits.getheader`.

    Returns
    -------
        `astropy.io.fits.header.Header`: The FITS header for the data.
    """
    ext = 0
    if fn.endswith('.fz'):
        ext = 1
    return fits.getheader(fn, *args, ext=ext, **kwargs)


def get_value(fn, *args, **kwargs):
    """Get a value from the FITS header.

    Small wrapper around `astropy.io.fits.getval` to auto-determine
    the FITS extension. This will return the value from the header
    associated with the image (not the compression header). If you need
    the compression header information use the astropy module directly.

    Parameters
    ----------
    fn: string
        Path to FITS file.

    Returns
    -------
        str or float: Value from header (with no type conversion).
    """
    ext = 0
    if fn.endswith('.fz'):
        ext = 1
    return fits.getval(fn, *args, ext=ext, **kwargs)


def logged(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(func.__name__ + " was called")
        return func(*args, **kwargs)
    return wrapper
