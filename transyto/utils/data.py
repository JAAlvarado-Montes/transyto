import os
import numpy as np
import tempfile

from transyto.utils import search_files_across_directories, get_header, fpack
from transyto.utils.wcs import plate_solve_frame
from ccdproc import combine, subtract_bias, subtract_dark, flat_correct
from astropy.nddata import CCDData
from astropy.wcs import WCS
from astropy import units as u

from contextlib import suppress


class Data:
    def __init__(self, filenames_path):
        """Initialize data for reduction.

        Parameters
        ----------
        filenames_path : str
            Top level path of .fits files to search for stars.
        """
        self.filenames_path = filenames_path

    @staticmethod
    def safe_load_ccdproc(fname, data_type):
        """Open fits file ensuring it has the right units for ccdproc.

        Make sure the units in BUNIT is what it should be. If BUNIT not in fits, then set to
        data_type input.

        Parameters
        ----------
        fname : string
            Fits file name to load.
        data_type : string
            Expected units of fits data.

        Returns
        -------
        data: ccdproc.CCDData
            Instance of ccdproc.CCDData with correct units specified.
        """
        try:
            data = CCDData.read(fname)
        except ValueError as err:
            if err.args[0] == "a unit for CCDData must be specified.":
                data = CCDData.read(fname, unit=data_type)
            else:
                raise err
        return data

    @staticmethod
    def create_master_image_stack(filenames_list, output_filename,
                                  min_number_files_in_directory=3, output_directory="./",
                                  method="median", scale=None, **kwargs):
        """Create a master image stack.

        Search for files into list to create a master file.

        Parameters
        ----------
        filenames_list : list,
            Absolute path to location of files. OK if no data or not enough
            data is in the directory
        output_filename : string,
            Name of the output fits file. Include valid file extension.
        min_number_files_in_directory : int, optional
            Minimum number of required raw files to create master image
        output_directory : string
            Name of output directory, optional. Default is working directory.
        method : string, optional
            Method to combine fits images. Default method is median
        scale : array, optional
            scale to be used when combining images. Default is None.
        **kwargs
            Description

        Returns
        -------
        output_filename : string or None
            Master file, otherwise returns None.
        """

        # Check minimum amount of files to combine
        if len(filenames_list) < min_number_files_in_directory:
            print('EXIT: Not enough files in list for combining (returns None)')
            return None

        # # Print files in list to combine
        print(f'About to combine {len(filenames_list)} files')

        # Make the output directory for master file

        os.makedirs(output_directory, exist_ok=True)

        # Get the ouput file name and path for master
        output_filename = os.path.join(output_directory, output_filename)

        # Remove existing file if it exists
        with suppress(FileNotFoundError):
            os.remove(output_filename)

        # Combine the file list to get the master data using any method
        combine(filenames_list, output_filename, method=method, scale=scale,
                combine_uncertainty_function=np.ma.std, unit="adu")

        # Print path of the master created
        print(f'CREATED (using {method}): {output_filename}\n')

        return output_filename

    def calibrate(self, bias_directory="", darks_directory="",
                  flats_directory="", flat_correction=True, plate_solve=False, verbose=True):
        """
        Does reduction of astronomical data by subtraction of dark noise
        and flat-fielding correction

        Parameters
        ----------
        bias_directory : str, optional
            Top level path of bias frames. Default empty.
        darks_directory : str, optional
            Top level path of dark frames. Default empty.
        flats_directory : str, optional
            Top level path of flat frames. Default empty.
        flat_correction : bool, optional
            Flag to perform flat/gain correction. Default True.
        plate_solve: boolean
            Flag to plate solve given frames. Default True.
        verbose : bool, optional
            Print each time an image is cleaned. Default True.

        """

        # Temporary directory to create intermediate master files
        with tempfile.TemporaryDirectory() as tmp_directory:

            # Create and charge masterdark
            darks_list = search_files_across_directories(darks_directory, "*.fit*")
            masterdark = self.create_master_image_stack(darks_list, "masterdark.fits",
                                                        output_directory=tmp_directory)
            masterdark = self.safe_load_ccdproc(masterdark, 'adu')

            if flat_correction:
                # Create and charge masterbias
                bias_list = search_files_across_directories(bias_directory, "*.fit*")
                masterbias = self.create_master_image_stack(bias_list, "masterbias.fits",
                                                            output_directory=tmp_directory)
                masterbias = self.safe_load_ccdproc(masterbias, 'adu')

                # Create and charge masterflat
                flats_list = search_files_across_directories(flats_directory, "*.fit*")
                masterflat = self.create_master_image_stack(flats_list, "masterflat.fits",
                                                            output_directory=tmp_directory)
                masterflat = self.safe_load_ccdproc(masterflat, 'adu')

                # Bias subtract the masterflat
                masterflat = subtract_bias(masterflat, masterbias)

                # Dark subtract the masterflat
                masterflat = subtract_dark(masterflat, masterdark,
                                           dark_exposure=(masterdark.
                                                          header["EXPTIME"] * u.s),
                                           data_exposure=(masterflat.
                                                          header["EXPTIME"] * u.s),
                                           scale=True)

            if plate_solve:
                print("Plate solving all fits files\n")
                plate_solve_frame(self.filenames_path)

            print("All fits files were plate solved\n")

            print("Starting data reduction process\n")

            # List of science exposures to clean
            files_list = search_files_across_directories(self.filenames_path, "*fit*")

            # Output directory for files after reduction
            output_directory = self.filenames_path + "Calibrated_data"
            os.makedirs(output_directory, exist_ok=True)

            # Reduce each science frame in files_list
            for fn in files_list:
                try:
                    # Charge data of raw file and make dark subtraction
                    raw_file = self.safe_load_ccdproc(fn, 'adu')

                    reduced_file = subtract_dark(raw_file, masterdark,
                                                 dark_exposure=(masterdark.
                                                                header["EXPTIME"] * u.s),
                                                 data_exposure=(raw_file.
                                                                header["EXPTIME"] * u.s), scale=False)

                    # Flat-field correction
                    if flat_correction:
                        reduced_file = flat_correct(reduced_file, masterflat)

                    # Save reduced science image to .fits file
                    file_name = os.path.basename(fn)
                    if fn.endswith(".fz"):
                        file_name = os.path.basename(fn).replace(".fz", "")
                    science_image_cleaned_name = os.path.join(output_directory, "calibrated_" + file_name)

                    # Get header and WCS of raw files checking any extension
                    header = get_header(fn)

                    # Parse the WCS keywords and header in the primary HDU
                    science_image_cleaned = CCDData(reduced_file, unit='adu',
                                                    header=raw_file.header, wcs=WCS(header))
                    # Write cleaned image
                    science_image_cleaned.write(science_image_cleaned_name, overwrite=True)

                    if os.path.isfile(science_image_cleaned_name) and verbose:
                        print("-> Reduced: {}".format(science_image_cleaned_name))
                        fpack(science_image_cleaned_name)
                except ValueError:
                    continue
