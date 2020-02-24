"""Defines TimeSeriesData"""

from __future__ import division
import os
import warnings
import logging

import pandas as pd
import numpy as np
import numpy.ma as ma
import time
import pyfiglet
import matplotlib.ticker as plticker

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
# from astropy.time import Time
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
# from astropy.io.fits import Undefined


from collections import namedtuple
from pathlib import Path
from operator import itemgetter
from wotan import flatten, t14
# from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib import dates

from photutils.aperture.circle import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
from photutils import centroid_2dg, centroid_1dg, centroid_com

from . import PACKAGEDIR
from .utils import (
    search_files_across_directories
)

__all__ = ['TimeSeriesData', 'LightCurve']

# Logger to track activity of the class
logger = logging.getLogger()

# warnings.filterwarnings('ignore', category=UserWarning, append=True)


class TimeSeriesData:
    """Photometry Class"""

    def __init__(self,
                 star_id,
                 data_directory,
                 search_pattern,
                 list_reference_stars,
                 aperture_radius,
                 telescope="TESS"):
        """Initialize class Photometry for a given target and reference stars.

        Parameters
        ----------
        star_id : string
            Name of target star to do aperture photometry
        data_directory : string
            Top level path of .fits files to search for stars.
        search_pattern : string
            Pattern for searching files
        list_reference_stars : list
            Reference stars to be used in aperture photometry.
        aperture_radius : float
            Radius of the inner circular perture.
        """

        # Positional Arguments
        self.star_id = star_id
        self.data_directory = data_directory
        self.search_pattern = search_pattern
        self.list_reference_stars = list_reference_stars
        self.telescope = telescope

        # Aperture parameters
        self.r = aperture_radius
        self.r_in = aperture_radius * 1.6
        self.r_out = aperture_radius * 2.2

        # Centroid bow width for centroid function.
        self.box_width = 2 * (self.r + 1)

        # Output directory for logs
        logs_dir = self.data_directory + "logs_photometry"
        os.makedirs(logs_dir, exist_ok=True)

        logger.addHandler(logging.FileHandler(filename=os.path.join(logs_dir,
                                              'photometry.log'), mode='w'))

        logger.info(pyfiglet.figlet_format("-*-*-*-\n{}\n-*-*-*-".format(self.pipeline)))

        logger.info("{} will use {} reference stars for the photometry\n".
                    format(self.pipeline, len(self.list_reference_stars)))

    @property
    def pipeline(self):
        return os.path.basename(PACKAGEDIR)

    @property
    def readout(self):
        return self.get_keyword_value().readout

    @property
    def obs_time(self):
        return self.get_keyword_value().obstime

    @property
    def exptime(self):
        return self.get_keyword_value().exp

    @property
    def instrument(self):
        return self.get_keyword_value().instr

    @property
    def gain(self):
        return self.get_keyword_value().gain

    @property
    def keyword_list(self):
        file = str(Path(__file__).parents[1]) + "/" + "telescope_keywords.csv"

        telescope = self.telescope

        (Huntsman,
         TESS,
         WASP,
         MEARTH,
         POCS) = np.loadtxt(file, skiprows=2,
                            delimiter=";", dtype=str,
                            usecols=(1, 2, 3, 4, 5),
                            unpack=True)

        if telescope == "Huntsman":
            kw_list = Huntsman
        elif telescope == "TESS":
            kw_list = TESS
        elif telescope == "WASP":
            kw_list = WASP
        elif telescope == "MEARTH":
            kw_list = MEARTH
        elif telescope == "POCS":
            kw_list = POCS

        return kw_list

    def _slice_data(self, data, origin, width):
        y, x = origin
        cutout = data[np.int(x - width / 2):np.int(x + width / 2),
                      np.int(y - width / 2):np.int(y + width / 2)]
        return cutout

    def _mask_data(self, image, sigma=1.0):
        threshold = np.median(image - (sigma * np.std(image)))
        masked_image = ma.masked_values(image, threshold)

        return masked_image

    def _estimate_centroid_via_2dgaussian(self, data, mask):
        """Computes the centroid of a data array using a 2D gaussian
        from photutils.
        """
        x, y = centroid_2dg(data, mask=mask)
        return x, y

    def _estimate_centroid_via_1dgaussian(self, data, mask):
        """Computes the centroid of a data array using a 2D gaussian
        from photutils.
        """
        x, y = centroid_1dg(data, mask=mask)
        return x, y

    def _estimate_centroid_via_moments(self, data, mask):
        """Computes the centroid of a data array using a 2D gaussian
        from photutils.
        """
        x, y = centroid_com(data, mask=mask)
        return x, y

    def find_centroid(self, prior_centroid, data, mask, method="2dgaussian"):

        prior_y, prior_x = prior_centroid
        with warnings.catch_warnings():
            # Ignore warning for the centroid_2dg function
            warnings.simplefilter('ignore', category=UserWarning)

            if method == "2dgaussian":
                x_cen, y_cen = self._estimate_centroid_via_2dgaussian(data,
                                                                      mask)
            elif method == "1dgaussian":
                x_cen, y_cen = self._estimate_centroid_via_1dgaussian(data,
                                                                      mask)

            elif method == "moments":
                x_cen, y_cen = self._estimate_centroid_via_moments(data, mask)

            # Compute the shifts in y and x.
            shift_y = self.box_width / 2 - y_cen
            shift_x = self.box_width / 2 - x_cen

            if shift_y < 0 and shift_x < 0:
                new_y = prior_y + np.abs(shift_y)
                new_x = prior_x + np.abs(shift_x)
            elif shift_y > 0 and shift_x > 0:
                new_y = prior_y - shift_y
                new_x = prior_x - shift_x
            elif shift_y < 0 and shift_x > 0:
                new_y = prior_y + np.abs(shift_y)
                new_x = prior_x - shift_x
            elif shift_y > 0 and shift_x < 0:
                new_y = prior_y - shift_y
                new_x = prior_x + np.abs(shift_x)
            else:
                new_y = prior_y
                new_x = prior_x

            return new_y, new_x

    def get_keyword_value(self, default=None):
        """Returns a header keyword value.

        If the keyword is Undefined or does not exist,
        then return ``default`` instead.
        """

        try:
            kw_values = itemgetter(*self.keyword_list)(self.header)
        except KeyError:
            logger.error("Header keyword does not exist")
            return default
        exp, obstime, instr, readout, gain = kw_values

        Outputs = namedtuple("Outputs", "exp obstime instr readout gain")

        return Outputs(exp, obstime, instr, readout, gain)

    def make_aperture(self, data, coordinates, radius, r_in, r_out,
                      method="exact", subpixels=10):
        """Make the aperture sum in each positions for a given star. It
           can be rectangular (e.g. square), circular or annular

        Parameters
        ----------
        data : numpy array or CCDData
            contains the data where the aperture is going to be done
        coordinates : tuple
            (x, y) position of the star to do aperture
        radius : float
            Radius of the central aperture
        method : str, optional
            Method to be used for the aperture photometry
        r_in : int,
            Pixels added to central radius to get the inner radius
            of background aperture
        r_out : int,
            Pixels added to central radius to get the outer radius
            of background aperture
        subpixels : int, optional
            Number of subpixels for subpixel method. Each pixel
            is divided into subpixels**2.0

        Returns
        -------
        float
            Sum inside the aperture (sky background subtracted)

        """

        # Circular inner aperture for the star
        target_apertures = CircularAperture(coordinates, r=radius)

        # Annular outer aperture for the sky background
        background_apertures = CircularAnnulus(coordinates,
                                               r_in=r_in,
                                               r_out=r_out)

        # Find median value of counts-per-pixel in the background
        background_mask = background_apertures.to_mask(method="center")
        background_data = background_mask.multiply(data)
        mask = background_mask.data
        annulus_data_1d = background_data[mask > 0]
        (mean_sigclip,
         median_sigclip,
         std_sigclip) = sigma_clipped_stats(annulus_data_1d,
                                            sigma=3, maxiters=10)
        # sky_bkg = 3 * median_sigclip - 2 * mean_sigclip

        # Make aperture photometry for the object and the background
        apertures = [target_apertures, background_apertures]
        phot_table = aperture_photometry(data, apertures,
                                         method=method,
                                         subpixels=subpixels)

        # For consistent outputs in table
        for col in phot_table.colnames:
            phot_table[col].info.format = "%.8g"

        # Find median value of counts-per-pixel in the sky background.
        # sky_bkg = phot_table["aperture_sum_1"] / background_apertures.area
        sky_bkg = median_sigclip
        phot_table['background_median'] = sky_bkg

        # Find background in object inner aperture and subtract it
        background_in_target = sky_bkg * target_apertures.area

        phot_table["background_in_target"] = background_in_target
        phot_table["background_in_target"].info.format = "%.8g"

        assert phot_table["aperture_sum_0"] > phot_table["background_in_target"]

        object_final_counts = phot_table["aperture_sum_0"] - background_in_target

        # For consistent outputs in table
        phot_table["target_aperture_bkg_subtracted"] = object_final_counts
        phot_table["target_aperture_bkg_subtracted"].info.format = "%.8g"

        logger.info(phot_table["target_aperture_bkg_subtracted"])

        return (phot_table["target_aperture_bkg_subtracted"].item(),
                phot_table["background_in_target"].item())

    # @logged
    def do_photometry(self, star_id, data_directory, search_pattern):
        """Get all data from plate-solved images (right ascention,
           declination, airmass, dates, etc). Then, it converts the
           right ascention and declination into image positions to
           call make_aperture and find its total counts.

        Parameters
        ----------
        star_id: string
            name of star to be localized in each file
        data_directory: list
            list of files (frames) where we want to get the counts
        search_pattern: string
            pattern to search files

        Returns
        --------
        Counts of a star, list of good frames and airmass: tuple

        """
        star = SkyCoord.from_name(star_id)

        # Search for files containing data to analyze
        fits_files = search_files_across_directories(data_directory,
                                                     search_pattern)

        # List of ADU counts for the source, background
        object_counts = list()
        background_in_object = list()

        # List of exposure times
        exptimes = list()

        # List of object positions
        x_pos = list()
        y_pos = list()

        # Observation dates list
        times = list()

        # List of good frames
        self.good_frames_list = list()

        for fn in fits_files[0:600]:
            # Get data, header and WCS of fits files with any extension
            ext = 0
            if fn.endswith(".fz"):
                ext = 1
            data, self.header = fits.getdata(fn, header=True, ext=ext)
            wcs = WCS(self.header)

            # Check if WCS exist in image
            if wcs.is_celestial:

                # Star pixel positions in the image
                center_yx = wcs.all_world2pix(star.ra, star.dec, 0)

                cutout = self._slice_data(data, center_yx, self.box_width)

                masked_data = self._mask_data(cutout)

                y_cen, x_cen = self.find_centroid(center_yx, cutout,
                                                  masked_data.mask,
                                                  method="2dgaussian")

                # Exposure time
                exptimes.append(self.exptime * 24 * 60 * 60)

                # Observation times
                time = self.obs_time

                # Sum of counts inside aperture
                (counts_in_aperture,
                 bkg_in_object) = self.make_aperture(data, (y_cen, x_cen),
                                                     radius=self.r,
                                                     r_in=self.r_in,
                                                     r_out=self.r_out)

                object_counts.append(counts_in_aperture)
                background_in_object.append(bkg_in_object)
                x_pos.append(center_yx[1])
                y_pos.append(center_yx[0])
                times.append(time)
                self.good_frames_list.append(fn)
            else:
                continue

        return (object_counts, background_in_object,
                exptimes, x_pos, y_pos, times)

    # @logged
    def get_relative_flux(self, save_rms=False):
        """Find the flux of a target star relative to some reference stars,
           using the counts inside an aperture

        Parameters
        ----------
        save_rms : bool, optional
            Save a txt file with the rms achieved for each time that
            the class is executed (defaul is False)

        Returns
        -------
        relative flux : float
            The ratio between the target flux and the
            integrated flux of the reference stars
        """
        start = time.time()

        logger.info(f"Starting aperture photometry for {self.star_id}\n")

        # Get flux of target star
        (target_flux,
         background_in_object,
         exptimes,
         x_pos_target,
         y_pos_target,
         self.times) = self.do_photometry(self.star_id,
                                          self.data_directory,
                                          self.search_pattern)

        self.times = np.asarray(self.times)

        logger.info("Finished aperture photometry on target star. "
                    f"{self.__class__.__name__} will compute now the "
                    "combined flux of the ensemble\n")

        # Positions of target star
        self.x_pos_target = np.array(x_pos_target) - np.nanmean(x_pos_target)
        self.y_pos_target = np.array(y_pos_target) - np.nanmean(y_pos_target)

        # Target and background counts per second
        exptimes = np.asarray(exptimes)
        target_flux = np.asarray(target_flux)
        self.target_flux_sec = target_flux / exptimes
        background_in_target_sec = np.asarray(background_in_object) / exptimes

        # CCD gain
        ccd_gain = self.gain

        readout_noise = (self.readout * self.r)**2 * np.pi * np.ones(len(self.good_frames_list))

        # Sigma readout noise
        ron = np.sqrt(readout_noise)
        self.sigma_ron = -2.5 * np.log10((self.target_flux_sec * ccd_gain * exptimes - ron)
                                         / (self.target_flux_sec * ccd_gain * exptimes))

        # Sigma photon noise
        # self.sigma_phot = 1 / np.sqrt(self.target_flux_sec * ccd_gain * self.exptimes)
        self.sigma_phot = -2.5 * np.log10((self.target_flux_sec * ccd_gain * exptimes
                                           - np.sqrt(self.target_flux_sec * ccd_gain
                                                     * exptimes))
                                          / (self.target_flux_sec * ccd_gain * exptimes))

        # Sigma sky-background noise
        self.sigma_sky = -2.5 * np.log10((self.target_flux_sec * ccd_gain * exptimes
                                          - np.sqrt(background_in_target_sec * ccd_gain
                                                    * exptimes))
                                         / (self.target_flux_sec * ccd_gain * exptimes))

        # Total photometric error for 1 mag in one observation
        self.sigma_total = np.sqrt(self.sigma_phot**2.0 + self.sigma_ron**2.0
                                   + self.sigma_sky**2.0)

        # Signal to noise: shot, sky noise (per second) and readout
        S_to_N_obj_sec = self.target_flux_sec / np.sqrt(self.target_flux_sec
                                                        + background_in_target_sec
                                                        + readout_noise
                                                        / (ccd_gain * exptimes))
        # Convert SN_sec to actual SN
        S_to_N_obj = S_to_N_obj_sec * np.sqrt(ccd_gain * exptimes)

        # Get the flux of each reference star
        self.reference_star_flux_sec = list()
        background_in_ref_star_sec = list()
        for ref_star in self.list_reference_stars:

            logger.info(f"Starting aperture photometry on ref_star {ref_star}\n")

            (refer_flux,
             background_in_ref_star,
             exptimes_ref,
             x_pos_ref,
             y_pos_ref,
             obs_dates) = self.do_photometry(ref_star,
                                             self.data_directory,
                                             self.search_pattern)
            self.reference_star_flux_sec.append(np.asarray(refer_flux) / exptimes)
            background_in_ref_star_sec.append(np.asarray(background_in_ref_star) / exptimes)
            logger.info(f"Finished aperture photometry on ref_star {ref_star}\n")

        self.reference_star_flux_sec = np.asarray(self.reference_star_flux_sec)
        background_in_ref_star_sec = np.asarray(background_in_ref_star_sec)

        sigma_squared_ref = (self.reference_star_flux_sec * exptimes
                             + background_in_ref_star_sec * exptimes
                             + readout_noise)

        weights_ref_stars = 1.0 / sigma_squared_ref

        ref_flux_averaged = np.average(self.reference_star_flux_sec * exptimes,
                                       weights=weights_ref_stars,
                                       axis=0)

        # Integrated flux per sec for ensemble of reference stars
        total_reference_flux_sec = np.sum(self.reference_star_flux_sec, axis=0)

        # Integrated sky background for ensemble of reference stars
        total_reference_bkg_sec = np.sum(background_in_ref_star_sec, axis=0)

        # S/N for reference star per second
        S_to_N_ref_sec = total_reference_flux_sec / np.sqrt(total_reference_flux_sec
                                                            + total_reference_bkg_sec
                                                            + readout_noise
                                                            / (ccd_gain * exptimes))
        # Convert S/N per sec for ensemble to total S/N
        S_to_N_ref = S_to_N_ref_sec * np.sqrt(ccd_gain * exptimes)

        # Relative flux per sec of target star
        differential_flux = target_flux / ref_flux_averaged

        # Normalized relative flux
        self.normalized_flux = differential_flux / np.nanmedian(differential_flux)

        # Find Differential S/N for object and ensemble
        S_to_N_diff = 1 / np.sqrt(S_to_N_obj**-2 + S_to_N_ref**-2)

        # Ending time of computatin analysis.
        end = time.time()
        exec_time = end - start

        # Print when all of the analysis ends
        logger.info(f"Differential photometry of {self.star_id} has been finished, "
                    f"with {len(self.good_frames_list)} frames "
                    f"of camera {self.instrument} (run time: {exec_time:.3f} sec)\n")

        # if detrend_data:
        #     logger.info("Removing trends from time series data\n")
        #     # Compute the transit duration
        #     transit_dur = t14(R_s=R_star, M_s=M_star,
        #                       P=Porb, small_planet=False)

        #     # Estimate the window length for the detrending
        #     wl = 3.0 * transit_dur

        #     # Detrend the time series data
        #     self.normalized_flux, self.lc_trend = flatten(self.times,
        #                                                   self.normalized_flux,
        #                                                   return_trend=True,
        #                                                   method="biweight",
        #                                                   window_length=wl)

        # Output directory
        self.output_dir_name = "TimeSeries_Analysis"

        if save_rms:
            # Output directory for files that contain photometric precisions
            output_directory = self.data_directory + self.output_dir_name + "/rms_precisions"
            os.makedirs(output_directory, exist_ok=True)

            # File with rms information
            file_rms_name = os.path.join(output_directory,
                                         f"rms_{self.instrument}.txt")

            with open(file_rms_name, "a") as file:
                file.write(f"{self.r} {self.std} {self.std_binned} "
                           f"{np.nanmedian(S_to_N_obj)} {np.nanmedian(S_to_N_ref)} "
                           f"{np.nanmedian(S_to_N_diff)}\n")

        return (self.times, self.target_flux_sec, self.sigma_total)


class LightCurve(TimeSeriesData):
    def __init__(self,
                 star_id,
                 data_directory,
                 search_pattern,
                 list_reference_stars,
                 aperture_radius,
                 telescope="TESS"):
        super(LightCurve, self).__init__(star_id=star_id,
                                         data_directory=data_directory,
                                         search_pattern=search_pattern,
                                         list_reference_stars=list_reference_stars,
                                         aperture_radius=aperture_radius,
                                         telescope=telescope)

    def clip_outliers(self, sigma=5.0, sigma_lower=None, sigma_upper=None,
                      return_mask=False, **kwargs):
        """ Covenience wrapper for sigma_clip function from astropy.
        """

        clipped_data = sigma_clip(data=self.normalized_flux,
                                  sigma=sigma, maxiters=10,
                                  cenfunc=np.median,
                                  masked=True,
                                  copy=True)

        mask = clipped_data.mask
        normalized_flux_clipped = self.normalized_flux[~mask]
        times_clipped = self.times[~mask]

        if return_mask:
            return normalized_flux_clipped, times_clipped, mask
        return normalized_flux_clipped, times_clipped

    def detrend_lightcurve(self, times, flux, R_star=None,
                           M_star=None, Porb=None):
        """Detrend time-series data

        Parameters
        ----------
        times : array
            Times of the observation
        flux : array
            Flux with trend to be removed
        R_star : None, optional
            Radius of the star (in solar units). It has to be specified if
            detrend_data is True.
        M_star : None, optional
            Mass of the star (in solar units). It has to be specified
            if detrend_data is True.
        Porb : None, optional
            Orbital period of the planet (in days). It has to be specified if
            detrend_data is True.

        Returns
        -------
        detrended and trended flux : numpy array
        """

        logger.info("Removing trends from time series data\n")
        # Compute the transit duration
        transit_dur = t14(R_s=R_star, M_s=M_star,
                          P=Porb, small_planet=False)

        # Estimate the window length for the detrending
        wl = 3.0 * transit_dur

        # Detrend the time series data
        detrended_flux, trended_flux = flatten(times, flux, return_trend=True,
                                               method="biweight",
                                               window_length=wl)
        return detrended_flux, trended_flux

    def bin_data(self, flux, times, bins):
        """Bin data into groups by usinf the mean of each group

        Parameters
        ----------
        flux : TYPE
            Description
        times : TYPE
            Description
        bins : TYPE
            Description

        Returns
        -------
        binned data: numpy array
            Data in bins

        """

        # Makes dataframe of given data
        df_flux = pd.DataFrame({"binned_flux": flux})
        binned_flux = df_flux.groupby(df_flux.index // bins).mean()
        binned_flux = binned_flux["binned_flux"]

        df_time = pd.DataFrame({"binned_times": times})
        binned_times = (df_time.groupby(df_time.index // bins).last()
                        + df_time.groupby(df_time.index // bins).first()) / 2
        binned_times = binned_times["binned_times"]

        return binned_flux, binned_times

    # @logged
    def plot(self, bins=4, detrend=False, R_star=None, M_star=None, Porb=None):
        """Plot a light curve using the flux time series

        Parameters
        ----------
        bins : int, optional
            Description
        detrend : bool, optional (default is False)
            If True, detrending of the time series data will be performed
        R_star : None, optional
            Radius of the star (in solar units). It has to be specified if
            detrend is True.
        M_star : None, optional
            Mass of the star (in solar units). It has to be specified
            if detrend is True.
        Porb : None, optional
            Orbital period of the planet (in days). It has to be specified if
            detrend is True.

        No Longer Returned
        ------------------
        """

        pd.plotting.register_matplotlib_converters()

        self.get_relative_flux()

        flux, times = self.clip_outliers(sigma=10)

        if detrend:
            flux, flux_tr = self.detrend_lightcurve(times, flux,
                                                    R_star=R_star,
                                                    M_star=M_star,
                                                    Porb=Porb)

        # Standard deviation in ppm for the observation
        std = np.nanstd(self.normalized_flux)

        # Binned data and times
        binned_flux, binned_times = self.bin_data(flux, times, bins)
        std_binned = np.nanstd(binned_flux)

        # Total time for binsize
        nbin_tot = self.exptime * bins

        # Output directory for lightcurves
        lightcurves_directory = self.data_directory + self.output_dir_name

        # lightcurve name
        lightcurve_name = os.path.join(lightcurves_directory, "Lightcurve_camera_"
                                       f"{self.instrument}_r{self.r}.png")

        fig, ax = plt.subplots(4, 1,
                               sharey="row", sharex="col", figsize=(10, 10))
        fig.suptitle(f"Differential Photometry\nTarget Star {self.star_id}, "
                     f"Aperture Radius = {self.r} pix", fontsize=13)

        ax[3].plot(times, flux, "k.", ms=3,
                   label=f"NBin = {self.exptime:.3f} d, std = {std:.2%}")
        ax[3].plot(binned_times, binned_flux, "ro", ms=4,
                   label=f"NBin = {nbin_tot:.3f} d, std = {std_binned:.2%}")
        # ax[3].errorbar(self.times, self.normalized_flux, yerr=self.sigma_total,
        #                fmt="none", ecolor="k", elinewidth=0.8,
        #                label="$\sigma_{\mathrm{tot}}=\sqrt{\sigma_{\mathrm{phot}}^{2} "
        #                "+ \sigma_{\mathrm{sky}}^{2} + \sigma_{\mathrm{scint}}^{2} + "
        #                "\sigma_{\mathrm{read}}^{2}}$",
        #                capsize=0.0)

        ax[3].set_ylabel("Relative\nFlux", fontsize=13)
        # ax[3].legend(fontsize=9.0, loc="lower left", ncol=3, framealpha=1.0)
        # ax[3].set_ylim((0.9995, 1.0004))
        ax[3].ticklabel_format(style="plain", axis="both", useOffset=False)
        loc_x3 = plticker.MultipleLocator(base=5)  # this locator puts ticks at regular intervals
        ax[3].xaxis.set_major_locator(loc_x3)
        ax[3].xaxis.set_major_formatter(plticker.FormatStrFormatter('%.1f'))

        # Plot of target star flux
        ax[2].plot(self.times,
                   self.target_flux_sec / np.nanmean(self.target_flux_sec),
                   "ro", label=f"Target star {self.star_id}", lw=0.0, ms=1.3)
        ax[2].set_ylabel("Normalized\nFlux", fontsize=13)
        ax[2].legend(fontsize=8.6, loc="lower left", ncol=1,
                     framealpha=1.0, frameon=True)
        ax[2].set_ylim((0.9, 1.05))

        ax[0].plot(self.times, self.x_pos_target, "ro-",
                   label="dx [Dec axis]", lw=0.5, ms=1.2)
        ax[0].plot(self.times, self.y_pos_target, "go-",
                   label="dy [RA axis]", lw=0.5, ms=1.2)
        ax[0].set_ylabel(r"$\Delta$ Pixel", fontsize=13)
        ax[0].legend(fontsize=8.6, loc="lower left", ncol=2, framealpha=1.0)
        ax[0].set_title(f"Camera: {self.instrument}", fontsize=13)

        for counter in range(len(self.list_reference_stars)):
            # ax[1].xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))

            # Colors for comparison stars
            # colors = ["blue", "magenta", "green", "cyan", "firebrick"]

            ax[1].plot(self.times, self.reference_star_flux_sec[counter]
                       / np.nanmean(self.reference_star_flux_sec[counter]),
                       "o", ms=1.3, label=f"Star {self.list_reference_stars[counter]}")
            ax[1].set_ylabel("Normalized\nFlux", fontsize=13)
            ax[1].set_ylim((0.9, 1.05))
            ax[1].legend(fontsize=8.1, loc="lower left",
                         ncol=len(self.list_reference_stars),
                         framealpha=1.0, frameon=True)

        ax[3].text(0.97, 0.07, "d)", fontsize=11, transform=ax[3].transAxes)
        ax[2].text(0.97, 0.07, "c)", fontsize=11, transform=ax[2].transAxes)
        ax[1].text(0.97, 0.07, "b)", fontsize=11, transform=ax[1].transAxes)
        ax[0].text(0.97, 0.07, "a)", fontsize=11, transform=ax[0].transAxes)

        # Wasp 29 times
        # ingress = datetime(2019, 10, 27, 10, 34, 00)
        # mid = datetime(2019, 10, 27, 11, 54, 00)
        # egress = datetime(2019, 10, 27, 13, 13, 00)

        # # Transit ingress, mid and egress times
        # plt.axvline(x=ingress, color="k", ls="--")
        # plt.axvline(x=mid, color="b", ls="--")
        # plt.axvline(x=egress, color="k", ls="--")
        # plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
        plt.xlabel("Time [BJD-2457000.0]", fontsize=13)
        plt.xticks(rotation=30, size=8.0)
        plt.savefig(lightcurve_name)

        logger.info(f"The light curve of {self.star_id} was plotted")

        fig, ax = plt.subplots(1, 1,
                               sharey="row", sharex="col", figsize=(13, 10))
        fig.suptitle(f"Evolution of Noise Sources for the Target Star {self.star_id} "
                     "($m_\mathrm{V}=10.9$)\n"
                     f"Huntsman Defocused Camera {self.instrument}, G Band Filter\n"
                     f"Sector 2", fontsize=15)
        ax.plot_date(self.times, self.sigma_total * 100, "k-",
                     label="$\sigma_{\mathrm{total}}$")
        ax.plot_date(self.times, self.sigma_phot * 100, color="firebrick", marker=None,
                     ls="-", label="$\sigma_{\mathrm{phot}}$")
        ax.plot_date(self.times, self.sigma_sky * 100,
                     "b-", label="$\sigma_{\mathrm{sky}}$")
        ax.plot_date(self.times, self.sigma_ron * 100,
                     "r-", label="$\sigma_{\mathrm{read}}$")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.998), fancybox=True,
                  ncol=5, frameon=True, fontsize=15)
        # ax.set_yscale("log")
        ax.tick_params(axis="both", direction="in", labelsize=15)
        ax.set_ylabel("Amplitude Error [%]", fontsize=17)
        plt.xticks(rotation=30)
        ax.set_xlabel("Time [UTC]", fontsize=17)
        # ax.set_ylim((0.11, 0.48))
        ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
        plt.grid(alpha=0.4)
        fig.savefig("noises.png")
