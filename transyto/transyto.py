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
import scipy
import matplotlib.ticker as plticker
import seaborn as sns

from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.time import Time
from barycorrpy import utc_tdb
from astropy.nddata import NDData
from astropy.table import Table
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats

from transitleastsquares import transitleastsquares
from collections import namedtuple
from tqdm import tqdm
from pathlib import Path
from operator import itemgetter
from wotan import flatten, t14
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib import dates

from photutils.aperture.circle import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
from photutils.centroids import centroid_2dg, centroid_1dg, centroid_com
from photutils.psf import extract_stars
from photutils.psf.epsf import EPSFBuilder

from . import PACKAGEDIR
from transyto.limbDC import ldc

from transyto.utils import (
    search_files_across_directories,
    catalog,
    get_data,
    get_header
)

from transyto.noise import (
    compute_scintillation,
    compute_noises
)

__all__ = ['TimeSeriesAnalysis', 'LightCurve']


warnings.simplefilter('ignore', category=AstropyWarning)


class TimeSeriesAnalysis:
    """Photometry Class"""

    def __init__(self, target_star="", data_directory="", search_pattern="*fit*",
                 list_reference_stars=[], aperture_radius=15, from_coordinates=None, ra_target=None,
                 dec_target=None, transit_times=[], ra_ref_stars=[], dec_ref_stars=[],
                 telescope=""):
        """Initialize class Photometry for a given target and reference stars.

        Parameters
        ----------
        star_id : str
            Name of target star to do aperture photometry
        data_directory : str
            Top level path of .fits files to search for stars.
        search_pattern : str
            Pattern for searching files
        list_reference_stars : list
            Reference stars to be used in aperture photometry.
        aperture_radius : float
            Radius of the inner circular perture.
        from_coordinates : None, optional
            Flag to find star by using its coordinates.
        ra_target : None, optional
            RA coords of target.
        dec_target : None, optional
            DEC of target.
        transit_times : list, optional
            Ingress, mid-transit, and egress time. In "isot" format.
            Example: ["2021-07-27T15:42:00", "2021-07-27T17:06:00", "2021-07-27T18:29:00"]
        ra_ref_stars : list, optional
            RA of ref stars.
        dec_ref_stars : list, optional
            DEC f ref. stars.
        telescope : str, optional
            Name of the telescope where the data come from.
        """

        # Positional Arguments
        self.target_star = target_star

        # Data directory.
        self._data_directory = data_directory

        # Output directory for light curves
        if self._data_directory:
            self._output_directory = data_directory + "Light_Curve_Analysis"
            os.makedirs(self._output_directory, exist_ok=True)

        self.search_pattern = search_pattern
        self.list_reference_stars = list_reference_stars
        self.telescope = telescope

        # Aperture parameters
        self.r = aperture_radius
        self.r_in = aperture_radius * 1.7
        self.r_out = aperture_radius * 2.3

        # RADEC of target and ref. stars if needed.
        self.ra_target = ra_target
        self.dec_target = dec_target
        self.ra_ref_stars = ra_ref_stars
        self.dec_ref_stars = dec_ref_stars

        # Transit times of target star: ingress, mid-transit, and egress time.
        self.transit_times = transit_times

        # Centroid bow width for centroid function.
        self._box_width = self.r_out + 0.5

        # Set possible positive answers to set some variables below.
        pos_answers = ['True', 'true', 'yes', 'y', 'Yes', True]
        if from_coordinates in pos_answers:
            self._from_coordinates = True
        else:
            self._from_coordinates = False

        # Output directory for logs
        if self._data_directory:
            logs_dir = self._data_directory + "logs_photometry"
            os.makedirs(logs_dir, exist_ok=True)

            # Logger to track activity of the class
            self.logger = logging.getLogger(f"{self.pipeline} logger")
            self.logger.addHandler(logging.FileHandler(filename=os.path.join(logs_dir,
                                                                             'photometry.log'),
                                                       mode='w'))
            self.logger.setLevel(logging.DEBUG)

            self.logger.info(pyfiglet.figlet_format(f"-*- {self.pipeline} -*-"))

    @classmethod
    def get_class_name(cls):
        return cls.__name__

    @property
    def pipeline(self):
        return os.path.basename(PACKAGEDIR)

    @property
    def readout(self):

        # if self.telescope == "Huntsman":
        #     return -0.070967 * self.gain**2 + 0.652094 * self.gain + 1.564342
        # else:
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
    def telescope_altitude(self):
        return self.get_keyword_value().altitude

    @property
    def telescope_latitude(self):
        return self.get_keyword_value().latitude

    @property
    def telescope_longitude(self):
        return self.get_keyword_value().longitude

    @property
    def gain(self):
        return self.get_keyword_value().gain

    @property
    def airmass(self):
        return self.get_keyword_value().airmass

    @property
    def keyword_list(self):
        file = str(Path(__file__).parents[1]) + "/" + "telescope_keywords.csv"

        telescope = self.telescope

        (Huntsman,
         MQ,
         TESS,
         WASP,
         MEARTH,
         POCS) = np.loadtxt(file, skiprows=1, delimiter=";", dtype=str,
                            usecols=(0, 1, 2, 3, 4, 5), unpack=True)

        if telescope == "Huntsman":
            kw_list = Huntsman
        elif telescope == "MQ":
            kw_list = MQ
        elif telescope == "TESS":
            kw_list = TESS
        elif telescope == "WASP":
            kw_list = WASP
        elif telescope == "MEARTH":
            kw_list = MEARTH
        elif telescope == "POCS":
            kw_list = POCS

        return kw_list

    def get_keyword_value(self, default=None):
        """Returns a header keyword value.

        If the keyword is Undefined or does not exist,
        then return ``default`` instead.
        """

        try:
            kw_values = itemgetter(*self.keyword_list)(self.header)
        except AttributeError:
            self.logger.error("Header keyword does not exist")
            return default
        exp, obstime, instr, readout, gain, airmass, altitude, latitude, longitude = kw_values

        Outputs = namedtuple("Outputs",
                             "exp obstime instr readout gain airmass altitude latitude longitude")

        return Outputs(exp, obstime, instr, readout, gain, airmass, altitude, latitude, longitude)

    def _slice_data(self, data, origin, width):
        y, x = origin
        cutout = data[np.int(x - width / 2.):np.int(x + width / 2.),
                      np.int(y - width / 2.):np.int(y + width / 2.)]
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
                x_cen, y_cen = self._estimate_centroid_via_2dgaussian(data, mask)
            elif method == "1dgaussian":
                x_cen, y_cen = self._estimate_centroid_via_1dgaussian(data, mask)

            elif method == "moments":
                x_cen, y_cen = self._estimate_centroid_via_moments(data, mask)

            # Compute the shifts in y and x.
            shift_y = self._box_width / 2 - y_cen
            shift_x = self._box_width / 2 - x_cen

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

            return new_x, new_y

    def make_effective_psf(self, nddatas, tables, plot_psf_profile=True):
        # Extract stars from all the frames
        stars = extract_stars(nddatas, tables, size=self.r_out)

        # Build the ePSF from all the cutouts extracted
        epsf_builder = EPSFBuilder(oversampling=1., maxiters=15, progress_bar=True,
                                   recentering_boxsize=self.r_out)
        epsf, fitted_star = epsf_builder(stars)

        masked_eff_psf = self._mask_data(epsf.data)

        x_cen, y_cen = self._estimate_centroid_via_2dgaussian(epsf.data, mask=masked_eff_psf.mask)

        # Output directory for ePSF
        output_directory = self._data_directory + "ePSF"
        os.makedirs(self._data_directory + "ePSF", exist_ok=True)

        epsf_name = os.path.join(output_directory, "ePSF.png")

        # Add subplot for fitted psf star
        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot(111)
        # ax.set_title(f"Huntsman {cam} Camera {instrume}\n"
        #             f"Star {star_id} " r"($m_\mathrm{V}=10.9$)", fontsize=15)
        norm = simple_norm(epsf.data, "sqrt", percent=99.9)
        epsf_img = ax.imshow(epsf.data, norm=norm, cmap="viridis", origin="lower")
        ax.scatter(x_cen, y_cen, c='k', marker='+', s=100)
        # ax.legend(loc="lower left", ncol=2, fontsize=10, framealpha=True)

        # Draw the apertures of object and background
        circ = Circle((x_cen, y_cen), self.r, alpha=0.7, facecolor="none",
                      edgecolor="k", lw=2.0, zorder=3)
        circ1 = Circle((x_cen, y_cen), self.r_in, alpha=0.7, facecolor="none",
                       edgecolor="r", ls="--", lw=2.0, zorder=3)
        circ2 = Circle((x_cen, y_cen), self.r_out, alpha=0.7, facecolor="none",
                       edgecolor="r", ls="--", lw=2.0, zorder=3)
        # ax.text(x_cen, self.r_out - self.r_in - 1.0, "Local Sky Background", color="w",
        #         ha="center", fontsize=13.0, zorder=5)  # 8.7

        n, radii = 50, [self.r_in, self.r_out]
        theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
        xs = np.outer(radii, np.cos(theta))
        ys = np.outer(radii, np.sin(theta))

        # in order to have a closed area, the circles
        # should be traversed in opposite directions
        xs[1, :] = xs[1, ::-1]
        ys[1, :] = ys[1, ::-1]

        # ax = plt.subplot(111, aspect='equal')
        ax.fill(np.ravel(xs) + x_cen, np.ravel(ys) + y_cen, facecolor="gray", alpha=0.6, zorder=4)
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.add_patch(circ)
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        ax.set_xlabel("X Pixels", fontsize=15)
        ax.set_ylabel("Y Pixels", fontsize=15)
        ax.set_xlim((0.0, self.r_out))
        ax.set_ylim((0.0, self.r_out))

        # Colorbar for the whole figure and new axes for it
        # fig.colorbar(epsf_img, orientation="vertical")

        fig.savefig(epsf_name, dpi=300)

        if plot_psf_profile:
            projections_list = list()
            sl = self.r
            pr = 5
            for nd in nddatas:
                projection_x = nd.data[np.int((x_cen - sl)):np.int(2.3 * (x_cen + sl)),
                                       np.int(y_cen - pr):np.int(y_cen + pr)]
                projection_y = nd.data[np.int(x_cen - pr):np.int(x_cen + pr),
                                       np.int((y_cen - sl)):np.int(2.3 * (y_cen + sl))]
                projection_x = np.mean(projection_x, axis=1)
                projection_y = np.mean(projection_y, axis=0)

                projection_average = (projection_x + projection_y) / 2

                projections_list.append(np.asarray(projection_average))

            # Name of PSF profile image
            fig_slices_name = os.path.join(output_directory, "{}_{}_profile.png".
                                           format(self.instrument, self.target_star))

            projection_average = np.sum(projections_list, axis=0) / len(projections_list)
            psf_half = (np.max(projection_average) + np.min(projection_average)) / 2
            pixs = np.linspace(-sl, sl, len(projection_average))

            peaks, _ = scipy.signal.find_peaks(projection_average)
            results_half = scipy.signal.peak_widths(projection_average, peaks, rel_height=0.5)

            idx_n, idx_p = -np.max(results_half[0]) / 4, np.max(results_half[0]) / 4

            fig = plt.figure(figsize=(6.5, 6.5))
            ax = fig.add_subplot(111)
            plt.title(f"PSF profile of {self.target_star} " r"($m_\mathrm{V}=10.0$)", fontsize=15)
            ax.plot(pixs, projection_average, "k-", ms=3)
            # ax.axhline(y=psf_half, xmin=0.36, xmax=0.67, c="r", ls="--", lw=1.5)
            ax.axvline(x=idx_n, c="b", ls="-.", lw=1.5)
            ax.axvline(x=idx_p, c="b", ls="-.", lw=1.5)
            ax.axvspan(idx_n, idx_p, facecolor='blue', alpha=0.15)
            ax.text(idx_p + 0.4, psf_half, rf"FWHM$\approx${np.max(results_half[0]) / 2:.3f} pix",
                    color="k", fontsize=13)

            ax.tick_params(axis="both", which="major", labelsize=15)
            ax.set_xlabel("Pixels", fontsize=15)
            ax.set_ylabel("Counts", fontsize=15)
            ax.grid(alpha=0.5)
            fig.savefig(fig_slices_name, dpi=300)

    def save_star_cutout(self, star_id, x, y, cutout, filename):
        """Save cutouts of a given star

        Parameters
        ----------
        star_id: str
            Name of star to do the cutout.
        x : float
            x-position of centroid.
        y : float
            y-position of centroid.
        cutout : array
            Cutout of the image.
        filename : str
            Name of file.
        """

        # Output directory for all the cutouts
        output_directory = self._data_directory + f"{star_id}_Cutouts"
        os.makedirs(output_directory, exist_ok=True)

        if filename.endswith(".fz"):
            filename = filename.replace(".fz", "")
        filename = os.path.splitext(os.path.basename(filename))[0]

        # Name of centroid image
        fig_cutouts_name = os.path.join(output_directory, "{}_{}_{}_centroid.png".
                                        format(self.header["INSTRUME"], star_id, filename))

        # Add subplot for normal star
        instrume = self.header["INSTRUME"]
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.set_title(f"Huntsman Camera {instrume}\n"
                     f"Star {star_id}", fontsize=15)
        norm = simple_norm(cutout, "sqrt", percent=99.7)
        ax.imshow(cutout, norm=norm, origin="lower", cmap="viridis")
        ax.scatter(x, y, c='k', marker='+', s=100)

        # Draw the apertures of object and background
        r = self.r
        # r_in = self.r_in
        # r_out = self.r_out
        circ = Circle((x, y), r, alpha=0.7, facecolor="none",
                      edgecolor="k", lw=2.0, zorder=3)
        # circ1 = Circle((x, y), r_in, alpha=0.7, facecolor="none",
        #                edgecolor="r", ls="--", lw=2.0, zorder=3)
        # circ2 = Circle((x, y), r_out, alpha=0.7, facecolor="none",
        #                edgecolor="r", ls="--", lw=2.0, zorder=3)

        # n, radii = 50, [r_in, r_out]
        # theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
        # xs = np.outer(radii, np.cos(theta))
        # ys = np.outer(radii, np.sin(theta))

        # # in order to have a closed area, the circles
        # # should be traversed in opposite directions
        # xs[1, :] = xs[1, ::-1]
        # ys[1, :] = ys[1, ::-1]

        # ax.fill(np.ravel(xs) + x, np.ravel(ys) + y, facecolor="gray",
        #         alpha=0.6, zorder=4)
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.add_patch(circ)
        # ax.add_patch(circ1)
        # ax.add_patch(circ2)
        ax.set_xlabel("X Pixels", fontsize=15)
        ax.set_ylabel("Y Pixels", fontsize=15)
        ax.set_xlim((0, self._box_width - 1.0))
        ax.set_ylim((0, self._box_width - 1.0))

        fig.savefig(fig_cutouts_name)
        plt.close(fig)

    @staticmethod
    def make_aperture(data, coordinates, radius, r_in, r_out,
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
        object_apertures = CircularAperture(coordinates, r=radius)

        # Annular outer aperture for the sky background
        background_apertures = CircularAnnulus(coordinates, r_in=r_in, r_out=r_out)

        # Find median value of counts-per-pixel in the background
        background_mask = background_apertures.to_mask(method="center")
        background_data = background_mask.multiply(data)
        mask = background_mask.data
        annulus_data_1d = background_data[mask > 0]
        (mean_sigclip,
         median_sigclip,
         std_sigclip) = sigma_clipped_stats(annulus_data_1d, sigma=3, maxiters=10)
        # sky_bkg = 3 * median_sigclip - 2 * mean_sigclip

        # Make aperture photometry for the object and the background
        apertures = [object_apertures, background_apertures]
        phot_table = aperture_photometry(data, apertures, method=method, subpixels=subpixels)

        # For consistent outputs in table
        for col in phot_table.colnames:
            phot_table[col].info.format = "%.8g"

        # Find median value of counts-per-pixel in the sky background.
        # sky_bkg = phot_table["aperture_sum_1"] / background_apertures.area
        sky_bkg = median_sigclip
        phot_table['bkg_median'] = sky_bkg

        # Find background in object inner aperture and subtract it
        object_background = sky_bkg * object_apertures.area

        phot_table["object_bkg"] = object_background
        phot_table["object_bkg"].info.format = "%.8g"

        # assert phot_table["aperture_sum_0"] > phot_table["object_bkg"]

        object_final_counts = phot_table["aperture_sum_0"] - object_background

        # Replace negative values by NaN
        if object_final_counts < 0:
            phot_table["object_bkg_subtracted"] = np.nan
        else:
            phot_table["object_bkg_subtracted"] = object_final_counts

        # For consistent outputs in table
        phot_table["object_bkg_subtracted"].info.format = "%.8g"

        return (phot_table["object_bkg_subtracted"].item(), phot_table["object_bkg"].item(),
                phot_table)

    # @logged
    def do_photometry(self, star_id, data_directory, search_pattern, ra_star=None, dec_star=None,
                      make_effective_psf=False, save_cutout=False):
        """Get all data from plate-solved images (right ascention,
           declination, airmass, dates, etc). Then, it converts the
           right ascention and declination into image positions to
           call make_aperture and find its total counts.

        Parameters
        ----------
        star_id: str
            Name of star to be localized in each file
        data_directory: list
            List of files (frames) where we want to get the counts
        search_pattern: str
            Pattern to search files

        Returns
        --------
        Counts of a star, list of good frames and airmass: tuple

        """
        if self._from_coordinates:
            star = SkyCoord(ra_star, dec_star, unit='deg', frame='icrs')
        else:
            star = SkyCoord.from_name(star_id)

        # Search for files containing data to analyze
        fits_files = search_files_across_directories(data_directory, search_pattern)

        # List of ADU counts for the source, background
        object_counts = list()
        background_in_object = list()

        # List of exposure times
        exptimes = list()

        airmasses = list()

        # List of object positions
        x_pos = list()
        y_pos = list()

        # Observation dates list
        times = list()

        # List of good frames
        self.good_frames_list = list()

        tables = list()
        nddatas = list()
        # projections_list = list()

        fmt = "{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} frames | {elapsed}<{remaining}"
        for fn in tqdm(fits_files, desc=f"{18 * ' ' }Progress: ", bar_format=fmt):
            # Get data, header and WCS of fits files with any extension
            data = get_data(fn)
            self.header = get_header(fn)

            wcs = WCS(self.header)
            # Check if WCS exist in image
            if wcs.is_celestial:

                # Star pixel positions in the image
                center_yx = wcs.all_world2pix(star.ra, star.dec, 0)

                cutout = self._slice_data(data, center_yx, self._box_width)

                masked_data = self._mask_data(cutout)

                x_cen, y_cen = self.find_centroid(center_yx, cutout, masked_data.mask,
                                                  method="2dgaussian")

                # Exposure time
                exptimes.append(self.exptime)
                airmasses.append(self.airmass)

                # Observation times
                time = self.obs_time

                # Sum of counts inside aperture
                (counts_in_aperture,
                    bkg_in_object,
                    phot_table) = self.make_aperture(data, (y_cen, x_cen), radius=self.r,
                                                     r_in=self.r_in, r_out=self.r_out)

                self.logger.debug(phot_table)

                object_counts.append(counts_in_aperture)
                background_in_object.append(bkg_in_object)
                x_pos.append(center_yx[1])
                y_pos.append(center_yx[0])
                times.append(time)
                self.good_frames_list.append(fn)

                if make_effective_psf:
                    cutout_psf = self._slice_data(data, center_yx, 2. * self.r_out)
                    masked_data = self._mask_data(cutout_psf)
                    x, y = self._estimate_centroid_via_2dgaussian(cutout_psf, mask=masked_data.mask)
                    positions = Table()
                    positions["x"] = [x]
                    positions["y"] = [y]

                    tables.append(positions)
                    nddatas.append(NDData(data=cutout_psf))

                if save_cutout:
                    x_cen, y_cen = self._estimate_centroid_via_2dgaussian(cutout,
                                                                          mask=masked_data.mask)
                    self.save_star_cutout(star_id, x_cen, y_cen, cutout, fn)

            else:
                continue

        if make_effective_psf:
            print(f"Building effective PSF for target star {self.target_star}")
            self.make_effective_psf(nddatas, tables)

        return (object_counts, background_in_object,
                exptimes, x_pos, y_pos, times, airmasses)

    # @logged
    def get_relative_flux(self, save_rms=False):
        """Find the flux of a target star relative to some reference stars,
           using the counts inside an aperture

        Parameters
        ----------
        save_rms : bool, optional (defaul is False)
            Save a txt file with the rms achieved for each time that the class is executed.

        Returns
        -------
        relative flux : float
            The ratio between the target flux and the ntegrated flux of the reference stars
        """
        start = time.time()

        print(pyfiglet.figlet_format(f"-*- {self.pipeline} -*-")
              + f"{16 * '#'}       by Jaime Andrés Alvarado Montes       {16 * '#'}\n")

        print(pyfiglet.figlet_format(f"1. Time Series")
              + "        Part of transyto package by Jaime A. Alvarado-Montes\n")

        print("{}>\t{} will use {} reference stars for differential photometry\n".
              format(8 * '-', self.pipeline, len(self.list_reference_stars)))

        print(f"{8 * '-'}>\tStarting aperture photometry on target star {self.target_star}:\n")

        self.logger.debug(f"-------------- Aperture photometry of {self.target_star} ---------------\n")
        # Get flux of target star
        (target_flux,
         background_in_object,
         exptimes,
         x_pos_target,
         y_pos_target,
         times,
         airmasses) = self.do_photometry(self.target_star, self._data_directory, self.search_pattern,
                                         ra_star=self.ra_target, dec_star=self.dec_target,
                                         make_effective_psf=False, save_cutout=True)

        print(f"\n{18 * ' '}Finished aperture photometry on target star {self.target_star}\n")

        # Get the date times anc compute the Barycentric Julian Date (Barycentric Dynamical Time)
        times = np.asarray(times)
        self.jdutc_times = Time(times, format='isot', scale='utc')
        bjdtdb_times = utc_tdb.JDUTC_to_BJDTDB(self.jdutc_times, hip_id=8102,
                                               lat=self.telescope_latitude,
                                               longi=self.telescope_longitude,
                                               alt=self.telescope_altitude)

        self.time_norm_factor = 2450000.
        times = bjdtdb_times[0] - self.time_norm_factor

        airmasses = np.asarray(airmasses)

        print(f"\n{8 * '-'}>\t{self.pipeline} will compute now the combined flux of the ensemble\n")

        # Positions of target star
        self.x_pos_target = np.array(x_pos_target) - np.nanmean(x_pos_target)
        self.y_pos_target = np.array(y_pos_target) - np.nanmean(y_pos_target)

        # Target and background counts per second
        exptimes = np.asarray(exptimes)
        target_flux = np.asarray(target_flux)
        target_flux_sec = target_flux / exptimes
        target_background_sec = np.asarray(background_in_object) / exptimes

        # CCD gain
        ccd_gain = self.gain

        noise_sources = compute_noises(ccd_gain, exptimes, target_flux_sec,
                                       target_background_sec, self.readout, self.r)

        # Sigma photon noise
        self.sigma_phot = noise_sources.sigma_photon

        # Sigma sky-background noise
        self.sigma_sky = noise_sources.sigma_sky

        # Sigma readout noise
        self.sigma_ron = noise_sources.sigma_readout

        # Sigma scintillation
        self.sigma_scint = compute_scintillation(0.143, self.telescope_altitude,
                                                 airmasses, exptimes)

        # Total photometric error for 1 mag in one observation
        self.sigma_total = np.sqrt(self.sigma_phot**2.0 + self.sigma_ron**2.0
                                   + self.sigma_sky**2.0 + self.sigma_scint**2.0)

        # Signal to noise: shot, sky noise (per second) and readout
        S_to_N_obj_sec = target_flux_sec / np.sqrt(target_flux_sec + target_background_sec
                                                   + (self.readout * self.r)**2 * np.pi
                                                   / (ccd_gain * exptimes))
        # Convert SN_sec to actual SN
        S_to_N_obj = S_to_N_obj_sec * np.sqrt(ccd_gain * exptimes)

        # Get the flux of each reference star
        reference_star_flux_sec = list()
        background_in_ref_star_sec = list()
        reference_airmasses = list()

        if self._from_coordinates:
            list_ra_ref_stars = self.ra_ref_stars
            list_dec_ref_stars = self.dec_ref_stars

        else:
            list_ra_ref_stars = list([1]) * len(self.list_reference_stars)
            list_dec_ref_stars = list([1]) * len(self.list_reference_stars)

        for ref_star, ra_ref_star, dec_ref_star in zip(self.list_reference_stars,
                                                       list_ra_ref_stars,
                                                       list_dec_ref_stars):

            print(f"{16 * ' '}• Starting aperture photometry on reference star {ref_star}\n")

            self.logger.debug(f"Aperture photometry of {ref_star}\n")
            (refer_flux,
             background_in_ref_star,
             _, _, _, _, ref_airmasses) = self.do_photometry(ref_star, self._data_directory,
                                                             self.search_pattern,
                                                             ra_ref_star, dec_ref_star,
                                                             make_effective_psf=False)
            reference_star_flux_sec.append(np.asarray(refer_flux) / exptimes)
            background_in_ref_star_sec.append(np.asarray(background_in_ref_star) / exptimes)
            reference_airmasses.append(np.asarray(ref_airmasses))
            print(f"\n{18 * ' '}Finished aperture photometry on reference star {ref_star}\n")

        self.reference_star_flux_sec = np.asarray(reference_star_flux_sec)
        background_in_ref_star_sec = np.asarray(background_in_ref_star_sec)

        sigma_squared_ref = (reference_star_flux_sec * exptimes
                             + background_in_ref_star_sec * exptimes
                             + (self.readout * self.r)**2 * np.pi / ccd_gain
                             + reference_airmasses)

        weights_ref_stars = 1.0 / sigma_squared_ref

        ref_flux_averaged = np.average(self.reference_star_flux_sec * exptimes,
                                       weights=weights_ref_stars, axis=0)

        # Integrated flux per sec for ensemble of reference stars
        total_reference_flux_sec = np.sum(self.reference_star_flux_sec, axis=0)

        # Integrated sky background for ensemble of reference stars
        total_reference_bkg_sec = np.sum(background_in_ref_star_sec, axis=0)

        # S/N for reference star per second
        S_to_N_ref_sec = total_reference_flux_sec / np.sqrt(total_reference_flux_sec
                                                            + total_reference_bkg_sec
                                                            + (self.readout * self.r)**2 * np.pi
                                                            / (ccd_gain * exptimes))
        # Convert S/N per sec for ensemble to total S/N
        S_to_N_ref = S_to_N_ref_sec * np.sqrt(ccd_gain * exptimes)

        # Relative flux per sec of target star
        differential_flux = target_flux / ref_flux_averaged
        # differential_flux = [-2.5 * np.log(target_flux / (r * exptimes)) for r in reference_star_flux_sec]
        # differential_flux = np.average(differential_flux, axis=0)

        # Find Differential S/N for object and ensemble
        S_to_N_diff = 1 / np.sqrt(S_to_N_obj**-2 + S_to_N_ref**-2)

        # Ending time of computatin analysis.
        end = time.time()
        exec_time = end - start

        # Print when all of the analysis ends
        print(f"{8 * '-'}>\tDifferential photometry of {self.target_star} has been finished, "
              f"with {len(self.good_frames_list)} frames "
              f"of camera {self.instrument} (run time: {exec_time:.3f} sec)\n")

        if save_rms:
            # Output directory for files that contain photometric precisions
            output_directory = self._output_directory + "/rms_precisions"
            os.makedirs(output_directory, exist_ok=True)

            # File with rms information
            file_rms_name = os.path.join(output_directory, f"rms_{self.instrument}.txt")

            with open(file_rms_name, "a") as file:
                file.write(f"{self.r} {self.std} {self.std_binned} "
                           f"{np.nanmedian(S_to_N_obj)} {np.nanmedian(S_to_N_ref)} "
                           f"{np.nanmedian(S_to_N_diff)}\n")

        return (times, differential_flux, self.sigma_total)


class LightCurve(TimeSeriesAnalysis):

    def __init__(self, target_star="", data_directory="", search_pattern="*.fit*",
                 list_reference_stars=[], aperture_radius=15, from_coordinates=True, ra_target=None,
                 dec_target=None, transit_times=[], ra_ref_stars=None, dec_ref_stars=None,
                 telescope=""):
        super(LightCurve, self).__init__(target_star=target_star, data_directory=data_directory,
                                         search_pattern=search_pattern,
                                         list_reference_stars=list_reference_stars,
                                         aperture_radius=aperture_radius,
                                         from_coordinates=from_coordinates,
                                         ra_target=ra_target,
                                         dec_target=dec_target,
                                         transit_times=transit_times,
                                         ra_ref_stars=ra_ref_stars,
                                         dec_ref_stars=dec_ref_stars,
                                         telescope=telescope)

    def clip_outliers(self, flux, sigma_lower=0, sigma_upper=0, **kwargs):
        """Clips out the outliers in flux.

        Parameters
        ----------
        flux : array
            Array of flux to be cleaned from outliers.
        sigma_lower : int, optional (default is 0)
            Lower threshold to select good values in flux.
        sigma_upper : int, optional (default is 0)
            Upper threshold to select good values in flux.
        **kwargs
            Description

        Returns
        -------
        masked arrays
            Flux array clipped out of outliers, mask with the indexes of good values in flux.
        """

        mask = np.where((flux > sigma_lower) * (flux < sigma_upper))

        return flux[mask], mask

    @staticmethod
    def clean_timeseries(time, flux, flux_error, return_mask=False, **kwargs):
        """Clean timeseries data of nans, infs, etc.

        Parameters
        ----------
        time : array
            Array of times
        flux : array
            Array of flux
        flux_error : array
            Aray of flux errors
        return_mask : boolean, optional
            Mask of nan values.
        **kwargs
            Description

        Returns
        -------
        arrays
            Time, flux, and flux errors cleaned arrays.
        """

        nan_mask = np.isnan(flux)

        clean_flux = flux[~nan_mask]
        clean_flux_errors = flux_error[~nan_mask]
        clean_time = time[~nan_mask]

        if return_mask:
            return clean_time, clean_flux, clean_flux_errors, nan_mask

        return clean_time, clean_flux, clean_flux_errors

    @staticmethod
    def detrend_timeseries(time, flux, R_star=None, M_star=None, Porb=None):
        """Detrend time-series data

        Parameters
        ----------
        time : array
            Times of the observation
        flux : array
            Flux with trend to be removed
        R_star : float, optional (defaul is None)
            Radius of the star (in solar units)
        M_star : float, optional (defaul is None)
            Mass of the star (in solar units)
        Porb : float, optional (defaul is None)
            Orbital period of the planet (in days).

        Returns
        -------
        detrended and trended flux : numpy array
        """

        trend = scipy.signal.medfilt(flux, 21)
        detrended_flux = flux / trend

        if Porb is not None:
            print(f"{8 * '-'}>\tDetrending time series with M_s = {M_star} M_sun, "
                  + f"R_s = {R_star} R_sun, and P_orb = {Porb:.3f} d\n")

            # Compute the transit duration
            transit_dur = t14(R_s=R_star, M_s=M_star,
                              P=Porb, small_planet=False)

            # Estimate the window length for the detrending
            wl = 3.0 * transit_dur

            # Detrend the time series data
            detrended_flux, _ = flatten(time, flux, return_trend=True, method="biweight",
                                        window_length=wl)

        return detrended_flux

    @staticmethod
    def bin_timeseries(time, flux, bins):
        """Bin data into groups by usinf the mean of each group

        Parameters
        ----------
        flux : array
            Array of flux
        time : array
            Array of time
        bins : array
            Number of bins to bin the data.

        Returns
        -------
        numpy array
            Data in bins

        """

        # Makes dataframe of given data
        df_flux = pd.DataFrame({"binned_flux": flux})
        binned_flux = df_flux.groupby(df_flux.index // bins).mean()
        binned_flux = binned_flux["binned_flux"]

        df_time = pd.DataFrame({"binned_times": time})
        binned_times = (df_time.groupby(df_time.index // bins).last()
                        + df_time.groupby(df_time.index // bins).first()) / 2
        binned_times = binned_times["binned_times"]

        return binned_times, binned_flux

    @staticmethod
    def model_lightcurve(time, flux, limb_dc):
        """Summary
        """

        model = transitleastsquares(time, flux)
        results = model.power(u=limb_dc)  # , oversampling_factor=5, duration_grid_step=1.02)

        print(f"\n{8 * '-'}>\tStarting model of light curve...\n")
        print(f"{8 * ' '}\t • Period: {results.period:.5f} d")
        print(f"{8 * ' '}\t • {len(results.transit_times)} transit times in time series: "
              f'{[f"{i:0.5f}" for i in results.transit_times]}')
        print(f"{8 * ' '}\t • Transit depth: {results.depth:.5f}")
        print(f"{8 * ' '}\t • Best duration (days): {results.duration: .5f}")
        print(f"{8 * ' '}\t • Signal detection efficiency (SDE): {results.SDE}\n")

        print("-------->\tFinished model of light curve. Plotting model...\n")

        return results

    # @logged
    def plot(self, time=[], flux=[], flux_uncertainty=[], bins=30, detrend=False, plot_tracking=False,
             plot_noise_sources=False, model_transit=False):
        """Plot a light curve using the flux time series

        Parameters
        ----------
        time : list or array,
            List of times of observations
        flux : list or array,
            List of target flux
        flux_uncertainty : list or array
            List of data errors
        bins : int, optional (default is 30)
            Number of bins to bin the data.
        detrend : bool, optional (default is False)
            If True, detrending of the time series data will be performed
        plot_tracking : bool, optional (default is False)
            Flag to plot the (x, y) position of target (i.e. tracking) of all
            the observation frames.
        plot_noise_sources : bool, optional (default is False)
            Flag to plot the noise sources throughout the night.
        model_transit : bool, optional (default is False)
            Flag for using transit least squares to model transits.

        No Longer Returned
        ------------------
        """

        print(pyfiglet.figlet_format(f"2. Light Curve")
              + "\t     Part of transyto package by Jaime A. Alvarado-Montes\n")

        # Get the data from the target star.
        star_data = catalog.StarData(self.target_star).query_from_mast()
        star_name = star_data["star_name"]
        star_vmag = star_data["Vmag"]
        star_tmag = star_data["Tmag"]

        pd.plotting.register_matplotlib_converters()

        # Remove invalid values such as nan, infs, etc
        time, flux, flux_uncertainty, nan_mask = self.clean_timeseries(time, flux, flux_uncertainty,
                                                                       return_mask=True)

        # Clip values outside sigma_upper
        if self.transit_times:
            jdutc_transit_times = Time(self.transit_times, format='isot', scale='utc')
            bjdtdb_transit_times = utc_tdb.JDUTC_to_BJDTDB(jdutc_transit_times, hip_id=8102,
                                                           lat=self.telescope_latitude,
                                                           longi=self.telescope_longitude,
                                                           alt=self.telescope_altitude)

            ingress = bjdtdb_transit_times[0][0] - self.time_norm_factor
            mid_transit = bjdtdb_transit_times[0][1] - self.time_norm_factor
            egress = bjdtdb_transit_times[0][2] - self.time_norm_factor

            ing_index = np.where(time <= ingress)[0][-1]
            egr_index = np.where(time >= egress)[0][0]

            # Get the pre-transit and post-transit data.
            pre_transit_flux = flux[:ing_index]
            post_transit_flux = flux[egr_index:]

            # Get the in-transit and out-of-transit data.
            in_transit_flux = flux[ing_index:egr_index]
            out_of_transit_flux = np.concatenate((pre_transit_flux, post_transit_flux))

            my_box_in = plt.boxplot(in_transit_flux)
            sigma_lower = my_box_in["caps"][0].get_ydata()[0]

            my_box_out = plt.boxplot(out_of_transit_flux)
            sigma_upper = my_box_out["caps"][1].get_ydata()[0]

            # Get the flux and mask after clipping the outliers.
            flux, clip_mask = self.clip_outliers(flux, sigma_lower=sigma_lower,
                                                 sigma_upper=sigma_upper)

            # Compute the relative/normalized flux of the target star (post-clipping).
            normalized_flux = flux / np.nanmedian(out_of_transit_flux[out_of_transit_flux
                                                                      < sigma_upper])

        if not self.transit_times:
            # Compute the relative/normalized flux of the target star (pre-clipping).
            normalized_flux = flux / np.nanmedian(flux)

            # Boxplot to identify outliers.
            my_box = plt.boxplot(normalized_flux)
            sigma_lower = my_box["caps"][0].get_ydata()[0]
            sigma_upper = my_box["caps"][1].get_ydata()[0]

            # Get the flux and mask after clipping the outliers.
            flux, clip_mask = self.clip_outliers(flux, sigma_lower=sigma_lower,
                                                 sigma_upper=sigma_upper)

            # Compute the relative/normalized flux of the target star (post-clipping).
            normalized_flux = flux / np.nanmedian(flux)

        # Select unclipped values in the array of flux errors.
        flux_uncertainty = flux_uncertainty[clip_mask]

        # Barycentric Julian Date (BJD)
        time = time[clip_mask]

        # Violin plot.
        fig = plt.figure(figsize=(6.0, 5.0))

        # Name for boxplot.
        violin_name = os.path.join(self._output_directory, "Violinplot_cam"
                                   f"{self.instrument}_rad{self.r}pix_"
                                   f"{len(self.list_reference_stars)}refstar.png")

        # ax = sns.swarmplot(y=normalized_flux, color=".25", zorder=3)
        if self.transit_times:
            flags = []
            for f in flux:
                if f in in_transit_flux:
                    flags.append("In-transit")
                else:
                    flags.append("Out-of-transit")

            flags = np.array(flags)

            df = pd.DataFrame({"flux": pd.Series(normalized_flux), "Data": pd.Series(flags)})
            df["all"] = ""

            # Violin plot to analyse distribution (with transit times).
            ax = sns.violinplot(x="all", y="flux", hue="Data", data=df, inner="stick",
                                linewidth=1.0, split=True, cut=2, bw="silverman")

        if not self.transit_times:
            # Violin plot to analyse distribution (without transit times).
            ax = sns.violinplot(y=normalized_flux, inner="stick", linewidth=1.0, cut=2,
                                bw="silverman")

        ax.set_xlabel("Density Distribution", fontsize=11)
        ax.set_ylabel("Relative flux", fontsize=11)
        ax.legend(loc=(0.23, 1.0), ncol=2, title="Data", frameon=False)

        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.4)

        fig.savefig(violin_name, dpi=300)
        plt.close(fig)

        print("The density distribution of transit data has been plotted\n")

        if model_transit or (detrend and model_transit):
            print("Starting transit modeling via TLS:\n")

            # flatten_flux = self.detrend_timeseries(time, flux)
            results = self.model_lightcurve(time, normalized_flux)

            # Name for folded light curve.
            model_lightcurve_name = os.path.join(self._output_directory, "Model_lightcurve_cam"
                                                 f"{self.instrument}_rad{self.r}pix_"
                                                 f"{len(self.list_reference_stars)}refstar.png")

            loc = plticker.MultipleLocator(base=5)  # this locator puts ticks at regular intervals

            fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
            ax.plot(results.model_folded_phase, results.model_folded_model, color='red')
            ax.scatter(results.folded_phase, results.folded_y, color='blue', s=10,
                       alpha=0.5, zorder=2)
            ax.set_xlim(0.48, 0.52)
            ax.set_xlabel('Phase')
            ax.set_ylabel('Relative flux')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.1f'))

            fig.savefig(model_lightcurve_name)

            print(f"Folded model of the light curve of {star_name} was plotted\n")

            # Name for periodogram.
            periodogram_name = os.path.join(self._output_directory, "Periodogram_cam"
                                            f"{self.instrument}_rad{self.r}pix_"
                                            f"{len(self.list_reference_stars)}refstar.png")

            fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
            ax.axvline(results.period, alpha=0.4, lw=3)
            # ax.set_xlim(np.min(results.periods), np.max(results.periods))
            for n in range(2, 10):
                ax.axvline(n * results.period, alpha=0.4, lw=1, linestyle="dashed")
                ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
            ax.set_ylabel(r'SDE')
            ax.set_xlabel('Period [d]')
            ax.plot(results.periods, results.power, color='black', lw=0.5)
            # ax.set_xlim(0, max(results.periods))

            fig.savefig(periodogram_name, dpi=300)

            # Detrend data using the previous transit model.
            # normalized_flux = self.detrend_data(time, normalized_flux, R_star=star_data["Rs"],
            #                                     M_star=star_data["Ms"], Porb=results.period)

        # Detrend data without using transit model.
        if detrend and not model_transit:
            normalized_flux = self.detrend_timeseries(time, flux)

        # Standard deviation in ppm for the observation
        std = np.nanstd(normalized_flux)

        # Name for light curve.
        lightcurve_name = os.path.join(self._output_directory, "Lightcurve_cam"
                                       f"{self.instrument}_rad{self.r}pix_"
                                       f"{len(self.list_reference_stars)}refstar.png")

        fig, ax = plt.subplots(2, 1,
                               sharey="row", sharex="col", figsize=(8.5, 6.3))
        fig.suptitle(f"Differential Photometry\nTarget Star {star_name}, "
                     f"Vmag={star_vmag} (Tmag={star_tmag}) Aperture = {self.r} pix, "
                     f"Focus: {self.header['FOC-POS']} eu", fontsize=13)

        ax[1].plot(time, normalized_flux, "k.", ms=3,
                   label=f"NBin = {self.exptime:.1f} s, std = {std:.2%}")

        ax[1].errorbar(time, normalized_flux, yerr=flux_uncertainty,
                       fmt="none", ecolor="k", elinewidth=0.8,
                       label=r"$\sigma_{\mathrm{tot}}=\sqrt{\sigma_{\mathrm{phot}}^{2} "
                       r"+ \sigma_{\mathrm{sky}}^{2} + \sigma_{\mathrm{scint}}^{2} + "
                       r"\sigma_{\mathrm{read}}^{2}}$", capsize=0.0)

        # Binned data and times
        if bins != 0:
            binned_times, binned_flux = self.bin_timeseries(time, normalized_flux, bins)
            std_binned = np.nanstd(binned_flux)

            # Total time for binsize
            nbin_tot = self.exptime * bins
            ax[1].plot(binned_times, binned_flux, "ro", ms=4,
                       label=f"NBin = {nbin_tot:.1f} s, std = {std_binned:.2%}")

        ax[1].set_ylabel("Relative Flux", fontsize=13)
        ax[1].legend(fontsize=8.0, loc=(0.0, 1.0), ncol=3, framealpha=1.0, frameon=False)

        for counter in range(len(self.list_reference_stars)):
            # ax[1].xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))

            # Colors for comparison stars
            # colors = ["blue", "magenta", "green", "cyan", "firebrick"]

            ax[0].plot(time, self.reference_star_flux_sec[counter][~nan_mask][clip_mask]
                       / np.nanmean(self.reference_star_flux_sec[counter][~nan_mask][clip_mask]),
                       "o", ms=1.3, label=f"Ref. {self.list_reference_stars[counter]}")
            ax[0].set_ylabel("Relative Flux", fontsize=13)
            # ax[0].set_ylim((0.9, 1.05))
            ax[0].legend(fontsize=8.1, loc="lower left", ncol=len(self.list_reference_stars),
                         framealpha=1.0, frameon=True)

        ax[1].text(0.97, 0.9, "b)", fontsize=11, transform=ax[1].transAxes)
        ax[0].text(0.97, 0.9, "a)", fontsize=11, transform=ax[0].transAxes)

        # Plot the ingress, mid-transit, and egress times.
        if self.transit_times:
            ax[1].axvline(ingress, c="k", ls="--", alpha=0.5)
            ax[1].axvline(mid_transit, c="k", ls="--", alpha=0.5)
            ax[1].axvline(egress, c="k", ls="--", alpha=0.5)

        ax[1].xaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))
        ax[1].yaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))

        plt.xlabel(r"BJD$_\mathrm{TDB}- $" f"{self.time_norm_factor}", fontsize=13)
        plt.xticks(rotation=30, size=8.0)

        fig.subplots_adjust(hspace=0.2)
        fig.savefig(lightcurve_name, dpi=300)

        print(f"The light curve of {star_name} was plotted")

        if plot_tracking:
            # Name for plot of tracking.
            plot_tracking_name = os.path.join(self._output_directory, "tracking_plot_cam"
                                              f"{self.instrument}_rad{self.r}pix_"
                                              f"{len(self.list_reference_stars)}refstar.png")

            fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.3))
            ax.plot(time, self.x_pos_target[~nan_mask][clip_mask], "ro-",
                    label="dx [Dec axis]", lw=0.5, ms=1.2)
            ax.plot(time, self.y_pos_target[~nan_mask][clip_mask], "go-",
                    label="dy [RA axis]", lw=0.5, ms=1.2)
            ax.set_ylabel(r"$\Delta$ Pixel", fontsize=13)
            ax.legend(fontsize=8.6, loc="lower right", ncol=1, framealpha=1.0)
            ax.set_title(f"Tracking of camera {self.instrument}", fontsize=13)
            # ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.3f'))
            ax.set_xlabel(f"Time [BJD-{self.time_norm_factor}]", fontsize=13)
            plt.xticks(rotation=30, size=8.0)
            plt.grid(alpha=0.4)
            fig.savefig(plot_tracking_name, dpi=300)

        if plot_noise_sources:
            # Name for plot of noise sources.
            plot_noise_name = os.path.join(self._output_directory, "noises_plot_cam"
                                           f"{self.instrument}_rad{self.r}pix_"
                                           f"{len(self.list_reference_stars)}refstar.png")

            fig, ax = plt.subplots(1, 1, sharey="row", sharex="col", figsize=(8.5, 6.3))
            ax.set_title(f"Noise Sources in {star_name} " r"($m_\mathrm{V}=10.0$)", fontsize=13)
            ax.plot_date(self.jdutc_times.plot_date, self.sigma_total[~nan_mask] * 100, "k-",
                         label=r"$\sigma_{\mathrm{total}}$")
            ax.plot_date(self.jdutc_times.plot_date, self.sigma_scint[~nan_mask] * 100,
                         "g-", label=r"$\sigma_{\mathrm{scint}}$")
            ax.plot_date(self.jdutc_times.plot_date, self.sigma_phot[~nan_mask] * 100,
                         color="firebrick", ls="-", marker=None, label=r"$\sigma_{\mathrm{phot}}$")
            ax.plot_date(self.jdutc_times.plot_date, self.sigma_sky[~nan_mask] * 100,
                         "b-", label=r"$\sigma_{\mathrm{sky}}$")
            ax.plot_date(self.jdutc_times.plot_date, self.sigma_ron[~nan_mask] * 100,
                         "r-", label=r"$\sigma_{\mathrm{read}}$")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.998), fancybox=True,
                      ncol=5, frameon=True, fontsize=8.1)
            # ax.set_yscale("log")
            ax.tick_params(axis="both", direction="in")
            ax.set_ylabel("Amplitude Error [%]", fontsize=13)
            plt.xticks(rotation=30, size=8.0)
            ax.set_xlabel("Time [UTC]", fontsize=13)
            # ax.set_ylim((0.11, 0.48))
            ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
            plt.grid(alpha=0.4)
            fig.savefig(plot_noise_name, dpi=300)

    def plot_from_table(self, table, exptime=30, bins=5, detrend=False, model_transit=False,
                        x_label="Time"):

        print(pyfiglet.figlet_format(f"* LightCurve *")
              + "\t     Part of transyto package by Jaime A. Alvarado-Montes\n")

        # Get the data from the target star.
        star_data = catalog.StarData(self.target_star).query_from_mast()
        star_name = star_data["star_name"]
        star_vmag = star_data["Vmag"]
        star_tmag = star_data["Tmag"]
        star_mass = star_data["Ms"]
        star_radius = star_data["Rs"]
        planet_period = star_data["orbital_period"]

        # Name for light curve.
        output_directory = os.path.dirname(table) + "/"
        os.makedirs(output_directory, exist_ok=True)

        if table.endswith(".dat"):
            sep = "\t"
        if table.endswith(".csv"):
            sep = ";"
        table = pd.read_csv(table, usecols=["time", "flux", "flux_uncertainty"], delimiter=sep)
        index = 10000
        time = np.array(table["time"]).astype(np.float)[:index]
        flux = np.array(table["flux"]).astype(np.float)[:index]
        flux_uncertainty = np.array(table["flux_uncertainty"]).astype(np.float)[:index]

        pd.plotting.register_matplotlib_converters()

        # Remove invalid values such as nan, infs, etc
        time, flux, flux_uncertainty, nan_mask = self.clean_timeseries(time, flux, flux_uncertainty,
                                                                       return_mask=True)

        # Clip values outside sigma_upper
        if self.transit_times:
            jdutc_transit_times = Time(self.transit_times, format='isot', scale='utc')
            bjdtdb_transit_times = utc_tdb.JDUTC_to_BJDTDB(jdutc_transit_times, hip_id=8102,
                                                           lat=self.telescope_latitude,
                                                           longi=self.telescope_longitude,
                                                           alt=self.telescope_altitude)

            ingress = bjdtdb_transit_times[0][0]
            mid_transit = bjdtdb_transit_times[0][1]
            egress = bjdtdb_transit_times[0][2]

            ing_index = np.where(time <= ingress)[0][-1]
            egr_index = np.where(time >= egress)[0][0]

            # Get the pre-transit and post-transit data.
            pre_transit_flux = flux[:ing_index]
            post_transit_flux = flux[egr_index:]

            # Get the in-transit and out-of-transit data.
            in_transit_flux = flux[ing_index:egr_index]
            out_of_transit_flux = np.concatenate((pre_transit_flux, post_transit_flux))

            my_box_in = plt.boxplot(in_transit_flux)
            sigma_lower = my_box_in["caps"][0].get_ydata()[0]

            my_box_out = plt.boxplot(out_of_transit_flux)
            sigma_upper = my_box_out["caps"][1].get_ydata()[0]

            # Get the flux and mask after clipping the outliers.
            flux, clip_mask = self.clip_outliers(flux, sigma_lower=sigma_lower,
                                                 sigma_upper=sigma_upper)

            # Compute the relative/normalized flux of the target star (post-clipping).
            normalized_flux = flux / np.nanmedian(out_of_transit_flux[out_of_transit_flux
                                                                      < sigma_upper])

        if not self.transit_times:
            # Compute the relative/normalized flux of the target star (pre-clipping).
            normalized_flux = flux / np.nanmedian(flux)

            # Boxplot to identify outliers.
            my_box = plt.boxplot(normalized_flux)
            sigma_lower = my_box["caps"][0].get_ydata()[0]
            sigma_upper = my_box["caps"][1].get_ydata()[0]

            # Get the flux and mask after clipping the outliers.
            flux, clip_mask = self.clip_outliers(flux, sigma_lower=sigma_lower,
                                                 sigma_upper=sigma_upper)

            # Compute the relative/normalized flux of the target star (post-clipping).
            normalized_flux = flux / np.nanmedian(flux)

        # Select unclipped values in the array of flux errors.
        flux_uncertainty = flux_uncertainty[clip_mask]

        # Barycentric Julian Date (BJD)
        time = time[clip_mask]

        # Violin plot.
        fig = plt.figure(figsize=(6.0, 5.0))

        # Name for boxplot.
        violin_name = os.path.join(output_directory, "Violinplot_cam"
                                   f"{self.telescope}_rad{self.r}pix.png")

        # ax = sns.swarmplot(y=normalized_flux, color=".25", zorder=3)
        if self.transit_times:
            flags = []
            for f in flux:
                if f in in_transit_flux:
                    flags.append("In-transit")
                else:
                    flags.append("Out-of-transit")

            flags = np.array(flags)

            df = pd.DataFrame({"flux": pd.Series(normalized_flux), "Data": pd.Series(flags)})
            df["all"] = ""

            # Violin plot to analyse distribution (with transit times).
            ax = sns.violinplot(x="all", y="flux", hue="Data", data=df, inner="stick",
                                linewidth=1.0, split=True, cut=2, bw="silverman")

        if not self.transit_times:
            # Violin plot to analyse distribution (without transit times).
            ax = sns.violinplot(y=normalized_flux, inner="box", linewidth=1.0, cut=2,
                                bw="silverman")

        ax.set_xlabel("Density Distribution", fontsize=11)
        ax.set_ylabel("Relative flux", fontsize=11)

        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.4)

        fig.savefig(violin_name, dpi=300)
        plt.close(fig)

        print(f"{8 * '-'}>\tThe density distribution of transit data has been plotted\n")

        if model_transit or (detrend and model_transit):
            print("-------->\tStarting transit modeling via Transit Least Squares (Hippke & "
                  "Heller 2019)\n\n"
                  "         \t • Computing LD Coefficients v.1.0 (Espinoza $ Jordan 2015)")

            # Calculate the limd darkening coefficients.
            limb_dc = ldc.compute(name="CoRot-5", Teff=star_data["Teff"], RF="KpHiRes", FT="A100",
                                  grav=star_data["stellar_gravity"], metal=star_data["Fe/H"])[0]
            ab = (limb_dc[1], limb_dc[2])

            # Detrend data using the previous transit model.
            normalized_flux = self.detrend_timeseries(time, normalized_flux, R_star=star_radius,
                                                      M_star=star_mass, Porb=planet_period)

            # flatten_flux = self.detrend_timeseries(time, flux)
            results = self.model_lightcurve(time, normalized_flux, limb_dc=ab)

            # Name for folded light curve.
            model_lightcurve_name = os.path.join(output_directory, "Model_lightcurve_cam"
                                                 f"{self.telescope}_rad{self.r}pix.png")

            # This locator puts ticks at regular intervals.
            loc = plticker.MultipleLocator(base=5)

            fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
            ax.plot(results.model_folded_phase, results.model_folded_model, color='red')
            ax.scatter(results.folded_phase, results.folded_y, color='blue', s=10,
                       alpha=0.5, zorder=2)
            # ax.set_xlim(0.48, 0.52)
            ax.set_xlabel('Phase')
            ax.set_ylabel('Relative flux')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.1f'))

            fig.savefig(model_lightcurve_name)

            print(f"-------->\tFolded model of the light curve of {star_name} was plotted\n")

            # Name for periodogram.
            periodogram_name = os.path.join(output_directory, "Periodogram_cam"
                                            f"{self.telescope}_rad{self.r}pix.png")

            fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
            ax.axvline(results.period, alpha=0.4, lw=3)
            # ax.set_xlim(np.min(results.periods), np.max(results.periods))
            for n in range(2, 10):
                ax.axvline(n * results.period, alpha=0.4, lw=1, linestyle="dashed")
                ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
            ax.set_ylabel(r'SDE')
            ax.set_xlabel('Period [d]')
            ax.plot(results.periods, results.power, color='black', lw=0.5)
            ax.set_xlim(0.0, np.max(results.periods))

            fig.savefig(periodogram_name, dpi=300)

        # Detrend data without using transit model.
        if detrend and not model_transit:
            normalized_flux = self.detrend_timeseries(time, flux)

        # Standard deviation in ppm for the observation
        std = np.nanstd(normalized_flux)

        lightcurve_name = os.path.join(output_directory, "Lightcurve_cam"
                                       f"{self.telescope}_rad{self.r}pix_.png")

        fig, ax = plt.subplots(1, 1,
                               sharey="row", sharex="col", figsize=(8.5, 6.3))
        fig.suptitle(f"Differential Photometry\nTarget Star {star_name}, Vmag={star_vmag} "
                     f"(Tmag={star_tmag}), Aperture = {self.r} pix", fontsize=13)

        ax.plot(time, normalized_flux, "k.", ms=3,
                label=f"NBin = {exptime:.1f} s, std = {std:.2%}")

        # ax.errorbar(time, normalized_flux, yerr=flux_uncertainty,
        #             fmt="none", ecolor="k", elinewidth=0.8,
        #             label=r"$\sigma_{\mathrm{tot}}=\sqrt{\sigma_{\mathrm{phot}}^{2} "
        #             r"+ \sigma_{\mathrm{sky}}^{2} + \sigma_{\mathrm{scint}}^{2} + "
        #             r"\sigma_{\mathrm{read}}^{2}}$", capsize=0.0)

        # Binned data and times
        if bins != 0:
            binned_times, binned_flux = self.bin_timeseries(time, normalized_flux, bins)
            std_binned = np.nanstd(binned_flux)

            # Total time for binsize
            nbin_tot = exptime * bins
            ax.plot(binned_times, binned_flux, "ro", ms=4,
                    label=f"NBin = {nbin_tot:.1f} s, std = {std_binned:.2%}")

        ax.set_ylabel("Relative Flux", fontsize=13)
        ax.legend(fontsize=8.0, loc=(0.0, 1.0), ncol=3, framealpha=1.0, frameon=False)

        # Plot the ingress, mid-transit, and egress times.
        if self.transit_times:
            ax.axvline(ingress, c="k", ls="--", alpha=0.5)
            ax.axvline(mid_transit, c="k", ls="--", alpha=0.5)
            ax.axvline(egress, c="k", ls="--", alpha=0.5)

        ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))

        plt.xlabel(f"{x_label}", fontsize=13)
        plt.xticks(rotation=30, size=8.0)

        fig.subplots_adjust(hspace=0.2)
        fig.savefig(lightcurve_name, dpi=300)

        print(f"-------->\tThe light curve of {star_name} was plotted\n")
