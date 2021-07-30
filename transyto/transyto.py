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

from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.time import Time
from astropy.nddata import NDData
from astropy.table import Table
from astropy.visualization import simple_norm
from astropy.stats import sigma_clip, sigma_clipped_stats

from transitleastsquares import transitleastsquares, cleaned_array
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
from transyto.utils import (
    search_files_across_directories,
    catalog
)
from transyto.utils.data import (
    get_data,
    get_header,
    get_value
)
from transyto.noise import (
    compute_scintillation,
    compute_noises
)

__all__ = ['TimeSeriesData', 'LightCurve']


warnings.simplefilter('ignore', category=AstropyWarning)


class TimeSeriesData:
    """Photometry Class"""

    def __init__(self, star_id, data_directory, search_pattern, list_reference_stars,
                 aperture_radius, from_coordinates=None, ra_target=None, dec_target=None,
                 ra_ref_stars=[], dec_ref_stars=[], telescope=""):
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
        from_coordinates : None, optional
            Flag to find star by using its coordinates.
        ra_target : None, optional
            RA coords of target.
        dec_target : None, optional
            DEC of target.
        ra_ref_stars : list, optional
            RA of ref stars.
        dec_ref_stars : list, optional
            DEC f ref. stars.
        telescope : str, optional
            Name of the telescope where the data come from.
        """

        # Positional Arguments
        self.star_id = star_id
        self.data_directory = data_directory
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

        # Centroid bow width for centroid function.
        self._box_width = 2. * (self.r + 1)

        pos_answers = ['True', 'true', 'yes', 'y', 'Yes', True]
        if from_coordinates in pos_answers:
            self._from_coordinates = True
        else:
            self._from_coordinates = False

        # Output directory for logs
        logs_dir = self.data_directory + "logs_photometry"
        os.makedirs(logs_dir, exist_ok=True)

        # Logger to track activity of the class
        self.logger = logging.getLogger(f"{self.pipeline} logger")
        self.logger.addHandler(logging.FileHandler(filename=os.path.join(logs_dir,
                                                                         'photometry.log'), mode='w'))
        self.logger.setLevel(logging.DEBUG)

        self.logger.info(pyfiglet.figlet_format(f"-*- {self.pipeline} -*-"))

        print(pyfiglet.figlet_format(f"-*- {self.pipeline} -*-\n"))
        print("{} will use {} reference stars for the differential photometry\n".
              format(self.pipeline, len(self.list_reference_stars)))

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
        exp, obstime, instr, readout, gain, airmass, altitude = kw_values

        Outputs = namedtuple("Outputs", "exp obstime instr readout gain airmass altitude")

        return Outputs(exp, obstime, instr, readout, gain, airmass, altitude)

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
                x_cen, y_cen = self._estimate_centroid_via_2dgaussian(data,
                                                                      mask)
            elif method == "1dgaussian":
                x_cen, y_cen = self._estimate_centroid_via_1dgaussian(data,
                                                                      mask)

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
        output_directory = self.data_directory + "ePSF"
        os.makedirs(output_directory, exist_ok=True)

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
                                           format(self.instrument, self.star_id))

            projection_average = np.sum(projections_list, axis=0) / len(projections_list)
            psf_half = (np.max(projection_average) + np.min(projection_average)) / 2
            pixs = np.linspace(-sl, sl, len(projection_average))

            peaks, _ = scipy.signal.find_peaks(projection_average)
            results_half = scipy.signal.peak_widths(projection_average, peaks, rel_height=0.5)

            idx_n, idx_p = -np.max(results_half[0]) / 4, np.max(results_half[0]) / 4

            fig = plt.figure(figsize=(6.5, 6.5))
            ax = fig.add_subplot(111)
            plt.title(f"PSF profile of {self.star_id} " r"($m_\mathrm{V}=10.0$)", fontsize=15)
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

    # def plot_cutouts(y, x, data_directory, search_pattern):
    #     """Make an effective psf and extract cutouts of a given star

    #     Parameters
    #     ----------
    #     star_id : string,
    #         Name of the star to find the positions
    #     data_directory : list
    #             list of files (frames) where we want to get the counts
    #     search_pattern : string
    #             pattern to search files
    #     cutouts_size : int, optional
    #         Number of pixels to extract cutouts of stars
    #     make_cutout_images : bool, optional
    #         Create new images with the star cutouts
    #     """
    #     for fn in fits_files[0:50]:
    #         ext = 0
    #         if fn.endswith(".fz"):
    #             ext = 1
    #         data, header = get_data(fn, header=True, ext=ext)

    #         x_pos = list()
    #         y_pos = list()

    #         wcs = WCS(header)
    #         if wcs.is_celestial:
    #             box_w = 40
    #             star_cutout = data[np.int(x) - box_w:np.int(x) + box_w,
    #                                np.int(y) - box_w:np.int(y) + box_w]

    #         else:
    #             continue

    #         positions = Table()
    #         positions["x"] = x_pos
    #         positions["y"] = y_pos

    #         # Output directory for all the cutouts
    #         output_directory = data_directory + "Cutouts"
    #         os.makedirs(output_directory, exist_ok=True)

    #         filename, extension = splitext_(os.path.basename(fn))

    #         # Name of centroid image
    #         fig_cutouts_name = os.path.join(output_directory, "{}_{}_{}_centroid.png".
    #                                         format(header["INSTRUME"],
    #                                                star_id, filename))

    #         # Add subplot for normal star
    #         instrume = header["INSTRUME"]
    #         fig, ax = plt.subplots(figsize=(6.5, 6.5))
    #         ax.set_title(f"Huntsman {cam} Camera {instrume}\n"
    #                      f"Star {star_id} " r"($m_\mathrm{V}=10.9$)", fontsize=15)
    #         norm = simple_norm(star_cutout, "sqrt", percent=99.7)
    #         # ax0.imshow(star[0], norm=norm, origin="lower", cmap="viridis")
    #         ax.imshow(star_cutout, norm=norm, origin="lower", cmap="viridis")
    #         ax.scatter(x_cen, y_cen, c='k', marker='+', s=100)
    #         # ax.legend(loc="lower left", ncol=2, fontsize=13, framealpha=True)

    #         # Draw the apertures of object and background
    #         r_in = r + 10
    #         r_out = r + 20
    #         circ = Circle((x_cen, y_cen), r, alpha=0.7, facecolor="none",
    #                       edgecolor="k", lw=2.0, zorder=3)
    #         circ1 = Circle((x_cen, y_cen), r_in, alpha=0.7, facecolor="none",
    #                        edgecolor="r", ls="--", lw=2.0, zorder=3)
    #         circ2 = Circle((x_cen, y_cen), r_out, alpha=0.7, facecolor="none",
    #                        edgecolor="r", ls="--", lw=2.0, zorder=3)
    #         ax.text(x_cen, r_out - (r_in / 9.0), "Local Sky Background", color="w",
    #                 ha="center", fontsize=13.0, zorder=5)  # 8.7

    #         n, radii = 50, [r_in, r_out]
    #         theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
    #         xs = np.outer(radii, np.cos(theta))
    #         ys = np.outer(radii, np.sin(theta))

    #         # in order to have a closed area, the circles
    #         # should be traversed in opposite directions
    #         xs[1, :] = xs[1, ::-1]
    #         ys[1, :] = ys[1, ::-1]

    #         # ax = plt.subplot(111, aspect='equal')
    #         ax.fill(np.ravel(xs) + x_cen, np.ravel(ys) + y_cen, facecolor="gray",
    #                 alpha=0.6, zorder=4)
    #         ax.tick_params(axis="both", which="major", labelsize=15)
    #         ax.add_patch(circ)
    #         ax.add_patch(circ1)
    #         ax.add_patch(circ2)
    #         ax.set_xlabel("X Pixels", fontsize=15)
    #         ax.set_ylabel("Y Pixels", fontsize=15)
    #         ax.set_xlim((2.0, 75))
    #         ax.set_ylim((2.0, 75))

    #         fig.savefig(fig_cutouts_name)
    #         print("Cutouts saved for {}".format(os.path.basename(fn)))

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

        assert phot_table["aperture_sum_0"] > phot_table["object_bkg"]

        object_final_counts = phot_table["aperture_sum_0"] - object_background

        # For consistent outputs in table
        phot_table["object_bkg_subtracted"] = object_final_counts
        phot_table["object_bkg_subtracted"].info.format = "%.8g"

        self.logger.debug(phot_table)

        return (phot_table["object_bkg_subtracted"].item(),
                phot_table["object_bkg"].item())

    # @logged
    def do_photometry(self, star_id, data_directory, search_pattern,
                      ra_star=None, dec_star=None, make_effective_psf=False):
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

        for fn in tqdm(fits_files[:60]):
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
                counts_in_aperture, bkg_in_object = self.make_aperture(data, (y_cen, x_cen),
                                                                       radius=self.r,
                                                                       r_in=self.r_in,
                                                                       r_out=self.r_out)

                object_counts.append(counts_in_aperture)
                background_in_object.append(bkg_in_object)
                x_pos.append(center_yx[1])
                y_pos.append(center_yx[0])
                times.append(time)
                self.good_frames_list.append(fn)

                if make_effective_psf:
                    cutout = self._slice_data(data, center_yx, 2. * self.r_out)
                    masked_data = self._mask_data(cutout)
                    x_cen, y_cen = self._estimate_centroid_via_2dgaussian(cutout,
                                                                          mask=masked_data.mask)
                    positions = Table()
                    positions["x"] = [x_cen]
                    positions["y"] = [y_cen]

                    tables.append(positions)
                    nddatas.append(NDData(data=cutout))
            else:
                continue

        if make_effective_psf:
            print(f"Building effective PSF for target star {self.star_id}")
            self.make_effective_psf(nddatas, tables)

        return (object_counts, background_in_object,
                exptimes, x_pos, y_pos, times, airmasses)

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

        print(f"Starting aperture photometry for {self.star_id}\n")

        self.logger.debug(f"-------------- Aperture photometry of {self.star_id} ---------------\n")
        # Get flux of target star
        (target_flux,
         background_in_object,
         exptimes,
         x_pos_target,
         y_pos_target,
         times,
         airmasses) = self.do_photometry(self.star_id, self.data_directory, self.search_pattern,
                                         ra_star=self.ra_target, dec_star=self.dec_target,
                                         make_effective_psf=True)

        times = np.asarray(times)

        airmasses = np.asarray(airmasses)

        print(f"Finished aperture photometry on target star. {self.pipeline}"
              " will compute now the combined flux of the ensemble\n")

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

        if self._from_coordinates:
            list_ra_ref_stars = self.ra_ref_stars
            list_dec_ref_stars = self.dec_ref_stars

        else:
            list_ra_ref_stars = list([1]) * len(self.list_reference_stars)
            list_dec_ref_stars = list([1]) * len(self.list_reference_stars)

        for ref_star, ra_ref_star, dec_ref_star in zip(self.list_reference_stars,
                                                       list_ra_ref_stars,
                                                       list_dec_ref_stars):

            print(f"Starting aperture photometry on ref_star {ref_star}\n")

            self.logger.debug(f"Aperture photometry of {ref_star}\n")
            (refer_flux,
             background_in_ref_star,
             _, _, _, _, _) = self.do_photometry(ref_star, self.data_directory, self.search_pattern,
                                                 ra_ref_star, dec_ref_star,
                                                 make_effective_psf=False)
            reference_star_flux_sec.append(np.asarray(refer_flux) / exptimes)
            background_in_ref_star_sec.append(np.asarray(background_in_ref_star) / exptimes)
            print(f"Finished aperture photometry on ref_star {ref_star}\n")

        self.reference_star_flux_sec = np.asarray(reference_star_flux_sec)
        background_in_ref_star_sec = np.asarray(background_in_ref_star_sec)

        # sigma_squared_ref = (reference_star_flux_sec * exptimes
        #                      + background_in_ref_star_sec * exptimes
        #                      + readout_noise)

        # weights_ref_stars = 1.0 / sigma_squared_ref

        ref_flux_averaged = np.average(self.reference_star_flux_sec * exptimes, axis=0)  # weights=weights_ref_stars, axis=0)

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
        print(f"Differential photometry of {self.star_id} has been finished, "
              f"with {len(self.good_frames_list)} frames "
              f"of camera {self.instrument} (run time: {exec_time:.3f} sec)\n")

        # Output directory
        self.output_dir_name = "TimeSeries_Analysis"
        output_directory = self.data_directory + self.output_dir_name
        os.makedirs(output_directory, exist_ok=True)

        if save_rms:
            # Output directory for files that contain photometric precisions
            output_directory = output_directory + "/rms_precisions"
            os.makedirs(output_directory, exist_ok=True)

            # File with rms information
            file_rms_name = os.path.join(output_directory,
                                         f"rms_{self.instrument}.txt")

            with open(file_rms_name, "a") as file:
                file.write(f"{self.r} {self.std} {self.std_binned} "
                           f"{np.nanmedian(S_to_N_obj)} {np.nanmedian(S_to_N_ref)} "
                           f"{np.nanmedian(S_to_N_diff)}\n")

        return (times, differential_flux, self.sigma_total)


class LightCurve(TimeSeriesData):
    def __init__(self, star_id, data_directory, search_pattern, list_reference_stars,
                 aperture_radius, from_coordinates=True, ra_target=None, dec_target=None,
                 ra_ref_stars=None, dec_ref_stars=None, telescope=""):
        super(LightCurve, self).__init__(star_id=star_id, data_directory=data_directory,
                                         search_pattern=search_pattern,
                                         list_reference_stars=list_reference_stars,
                                         aperture_radius=aperture_radius,
                                         from_coordinates=from_coordinates,
                                         ra_target=ra_target,
                                         dec_target=dec_target,
                                         ra_ref_stars=ra_ref_stars,
                                         dec_ref_stars=dec_ref_stars,
                                         telescope=telescope)

    def clip_outliers(self, time, flux, error, sigma_upper, return_mask=False, **kwargs):
        """ Covenience wrapper for sigma_clip function from astropy.
        """

        clipped_data = sigma_clip(data=flux, sigma_upper=sigma_upper, sigma_lower=float('inf'),
                                  maxiters=10, cenfunc=np.median, masked=True, copy=True)

        mask = clipped_data.mask
        normalized_flux_clipped = flux[~mask]
        times_clipped = time[~mask]
        errors_clipped = error[~mask]

        if return_mask:
            return normalized_flux_clipped, times_clipped, errors_clipped, mask
        return times_clipped, normalized_flux_clipped, errors_clipped

    def detrend_data(self, time, flux, R_star=None, M_star=None, Porb=None):
        """Detrend time-series data

        Parameters
        ----------
        time : array
            Times of the observation
        flux : array
            Flux with trend to be removed
        R_star : float
            Radius of the star (in solar units)
        M_star : float
            Mass of the star (in solar units)
        Porb : float, optional
            Orbital period of the planet (in days).

        Returns
        -------
        detrended and trended flux : numpy array
        """

        trend = scipy.signal.medfilt(flux, 15)
        detrended_flux = flux / trend

        if Porb is not None:
            self.logger.info("Now detrending the time series using queried "
                             + f"M_s = {M_star} M_sun, R_s = {R_star} R_sun, and "
                             + f"P_orb = {Porb} d found from previous model\n")

            # Compute the transit duration
            transit_dur = t14(R_s=R_star, M_s=M_star,
                              P=Porb, small_planet=False)

            # Estimate the window length for the detrending
            wl = 3.0 * transit_dur

            # Detrend the time series data
            detrended_flux, _ = flatten(time, flux, return_trend=True, method="biweight",
                                        window_length=wl)

        return detrended_flux

    def bin_timeseries(self, times, flux, bins):
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

        return binned_times, binned_flux

    def model_lightcurve(self, time, flux):
        """Summary
        """

        model = transitleastsquares(time, flux)
        results = model.power()

        print("Starting model of light curve...\n")
        print(f"Period: {results.period:.5f} d")
        print(f"{len(results.transit_times)} transit times in time series: "
              f'{[f"{i:0.5f}" for i in results.transit_times]}')
        print(f"Transit depth: {results.depth:.5f}")
        print(f"Best duration (days): {results.duration: .5f}")
        print(f"Signal detection efficiency (SDE): {results.SDE}\n")

        print("Finished model of light curve. Plotting model...")

        return results

    # @logged
    def plot(self, time, flux, std_errors, sigma=3, bins=30, detrend=False, plot_tracking=False,
             plot_noise_sources=False, model_transit=False, Porb=None):
        """Plot a light curve using the flux time series

        Parameters
        ----------
        time : list or array,
            List of times of observations
        flux : list or array,
            List of target flux
        std_errors : TYPE
            List of data errors
        sigma : float, optional
            Number of sigmas to clip outliers from time series data
        bins : int, optional
            Description
        detrend : bool, optional (default is False)
            If True, detrending of the time series data will be performed
        plot_tracking : bool, optional
            Flag to plot the (x, y) position of target (i.e. tracking) of all
            the observation frames.
        plot_noise_name : bool, optional
            Flag to plot the noise sources throughout the night.
        Porb : float, optional
            Orbital period of the planet (in days).

        No Longer Returned
        ------------------
        """

        # Output directory for lightcurves
        lightcurves_directory = self.data_directory + self.output_dir_name

        pd.plotting.register_matplotlib_converters()

        star_data = catalog.StarData(self.star_id).query_from_mast()

        normalized_flux = flux / np.nanmedian(flux)

        # Make time an astropy object
        time_object = Time(time)

        # Barycentric Julian Date (BJD)
        time_jd = time_object.jd - 2457000.0

        if detrend:
            normalized_flux = self.detrend_data(time_jd, flux)  # , R_star=star_data["Rs"], M_star=star_data["Ms"], Porb=Porb)

        # Remove invalid values such as nan, inf, non, negative
        # time, flux = cleaned_array(time, flux)

        # Clip values outside sigma_upper
        time_jd, normalized_flux, std_error = self.clip_outliers(time_jd, normalized_flux,
                                                                 std_errors, sigma_upper=sigma)

        # Standard deviation in ppm for the observation
        std = np.nanstd(normalized_flux)

        # Light curve name
        lightcurve_name = os.path.join(lightcurves_directory, "Lightcurve_camera_"
                                       f"{self.instrument}_r{self.r}.png")

        fig, ax = plt.subplots(2, 1,
                               sharey="row", sharex="col", figsize=(8.5, 6.3))
        fig.suptitle(f"Differential Photometry\nTarget Star {self.star_id}, "
                     f"Aperture radius = {self.r} pixels, focus: {self.header['FOC-POS']} eu", fontsize=13)

        ax[1].plot(time_jd, normalized_flux, "k.", ms=3,
                   label=f"NBin = {self.exptime:.1f} s, std = {std:.2%}")

        fig.subplots_adjust(hspace=0.2)

        # Binned data and times
        if bins != 0:
            binned_times, binned_flux = self.bin_timeseries(time_jd, normalized_flux, bins)
            std_binned = np.nanstd(binned_flux)

            # Total time for binsize
            nbin_tot = self.exptime * bins
            ax[1].plot(binned_times, binned_flux, "ro", ms=4,
                       label=f"NBin = {nbin_tot:.1f} s, std = {std_binned:.2%}")

        ax[1].errorbar(time_jd, normalized_flux, yerr=std_error,
                       fmt="none", ecolor="k", elinewidth=0.8,
                       label=r"$\sigma_{\mathrm{tot}}=\sqrt{\sigma_{\mathrm{phot}}^{2} "
                       r"+ \sigma_{\mathrm{sky}}^{2} + \sigma_{\mathrm{scint}}^{2} + "
                       r"\sigma_{\mathrm{read}}^{2}}$", capsize=0.0)

        ax[1].set_ylabel("Relative Flux", fontsize=13)
        ax[1].legend(fontsize=8.0, loc=(0.0, 1.0), ncol=3, framealpha=1.0, frameon=False)

        # ax[1].axvline(2278.0618, c="k", ls="--", alpha=0.5)
        # ax[1].axvline(2278.1241, c="k", ls="--", alpha=0.5)
        # ax[1].axvline(2278.1864, c="k", ls="--", alpha=0.5)

        ax[1].xaxis.set_major_formatter(plticker.FormatStrFormatter('%.3f'))
        ax[1].yaxis.set_major_formatter(plticker.FormatStrFormatter('%.3f'))

        for counter in range(len(self.list_reference_stars)):
            # ax[1].xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))

            # Colors for comparison stars
            # colors = ["blue", "magenta", "green", "cyan", "firebrick"]

            ax[0].plot(time_object.jd - 2457000.0, self.reference_star_flux_sec[counter]
                       / np.nanmean(self.reference_star_flux_sec[counter]),
                       "o", ms=1.3, label=f"Ref. star {self.list_reference_stars[counter]}")
            ax[0].set_ylabel("Normalized Flux", fontsize=13)
            # ax[0].set_ylim((0.9, 1.05))
            ax[0].legend(fontsize=8.1, loc="lower left",
                         ncol=len(self.list_reference_stars),
                         framealpha=1.0, frameon=True)

        ax[1].text(0.97, 0.9, "b)", fontsize=11, transform=ax[1].transAxes)
        ax[0].text(0.97, 0.9, "a)", fontsize=11, transform=ax[0].transAxes)

        plt.xlabel("Time [BJD-2457000.0]", fontsize=13)
        plt.xticks(rotation=30, size=8.0)
        plt.savefig(lightcurve_name)

        print(f"The light curve of {self.star_id} was plotted")

        if model_transit:
            results = self.model_lightcurve(time_jd)

            # Folded light curve name
            model_lightcurve_name = os.path.join(lightcurves_directory, "model_lightcurve_camera_"
                                                 f"{self.instrument}_r{self.r}.png")

            loc = plticker.MultipleLocator(base=5)  # this locator puts ticks at regular intervals

            fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
            ax.plot(results.model_folded_phase, results.model_folded_model, color='red')
            ax.scatter(results.folded_phase, results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
            ax.set_xlim(0.48, 0.52)
            ax.set_xlabel('Phase')
            ax.set_ylabel('Relative flux')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.1f'))

            fig.savefig(model_lightcurve_name)

            print(f"Folded model of the light curve of {self.star_id} was plotted\n")

        if plot_tracking:
            plot_tracking_name = os.path.join(lightcurves_directory, "tracking_plot_"
                                              f"{self.instrument}_r{self.r}.png")

            fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.3))
            ax.plot(time_object.jd - 2457000.0, self.x_pos_target, "ro-",
                    label="dx [Dec axis]", lw=0.5, ms=1.2)
            ax.plot(time_object.jd - 2457000.0, self.y_pos_target, "go-",
                    label="dy [RA axis]", lw=0.5, ms=1.2)
            ax.set_ylabel(r"$\Delta$ Pixel", fontsize=13)
            ax.legend(fontsize=8.6, loc="lower left", ncol=2, framealpha=1.0)
            ax.set_title(f"Tracking of camera {self.instrument}", fontsize=13)
            # ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.3f'))
            ax.set_xlabel("Time [BJD-2457000.0]", fontsize=13)
            plt.xticks(rotation=30, size=8.0)
            plt.grid(alpha=0.4)
            fig.savefig(plot_tracking_name)

        if plot_noise_sources:
            plot_noise_name = os.path.join(lightcurves_directory, "noises_plot_"
                                           f"{self.instrument}_r{self.r}.png")

            fig, ax = plt.subplots(1, 1, sharey="row", sharex="col", figsize=(8.5, 6.3))
            ax.set_title(f"Noise Sources in {self.star_id} " r"($m_\mathrm{V}=10.0$)", fontsize=13)
            ax.plot_date(time_object.plot_date, self.sigma_total * 100, "k-",
                         label=r"$\sigma_{\mathrm{total}}$")
            ax.plot_date(time_object.plot_date, self.sigma_scint * 100,
                         "g-", label=r"$\sigma_{\mathrm{scint}}$")
            ax.plot_date(time_object.plot_date, self.sigma_phot * 100, color="firebrick", marker=None,
                         ls="-", label=r"$\sigma_{\mathrm{phot}}$")
            ax.plot_date(time_object.plot_date, self.sigma_sky * 100,
                         "b-", label=r"$\sigma_{\mathrm{sky}}$")
            ax.plot_date(time_object.plot_date, self.sigma_ron * 100,
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
            fig.savefig(plot_noise_name)
