#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Work with time series data"""

from __future__ import division
import os
import warnings
import logging

import pandas as pd
import numpy as np
import time
import pyfiglet
import scipy
import matplotlib.ticker as plticker
import seaborn as sns

from astroquery.simbad import Simbad

from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.time import Time
from barycorrpy import utc_tdb
from astropy.nddata import NDData
from astropy.table import Table
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats
from astropy.stats import SigmaClip
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.modeling import models, fitting

#from transitleastsquares import transitleastsquares
from collections import namedtuple
from tqdm import tqdm
from pathlib import Path
from operator import itemgetter
from wotan import flatten, t14
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib import dates

from photutils.detection import DAOStarFinder
from photutils.aperture.circle import CircularAperture, CircularAnnulus
from photutils.aperture import ApertureStats
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

    def __init__(self, target_star='', data_directory='', search_pattern='*fit*',
                 from_coordinates=None, ra_target=None, dec_target=None,
                 transit_times=[], telescope='', centroid_box=30):
        """Initialize class Photometry for a given target and reference stars.

        Parameters
        ----------
        star_id : str
            Name of target star to do aperture photometry
        data_directory : str
            Top level path of .fits files to search for stars.
        search_pattern : str
            Pattern for searching files
        from_coordinates : None, optional
            Flag to find star by using its coordinates.
        ra_target : None, optional
            RA coords of target.
        dec_target : None, optional
            DEC of target.
        transit_times : list, optional
            Ingress, mid-transit, and egress time. In "isot" format.
            Example: ["2021-07-27T15:42:00", "2021-07-27T17:06:00", "2021-07-27T18:29:00"]
        telescope : str, optional
            Name of the telescope where the data come from.
        centroid_box : int, optional
            Initial box to calculate centroid.
        """

        # Name of target star and number of reference stars to be used.
        self.target_star_id = target_star

        # Data directory and search pattern for files.
        self._data_directory = data_directory
        self._search_pattern = search_pattern

        # List of files to be used by transyto to perform differential photometry.
        self.fits_files = search_files_across_directories(self._data_directory,
                                                          self._search_pattern)

        # Output directory for light curves
        if self._data_directory:
            self._output_directory = Path(self._data_directory, 'Light_Curve_Analysis')
            os.makedirs(self._output_directory, exist_ok=True)

        # Name of the telescope with which data was collected.
        self.telescope = telescope

        # RADEC of target and ref. stars if needed.
        self.ra_target = ra_target
        self.dec_target = dec_target

        # Transit times of target star: ingress, mid-transit, and egress time.
        self.transit_times = transit_times

        # Set possible positive answers to set some variables below.
        pos_answers = ['True', 'true', 'yes', 'y', 'Yes', True]
        if from_coordinates in pos_answers:
            self._from_coordinates = True
        else:
            self._from_coordinates = False

        # Initial box to slice data and calculate centroid
        self._initial_centroid_box = centroid_box

        # Output directory for logs
        if self._data_directory:
            logs_dir = Path(self._data_directory, 'logs_photometry')
            os.makedirs(logs_dir, exist_ok=True)

            # Logger to track activity of the class
            self.logger = logging.getLogger(f'{self.pipeline} logger')
            self.logger.addHandler(logging.FileHandler(filename=Path(logs_dir, 'photometry.log'),
                                                       mode='w'))
            self.logger.setLevel(logging.DEBUG)

            self.logger.info(pyfiglet.figlet_format(f'-*- {self.pipeline} -*-'))

    @classmethod
    def __get_class_name(cls):
        return cls.__name__

    @classmethod
    def __getattr__(self, name):
        return f'{self.__get_class_name()} does not have "{str(name)}" attribute.'

    @property
    def pipeline(self):
        return os.path.basename(PACKAGEDIR)

    @property
    def readout(self):

        # if self.telescope == "Huntsman":
        #     return -0.070967 * self.detector_gain**2 + 0.652094 * self.detector_gain + 1.564342
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
    def detector_gain(self):
        return self.get_keyword_value().gain

    @property
    def airmass(self):
        return self.get_keyword_value().airmass

    @property
    def keyword_list(self):
        file = os.path.join(str(Path(__file__).parents[1]), 'telescope_keywords.csv')

        (Huntsman,
         MQ,
         TESS,
         WASP,
         MEARTH,
         POCS) = np.loadtxt(file, skiprows=1, delimiter=';', dtype=str,
                            usecols=(0, 1, 2, 3, 4, 5), unpack=True)

        if self.telescope in ['Huntsman', 'MQ', 'TESS', 'WASP', 'MEARTH', 'POCS']:
            kw_list = eval(self.telescope)

        return kw_list

    def get_keyword_value(self, default=None):
        """Returns a header keyword value.

        If the keyword is Undefined or does not exist,
        then return ``default`` instead.
        """

        try:
            kw_values = itemgetter(*self.keyword_list)(self.header)
        except AttributeError:
            self.logger.error('Header keyword does not exist')
            return default
        exp, obstime, instr, readout, gain, airmass, altitude, latitude, longitude = kw_values

        Outputs = namedtuple('Outputs',
                             'exp obstime instr readout gain airmass altitude latitude longitude')

        return Outputs(exp, obstime, instr, readout, gain, airmass, altitude, latitude, longitude)

    def _slice_data(self, data, origin, width):
        y, x = origin
        cutout = data[int(x - width / 2.):int(x + width / 2.),
                      int(y - width / 2.):int(y + width / 2.)]
        return cutout

    def _mask_noise(self, data, noise_mean, noise_std, threshold_sigma=3.0):
        mask = data < (noise_mean + threshold_sigma * noise_std)

        return mask

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

    def _find_target_star(self):
        """Find the target star
        """
        # Get coordinates of target star as given by user (if any).
        if self._from_coordinates:
            self.target_star_coord = SkyCoord(self.ra_target, self.dec_target, frame='icrs')
        # Get coordinates of target star using its name.
        else:
            while True:
                try:
                    self.target_star_coord = SkyCoord.from_name(self.target_star_id)
                except NameResolveError:
                    self.target_star_id = input(f'{9 * " "}\tName syntax is incorrect, please use a'
                                                ' different name syntax for the target star: ')
                    continue
                break

        # Convert target star coordinates to 'hmsdms' format.
        self.target_star_coord = self.target_star_coord.to_string('hmsdms')

        # Split coordinates string to get RA and DEC by separate.
        self.target_star_coord = self.target_star_coord.split(' ')

        # Get RA and DEC from list
        self.target_star_coord_ra = self.target_star_coord[0]
        self.target_star_coord_dec = self.target_star_coord[1]

        print(f'\n{9 * " "}\tThe target star was found, {self.pipeline} will proceed'
              ' with the photometry:\n')

    def _find_ref_stars_coordinates(self):
        """Get all data from plate-solved images (right ascention,
           declination, airmass, dates, etc). Then, it converts the
           right ascention and declination into image positions to
           call make_aperture and find its total counts.

        """

        # Get data, header, and WCS of first frame (fits file)
        fn = self.fits_files[0]
        data = get_data(fn)
        header = get_header(fn)
        wcs = WCS(header)
        # Check if WCS exist in image
        if wcs.is_celestial:

            # Build SkyCoord object for target star.
            target_star = SkyCoord(self.target_star_coord_ra,
                                   self.target_star_coord_dec, frame='icrs')

            # Target star pixel positions in the image
            center_yx = wcs.all_world2pix(target_star.ra, target_star.dec, 0)

            # Slice the data to mask the target from DAOSTAR algorithm
            cutout = self._slice_data(data, center_yx, self._centroid_box)

            # Set sigma clipping algorithm
            sigclip = SigmaClip(sigma=3.0, maxiters=50)
            # Find noise local level of the target star.
            noise_level = CircularAnnulus(center_yx, r_in=20, r_out=40)
            # Get the statistics of the annulus (bakgroound) aperture.
            bkg_stats = ApertureStats(data, noise_level, sigma_clip=sigclip)
            noise_median, noise_std = bkg_stats.median, bkg_stats.std

            # Create the mask using the statistics from the aperture.
            cutout_mask = self._mask_noise(cutout, noise_median, noise_std)

            # Calculate the centroid of the target star (and use the mask to remove noise level).
            x_cen, y_cen = self._find_centroid(center_yx, cutout, cutout_mask, method='2dgaussian')

            # Create the mask that will be used to remove the target from DAOSTAR algorithm.
            target_mask = np.zeros(data.shape, dtype=bool)
            target_mask[int(x_cen - self._centroid_box / 2.):
                        int(x_cen + self._centroid_box / 2.),
                        int(y_cen - self._centroid_box / 2.):
                        int(y_cen + self._centroid_box / 2.)] = True

            # We clipped and clean the background to leave the stars only
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            # Set the DAOStarFinder algorithm
            daofind = DAOStarFinder(fwhm=5.0, threshold=5 * std)
            # Find the target star only.
            target = daofind(data - median, mask=~target_mask)
            # Get the magnitude of the target star.
            target_magnitude = target['mag'][0]

            # Change format in columns of target star daofind.
            for col in target.colnames:
                target[col].info.format = '%.8g'  # for consistent table output

            # Find the reference stars (avoiding the target star).
            ref_stars = daofind(data - median, mask=target_mask)
            for col in ref_stars.colnames:
                ref_stars[col].info.format = '%.8g'  # for consistent table output

            # Now we filter the reference stars that have similar magnitude to the target
            ref_stars = ref_stars.to_pandas()
            ref_stars['resid'] = ref_stars['mag'].sub(target_magnitude).abs()
            filtered_ref_stars = ref_stars.loc[ref_stars['resid'] <= 0.2]

            self.logger.info(filtered_ref_stars)

            target_star_position = np.transpose((target['xcentroid'], target['ycentroid']))

            ref_stars_positions = np.transpose((filtered_ref_stars['xcentroid'],
                                                filtered_ref_stars['ycentroid']))

            # Use only the requested number of reference stars.
            if len(ref_stars_positions) == 0:
                print(f'{8 * "-"}>\t{self.pipeline} did not found any suitable reference stars')

            elif len(ref_stars_positions) == 1:
                print(f'{8 * "-"}>\t{self.pipeline} found a single suitable reference star...')

            else:
                print(f'{8 * "-"}>\t{self.pipeline} found {len(ref_stars_positions)} suitable reference stars...')

            # Use (x, y) positions to create lists of ras and decs for the chosen reference stars.
            ref_stars_ra_list = list()
            ref_stars_dec_list = list()
            for ref_star_pos in ref_stars_positions:
                rx, ry = ref_star_pos
                sky = wcs.pixel_to_world(rx, ry).to_string('hmsdms')
                sky = sky.split(' ')
                ref_stars_ra_list.append(sky[0])
                ref_stars_dec_list.append(sky[1])

            self.ref_stars_coordinates_list = pd.DataFrame([ref_stars_ra_list, ref_stars_dec_list])
            self.ref_stars_coordinates_list = self.ref_stars_coordinates_list.transpose()
            self.ref_stars_coordinates_list.columns = ['RA', 'DEC']
            self.ref_stars_coordinates_list.index += 1

            print(self.ref_stars_coordinates_list)

            time.sleep(2)

            output_directory = os.path.join(self._data_directory, 'Photometry_field_stars')
            os.makedirs(output_directory, exist_ok=True)

            # Add subplot for fitted psf star
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # Plot the target and available reference stars in the field just as a reference.
            phot_available_ref_stars_name = os.path.join(output_directory, 'photometry_available_ref_stars.png')

            ref_apertures = CircularAperture(ref_stars_positions, r=4.)
            target_aperture = CircularAperture(target_star_position, r=4.)
            norm = ImageNormalize(stretch=SqrtStretch())
            ax.imshow(data, cmap='Greys', origin='lower',
                      norm=norm, interpolation='nearest')
            target_patches = target_aperture.plot(color='red', lw=1.5, alpha=0.5,
                                                  label=f'Target: {self.target_star_id}')
            ref_patches = ref_apertures.plot(color='blue', lw=1.5, alpha=0.5,
                                             label='Available reference stars')

            handles = (target_patches[0], ref_patches[0])

            plt.legend(ncol=2, loc='upper center', fontsize=8.4, bbox_to_anchor=(0.5, 1.1),
                       fancybox=True, frameon=True, handles=handles, prop={'weight': 'bold'})

            plt.grid(alpha=0.4)
            ax.set_xlabel('X pixels', fontsize=9)
            ax.set_ylabel('Y pixels', fontsize=9)

            # Get the (x, y) coordinates of each reference star to visually check them.
            for i, xy_pos in zip(self.ref_stars_coordinates_list.index.values, ref_stars_positions):
                ax.annotate(f"{i}", xy_pos)

            plt.show(block=False)
            fig.savefig(phot_available_ref_stars_name, dpi=300)

            # Ask what reference stars are going to be used (no variable, high proper motion, etc.)
            index_list = [int(index) for index in
                          input(f'{8 * " "}\tList of reference stars to be used (separate with space): ').split()]
            self.ref_stars_coordinates_list = self.ref_stars_coordinates_list.loc[index_list]

            # Remove (x, y) positions of reference stars not used, to plot only the ones selected.
            new_ref_stars_positions = list()
            for i in range(len(ref_stars_positions)):
                if i in (np.array(index_list) - 1):
                    new_ref_stars_positions.append(ref_stars_positions[i])
                else:
                    continue
            new_ref_stars_positions = np.array(new_ref_stars_positions)

            # Add subplot for fitted psf star
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # Plot the target and selected reference stars in the field just as a reference.
            phot_selected_ref_stars_name = Path(output_directory,
                                                'photometry_selected_ref_stars.png')

            ref_apertures = CircularAperture(new_ref_stars_positions, r=4.)
            target_aperture = CircularAperture(target_star_position, r=4.)
            norm = ImageNormalize(stretch=SqrtStretch())
            ax.imshow(data, cmap='Greys', origin='lower',
                      norm=norm, interpolation='nearest')
            target_patches = target_aperture.plot(color='red', lw=1.5, alpha=0.5,
                                                  label=f'Target: {self.target_star_id}')
            ref_patches = ref_apertures.plot(color='blue', lw=1.5, alpha=0.5,
                                             label='Selected reference stars')

            handles = (target_patches[0], ref_patches[0])

            for i, xy_pos in zip(self.ref_stars_coordinates_list.index.values,
                                 new_ref_stars_positions):
                ax.annotate(f"{i}", xy_pos)

            plt.legend(ncol=2, loc='upper center', fontsize=8.4, bbox_to_anchor=(0.5, 1.1),
                       fancybox=True, frameon=True, handles=handles, prop={'weight': 'bold'})
            plt.grid(alpha=0.4)
            ax.set_xlabel('X pixels', fontsize=9)
            ax.set_ylabel('Y pixels', fontsize=9)
            fig.savefig(phot_selected_ref_stars_name, dpi=300)
        else:
            print(f"{16 * ' '}• Not possible to find reference stars, first frame has no WCS\n")

    def _find_centroid(self, prior_centroid, data, mask, method='2dgaussian'):

        prior_y, prior_x = prior_centroid
        with warnings.catch_warnings():
            # Ignore warning for the centroid_2dg function
            warnings.simplefilter('ignore', category=UserWarning)

            if method == '2dgaussian':
                x_cen, y_cen = self._estimate_centroid_via_2dgaussian(data, mask)
            elif method == '1dgaussian':
                x_cen, y_cen = self._estimate_centroid_via_1dgaussian(data, mask)
            elif method == 'moments':
                x_cen, y_cen = self._estimate_centroid_via_moments(data, mask)

            # Compute the shifts in y and x.
            shift_y = self._centroid_box / 2 - y_cen
            shift_x = self._centroid_box / 2 - x_cen

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

    def make_effective_psf(self, nddatas, tables, plot_psf_profile=False):
        # Extract stars from all the frames
        stars = extract_stars(nddatas, tables, size=self.r_out)

        # Build the ePSF from all the cutouts extracted
        epsf_builder = EPSFBuilder(oversampling=1., maxiters=15, progress_bar=True,
                                   recentering_boxsize=self.r_out)
        epsf, fitted_star = epsf_builder(stars)

        masked_eff_psf = self._mask_data(epsf.data)

        x_cen, y_cen = self._estimate_centroid_via_2dgaussian(epsf.data, mask=masked_eff_psf.mask)

        # Output directory for ePSF
        output_directory = self._data_directory + 'ePSF'
        os.makedirs(self._data_directory + 'ePSF', exist_ok=True)

        epsf_name = Path(output_directory, 'ePSF.png')

        # Add subplot for fitted psf star
        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot(111)
        # ax.set_title(f"Huntsman {cam} Camera {instrume}\n"
        #             f"Star {star_id} " r"($m_\mathrm{V}=10.9$)", fontsize=15)
        norm = simple_norm(epsf.data, 'sqrt', percent=99.9)
        epsf_img = ax.imshow(epsf.data, norm=norm, cmap='viridis', origin='lower')
        ax.scatter(x_cen, y_cen, c='k', marker='+', s=100)
        # ax.legend(loc="lower left", ncol=2, fontsize=10, framealpha=True)

        # Draw the apertures of object and background
        circ = Circle((x_cen, y_cen), self.r, alpha=0.7, facecolor='none',
                      edgecolor='k', lw=2.0, zorder=3)
        circ1 = Circle((x_cen, y_cen), self.r_in, alpha=0.7, facecolor='none',
                       edgecolor="r", ls='--', lw=2.0, zorder=3)
        circ2 = Circle((x_cen, y_cen), self.r_out, alpha=0.7, facecolor='none',
                       edgecolor='r', ls='--', lw=2.0, zorder=3)
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
        ax.fill(np.ravel(xs) + x_cen, np.ravel(ys) + y_cen, facecolor='gray', alpha=0.6, zorder=4)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.add_patch(circ)
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        ax.set_xlabel('X Pixels', fontsize=15)
        ax.set_ylabel('Y Pixels', fontsize=15)
        ax.set_xlim((0.0, self.r_out))
        ax.set_ylim((0.0, self.r_out))

        # Colorbar for the whole figure and new axes for it
        # fig.colorbar(epsf_img, orientation="vertical")

        fig.savefig(epsf_name, dpi=300)

        projections_list = list()
        sl = self.r
        pr = 5
        for nd in nddatas:
            projection_x = nd.data[int((x_cen - sl)):int(2.3 * (x_cen + sl)),
                                   int(y_cen - pr):int(y_cen + pr)]
            projection_y = nd.data[int(x_cen - pr):int(x_cen + pr),
                                   int((y_cen - sl)):int(2.3 * (y_cen + sl))]
            projection_x = np.mean(projection_x, axis=1)
            projection_y = np.mean(projection_y, axis=0)

            projection_average = (projection_x + projection_y) / 2

            projections_list.append(np.asarray(projection_average))

        # Name of PSF profile image
        fig_name = Path(output_directory, f'{self.instrument}_{self.target_star}_profile.png')

        projection_average = np.sum(projections_list, axis=0) / len(projections_list)
        psf_half = (np.max(projection_average) + np.min(projection_average)) / 2
        pixs = np.linspace(-sl, sl, len(projection_average))

        peaks, _ = scipy.signal.find_peaks(projection_average)
        results_half = scipy.signal.peak_widths(projection_average, peaks, rel_height=0.5)

        idx_n, idx_p = -np.max(results_half[0]) / 4, np.max(results_half[0]) / 4

        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot(111)
        plt.title(f'PSF profile of {self.target_star} ' r'($m_\mathrm{V}=10.0$)', fontsize=15)
        ax.plot(pixs, projection_average, 'k-', ms=3)
        # ax.axhline(y=psf_half, xmin=0.36, xmax=0.67, c="r", ls="--", lw=1.5)
        ax.axvline(x=idx_n, c='b', ls='-.', lw=1.5)
        ax.axvline(x=idx_p, c='b', ls='-.', lw=1.5)
        ax.axvspan(idx_n, idx_p, facecolor='blue', alpha=0.15)
        ax.text(idx_p + 0.4, psf_half, rf'FWHM$\approx${np.max(results_half[0]) / 2:.3f} pix',
                color='k', fontsize=13)

        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel('Pixels', fontsize=15)
        ax.set_ylabel('Counts', fontsize=15)
        ax.grid(alpha=0.5)
        fig.savefig(fig_name, dpi=300)
        plt.close(fig)

    def save_star_cutout(self, star_id='', star_ra='', star_dec='', x=None, y=None,
                         cutout=None, num_frame=None, filename=''):
        """Save cutouts of a given star

        Parameters
        ----------
        star_id: str, optional
            Name of star to do the cutout.
        star_ra: str, optional
            RA of star.
        star_dec: str, optional
            DEC of star.
        x : float, optional
            x-position of centroid.
        y : float, optional
            y-position of centroid.
        cutout : array, optional
            Cutout of the image.
        filename : str, optional
            Name of file.
        """

        # Output directory for all the cutouts
        output_directory = Path(f'{self._data_directory}/Cutouts/{star_id}')
        os.makedirs(output_directory, exist_ok=True)

        if filename.endswith('.fz'):
            filename = filename.replace('.fz', '')
        filename = os.path.splitext(os.path.basename(filename))[0]

        # Name of centroid image
        fig_name = Path(output_directory, f'{self.instrument}_{star_id}_{filename}_centroid.png')

        # Add subplot for normal star
        fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(9, 5.5))
        fig.suptitle(f'Huntsman Camera {self.instrument} (frame {num_frame})\n'
                     f'Star {star_id}, RA: {star_ra}, DEC: {star_dec}', fontsize=15, y=0.995)
        ax.set_title('Photometric Data\n\n'
                     r'$r_\mathrm{inner\_aperture}$=1 x FWHM$_\mathrm{mean}$' '\n'
                     r'$r_\mathrm{inner\_annulus}$='
                     f"{self.r_in / self.r:.0f} x " r'$r_\mathrm{inner\_aperture}$' '\n'
                     r'$r_\mathrm{outer\_annulus}$='
                     f'{self.r_out / self.r:.0f} x ' r'$r_\mathrm{inner\_aperture}$')

        # Create norm for visualization
        norm = simple_norm(self.new_cutout, 'sqrt', percent=99.7)

        # Calculate the residuals
        residuals = self.new_cutout - self.psf_model(self.x_model, self.y_model)

        ax.imshow(cutout, origin='lower', cmap='viridis', norm=norm)
        ax.scatter(x, y, c='k', marker='+', s=100)

        # Draw the apertures of object and background
        circ = Circle((x, y), self.r, alpha=0.7, facecolor='none',
                      edgecolor='k', lw=2.0, zorder=3)
        circ1 = Circle((x, y), self.r_in, alpha=0.7, facecolor='none',
                       edgecolor='r', ls='--', lw=2.0, zorder=3)
        circ2 = Circle((x, y), self.r_out, alpha=0.7, facecolor='none',
                       edgecolor='r', ls='--', lw=2.0, zorder=3)

        n, radii = 50, [self.r_in, self.r_out]
        theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
        xs = np.outer(radii, np.cos(theta))
        ys = np.outer(radii, np.sin(theta))

        # in order to have a closed area, the circles should be traversed in opposite directions
        xs[1, :] = xs[1, ::-1]
        ys[1, :] = ys[1, ::-1]

        ax.fill(np.ravel(xs) + x, np.ravel(ys) + y, facecolor='gray',
                alpha=0.6, zorder=4)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.add_patch(circ)
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        ax.set_xlabel('X Pixels', fontsize=15)
        ax.set_ylabel('Y Pixels', fontsize=15)
        ax.set_xlim((0, 2.25 * self._centroid_box))
        ax.set_ylim((0, 2.25 * self._centroid_box))

        ax1.set_title(f'PSF Model: 2D Gaussian\n\n'
                      r'FWHM$_{x}$' f'={self.psf_model_x_fwhm:.2f} pixels' '\n'
                      r'FWHM$_{y}$' f'={self.psf_model_y_fwhm:.2f} pixels' '\n'
                      r'FWHM$_\mathrm{mean}$' f'={self.r:.2f} pixels')
        ax1.imshow(self.psf_model(self.x_model, self.y_model), origin="lower",
                   cmap='viridis', norm=norm)
        ax1.tick_params(axis='both', which='major', labelsize=15)
        ax1.yaxis.set_ticks_position('none')
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2.set_title('Residuals')
        ax2.imshow(residuals, origin='lower', cmap='viridis', norm=norm)
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax2.yaxis.set_ticks_position('none')
        ax2.set_xticks([])
        ax2.set_yticks([])

        fig.tight_layout()
        fig.savefig(fig_name)
        plt.close(fig)

    @staticmethod
    def make_aperture(data, coordinates, radius, r_in, r_out,
                      method='exact', subpixels=10):
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
        background_mask = background_apertures.to_mask(method='center')
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

        phot_table['object_bkg'] = object_background
        phot_table['object_bkg'].info.format = "%.8g"

        # assert phot_table["aperture_sum_0"] > phot_table["object_bkg"]

        object_final_counts = phot_table['aperture_sum_0'] - object_background

        # Replace negative values by NaN
        if object_final_counts < 0:
            phot_table['object_bkg_subtracted'] = np.nan
        else:
            phot_table['object_bkg_subtracted'] = object_final_counts

        # For consistent outputs in table
        phot_table['object_bkg_subtracted'].info.format = '%.8g'

        return (phot_table['object_bkg_subtracted'].item(), phot_table['object_bkg'].item(),
                phot_table)

    # @logged
    def do_photometry(self, make_effective_psf=False):
        """Get all data from plate-solved images (right ascention,
           declination, airmass, dates, etc). Then, it converts the
           right ascention and declination into image positions to
           call make_aperture and find its total counts.

        Parameters
        ----------
        star_id: str
            Name of star to be localized in each file

        Returns
        --------
        Counts of a star, list of good frames and airmass: tuple

        """

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

        fmt = '{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} frames | {elapsed}<{remaining}'
        for fn in tqdm(self.fits_files, desc=f'{18 * " " }Progress: ', bar_format=fmt):
            # Get data, header and WCS of fits files with any extension
            data = get_data(fn)
            self.header = get_header(fn)

            # Centroid box width for initial centroid function.
            self._centroid_box = self._initial_centroid_box

            wcs = WCS(self.header)
            # Check if WCS exist in image
            if wcs.is_celestial:

                # Build SkyCoord object for target star.
                target_star = SkyCoord(self.target_star_coord_ra,
                                       self.target_star_coord_dec, frame='icrs')

                # Get the initial  (x, y) positions of the target star in the image.
                center_yx = wcs.all_world2pix(target_star.ra, target_star.dec, 0)

                # Do a first cutout to refine the centroid of the target star.
                first_cutout = self._slice_data(data, center_yx, self._centroid_box)

                # Set the sigma clipping algorithm to mask pixels of the local noise level.
                sigclip = SigmaClip(sigma=3.0, maxiters=50)
                # Create the annulus aperture to get the local noise level.
                noise_level = CircularAnnulus(center_yx, r_in=20, r_out=40)
                # Get the statistics of the noise level.
                bkg_stats = ApertureStats(data, noise_level, sigma_clip=sigclip)
                noise_median, noise_std = bkg_stats.median, bkg_stats.std

                # Create the mask of the noise level pixes.
                first_cutout_mask = self._mask_noise(first_cutout, noise_median, noise_std)

                # Calculate the centroid of the star and return (x, y) in original data coordinates.
                x_cen, y_cen = self._find_centroid(center_yx, first_cutout, first_cutout_mask,
                                                   method='2dgaussian')

                # Create a new cutout around the new centroid to model the PSF of the target star.
                self.new_cutout = self._slice_data(data, (y_cen, x_cen), self._centroid_box)

                # Get the dimensions of the new cutout.
                yp, xp = self.new_cutout.shape

                # Generate grid of same size like box to put the fit on
                self.y_model, self.x_model, = np.mgrid[:yp, :xp]

                # Declare what function you want to fit to your data
                gaussian_2d = models.Gaussian2D()

                # Declare what fitting function you want to use
                fit_gauss_2d = fitting.LevMarLSQFitter()

                # Fit the model to your data (new_cutout)
                self.psf_model = fit_gauss_2d(gaussian_2d, self.x_model,
                                              self.y_model, self.new_cutout)

                # Get the FWHM in both x and y axes.
                self.psf_model_x_fwhm = self.psf_model.x_fwhm
                self.psf_model_y_fwhm = self.psf_model.y_fwhm

                # Calculate the radius of the aperture and the annulus for aperture photometry.
                self.r = 1. * (self.psf_model_x_fwhm + self.psf_model_y_fwhm) / 2.
                self.r_in = 3. * self.r
                self.r_out = 5. * self.r
                # Redefine the box_width to extract a cutout that can be used as a diagnosis plot.
                self._centroid_box = self.r_out + 0.5

                # Get the exposure time and airmass for the current frame.
                exptimes.append(self.exptime)
                airmasses.append(self.airmass)

                # Get the observation time for the current frame.
                time = self.obs_time

                # Calculate the sum of counts inside inner aperture and annulus for current frame.
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

                # If True, an effective instrumental PSF will be calculated using all the frames.
                if make_effective_psf:
                    cutout_psf = self._slice_data(data, center_yx, 2. * self.r_out)
                    masked_data = self._mask_data(cutout_psf)
                    x, y = self._estimate_centroid_via_2dgaussian(cutout_psf, mask=masked_data.mask)
                    positions = Table()
                    positions['x'] = [x]
                    positions['y'] = [y]

                    tables.append(positions)
                    nddatas.append(NDData(data=cutout_psf))

                # Make a bigger cutout to include both apertures: star and local background.
                cutout = self._slice_data(data, (y_cen, x_cen), 2.3 * self._centroid_box)

                # Find a rough center of the cutout.
                mid_point = self._centroid_box / 2.
                x_cen, y_cen = 2.3 * mid_point, 2.3 * mid_point

                # Filter and choose the pixels only in the middle region of the cutout.
                mask = np.zeros(cutout.shape, dtype=bool)
                mask[int(x_cen - mid_point):int(x_cen + mid_point),
                     int(y_cen - mid_point):int(y_cen + mid_point)] = True

                # Recalculate the centroid for the selected area to improve precision.
                new_x_cen, new_y_cen = self._estimate_centroid_via_2dgaussian(cutout, mask=~mask)
                num_frame = self.fits_files.index(fn) + 1

                # Save cutout
                self.save_star_cutout(star_id=self.target_star_id, star_ra=self.target_star_coord_ra,
                                      star_dec=self.target_star_coord_dec, x=new_x_cen, y=new_y_cen,
                                      cutout=cutout, num_frame=num_frame, filename=fn)

            else:
                continue

        if make_effective_psf:
            print(f'Building effective PSF for target star {self.target_star}')
            self.make_effective_psf(nddatas, tables)

        self.exptimes = np.asarray(exptimes)
        self.airmasses = np.asarray(airmasses)

        return (object_counts, background_in_object, x_pos, y_pos, times)

    # @logged
    def do_photometry_ref_stars(self, make_effective_psf=False):
        """Get all data from plate-solved images (right ascention,
           declination, airmass, dates, etc). Then, it converts the
           right ascention and declination into image positions to
           call make_aperture and find its total counts.

        """
        # Get the flux of each reference star
        ref_stars_flux_sec = list()
        ref_stars_background_sec = list()

        ref_stars_ra = self.ref_stars_coordinates_list['RA']
        ref_stars_dec = self.ref_stars_coordinates_list['DEC']

        for ref_star_ra, ref_star_dec in zip(ref_stars_ra, ref_stars_dec):

            df = self.ref_stars_coordinates_list
            ref_index = df[df['RA'] == ref_star_ra].index.values[0]
            self.logger.debug(f'Aperture photometry of reference star {ref_index}\n')
            print(f'{16 * " "}• Starting aperture photometry on reference star {ref_index}\n')

            tables = list()
            nddatas = list()

            star = SkyCoord(ref_star_ra, ref_star_dec, frame='icrs')

            # List of ADU counts for the source, background
            object_counts = list()
            background_in_object = list()

            fmt = '{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} frames | {elapsed}<{remaining}'
            for fn in tqdm(self.fits_files, desc=f'{18 * " " }Progress: ', bar_format=fmt):
                # Get data, header and WCS of fits files with any extension
                data = get_data(fn)
                self.header = get_header(fn)

                # Centroid box width for initial centroid function.
                self._centroid_box = self._initial_centroid_box

                wcs = WCS(self.header)

                # Star pixel positions in the image
                center_yx = wcs.all_world2pix(star.ra, star.dec, 0)

                # Do a first cutout to refine the centroid of each reference star.
                first_cutout = self._slice_data(data, center_yx, self._centroid_box)

                # Set the sigma clipping algorithm to mask pixels of the local noise level.
                sigclip = SigmaClip(sigma=3.0, maxiters=50)
                # Create the annulus aperture to get the local noise level.
                noise_level = CircularAnnulus(center_yx, r_in=20, r_out=40)
                # Get the statistics of the noise level.
                bkg_stats = ApertureStats(data, noise_level, sigma_clip=sigclip)
                noise_median, noise_std = bkg_stats.median, bkg_stats.std

                # Create the mask of the noise level pixes.
                first_cutout_mask = self._mask_noise(first_cutout, noise_median, noise_std)

                # Calculate the centroid of each reference star (removing the noise level pixels).
                x_cen, y_cen = self._find_centroid(center_yx, first_cutout, first_cutout_mask,
                                                   method='2dgaussian')

                # Create new cutout to calculate the FWHM by doing a 2D Gaussian fit to the PSF.
                self.new_cutout = self._slice_data(data, (y_cen, x_cen), self._centroid_box)

                yp, xp = self.new_cutout.shape

                # Generate grid of same size like box to put the fit on
                self.y_model, self.x_model, = np.mgrid[:yp, :xp]
                # Declare what function you want to fit to your data
                gaussian_2d = models.Gaussian2D()
                # Declare what fitting function you want to use
                fit_gauss_2d = fitting.LevMarLSQFitter()

                # Fit the model to your data (box)
                self.psf_model = fit_gauss_2d(gaussian_2d, self.x_model,
                                              self.y_model, self.new_cutout)

                self.psf_model_x_fwhm = self.psf_model.x_fwhm
                self.psf_model_y_fwhm = self.psf_model.y_fwhm

                self.r = 1. * (self.psf_model_x_fwhm + self.psf_model_y_fwhm) / 2.
                self.r_in = 3. * self.r
                self.r_out = 5. * self.r
                self._centroid_box = self.r_out + 0.5

                # Sum of counts inside aperture
                (counts_in_aperture,
                 bkg_in_object,
                 phot_table) = self.make_aperture(data, (y_cen, x_cen), radius=self.r,
                                                  r_in=self.r_in, r_out=self.r_out)

                self.logger.debug(phot_table)

                object_counts.append(counts_in_aperture)
                background_in_object.append(bkg_in_object)

                if make_effective_psf:
                    cutout_psf = self._slice_data(data, center_yx, 2. * self.r_out)
                    masked_data = self._mask_data(cutout_psf)
                    x, y = self._estimate_centroid_via_2dgaussian(cutout_psf, mask=masked_data.mask)
                    positions = Table()
                    positions['x'] = [x]
                    positions['y'] = [y]

                    tables.append(positions)
                    nddatas.append(NDData(data=cutout_psf))

                # Make a bigger cutout to include both apertures: star and local background.
                cutout = self._slice_data(data, (y_cen, x_cen), 2.3 * self._centroid_box)

                # Find a rough center of the cutout.
                mid_point = self._centroid_box / 2.
                x_cen, y_cen = 2.3 * mid_point, 2.3 * mid_point

                # Filter and choose the pixels only in the middle region of the cutout.
                mask = np.zeros(cutout.shape, dtype=bool)
                mask[int(x_cen - mid_point):int(x_cen + mid_point),
                     int(y_cen - mid_point):int(y_cen + mid_point)] = True

                # Recalculate the centroid for the selected area to improve precision.
                new_x_cen, new_y_cen = self._estimate_centroid_via_2dgaussian(cutout, mask=~mask)
                num_frame = self.fits_files.index(fn) + 1

                # Save cutout
                self.save_star_cutout(star_id=f'Ref_{ref_index}', x=new_x_cen, y=new_y_cen,
                                      cutout=cutout, num_frame=num_frame, filename=fn,
                                      star_ra=ref_star_ra, star_dec=ref_star_dec)

            ref_stars_flux_sec.append(np.asarray(object_counts) / self.exptimes)
            ref_stars_background_sec.append(np.asarray(background_in_object) / self.exptimes)

            print(f'\n{18 * " "}Finished aperture photometry on reference star {ref_index}\n')

            if make_effective_psf:
                print(f'Building effective PSF for target star {self.target_star}')
                self.make_effective_psf(nddatas, tables)

        self.ref_stars_flux_sec = np.asarray(ref_stars_flux_sec)

        self.ref_stars_background_sec = np.asarray(ref_stars_background_sec)

        sigma_squared_ref = (self.ref_stars_flux_sec * self.exptimes
                             + self.ref_stars_background_sec * self.exptimes
                             + (self.readout * self.r)**2 * np.pi / self.detector_gain
                             + self.airmasses)

        weights_ref_stars = 1.0 / sigma_squared_ref

        self.ref_stars_flux_averaged = np.average(self.ref_stars_flux_sec * self.exptimes,
                                                  weights=weights_ref_stars, axis=0)

        # Integrated flux per sec for ensemble of reference stars
        self.ref_stars_total_flux_sec = np.sum(self.ref_stars_flux_sec, axis=0)

        # Integrated sky background for ensemble of reference stars
        self.ref_stars_total_bkg_sec = np.sum(self.ref_stars_background_sec, axis=0)

        # S/N for reference star per second
        ref_stars_S_to_N_sec = self.ref_stars_total_flux_sec / np.sqrt(self.ref_stars_total_flux_sec
                                                                       + self.ref_stars_total_bkg_sec
                                                                       + (self.readout * self.r)**2 * np.pi
                                                                       / (self.detector_gain * self.exptimes))
        # Convert S/N per sec for ensemble to total S/N
        self.S_to_N_ref = ref_stars_S_to_N_sec * np.sqrt(self.detector_gain * self.exptimes)

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

        print(pyfiglet.figlet_format(f'-*- {self.pipeline} -*-')
              + f'{16 * "#"}       by Jaime Andrés Alvarado Montes       {16 * "#"}\n')

        print(pyfiglet.figlet_format(f'1. Time Series')
              + '        Part of transyto package by Jaime A. Alvarado-Montes\n')

        print(f'{8 * "-"}>\tStarting aperture photometry on target star {self.target_star_id}:\n')

        self._find_target_star()

        self.logger.debug(f'-------------- Aperture photometry of {self.target_star_id} ---------------\n')
        # Get flux of target star
        (target_flux,
         background_in_object,
         x_pos_target,
         y_pos_target,
         times) = self.do_photometry(make_effective_psf=False)

        print(f'\n{18 * " "}Finished aperture photometry on target star {self.target_star_id}\n')

        # Get the date times anc compute the Barycentric Julian Date (Barycentric Dynamical Time)
        times = np.asarray(times)
        self.jdutc_times = Time(times, format='isot', scale='utc')
        bjdtdb_times = utc_tdb.JDUTC_to_BJDTDB(self.jdutc_times, hip_id=8102,
                                               lat=self.telescope_latitude,
                                               longi=self.telescope_longitude,
                                               alt=self.telescope_altitude)

        self.time_norm_factor = 2450000.
        times = bjdtdb_times[0] - self.time_norm_factor

        # Positions of target star
        self.x_pos_target = np.array(x_pos_target) - np.nanmean(x_pos_target)
        self.y_pos_target = np.array(y_pos_target) - np.nanmean(y_pos_target)

        # Target and background counts per second
        target_flux = np.asarray(target_flux)
        target_flux_sec = target_flux / self.exptimes
        target_background_sec = np.asarray(background_in_object) / self.exptimes

        noise_sources = compute_noises(self.detector_gain, self.exptimes, target_flux_sec,
                                       target_background_sec, self.readout, self.r)

        # Sigma photon noise
        self.sigma_phot = noise_sources.sigma_photon

        # Sigma sky-background noise
        self.sigma_sky = noise_sources.sigma_sky

        # Sigma readout noise
        self.sigma_ron = noise_sources.sigma_readout

        # Sigma scintillation
        self.sigma_scint = compute_scintillation(0.143, self.telescope_altitude,
                                                 self.airmasses, self.exptimes)

        # Total photometric error for 1 mag in one observation
        self.sigma_total = np.sqrt(self.sigma_phot**2.0 + self.sigma_ron**2.0
                                   + self.sigma_sky**2.0 + self.sigma_scint**2.0)

        # Signal to noise: shot, sky noise (per second) and readout
        S_to_N_obj_sec = target_flux_sec / np.sqrt(target_flux_sec + target_background_sec
                                                   + (self.readout * self.r)**2 * np.pi
                                                   / (self.detector_gain * self.exptimes))
        # Convert SN_sec to actual SN
        S_to_N_obj = S_to_N_obj_sec * np.sqrt(self.detector_gain * self.exptimes)

        # Find reference stars and do photometry of the ensemble.
        print(f'\n{8 * "-"}>\t{self.pipeline} will find suitable reference stars for '
              'differential photometry \n')
        self._find_ref_stars_coordinates()

        # Do photometry of the ensemble of reference stars.
        print(f'\n{8 * "-"}>\t{self.pipeline} will compute now the combined flux of the ensemble\n')
        self.do_photometry_ref_stars(make_effective_psf=False)

        # Relative flux per sec of target star
        differential_flux = target_flux / self.ref_stars_flux_averaged
        # differential_flux = [-2.5 * np.log(target_flux / (r * exptimes)) for r in reference_star_flux_sec]
        # differential_flux = np.average(differential_flux, axis=0)
        normalized_flux = differential_flux / np.nanmedian(differential_flux)

        # Find Differential S/N for object and ensemble
        S_to_N_diff = 1 / np.sqrt(S_to_N_obj**-2 + self.S_to_N_ref**-2)

        # Ending time of computatin analysis.
        end = time.time()
        exec_time = end - start

        # Print when all of the analysis ends
        print(f'{8 * "-"}>\tDifferential photometry of {self.target_star_id} has been finished, '
              f'with {len(self.good_frames_list)} frames '
              f'of camera {self.instrument} (run time: {exec_time:.3f} sec)\n')

        if save_rms:
            # Output directory for files that contain photometric precisions
            output_directory = os.path.join(self._output_directory, 'rms_precisions')
            os.makedirs(output_directory, exist_ok=True)

            # File with rms information
            file_rms_name = os.path.join(output_directory, f'rms_{self.instrument}.txt')

            with open(file_rms_name, 'a') as file:
                file.write(f'{self.r} {self.std} {self.std_binned} '
                           f'{np.nanmedian(S_to_N_obj)} {np.nanmedian(self.S_to_N_ref)} '
                           f'{np.nanmedian(S_to_N_diff)}\n')

        return (times, normalized_flux, self.sigma_total)


class LightCurve(TimeSeriesAnalysis):

    def __init__(self, target_star='', data_directory='', search_pattern='*.fit*',
                 from_coordinates=True, ra_target=None, dec_target=None,
                 transit_times=[], telescope='', centroid_box=30):
        super(LightCurve, self).__init__(target_star=target_star,
                                         data_directory=data_directory,
                                         search_pattern=search_pattern,
                                         from_coordinates=from_coordinates,
                                         ra_target=ra_target,
                                         dec_target=dec_target,
                                         transit_times=transit_times,
                                         telescope=telescope,
                                         centroid_box=centroid_box)

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
        mask = np.where(np.logical_and(flux >= sigma_lower, flux <= sigma_upper))

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
            print(f'{8 * "-"}>\tDetrending time series with M_s = {M_star} M_sun, '
                  + f'R_s = {R_star} R_sun, and P_orb = {Porb:.3f} d\n')

            # Compute the transit duration
            transit_dur = t14(R_s=R_star, M_s=M_star,
                              P=Porb, small_planet=False)

            # Estimate the window length for the detrending
            wl = 3.0 * transit_dur

            # Detrend the time series data
            detrended_flux, _ = flatten(time, flux, return_trend=True, method='biweight',
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
        df_flux = pd.DataFrame({'binned_flux': flux})
        binned_flux = df_flux.groupby(df_flux.index // bins).mean()
        binned_flux = binned_flux['binned_flux']

        df_time = pd.DataFrame({'binned_times': time})
        binned_times = (df_time.groupby(df_time.index // bins).last()
                        + df_time.groupby(df_time.index // bins).first()) / 2
        binned_times = binned_times['binned_times']

        return binned_times, binned_flux

    @staticmethod
    def model_lightcurve(time, flux, limb_dc):
        """Summary
        """

        model = transitleastsquares(time, flux)
        results = model.power(u=limb_dc)  # , oversampling_factor=5, duration_grid_step=1.02)

        print(f'\n{8 * "-"}>\tStarting model of light curve...\n')
        print(f'{8 * " "}\t • Period: {results.period:.5f} d')
        print(f'{8 * " "}\t • {len(results.transit_times)} transit times in time series: '
              f'{[f"{i:0.5f}" for i in results.transit_times]}')
        print(f'{8 * " "}\t • Transit depth: {results.depth:.5f}')
        print(f'{8 * " "}\t • Best duration (days): {results.duration: .5f}')
        print(f'{8 * " "}\t • Signal detection efficiency (SDE): {results.SDE}\n')

        print('-------->\tFinished model of light curve. Plotting model...\n')

        return results

    # @logged
    def plot(self, time=[], flux=[], flux_uncertainty=[], bins=30, detrend=False,
             plot_tracking=False, plot_noise_sources=False, model_transit=False):
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

        print(pyfiglet.figlet_format(f'2. Light Curve')
              + '\t     Part of transyto package by Jaime A. Alvarado-Montes\n')

        # Refine the name of the target star using the main ID from Simbad
        simbad_result_table = Simbad.query_object(self.target_star_id)
        self.target_star_id = simbad_result_table['MAIN_ID'][0]

        # Get the data from the target star.
        star_data = catalog.StarData(self.target_star_id).query_from_mast()

        try:
            star_vmag = star_data['Vmag']
        except TypeError:
            star_vmag = input('Target star V magnitude: ')

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
            sigma_lower = my_box_in['caps'][0].get_ydata()[0]

            my_box_out = plt.boxplot(out_of_transit_flux)
            sigma_upper = my_box_out['caps'][1].get_ydata()[0]

            # Get the flux and mask after clipping the outliers.
            flux, clip_mask = self.clip_outliers(flux, sigma_lower=sigma_lower,
                                                 sigma_upper=sigma_upper)

            # Compute the relative/normalized flux of the target star (post-clipping).
            # normalized_flux = flux / np.nanmedian(out_of_transit_flux[out_of_transit_flux
            #                                                           < sigma_upper])

        if not self.transit_times:
            # Boxplot to identify outliers.
            my_box = plt.boxplot(flux)
            sigma_lower = my_box['caps'][0].get_ydata()[0]
            sigma_upper = my_box['caps'][1].get_ydata()[0]

            # Get the flux and mask after clipping the outliers.
            flux, clip_mask = self.clip_outliers(flux, sigma_lower=sigma_lower,
                                                 sigma_upper=sigma_upper)

        # Select unclipped values in the array of flux errors.
        flux_uncertainty = flux_uncertainty[clip_mask]

        # Barycentric Julian Date (BJD)
        time = time[clip_mask]

        # Violin plot.
        fig = plt.figure(figsize=(6.0, 5.0))

        # Name for boxplot.
        violin_name = Path(self._output_directory,
                           f'Violinplot_cam_{self.instrument}_rad{self.r}pix'
                           f'_{len(self.ref_stars_coordinates_list)}refstar.png')

        # ax = sns.swarmplot(y=normalized_flux, color=".25", zorder=3)
        if self.transit_times:
            flags = []
            for f in flux:
                if f in in_transit_flux:
                    flags.append('In-transit')
                else:
                    flags.append('Out-of-transit')

            flags = np.array(flags)

            df = pd.DataFrame({'flux': pd.Series(flux), 'Data': pd.Series(flags)})
            df['all'] = ''

            # Violin plot to analyse distribution (with transit times).
            ax = sns.violinplot(x='all', y='flux', hue='Data', data=df, inner='stick',
                                linewidth=1.0, split=True, cut=2, bw='silverman')

        if not self.transit_times:
            # Violin plot to analyse distribution (without transit times).
            ax = sns.violinplot(y=flux, inner='stick', linewidth=1.0, cut=2,
                                bw='silverman')

        ax.set_xlabel('Density Distribution', fontsize=11)
        ax.set_ylabel('Relative flux', fontsize=11)
        ax.legend(loc=(0.23, 1.0), ncol=2, title='Data', frameon=False)

        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=0.4)

        fig.savefig(violin_name, dpi=300)
        plt.close(fig)

        print('The density distribution of transit data has been plotted\n')

        if model_transit or (detrend and model_transit):
            print('Starting transit modeling via TLS:\n')

            # flatten_flux = self.detrend_timeseries(time, flux)
            results = self.model_lightcurve(time, flux)

            # Name for folded light curve.
            model_lightcurve_name = Path(self._output_directory, 'Model_lightcurve_cam'
                                                                 f'{self.instrument}_rad{self.r}pix_'
                                                                 f'{len(self.ref_stars_coordinates_list)}refstar.png')

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

            print(f'Folded model of the light curve of {self.target_star_id} was plotted\n')

            # Name for periodogram.
            periodogram_name = Path(self._output_directory, 'Periodogram_cam'
                                                            f'{self.instrument}_rad{self.r}pix_'
                                                            f'{len(self.ref_stars_coordinates_list)}refstar.png')

            fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
            ax.axvline(results.period, alpha=0.4, lw=3)
            # ax.set_xlim(np.min(results.periods), np.max(results.periods))
            for n in range(2, 10):
                ax.axvline(n * results.period, alpha=0.4, lw=1, linestyle='dashed')
                ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle='dashed')
            ax.set_ylabel(r'SDE')
            ax.set_xlabel('Period [d]')
            ax.plot(results.periods, results.power, color='black', lw=0.5)
            # ax.set_xlim(0, max(results.periods))

            fig.savefig(periodogram_name, dpi=300)

            # Detrend data using the previous transit model.
            # flux = self.detrend_data(time, flux, R_star=star_data["Rs"],
            #                                     M_star=star_data["Ms"], Porb=results.period)

        # Detrend data without using transit model.
        if detrend and not model_transit:
            flux = self.detrend_timeseries(time, flux)

        # Standard deviation in ppm for the observation
        std = np.nanstd(flux)

        # Name for light curve.
        lightcurve_name = Path(self._output_directory, 'Lightcurve_cam'
                                                       f'{self.instrument}_rad{self.r}pix_'
                                                       f'{len(self.ref_stars_coordinates_list)}refstar.png')

        fig, ax = plt.subplots(2, 1,
                               sharey='row', sharex='col', figsize=(8.5, 7.3))
        fig.suptitle(f'Differential Photometry\nTarget Star {self.target_star_id} '
                     f'(Vmag={star_vmag})', fontsize=13)

        ax[1].plot(time, flux, 'k.', ms=3,
                   label=f'NBin = {self.exptime:.1f} s, std = {std:.2%}')

        ax[1].errorbar(time, flux, yerr=flux_uncertainty,
                       fmt='none', ecolor="k", elinewidth=0.8,
                       label=r"$\sigma_{\mathrm{tot}}=\sqrt{\sigma_{\mathrm{phot}}^{2} "
                       r"+ \sigma_{\mathrm{sky}}^{2} + \sigma_{\mathrm{scint}}^{2} + "
                       r"\sigma_{\mathrm{read}}^{2}}$", capsize=0.0)
        # Binned data and times
        if bins != 0:
            binned_times, binned_flux = self.bin_timeseries(time, flux, bins)
            std_binned = np.nanstd(binned_flux)

            # Total time for binsize
            nbin_tot = self.exptime * bins
            ax[1].plot(binned_times, binned_flux, 'ro', ms=4,
                       label=f'NBin = {nbin_tot:.1f} s, std = {std_binned:.2%}')

        ax[1].set_ylabel('Relative Flux', fontsize=13)
        ax[1].legend(fontsize=8.0, loc=(0.0, 2.34), ncol=3, framealpha=1.0, frameon=False)
        fig.tight_layout(pad=2.85)

        for counter, ref_star_flux_sec in zip(self.ref_stars_coordinates_list.index.values.tolist(),
                                              self.ref_stars_flux_sec):
            # ax[1].xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))

            # Colors for comparison stars
            # colors = ["blue", "magenta", "green", "cyan", "firebrick"]

            ax[0].plot(time, ref_star_flux_sec[~nan_mask][clip_mask]
                       / np.nanmean(ref_star_flux_sec[~nan_mask][clip_mask]),
                       'o', ms=1.3, label=f'Ref. {counter}')
            ax[0].set_ylabel('Normalised Flux', fontsize=13)
            # ax[0].set_ylim((0.9, 1.05))
            ax[0].legend(fontsize=8.1, loc='lower left', ncol=len(self.ref_stars_coordinates_list),
                         framealpha=1.0, frameon=True)

        ax[1].text(0.97, 0.9, 'b)', fontsize=11, transform=ax[1].transAxes)
        ax[0].text(0.97, 0.9, 'a)', fontsize=11, transform=ax[0].transAxes)
        ax[0].set_title('Normalised Flux of Reference Stars')
        ax[1].set_title(f'Relative Flux of {self.target_star_id}')
        # Plot the ingress, mid-transit, and egress times.
        if self.transit_times:
            ax[1].axvline(ingress, c='k', ls='--', alpha=0.5)
            ax[1].axvline(mid_transit, c='k', ls='--', alpha=0.5)
            ax[1].axvline(egress, c='k', ls='--', alpha=0.5)

        ax[1].xaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))
        ax[1].yaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))

        plt.xlabel(r'BJD$_\mathrm{TDB}- $' f'{self.time_norm_factor}', fontsize=13)
        plt.xticks(rotation=30, size=8.0)

        fig.subplots_adjust(hspace=0.2)
        fig.savefig(lightcurve_name, dpi=300)

        print(f'The light curve of {self.target_star_id} was plotted')

        if plot_tracking:
            # Name for plot of tracking.
            plot_tracking_name = Path(self._output_directory, 'tracking_plot_cam'
                                                              f'{self.instrument}_rad{self.r}pix_'
                                                              f'{len(self.ref_stars_coordinates_list)}refstar.png')

            fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.3))
            ax.plot(time, self.x_pos_target[~nan_mask][clip_mask], 'ro-',
                    label='dx [Dec axis]', lw=0.5, ms=1.2)
            ax.plot(time, self.y_pos_target[~nan_mask][clip_mask], 'go-',
                    label='dy [RA axis]', lw=0.5, ms=1.2)
            ax.set_ylabel(r'$\Delta$ Pixel', fontsize=13)
            ax.legend(fontsize=8.6, loc='lower right', ncol=1, framealpha=1.0)
            ax.set_title(f'Tracking of camera {self.instrument}', fontsize=13)
            # ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.3f'))
            ax.set_xlabel(f'Time [BJD-{self.time_norm_factor}]', fontsize=13)
            plt.xticks(rotation=30, size=8.0)
            plt.grid(alpha=0.4)
            fig.savefig(plot_tracking_name, dpi=300)

        if plot_noise_sources:
            # Name for plot of noise sources.
            plot_noise_name = Path(self._output_directory, 'noises_plot_cam'
                                   f'{self.instrument}_rad{self.r}pix_'
                                   f'{len(self.ref_stars_coordinates_list)}refstar.png')

            fig, ax = plt.subplots(1, 1, sharey="row", sharex="col", figsize=(8.5, 6.3))
            ax.set_title(f'Noise Sources in {self.target_star_id} ' f'(Vmag={star_vmag})',
                         fontsize=13)
            ax.plot_date(self.jdutc_times.plot_date, self.sigma_total[~nan_mask] * 100, 'k-',
                         label=r"$\sigma_{\mathrm{total}}$")
            ax.plot_date(self.jdutc_times.plot_date, self.sigma_scint[~nan_mask] * 100, 'g-',
                         label=r'$\sigma_{\mathrm{scint}}$')
            ax.plot_date(self.jdutc_times.plot_date, self.sigma_phot[~nan_mask] * 100, '-',
                         color='firebrick', label=r'$\sigma_{\mathrm{phot}}$')
            ax.plot_date(self.jdutc_times.plot_date, self.sigma_sky[~nan_mask] * 100, 'b-',
                         label=r'$\sigma_{\mathrm{sky}}$')
            ax.plot_date(self.jdutc_times.plot_date, self.sigma_ron[~nan_mask] * 100, 'r-',
                         label=r'$\sigma_{\mathrm{read}}$')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.998), fancybox=True,
                      ncol=5, frameon=True, fontsize=8.1)
            # ax.set_yscale("log")
            ax.tick_params(axis='both', direction='in')
            ax.set_ylabel('Amplitude Error [%]', fontsize=13)
            plt.xticks(rotation=30, size=8.0)
            ax.set_xlabel('Time [UTC]', fontsize=13)
            # ax.set_ylim((0.11, 0.48))
            ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
            plt.grid(alpha=0.4)
            fig.savefig(plot_noise_name, dpi=300)

    def plot_from_table(self, table, exptime=30, bins=5, detrend=False, model_transit=False,
                        x_label='Time'):

        print(pyfiglet.figlet_format(f'* LightCurve *')
              + '\t     Part of transyto package by Jaime A. Alvarado-Montes\n')

        # Get the data from the target star.
        star_data = catalog.StarData(self.target_star_name).query_from_mast()
        star_name = star_data['star_name']
        star_vmag = star_data['Vmag']
        star_mass = star_data['Ms']
        star_radius = star_data['Rs']
        planet_period = star_data['orbital_period']

        # Name for light curve.
        output_directory = os.path.dirname(table) + '/'
        os.makedirs(output_directory, exist_ok=True)

        if table.endswith('.dat'):
            sep = '\t'
        if table.endswith('.csv'):
            sep = ';'
        table = pd.read_csv(table, usecols=['time', 'flux', 'flux_uncertainty'], delimiter=sep)
        index = 10000
        time = np.array(table['time']).astype(np.float)[:index]
        flux = np.array(table['flux']).astype(np.float)[:index]
        flux_uncertainty = np.array(table['flux_uncertainty']).astype(np.float)[:index]

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
            sigma_lower = my_box_in['caps'][0].get_ydata()[0]

            my_box_out = plt.boxplot(out_of_transit_flux)
            sigma_upper = my_box_out['caps'][1].get_ydata()[0]

            # Get the flux and mask after clipping the outliers.
            flux, clip_mask = self.clip_outliers(flux, sigma_lower=sigma_lower,
                                                 sigma_upper=sigma_upper)

            # Compute the relative/normalized flux of the target star (post-clipping).
            # flux = flux / np.nanmedian(out_of_transit_flux[out_of_transit_flux < sigma_upper])

        if not self.transit_times:
            # Boxplot to identify outliers.
            my_box = plt.boxplot(flux)
            sigma_lower = my_box['caps'][0].get_ydata()[0]
            sigma_upper = my_box['caps'][1].get_ydata()[0]

            # Get the flux and mask after clipping the outliers.
            flux, clip_mask = self.clip_outliers(flux, sigma_lower=sigma_lower,
                                                 sigma_upper=sigma_upper)

        # Select unclipped values in the array of flux errors.
        flux_uncertainty = flux_uncertainty[clip_mask]

        # Barycentric Julian Date (BJD)
        time = time[clip_mask]

        # Violin plot.
        fig = plt.figure(figsize=(6.0, 5.0))

        # Name for boxplot.
        violin_name = Path(output_directory, f'Violinplot_cam{self.telescope}_rad{self.r}pix.png')

        # ax = sns.swarmplot(y=flux, color=".25", zorder=3)
        if self.transit_times:
            flags = []
            for f in flux:
                if f in in_transit_flux:
                    flags.append('In-transit')
                else:
                    flags.append('Out-of-transit')

            flags = np.array(flags)

            df = pd.DataFrame({'flux': pd.Series(flux), 'Data': pd.Series(flags)})
            df['all'] = ''

            # Violin plot to analyse distribution (with transit times).
            ax = sns.violinplot(x='all', y='flux', hue='Data', data=df, inner='stick',
                                linewidth=1.0, split=True, cut=2, bw='silverman')

        if not self.transit_times:
            # Violin plot to analyse distribution (without transit times).
            ax = sns.violinplot(y=flux, inner='box', linewidth=1.0, cut=2,
                                bw='silverman')

        ax.set_xlabel('Density Distribution', fontsize=11)
        ax.set_ylabel('Relative flux', fontsize=11)

        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=0.4)

        fig.savefig(violin_name, dpi=300)
        plt.close(fig)

        print(f"{8 * '-'}>\tThe density distribution of transit data has been plotted\n")

        if model_transit or (detrend and model_transit):
            print("-------->\tStarting transit modeling via Transit Least Squares (Hippke & "
                  'Heller 2019)\n\n'
                  '         \t • Computing LD Coefficients v.1.0 (Espinoza $ Jordan 2015)')

            # Calculate the limd darkening coefficients.
            limb_dc = ldc.compute(name='CoRot-5', Teff=star_data["Teff"], RF='KpHiRes', FT='A100',
                                  grav=star_data['stellar_gravity'], metal=star_data['Fe/H'])[0]
            ab = (limb_dc[1], limb_dc[2])

            # Detrend data using the previous transit model.
            flux = self.detrend_timeseries(time, flux, R_star=star_radius,
                                           M_star=star_mass, Porb=planet_period)

            # flatten_flux = self.detrend_timeseries(time, flux)
            results = self.model_lightcurve(time, flux, limb_dc=ab)

            # Name for folded light curve.
            model_lightcurve_name = Path(output_directory, 'Model_lightcurve_cam'
                                                           f'{self.telescope}_rad{self.r}pix.png')

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

            print(f'-------->\tFolded model of the light curve of {star_name} was plotted\n')

            # Name for periodogram.
            periodogram_name = Path(output_directory, 'Periodogram_cam'
                                                      f'{self.telescope}_rad{self.r}pix.png')

            fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
            ax.axvline(results.period, alpha=0.4, lw=3)
            # ax.set_xlim(np.min(results.periods), np.max(results.periods))
            for n in range(2, 10):
                ax.axvline(n * results.period, alpha=0.4, lw=1, linestyle='dashed')
                ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle='dashed')
            ax.set_ylabel(r'SDE')
            ax.set_xlabel('Period [d]')
            ax.plot(results.periods, results.power, color='black', lw=0.5)
            ax.set_xlim(0.0, np.max(results.periods))

            fig.savefig(periodogram_name, dpi=300)

        # Detrend data without using transit model.
        if detrend and not model_transit:
            flux = self.detrend_timeseries(time, flux)

        # Standard deviation in ppm for the observation
        std = np.nanstd(flux)

        lightcurve_name = Path(output_directory, 'Lightcurve_cam'
                                                 f'{self.telescope}_rad{self.r}pix_.png')

        fig, ax = plt.subplots(1, 1,
                               sharey='row', sharex='col', figsize=(8.5, 6.3))
        fig.suptitle(f'Differential Photometry\nTarget Star {star_name}, Vmag={star_vmag}, '
                     f'Aperture = {self.r} pix', fontsize=13)

        ax.plot(time, flux, 'k.', ms=3,
                label=f'NBin = {exptime:.1f} s, std = {std:.2%}')

        # ax.errorbar(time, flux, yerr=flux_uncertainty,
        #             fmt="none", ecolor="k", elinewidth=0.8,
        #             label=r"$\sigma_{\mathrm{tot}}=\sqrt{\sigma_{\mathrm{phot}}^{2} "
        #             r"+ \sigma_{\mathrm{sky}}^{2} + \sigma_{\mathrm{scint}}^{2} + "
        #             r"\sigma_{\mathrm{read}}^{2}}$", capsize=0.0)

        # Binned data and times
        if bins != 0:
            binned_times, binned_flux = self.bin_timeseries(time, flux, bins)
            std_binned = np.nanstd(binned_flux)

            # Total time for binsize
            nbin_tot = exptime * bins
            ax.plot(binned_times, binned_flux, 'ro', ms=4,
                    label=f'NBin = {nbin_tot:.1f} s, std = {std_binned:.2%}')

        ax.set_ylabel('Relative Flux', fontsize=13)
        ax.legend(fontsize=8.0, loc=(0.0, 1.0), ncol=3, framealpha=1.0, frameon=False)

        # Plot the ingress, mid-transit, and egress times.
        if self.transit_times:
            ax.axvline(ingress, c='k', ls='--', alpha=0.5)
            ax.axvline(mid_transit, c='k', ls="--", alpha=0.5)
            ax.axvline(egress, c='k', ls='--', alpha=0.5)

        ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))

        plt.xlabel(f'{x_label}', fontsize=13)
        plt.xticks(rotation=30, size=8.0)

        fig.subplots_adjust(hspace=0.2)
        fig.savefig(lightcurve_name, dpi=300)

        print(f'-------->\tThe light curve of {star_name} was plotted\n')
