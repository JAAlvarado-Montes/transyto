import logging
import numpy as np

from collections import namedtuple

log = logging.getLogger(__name__)

__all__ = ['compute_scintillation', 'compute_noises']


def compute_scintillation(telescope_aperture, telescope_altitude, airmass, exptime):
    """Compute scintillation noise for a given observation

    Parameters
    ----------
    telescope_aperture : float
        Aperture of the telescope [m]
    telescpe_altitude : float
        Altitude of the telescope [m]
    airmass : float or array
        airmass of the observation
    exptime : float or array
        Exposure time of the observation [s]

    Returns
    -------
    float
       Amplitude error (sigma) of scintillation for a given observation.
    """
    scintillation = (0.004 * telescope_aperture**(-2. / 3.) * airmass**(7. / 4.)
                     * np.exp(-telescope_altitude / 8000.) * (2 * exptime)**(-0.5))

    return scintillation


def compute_noises(gain, exptime, target_flux_sec, target_background_sec,
                   pixel_readout, aperture_radius):
    """Compute photon noise for a given observation

    Parameters
    ----------
    gain : float
        Gain of the CCD/CMOS detector
    exptime : float or array
        Exposure time of the observation [s]
    target_flux_sec : float or array
        Counts per second in aperture for target star [ADU/s]
    target_background_sec : float or array
        Background counts per second in aperture for target star [ADU/s]
    pixel_readout : float
        Readout noise of pixels in detector [e-]
    aperture_radius : int
        Radius used for the aperture photometry [pix]

    Returns
    -------
    TYPE
        Amplitude errors for photon, background, and readout noise.
    """
    # Sigma of photon noise
    target_signal = target_flux_sec * gain * exptime
    sigma_photon = -2.5 * np.log10((target_signal - np.sqrt(target_signal)) / target_signal)

    # Sigma of sky background
    sigma_sky = -2.5 * np.log10((target_signal - np.sqrt(target_background_sec
                                                         * gain * exptime)) / target_signal)

    # Sigma readout noise
    readout_noise = (pixel_readout * aperture_radius)**2 * np.pi
    sigma_readout = -2.5 * np.log10((target_signal - np.sqrt(readout_noise)) / target_signal)

    Outputs = namedtuple('Outputs', 'sigma_photon sigma_sky sigma_readout')

    return Outputs(sigma_photon, sigma_sky, sigma_readout)
