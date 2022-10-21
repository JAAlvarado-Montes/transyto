"""Use transyto package to perform time series photometry"""
from transyto import LightCurve

# String pattern to look for data frames.
search_pattern = "*fit*"

# Get position of the target star using its coordinates.
from_coords = False

# Name of the telescope
telescope = "Huntsman"

# Name or coordinates of target star.
target_name = "Wasp-50"
target_ra = None
target_dec = None

# Transit times: ingress, mid-transit, and egress.
# transit_times = ["2021-07-27T15:42:00", "2021-07-27T17:06:00", "2021-07-27T18:29:00"]
transit_times = []

if __name__ == "__main__":
    import argparse

    # Get the command line option
    parser = argparse.ArgumentParser(description="Differential photometry for a star")

    # Get the command line option
    parser.add_argument("--data-directory", dest="data_directory",
                        help="Directories containing raw data",
                        default=None)
    parser.add_argument("--search-pattern", dest="search_pattern",
                        help="String search pattern", type=str,
                        default=search_pattern)
    parser.add_argument("--telescope", dest="telescope",
                        help="Name of the telescope", type=str,
                        default=telescope)
    parser.add_argument("--target-star", dest="target_star",
                        help="Name of target star", type=str,
                        default=target_name)
    parser.add_argument("--ra-target", dest="ra_target",
                        help="RA of target", type=float,
                        default=target_ra)
    parser.add_argument("--dec-target", dest="dec_target",
                        help="DEC of target", type=float,
                        default=target_dec)
    parser.add_argument("--transit-times", dest="transit_times",
                        help="Transit times of target star", type=list,
                        default=transit_times)
    parser.add_argument("--coords", dest="from_coordinates",
                        help="Flag to use coordinates", type=bool,
                        default=from_coords)

    args = parser.parse_args()

    lightcurve = LightCurve(**vars(args))

    time, flux, flux_uncertainty = lightcurve.get_relative_flux()

    lightcurve.plot(time=time, flux=flux, flux_uncertainty=flux_uncertainty, plot_tracking=True,
                    plot_noise_sources=True, bins=2, detrend=False, model_transit=False)
