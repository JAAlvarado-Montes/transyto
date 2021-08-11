"""Use transyto package to perform time series photometry"""
from transyto import LightCurve

# String pattern to look for data frames.
search_pattern = "*fit*"

# Get position of the stars using coordiates and not their name.
from_coords = False

# Name or coordinates of reference stars.
# reference_stars = ["CD-48 14211", "CD-48 14215", "CD-48 14225", "CD-48 14210"]
reference_stars = ["CD-48 14210", "CD-48 14211", "CD-48 14225"]

refs_ra = [336.69230777, 336.7437907, 336.90741707, 337.5871838]
refs_dec = [-48.35990944, -48.32531778, -47.95336391, -47.63612619]

# Name of the telescope
telescope = "Huntsman"

# Name or coordinates of target star.
target_name = "Wasp-95"
target_ra = 337.4572324
target_dec = -48.00306924

# Transit times: ingress, mid-transit, and egress.
transit_times = ["2021-07-27T15:42:00", "2021-07-27T17:06:00", "2021-07-27T18:29:00"]

# Radius for the aperture photometry.
aperture_radius = 22

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
    parser.add_argument("--transit-times", dest="transit_times",
                        help="Transit times of target star", type=list,
                        default=transit_times)
    parser.add_argument("--ref-stars", dest="list_reference_stars",
                        help="List of reference stars", type=list,
                        default=reference_stars)
    parser.add_argument("--radius", dest="aperture_radius",
                        help="aperture radius", type=float,
                        default=aperture_radius)
    parser.add_argument("--coords", dest="from_coordinates",
                        help="Flag to use coordinates", type=bool,
                        default=from_coords)
    parser.add_argument("--ra_target", dest="ra_target",
                        help="RA of target", type=float,
                        default=target_ra)
    parser.add_argument("--dec_target", dest="dec_target",
                        help="DEC of target", type=float,
                        default=target_dec)
    parser.add_argument("--ra_refs", dest="ra_ref_stars",
                        help="RA of target", type=list,
                        default=refs_ra)
    parser.add_argument("--dec_refs", dest="dec_ref_stars",
                        help="DEC of ref. stars", type=list,
                        default=refs_dec)

    args = parser.parse_args()

    lightcurve = LightCurve(**vars(args))

    time, flux, flux_uncertainty = lightcurve.get_relative_flux()

    lightcurve.plot(time=time, flux=flux, flux_uncertainty=flux_uncertainty, plot_tracking=True,
                    plot_noise_sources=True, bins=3, detrend=False, model_transit=False)
