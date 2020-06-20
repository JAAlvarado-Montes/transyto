from .utils import search_files_across_directories
from astropy.wcs import WCS
from pocs.utils.images.fits import getheader
from pocs.utils import error
from warnings import warn

import os
import subprocess


def plate_solve_frame(fname, timeout=1200, solve_opts=None,
                      replace=True, remove_extras=True,
                      skip_solved=True, verbose=True, **kwargs):
    """ Plate solves an image.

    Args:
        fname(str, required):       Filename to solve in .fits extension.
        timeout(int, optional):     Timeout for the solve-field command,
                                    defaults to 60 seconds.
        solve_opts(list, optional): List of options for solve-field.
        verbose(bool, optional):    Show output, defaults to False.
    """

    verbose = kwargs.get('verbose', False)
    skip_solved = kwargs.get('skip_solved', True)

    out_dict = {}

    file_path, file_ext = os.path.splitext(fname)

    header = getheader(fname)
    wcs = WCS(header)

    # Check for solved file
    if skip_solved and wcs.is_celestial:

        if verbose:
            print("Solved file exists, skipping",
                  "(pass skip_solved=False to solve again):",
                  fname)

        out_dict.update(header)
        out_dict['solved_fits_file'] = fname
        return out_dict

    if verbose:
        print("Entering solve_field")

    # solve_field_script = os.path.join(os.getenv(''), 'scripts', 'solve_field.sh')
    solve_field_script = "solve-field"

    # solve_field_script = os.system(solve_field_script)

    # print(f"{solve_field_script}")
    # exit(0)

    # if not os.path.exists(solve_field_script):  # pragma: no cover
    #     raise error.InvalidSystemCommand(
    #         "Can't find solve-field: {}".format(solve_field_script))

    # Add the options for solving the field
    if solve_opts is not None:
        options = solve_opts
    else:
        options = [
            '--guess-scale',
            '--cpulimit', str(timeout),
            '--no-verify',
            '--no-plots',
            '--crpix-center',
            '--match', 'none',
            '--corr', 'none',
            '--wcs', 'none',
            '--downsample', '4',
        ]

        if kwargs.get('overwrite', False):
            options.append('--overwrite')
        if kwargs.get('skip_solved', False):
            options.append('--skip-solved')

        if 'ra' in kwargs:
            options.append('--ra')
            options.append(str(kwargs.get('ra')))
        if 'dec' in kwargs:
            options.append('--dec')
            options.append(str(kwargs.get('dec')))
        if 'radius' in kwargs:
            options.append('--radius')
            options.append(str(kwargs.get('radius')))

    if fname.endswith('.fz'):
        options.append('--extension=1')

    cmd = [solve_field_script] + options + [fname]
    if verbose:
        print("Cmd:", cmd)

    try:
        proc = subprocess.run(cmd, universal_newlines=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as e:
        raise error.InvalidCommand(
            "Can't send command to solve_field.sh: {} \t {}".format(e, cmd))
    except ValueError as e:
        raise error.InvalidCommand(
            "Bad parameters to solve_field: {} \t {}".format(e, cmd))
    except Exception as e:
        raise error.PanError("Timeout on plate solving: {}".format(e))

    if verbose:
        print("Returning proc from solve_field")

    try:
        # Handle extra files created by astrometry.net
        new = fname.replace(file_ext, '.new')
        rdls = fname.replace(file_ext, '.rdls')
        axy = fname.replace(file_ext, '.axy')
        xyls = fname.replace(file_ext, '-indx.xyls')

        if replace and os.path.exists(new):
            # Remove converted fits
            os.remove(fname)
            # Rename solved fits to proper extension
            os.rename(new, fname)

            out_dict['solved_fits_file'] = fname
        else:
            out_dict['solved_fits_file'] = new

        if remove_extras:
            for f in [rdls, xyls, axy]:
                if os.path.exists(f):
                    os.remove(f)

    except Exception as e:
        warn('Cannot remove extra files: {}'.format(e))

    return proc


if __name__ == '__main__':  # pragma: no cover
    """
    Build WCS for the given frames
    """

    import argparse

    # Get the command line option
    parser = argparse.ArgumentParser(description="Build WCS for fits files")

    # Get the command line option
    parser.add_argument("--search-frames", dest="search_frames", default=None,
                        help="Directories containing raw data")
    parser.add_argument("--pattern", dest="search_pattern", default=None,
                        help="Directories containing raw data")

    args = parser.parse_args()

    files_list = search_files_across_directories(args.search_frames,
                                                 args.search_pattern)
    for fn in files_list:
        plate_solve_frame(fn, verbose=True)
