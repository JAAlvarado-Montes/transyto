from transyto.utils import search_files_across_directories, get_header
from astropy.wcs import WCS
from astropy.io import fits

from warnings import warn

import os
import subprocess
import numpy as np


def plate_solve_frame(filenames_path, timeout=100, solve_opts=None, replace=True, skip_solved=True,
                      remove_extras=True, verbose=True, compute_wcs_uncertainty=True,
                      file_search_pattern='*.fit*', **kwargs):
    """Plate solve an image.

    Parameters
    ----------
    filenames_path : TYPE
        Description
    timeout : int, optional
        Timeout for the solve-field command. Default 1200 seconds.
    solve_opts : list, optional
        List of options for solve-field. Default True.
    replace : boolean, optional
        Replace the unsolved file by the solved one. Default True.
    skip_solved : boolean, optional
        If file is solved then skip it. Defaul True.
    remove_extras : boolean, optional
        Remoce extra files produced by solve-field. Default True.
    verbose : boolean, optional
        Show process by solve-field. Defaul True.
    compute_wcs_uncertainty : bool, optional
        Description
    file_search_pattern : str, optional
        Description
    **kwargs
        Description

    Returns
    -------
    list: All the pathnames of solved files.
    """
    files_list = search_files_across_directories(filenames_path, file_search_pattern)

    for fname in files_list:

        verbose = kwargs.get('verbose', verbose)
        skip_solved = kwargs.get('skip_solved', skip_solved)

        out_dict = {}

        file_path, file_ext = os.path.splitext(fname)

        header = get_header(fname)
        wcs = WCS(header)

        # Check for solved file
        if skip_solved and wcs.is_celestial:
            print(verbose)
            if verbose:
                print('Solved file exists, skipping',
                      '(pass skip_solved=False to solve again):',
                      fname)

            out_dict.update(header)
            out_dict['solved_fits_file'] = fname
            continue

        if verbose:
            print('Entering solve_field...')

        solve_field_script = 'solve-field'

        # if not os.path.exists(solve_field_script):  # pragma: no cover
        #     raise error.InvalidSystemCommand(
        #         "Can't find solve-field: {}".format(solve_field_script))

        if compute_wcs_uncertainty:
            # Set name for correlation file
            corr_filename = f'corr_{os.path.splitext(os.path.basename(fname))[0]}.fits'
            corr_filepath = os.path.join(os.path.dirname(fname), corr_filename)
        else:
            corr_filepath = 'none'

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
                '--corr', corr_filepath,
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
            subprocess.run(cmd, universal_newlines=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        except OSError as e:
            raise 'Cannot send command to solve_field.sh: {} \t {}'.format(e, cmd)
        except ValueError as e:
            raise 'Bad parameters to solve_field: {} \t {}'.format(e, cmd)
        except Exception as e:
            raise 'Timeout on plate solving: {}'.format(e)
            continue

        if verbose:
            print(f'Returning proc from solve_field. WCS built for {fname}\n')

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

        try:
            if compute_wcs_uncertainty:
                compute_wcs_delta(fname, corr_filepath, **kwargs)
            else:
                pass
        except Exception as e:
            warn('WCS delta cannot be computed: {}'.format(e))

    return files_list


def compute_wcs_delta(filename, corr_filepath, remove_corr_file=True, focal_length=None,
                      pixel_size_keyword='XPIXSZ', **kwargs):
    """Summary

    Parameters
    ----------
    filename : str
        Name (path) of original fits file
    corr_filepath : str
        Name (path) of correlation fits file from astrometry
    remove_corr_file : bool, optional
        Flag to remove correlation file. Default True
    focal_length : None, optional
        Focal length of the telescope (in mm) used for the observations. Default None
    pixel_size_keyword : str, optional
        String of the header keyword that contains the pixel size information. Defaul 'XPIXSZ'
    **kwargs
        Description
    """
    # Get the pixel size from header (in microns)
    header = get_header(filename)
    pixel_size = header[pixel_size_keyword]

    # Get pixel size in arcsec
    arcsec_pix = 206.265 * (pixel_size / focal_length)

    try:
        # Calculate the astrometric uncertainty in RA and DEC
        tbl = fits.open(f'{corr_filepath}')[1].data
        rmserr = np.sqrt(np.mean((tbl.index_x - tbl.field_x)**2
                                 + (tbl.index_y - tbl.field_y)**2))

        # Convert rmserr to arcsec
        rmserr = rmserr * arcsec_pix

        # Write astrometric uncertainty to file's header
        fits.setval(filename, 'WCSDELTA', value=float(f'{rmserr:.20f}'),
                    comment='Astrometric Uncertainty in RA and DEC (arcsec)', before='WCSAXES')
    except Exception as e:
        warn('WCS delta cannot be computed: {}'.format(e))
    try:
        if remove_corr_file:
            os.remove(corr_filepath)

    except Exception as e:
        warn('Cannot remove correlation file: {}'.format(e))
