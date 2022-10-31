from transyto.utils import search_files_across_directories, get_header
from astropy.wcs import WCS

from warnings import warn

import os
import subprocess


def plate_solve_frame(filenames_path, timeout=100, solve_opts=None,
                      replace=True, remove_extras=True,
                      skip_solved=True, verbose=True, **kwargs):
    """Plate solves an image.

    Parameters
    ----------
    fits_file: string or list
        Filename to solve in .fits extension.
    timeout: int, optional
        Timeout for the solve-field command. Default 1200 seconds.
    verbose: boolean, optional
        Show output, defaults to False. Default True.
    solve_opts: list, optional
        List of options for solve-field. Default True.
    replace: boolean, optional
        Replace the unsolved file by the solved one. Default True.
    remove_extras: boolean, optional
        Remoce extra files produced by solve-field. Default True.
    skip_solved: boolean, optional
        If file is solved then skip it. Defaul True.
    verbose: boolean, optional
        Show process by solve-field. Defaul True.
    **kwargs: Description

    Returns
    -------
    list: All the pathnames of solved files.

    """
    files_list = search_files_across_directories(filenames_path, '*.fit*')

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

        # solve_field_script = os.path.join(os.getenv(''), 'scripts', 'solve_field.sh')
        solve_field_script = 'solve-field'

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

    return files_list
