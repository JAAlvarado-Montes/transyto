#! /usr/bin/env python
import sys
import os
import numpy as np
import glob
import subprocess
if sys.version_info.major == 2:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen
import scipy.interpolate as si

try:
    import pyfits as fits
except ImportError:
    import astropy.io.fits as fits

rootdir = os.path.dirname(os.path.realpath(__file__))


def FixSpaces(intervals):
    s = ''
    i = 0
    while True:
        if intervals[i] == '':
            intervals.pop(i)
        else:
            i = i + 1
            if len(intervals) == i:
                break
        if len(intervals) == i:
            break
    for i in range(len(intervals)):
        if i != len(intervals) - 1:
            s = f'{s}{np.double(intervals[i])}\t'
        else:
            s = f'{s}{np.double(intervals[i])}\n'
    return s


def getFileLines(fname):
    with open(fname, 'r') as f:
        line = f.readline()
        if line.find('\n') == -1:
            lines = line.split('\r')
        else:
            f.seek(0)
            line = f.read()
            lines = line.split('\n')
    return lines


def getATLASStellarParams(lines):
    for i in range(len(lines)):
        line = lines[i]
        idx = line.find('EFF')
        if idx != -1:
            idx2 = line.find('GRAVITY')
            TEFF = line[idx + 4:idx2 - 1]
            GRAVITY = line[idx2 + 8:idx2 + 8 + 5]
            idx = line.find('L/H')
            if idx == -1:
                LH = '1.25'
            else:
                LH = line[idx + 4:]
            break
    return str(int(np.double(TEFF))), str(np.round(np.double(GRAVITY), 2)), str(np.round(np.double(LH), 2))


def getIntensitySteps(lines):
    for j in range(len(lines)):
        line = lines[j]
        idx = line.find('intervals')
        if idx != -1:
            line = lines[j + 1]
            intervals = line.split(' ')
            break

    s = FixSpaces(intervals)
    return j + 2, s


version = 'v.1.0.'


def get_derivatives(rP, IP):
    """
    This function calculates the derivatives in an intensity profile I(r).
    For a detailed explaination, see Section 2.2 in Espinoza & Jordan (2015).

    INPUTS:
      rP:   Normalized radii, given by r = sqrt(1-mu**2)
      IP:   Intensity at the given radii I(r).

    OUTPUTS:
      rP:      Output radii at which the derivatives are calculated.
      dI/dr:   Measurement of the derivative of the intensity profile.
    """
    ri = rP[1:-1]  # Points
    rib = rP[:-2]  # Points inmmediately before
    ria = rP[2:]  # Points inmmediately after
    Ii = IP[1:-1]
    Iib = IP[:-2]
    Iia = IP[2:]

    rbar = (ri + rib + ria) / 3.0
    Ibar = (Ii + Iib + Iia) / 3.0
    num = (ri - rbar) * (Ii - Ibar) + (rib - rbar) * (Iib - Ibar) + (ria - rbar) * (Iia - Ibar)
    den = (ri - rbar)**2 + (rib - rbar)**2 + (ria - rbar)**2

    return rP[1:-1], num / den


def fix_spaces(the_string):
    """
    This function fixes some spacing issues in the ATLAS model files.
    """
    splitted = the_string.split(' ')
    for s in splitted:
        if s != '':
            return s
    return the_string


def fit_exponential(mu, inten):
    """
    Calculate the coefficients for the exponential LD law.
    It assumes input intensities are normalized.  For a derivation of the
    least-squares problem solved, see Espinoza & Jordan (2015).

    INPUTS:
      mu:  Angles at which each intensity is calculated (numpy array).
      inten:   Normalized intensities (i.e., inten(mu)/inten(1)) (numpy array).

    OUTPUTS:
      e1:   Coefficient of the linear term of the exponential law.
      e2:   Coefficient of the exponential term of the exponential law.
    """
    # Define A matrix for the linear system:
    A = np.zeros([2, 2])
    # Define b vector for the linear system:
    b = np.zeros(2)
    # Obtain alpha_n_k and beta_k that fill the A matrix and b vector.
    # In this case, g_1 = 1-mu, g_2 = 1/(1-exp(mu)):
    A[0, 0] = sum((1.0 - mu)**2)                    # alpha_{1,1}
    A[0, 1] = sum((1.0 - mu) * (1. / (1. - np.exp(mu))))  # alpha_{1,2}
    A[1, 0] = A[0, 1]                              # alpha_{2,1} = alpha_{1,2}
    A[1, 1] = sum((1. / (1. - np.exp(mu)))**2)        # alpha_{2,2}
    b[0] = sum((1.0 - mu) * (1.0 - inten))                 # beta_1
    b[1] = sum((1. / (1. - np.exp(mu))) * (1.0 - inten))     # beta_2
    return np.linalg.solve(A, b)


def fit_logarithmic(mu, inten):
    """
    Calculate the coefficients for the logarithmic LD law.
    It assumes input intensities are normalized.  For a derivation of the
    least-squares problem solved, see Espinoza & Jordan (2015).

    INPUTS:
      mu:  Angles at which each intensity is calculated (numpy array).
      inten:   Normalized intensities (i.e., inten(mu)/inten(1)) (numpy array).

    OUTPUTS:
      l1:   Coefficient of the linear term of the logarithmic law.
      l2:   Coefficient of the logarithmic term of the logarithmic law.
    """
    # Define A matrix for the linear system:
    A = np.zeros([2, 2])
    # Define b vector for the linear system:
    b = np.zeros(2)
    # Obtain the alpha_n_k and beta_k that fill the A matrix and b vector.
    # In this case, g_1 = 1-mu, g_2 = mu*ln(mu):
    A[0, 0] = sum((1.0 - mu)**2)               # alpha_{1,1}
    A[0, 1] = sum((1.0 - mu) * (mu * np.log(mu)))  # alpha_{1,2}
    A[1, 0] = A[0, 1]                         # alpha_{2,1} = alpha_{1,2}
    A[1, 1] = sum((mu * np.log(mu))**2)        # alpha_{2,2}
    b[0] = sum((1.0 - mu) * (1.0 - inten))            # beta_1
    b[1] = sum((mu * np.log(mu)) * (1.0 - inten))     # beta_2
    return np.linalg.solve(A, b)


def fit_square_root(mu, inten):
    """
    Calculates the coefficients for the square-root LD law.
    It assumes input intensities are normalized.  For a derivation of the
    least-squares problem solved, see Espinoza & Jordan (2015).

    INPUTS:
      mu:  Angles at which each intensity is calculated (numpy array).
      inten:   Normalized intensities (i.e., inten(mu)/inten(1)) (numpy array).

    OUTPUTS:
      s1:   Coefficient of the linear term of the square-root law.
      s2:   Coefficient of the square-root term of the square-root law.
    """
    # Define A matrix for the linear system:
    A = np.zeros([2, 2])
    # Define b vector for the linear system:
    b = np.zeros(2)
    # Obtain the alpha_n_k and beta_k that fill the A matrix and b vector:
    for n in range(1, 3, 1):
        for k in range(1, 3, 1):
            A[n - 1, k - 1] = sum((1.0 - mu**(n / 2.0)) * (1.0 - mu**(k / 2.0)))
        b[n - 1] = sum((1.0 - mu**(n / 2.0)) * (1.0 - inten))
    x = np.linalg.solve(A, b)
    return x[1], x[0]  # x[1] = s1, x[0] = s2


def fit_non_linear(mu, inten):
    """
    Calculate the coefficients for the non-linear LD law.
    It assumes input intensities are normalized.  For a derivation of the
    least-squares problem solved, see Espinoza & Jordan (2015).

    INPUTS:
      mu:  Angles at which each intensity is calculated (numpy array).
      inten:   Normalized intensities (i.e., inten(mu)/inten(1)) (numpy array).

    OUTPUTS:
      c1:   Coefficient of the square-root term of the non-linear law.
      c2:   Coefficient of the linear term of the non-linear law.
      c3:   Coefficient of the (1-mu^{3/2}) term of the non-linear law.
      c4:   Coefficient of the quadratic term of the non-linear law.
    """
    # Define A matrix for the linear system:
    A = np.zeros([4, 4])
    # Define b vector for the linear system:
    b = np.zeros(4)
    # Obtain the alpha_n_k and beta_k that fill the A matrix and b vector:
    for n in range(1, 5, 1):
        for k in range(1, 5, 1):
            A[n - 1, k - 1] = sum((1.0 - mu**(n / 2.0)) * (1.0 - mu**(k / 2.0)))
        b[n - 1] = sum((1.0 - mu**(n / 2.0)) * (1.0 - inten))
    return np.linalg.solve(A, b)


def fit_three_parameter(mu, inten):
    """
    Calculate the coefficients for the three-parameter LD law.
    It assumes input intensities are normalized.  For a derivation of the
    least-squares problem solved, see Espinoza & Jordan (2015).

    INPUTS:
      mu:  Angles at which each intensity is calculated (numpy array).
      inten:   Normalized intensities (i.e., inten(mu)/inten(1)) (numpy array).

    OUTPUTS:
      b1:   Coefficient of the linear term of the three-parameter law.
      b2:   Coefficient of the (1-mu^{3/2}) part of the three-parameter law.
      b3:   Coefficient of the quadratic term of the three-parameter law.
    """
    # Define A matrix for the linear system:
    A = np.zeros([3, 3])
    # Define b vector for the linear system:
    b = np.zeros(3)
    # Obtain the alpha_n_k and beta_k that fill the A matrix and b vector.
    # In this case we skip c1 (i.e., set c1=0):
    for n in range(2, 5, 1):
        for k in range(2, 5, 1):
            A[n - 2, k - 2] = sum((1.0 - mu**(n / 2.0)) * (1.0 - mu**(k / 2.0)))
        b[n - 2] = sum((1.0 - mu**(n / 2.0)) * (1.0 - inten))
    return np.linalg.solve(A, b)


def fit_quadratic(mu, inten):
    """
    Calculate the coefficients for the quadratic LD law.
    It assumes input intensities are normalized.  For a derivation of the
    least-squares problem solved, see Espinoza & Jordan (2015).

    INPUTS:
      mu:  Angles at which each intensity is calculated (numpy array).
      inten:   Normalized intensities (i.e., inten(mu)/inten(1)) (numpy array).

    OUTPUTS:
      u1:   Linear coefficient of the quadratic law.
      u2:   Quadratic coefficient of the quadratic law.
    """
    # Define A matrix for the linear system:
    A = np.zeros([2, 2])
    # Define b vector for the linear system:
    b = np.zeros(2)
    # Obtain the alpha_n_k and beta_k that fill the A matrix and b vector:
    for n in range(1, 3, 1):
        for k in range(1, 3, 1):
            A[n - 1, k - 1] = sum(((1.0 - mu)**n) * ((1.0 - mu)**k))
        b[n - 1] = sum(((1.0 - mu)**n) * (1.0 - inten))
    return np.linalg.solve(A, b)


def fit_linear(mu, inten):
    """
    Calculate the coefficients for the linear LD law.
    It assumes input intensities are normalized.  For a derivation of the
    least-squares problem solved, see Espinoza & Jordan (2015).

    INPUTS:
      mu:  Angles at which each intensity is calculated (numpy array).
      inten:   Normalized intensities (i.e., inte(mu)/inten(1)) (numpy array).

    OUTPUTS:
      a:   Coefficient of the linear law.
    """
    alpha_1_1 = sum((1.0 - mu)**2)
    beta_1 = sum((1.0 - mu) * (1.0 - inten))
    a = beta_1 / alpha_1_1
    return a


def downloader(url):
    """
    This function downloads a file from the given url using wget.
    """
    file_name = url.split('/')[-1]
    print(f'\t             > Downloading file {file_name} from {url}.')
    os.system('wget -q ' + url)


def get_closest(value, allowed_values):
    """
    Returns the closest variable from a list of numerics
    """
    if value in allowed_values:
        return value

        # Get closest value to input
    diffs = np.abs(value - allowed_values)
    closest = allowed_values[diffs.argmin()]

    return closest


def ATLAS_model_search(s_met, s_grav, s_teff, s_vturb):
    """
    Given input metallicities, gravities, effective temperature and
    microturbulent velocity, this function estimates which model is
    the most appropiate (i.e., the closer one in parameter space).
    If the model is not present in the system, it downloads it from
    Robert L. Kurucz's website (kurucz.harvard.edu/grids.html).
    """
    model_path = os.path.join(rootdir, 'atlas_models')

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(os.path.join(model_path, 'raw_models'), exist_ok=True)

    # This is the list of all the available metallicities in Kurucz's website:
    possible_mets = np.array([-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5,
                              -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
    # And this is the list of all possible vturbs:
    possible_vturb = np.array([0.0, 2.0, 4.0, 8.0])

    # Check if turbulent velocity is given. If not, set to 2 km/s:
    if s_vturb == -1:
        print('\t           > No known turbulent velocity. Setting it to 2 km/s.')
        s_vturb = 2.0

    chosen_vturb = get_closest(s_vturb, possible_vturb)
    if chosen_vturb != s_vturb:
        print(f'\t > For input vturb {s_vturb} km/s, closest vturb is {chosen_vturb} km/s.')

    chosen_met = get_closest(s_met, possible_mets)
    if chosen_met != s_met:
        print(f'\t > For input metallicity {s_met}, closest metallicity is {chosen_met}.')

    # Check if the intensity file for the calculated metallicity and
    # vturb is on the atlas_models folder:
    if chosen_met == 0.0:
        met_dir = 'p00'
    elif chosen_met < 0:
        met_string = str(np.abs(chosen_met)).split('.')
        met_dir = f'm{met_string[0]}{met_string[1]}'
    else:
        met_string = str(np.abs(chosen_met)).split('.')
        met_dir = f'p{met_string[0]}{met_string[1]}'

    print('\t           > Checking if ATLAS model file is on the system ...')
    # This will make the code below easier to follow:
    amodel = f'{met_dir}k{chosen_vturb:.0f}'
    afile = os.path.join(model_path, 'raw_models', f'i{amodel}')

    if os.path.exists(f'{afile}new.pck') or os.path.exists(f'{afile}.pck19') or os.path.exists(f'{afile}.pck'):
        print('\t           > Model file found.')
    else:
        # If not in the system, download it from Kurucz's website.
        # First, check all possible files to download:
        print('\t           > Model file not found.')
        response = urlopen(f'http://kurucz.harvard.edu/grids/grid{met_dir}/')
        html = str(response.read())
        ok = True
        filenames = []
        while ok:
            idx = html.find(f'>i{met_dir.lower()}')
            if idx == -1:
                ok = False
            else:
                for i in range(30):
                    if html[idx + i] == '<':
                        filenames.append(html[idx + 1:idx + i])
            html = html[idx + 1:]

        hasnew = False
        gotit = False
        araw = os.path.join(model_path, 'raw_models')
        # Check that filenames have the desired vturb and prefer *new* models:
        for afname in filenames:
            if 'new' in afname and amodel in afname:
                hasnew = True
                gotit = True
                downloader(f'http://kurucz.harvard.edu/grids/grid{met_dir}/{afname}')
                if not os.path.exists(araw):
                    os.makedirs(araw, exist_ok=True)
                os.rename(afname, os.path.join(araw, afname))

        if not hasnew:
            for afname in filenames:
                if '.pck19' in afname and amodel in afname:
                    gotit = True
                    downloader(f'http://kurucz.harvard.edu/grids/grid{met_dir}/{afname}')
                    if not os.path.exists(araw):
                        os.makedirs(araw, exist_ok=True)
                    os.rename(afname, os.path.join(araw, afname))
            if not gotit:
                for afname in filenames:
                    if f'{amodel}.pck' in afname:
                        gotit = True
                        downloader(f'http://kurucz.harvard.edu/grids/grid{met_dir}/{afname}')
                        if not os.path.exists(araw):
                            os.makedirs(araw, exist_ok=True)
                        os.rename(afname, os.path.join(araw, afname))
        if not gotit:
            print(f'\t > No model with closest metallicity of {chosen_met} and closest '
                  f'vturb of {chosen_vturb} km/s found.\n\t   Please modify the input '
                  'values of the target and select other stellar parameters '
                  'for it.')
            sys.exit()

    # Check if the models in machine readable form have been generated.
    # If not, generate them:
    if not os.path.exists(model_path + amodel):
        # Now read the files and generate machine-readable files:
        possible_paths = [f'{afile}new.pck', f'{afile}.pck19', f'{afile}.pck']

        for i in range(len(possible_paths)):
            possible_path = possible_paths[i]
            if os.path.exists(possible_path):
                lines = getFileLines(possible_path)
                # Create folder for current metallicity and turbulent
                # velocity if not created already:
                amodel_path = os.path.join(model_path, amodel)
                if not os.path.exists(amodel_path):
                    os.makedirs(amodel_path, exist_ok=True)
                # Save files in the folder:
                while True:
                    TEFF, GRAVITY, LH = getATLASStellarParams(lines)
                    teff_path = os.path.join(amodel_path, TEFF)
                    if not os.path.exists(teff_path):
                        os.makedirs(teff_path, exist_ok=True)
                    idx, mus = getIntensitySteps(lines)

                    mr_file = os.path.join(teff_path, f'grav_{GRAVITY}_lh_{LH}.dat')
                    save_mr_file = not os.path.exists(mr_file)

                    if save_mr_file:
                        f = open(mr_file, 'w')
                        f.write(f'#TEFF:{TEFF}'
                                f' METALLICITY:{met_dir}'
                                f' GRAVITY:{GRAVITY}'
                                f' VTURB:{int(chosen_vturb)}'
                                f' L/H: {LH}\n')
                        f.write(f'#wav (nm) \t cos(theta):{mus}')

                    for i in range(idx, len(lines)):
                        line = lines[i]
                        idx = line.find('EFF')
                        idx2 = line.find('\x0c')
                        if idx2 != -1 or line == '':
                            pass
                        elif idx != -1:
                            lines = lines[i:]
                            break
                        else:
                            wav_p_intensities = line.split(' ')
                            s = FixSpaces(wav_p_intensities)
                            if save_mr_file:
                                f.write(f'{s}\n')

                    if save_mr_file:
                        f.close()

                    if i == len(lines) - 1:
                        break

    # Now, assuming models are written in machine readable form, we can work:
    chosen_met_folder = os.path.join(model_path, amodel)

    # Now check closest Teff for input star:
    t_diff = np.inf
    chosen_teff = np.inf
    chosen_teff_folder = ''
    tefffolders = glob.glob(chosen_met_folder + '/*')
    for tefffolder in tefffolders:
        fname = tefffolder.split('/')[-1]
        teff = np.double(fname)
        c_t_diff = abs(teff - s_teff)
        if c_t_diff < t_diff:
            chosen_teff = teff
            chosen_teff_folder = tefffolder
            t_diff = c_t_diff

    print('\t           > For input effective temperature {:.1f} K, closest value '
          'is {:.0f} K.'.format(s_teff, chosen_teff))
    # Now check closest gravity and turbulent velocity:
    grav_diff = np.inf
    chosen_grav = 0.0
    chosen_filename = ''
    all_files = glob.glob(os.path.join(chosen_teff_folder, '*'))

    for filename in all_files:
        grav = np.double((filename.split('grav')[1]).split('_')[1])
        c_g_diff = abs(grav - s_grav)
        if c_g_diff < grav_diff:
            chosen_grav = grav
            grav_diff = c_g_diff
            chosen_filename = filename

    # Summary:
    model_root_len = len(model_path)
    print(f'\t           > For input metallicity {s_met}, effective temperature {s_teff} K, and\n'
          f'\t             log-gravity {s_grav}, and turbulent velocity {s_vturb} km/s, closest\n'
          f'\t             combination is metallicity: {chosen_met}, effective temperature: {chosen_teff} K,\n'
          f'\t             log-gravity {chosen_grav} and turbulent velocity of {chosen_vturb} km/s.\n\n'
          f'\t           > Chosen model file to be used: {chosen_filename[model_root_len:]}.\n')

    return chosen_filename, chosen_teff, chosen_grav, chosen_met, chosen_vturb


def PHOENIX_model_search(s_met, s_grav, s_teff, s_vturb):
    """
    Given input metallicities, gravities, effective temperature and
    microturbulent velocity, this function estimates which model is
    the most appropiate (i.e., the closer one in parameter space).
    If the model is not present in the system, it downloads it from
    the PHOENIX public library (phoenix.astro.physik.uni-goettingen.de).
    """
    # Path to the PHOENIX models
    phoenix_path = os.path.join(rootdir, 'phoenix_models')
    model_path = os.path.join(phoenix_path, 'raw_models')

    if not os.path.exists(phoenix_path):
        os.makedirs(phoenix_path, exist_ok=True)

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    # In PHOENIX models, all of them are computed with vturb = 2 km/2
    if s_vturb == -1:
        print('\t    + No known turbulent velocity. Setting it to 2 km/s.')
        s_vturb = 2.0

    possible_mets = np.array([0.0, -0.5, -1.0, 1.0, -1.5, -2.0, -3.0, -4.0])
    chosen_met = get_closest(s_met, possible_mets)
    if chosen_met != s_met:
        print(f'\t > For input metallicity {s_met}, closest metallicity is {chosen_met}.')

    # Generate the folder name:
    if chosen_met == 0.0:
        met_folder = 'm00'
        model = 'Z-0.0'
    else:
        abs_met = str(np.abs(chosen_met)).split('.')
        if chosen_met < 0:
            met_folder = f'm{abs_met[0]}{abs_met[1]}'
            model = f'Z-{abs_met[0]}{abs_met[1]}'
        else:
            met_folder = f'p{abs_met[0]}{abs_met[1]}'
            model = f'Z+{abs_met[0]}{abs_met[1]}'

    chosen_met_folder = os.path.join(model_path, met_folder)

    # Check if folder exists. If it does not, create it and download the
    # PHOENIX models that are closer in temperature and gravity to the
    # user input values:
    if not os.path.exists(chosen_met_folder):
        os.makedirs(chosen_met_folder, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(chosen_met_folder)

    # See if in a past call the file list for the given metallicity was
    # saved; if not, retrieve it from the PHOENIX website:
    m_file_list = 'file_list.dat'

    if os.path.exists(m_file_list):
        with open(m_file_list) as f:
            all_files = f.readlines()
            for i in np.arange(len(all_files)):
                all_files[i] = all_files[i].strip()
    else:
        response = urlopen('ftp://phoenix.astro.physik.uni-goettingen.de/SpecIntFITS/'
                           f'PHOENIX-ACES-AGSS-COND-SPECINT-2011/{model}/')
        html = str(response.read())
        all_files = []
        while True:
            idx = html.find('lte')
            if idx == -1:
                break
            else:
                idx2 = html.find('.fits')
                all_files.append(html[idx:idx2 + 5])
            html = html[idx2 + 5:]
        f = open(m_file_list, 'w')
        for file in all_files:
            f.write(f'{file}\n')
        f.close()

    # Now check closest Teff for input star:
    possible_teff = [np.double(f[3:8]) for f in all_files]
    chosen_teff = get_closest(s_teff, possible_teff)

    if chosen_teff != s_teff:
        print(f'\t    + For input effective temperature {s_teff:.1f} K, closest value is {chosen_teff:.0f} K.')

    teff_string = f'{chosen_teff:05.0f}'
    teff_files = [f for f in all_files if teff_string in f]

    # Now check closest gravity:
    possible_grav = [np.double(f[9:13]) for f in teff_files]
    chosen_grav = get_closest(s_grav, possible_grav)
    chosen_fname = teff_files[possible_grav.index(chosen_grav)]

    print('\t    + Checking if PHOENIX model file is on the system...')

    # Check if file is already downloaded. If not, download it from the PHOENIX website:
    if not os.path.exists(chosen_fname):
        print('\t    + Model file not found.')
        downloader('ftp://phoenix.astro.physik.uni-goettingen.de/SpecIntFITS/'
                   f'PHOENIX-ACES-AGSS-COND-SPECINT-2011/{model}/{chosen_fname}')
    else:
        print('\t    + Model file found.')

    os.chdir(cwd)
    chosen_path = os.path.join(chosen_met_folder, chosen_fname)

    # Summary:
    print(f'\t + For input metallicity {s_met}, effective temperature {s_teff} K, and\n'
          f'\t   log-gravity {s_grav}, closest combination is metallicity: {chosen_met},\n'
          f'\t   effective temperature: {chosen_teff} K, and log-gravity {chosen_grav}\n\n'
          f'\t + Chosen model file to be used:\n\t\t{chosen_fname}\n')

    return chosen_path, chosen_teff, chosen_grav, chosen_met, s_vturb


def get_response(min_w, max_w, response_function):
    root = os.path.join(rootdir, 'response_functions')

    # Standard response functions:
    lower_function_name = response_function.lower()
    if lower_function_name == 'kphires':
        response_file = os.path.join(root, 'standard', 'kepler_response_hires1.txt')
    elif lower_function_name == 'kplowres':
        response_file = os.path.join(root, 'standard', 'kepler_response_lowres1.txt')
    elif lower_function_name == 'irac1':
        response_file = os.path.join(root, 'standard', 'IRAC1_subarray_response_function.txt')
    elif lower_function_name == 'irac2':
        response_file = os.path.join(root, 'standard', 'RAC2_subarray_response_function.txt')
    elif lower_function_name == 'wfc3':
        response_file = os.path.join(root, 'standard', 'WFC3_response_function.txt')

    # User-defined response functions:
    else:
        user_file = os.path.join(root, response_function)
        if os.path.exists(user_file):
            response_file = user_file
        elif os.path.exists(response_function):  # RF not in RF folder:
            response_file = response_function
        else:
            print(f'Error: {response_function} is not valid.')
            sys.exit()

    # Open the response file, which we assume has as first column wavelength
    # and second column the response:
    w, r = np.loadtxt(response_file, unpack=True)
    if 'kepler' in response_file:
        w = 10 * w
        if min_w is None:
            min_w = min(w)
        if max_w is None:
            max_w = max(w)
        print('\t           > Kepler response file detected.  Switch from nanometers to Angstroms.')
        print(f'\t           > Minimum wavelength: {min(w)} A.\n'
              f'\t           > Maximum wavelength: {max(w)} A.')
    elif 'IRAC' in response_file:
        w = 1e4 * w
        if min_w is None:
            min_w = min(w)
        if max_w is None:
            max_w = max(w)
        print('\t > IRAC response file detected.  Switch from microns to '
              'Angstroms.')
        print(f'\t > Minimum wavelength: {min(w)} A.\n'
              f'\t > Maximum wavelength: {max(w)} A.')
    else:
        if min_w is None:
            min_w = min(w)
        if max_w is None:
            max_w = max(w)

    # Fit a univariate linear spline (k=1) with s=0 (a node in each data-point):
    S = si.UnivariateSpline(w, r, s=0, k=1)
    if type(min_w) is list:
        S_wav = []
        S_res = []
        for i in range(len(min_w)):
            c_idx = np.where((w > min_w[i]) & (w < max_w[i]))[0]
            c_S_wav = np.append(np.append(min_w[i], w[c_idx]), max_w[i])
            c_S_res = np.append(np.append(S(min_w[i]), r[c_idx]), S(max_w[i]))
            S_wav.append(np.copy(c_S_wav))
            S_res.append(np.copy(c_S_res))
    else:
        idx = np.where((w > min_w) & (w < max_w))[0]
        S_wav = np.append(np.append(min_w, w[idx]), max_w)
        S_res = np.append(np.append(S(min_w), r[idx]), S(max_w))

    return min_w, max_w, S_wav, S_res


def read_ATLAS(chosen_filename, model):
    # Define the ATLAS grid in mu = cos(theta):
    mu = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25,
                   0.2, 0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.01])
    mu100 = np.arange(1.0, 0.0, -0.01)

    # Now prepare files and read data from the ATLAS models:
    with open(chosen_filename, 'r') as f:
        lines = f.readlines()
    # Remove comments and blank lines:
    for i in np.flipud(np.arange(len(lines))):
        if lines[i].strip() == '' or lines[i].strip().startswith('#'):
            lines.pop(i)

    nwave = len(lines)
    wavelengths = np.zeros(nwave)
    intensities = np.zeros((nwave, len(mu)))
    I100 = np.zeros((nwave, len(mu100)))
    for i in np.arange(nwave):
        # If no jump of line or comment, save the intensities:
        splitted = lines[i].split()
        if len(splitted) == 18:
            wavelengths[i] = np.double(splitted[0]) * 10  # nano to angstrom
            intensities[i] = np.array(splitted[1:], np.double)

            # Only if I(1) is different from zero, fit the LDs:
            if intensities[i, 0] != 0.0:
                # Kurucz doesn't put points on his files (e.g.: 0.8013 is 8013).
                intensities[i, 1:] = intensities[i, 1:] / 1e5
                # Normalzie intensities wrt the first one:
                intensities[i, 1:] = intensities[i, 1:] * intensities[i, 0]
                # If requested, extract the 100 mu-points, with cubic spline
                # interpolation (k=3) through all points (s=0) as CB11:
                if model == 'A100':
                    II = si.UnivariateSpline(mu[::-1], intensities[i, ::-1],
                                             s=0, k=3)
                    I100[i] = II(mu100)

    # Select only those with non-zero intensity:
    flag = intensities[:, 0] != 0.0
    if model == 'A100':
        return wavelengths[flag], I100[flag], mu100
    else:
        return wavelengths[flag], intensities[flag], mu


def read_PHOENIX(chosen_path):
    mu = fits.getdata(chosen_path, 'MU')
    data = fits.getdata(chosen_path)
    CDELT1 = fits.getval(chosen_path, 'CDELT1')
    CRVAL1 = fits.getval(chosen_path, 'CRVAL1')
    wavelengths = np.arange(data.shape[1]) * CDELT1 + CRVAL1
    inten = data.transpose()
    return wavelengths, inten, mu


def integrate_response_ATLAS(wavelengths, inten, mu, S_res, S_wav,
                             atlas_correction, photon_correction,
                             interpolation_order, model):
    # Define the number of mu angles at which we will perform the integrations:
    nmus = len(mu)

    # Integrate intensity through each angle:
    I_l = np.array([])
    for i in range(nmus):
        # Interpolate the intensities:
        Ifunc = si.UnivariateSpline(wavelengths, inten[:, i], s=0, k=interpolation_order)
        # If several wavelength ranges where given, integrate through
        # each chunk one at a time.  If not, integrate the given chunk:
        if type(S_res) is list:
            integration_results = 0.0
            for j in range(len(S_res)):
                if atlas_correction and photon_correction:
                    integrand = (S_res[j] * Ifunc(S_wav[j])) / S_wav[j]
                elif atlas_correction and not photon_correction:
                    integrand = (S_res[j] * Ifunc(S_wav[j])) / (S_wav[j]**2)
                elif not atlas_correction and photon_correction:
                    integrand = (S_res[j] * Ifunc(S_wav[j])) * (S_wav[j])
                else:
                    integrand = S_res[j] * Ifunc(S_wav[j]) * S_wav[j]
                integration_results += np.trapz(integrand, x=S_wav[j])
        else:
            if atlas_correction and photon_correction:
                integrand = (S_res * Ifunc(S_wav)) / S_wav
            elif atlas_correction and not photon_correction:
                integrand = (S_res * Ifunc(S_wav)) / (S_wav**2)
            elif not atlas_correction and photon_correction:
                integrand = S_res * Ifunc(S_wav) * S_wav
            else:
                integrand = S_res * Ifunc(S_wav)
            integration_results = np.trapz(integrand, x=S_wav)
        I_l = np.append(I_l, integration_results)

    I0 = I_l / (I_l[0])

    return I0


def integrate_response_PHOENIX(wavelengths, inten, mu, S_res, S_wav, correction,
                               interpolation_order):
    I_l = np.array([])
    for i in range(len(mu)):
        Ifunc = si.UnivariateSpline(wavelengths, inten[:, i], s=0,
                                    k=interpolation_order)
        if type(S_res) is list:
            integration_results = 0.0
            for j in range(len(S_res)):
                if correction:
                    integrand = S_res[j] * Ifunc(S_wav[j]) * S_wav[j]
                else:
                    integrand = S_res[j] * Ifunc(S_wav[j])
                integration_results += np.trapz(integrand, x=S_wav[j])

        else:
            integrand = S_res * Ifunc(S_wav)  # lambda x,I,S: I(x)*S(x)
            if correction:
                integrand *= S_wav  # lambda x,I,S: (I(x)*S(x))*x
            # Integral of Intensity_nu*(Response Function*lambda)*c/lambda**2
            integration_results = np.trapz(integrand, x=S_wav)
        I_l = np.append(I_l, integration_results)

    return I_l / (I_l[-1])


def get_rmax(mu, I0):
    # Apply correction due to spherical extension. First, estimate the r:
    r = np.sqrt(1.0 - (mu**2))
    # Estimate the derivatives at each point:
    rPi, m = get_derivatives(r, I0)
    # Estimate point of maximum (absolute) derivative:
    idx_max = np.argmax(np.abs(m))
    r_max = rPi[idx_max]
    # To refine this value, take 20 points to the left and 20 to the right
    # of this value, generate spline and search for roots:
    ndata = 20
    idx_lo = np.max([idx_max - ndata, 0])
    idx_hi = np.min([idx_max + ndata, len(mu) - 1])
    r_maxes = rPi[idx_lo:idx_hi]
    m_maxes = m[idx_lo:idx_hi]
    spl = si.UnivariateSpline(r_maxes[::-1], m_maxes[::-1], s=0, k=4)
    fine_r_max = spl.derivative().roots()
    if len(fine_r_max) > 1:
        abs_diff = np.abs(fine_r_max - r_max)
        iidx_min = np.where(abs_diff == np.min(abs_diff))[0]
        fine_r_max = fine_r_max[iidx_min]

    return r, fine_r_max


def get100_PHOENIX(wavelengths, inten, new_mu, idx_new):
    mu100 = np.arange(0.01, 1.01, 0.01)
    I100 = np.zeros((len(wavelengths), len(mu100)))
    for i in range(len(wavelengths)):
        # Cubic splines (k=3), interpolation through all points (s=0) ala CB11.
        II = si.UnivariateSpline(new_mu, inten[i, idx_new], s=0, k=3)
        I100[i] = II(mu100)
    return mu100, I100


def calc_lds(name, response_function, model, s_met, s_grav, s_teff,
             s_vturb, min_w=None, max_w=None, atlas_correction=True,
             photon_correction=True, interpolation_order=1):
    """
    Generate the limb-darkening coefficients.  Note that response_function
    can be a string with the filename of a response function not in the
    list. The file has to be in the response_functions folder.

    Parameters
    ----------
    name: String
       Name of the object we are working on.
    response_function: String
       Number of a standard response function or filename of a response
       function under the response_functions folder.
    model: String
       Fitting technique model.
    s_met: Float
       Metallicity of the star.
    s_grav: Float
       log_g of the star (cgs).
    s_teff: Float
       Effective temperature of the star (K).
    s_vturb: Float
       Turbulent velocity in the star (km/s)
    min_w: Float
       Minimum wavelength to integrate (if None, use the minimum wavelength
       of the response function).
    max_w: Float
       Maximum wavelength to integrate (if None, use the maximum wavelength
       of the response function).
    atlas_correction: Bool
       True if corrections in the integrand of the ATLAS models should
       be applied (i.e., transformation of ATLAS intensities given in
       frequency to per wavelength)
    photon_correction: Bool
       If True, correction for photon-counting devices is used.
    interpolation_order: Integer
       Degree of the spline interpolation order.

    Returns
    -------
    LDC: 1D float tuple
       The linear (a), quadratic (u1, u2), three-parameter (b1, b2, b3),
       non-linear (c1, c2, c3, c4), logarithmic (l1, l2),
       exponential (e1, e2), and square-root laws (s1, s2).
    """
    print('\n\t           Reading response functions\n\t           --------------------------')

    # Get the response file minimum and maximum wavelengths and all the
    # wavelengths and values:
    min_w, max_w, S_wav, S_res = get_response(min_w, max_w, response_function)

    ######################################################################
    # IF USING ATLAS MODELS....
    ######################################################################
    if 'A' in model:
        # Search for best-match ATLAS9 model for the input stellar parameters:
        print('\n\t           ATLAS modelling\n\t           ---------------\n'
              '\t           > Searching for best-match Kurucz model ...')
        (chosen_filename, chosen_teff, chosen_grav,
            chosen_met,
            chosen_vturb) = ATLAS_model_search(s_met, s_grav, s_teff, s_vturb)

        # Read wavelengths and intensities (I) from ATLAS models.
        # If model is 'A100', it also returns the interpolated
        # intensities (I100) and the associated mu values (mu100).
        # If not, those arrays are empty:
        wavelengths, I, mu = read_ATLAS(chosen_filename, model)

        # Now use these intensities to obtain the (normalized) integrated
        # intensities with the response function:
        I0 = integrate_response_ATLAS(wavelengths, I, mu, S_res,
                                      S_wav, atlas_correction, photon_correction,
                                      interpolation_order, model)

        # Finally, obtain the limb-darkening coefficients:
        if model == 'AS':
            idx = mu >= 0.05  # Select indices as in Sing (2010)
        else:
            idx = mu >= 0.0  # Select all

    ######################################################################
    # IF USING PHOENIX MODELS....
    ######################################################################
    elif 'P' in model:
        # Search for best-match PHOENIX model for the input stellar parameters:
        print('\n\t PHOENIX modelling\n\t -----------------\n'
              '\t > Searching for best-match PHOENIX model ...')
        (chosen_path, chosen_teff, chosen_grav,
            chosen_met,
            chosen_vturb) = PHOENIX_model_search(s_met, s_grav, s_teff, s_vturb)

        # Read PHOENIX model wavelenghts, intensities and mus:
        wavelengths, I, mu = read_PHOENIX(chosen_path)

        # Now use these intensities to obtain the (normalized) integrated
        # intensities with the response function:
        I0 = integrate_response_PHOENIX(wavelengths, I, mu, S_res, S_wav,
                                        photon_correction, interpolation_order)

        # Obtain correction due to spherical extension. First, get r_max:
        r, fine_r_max = get_rmax(mu, I0)

        # Now get r for each intensity point and leave out those that have r>1:
        new_r = r / fine_r_max
        idx_new = new_r <= 1.0
        new_r = new_r[idx_new]
        # Reuse variable names:
        mu = np.sqrt(1.0 - (new_r**2))
        I0 = I0[idx_new]

        # Now, if the model requires it, obtain 100-mu points interpolated
        # in this final range of "usable" intensities:
        if model == 'P100':
            mu, I100 = get100_PHOENIX(wavelengths, I, mu, idx_new)
            I0 = integrate_response_PHOENIX(wavelengths, I100, mu,
                                            S_res, S_wav, photon_correction,
                                            interpolation_order)

        # Now define each possible model and fit LDs:
        if model == 'PQS':  # Quasi-spherical model (Claret et al. 2012)
            idx = mu >= 0.1
        elif model == 'PS':  # Sing method
            idx = mu >= 0.05
        else:
            idx = mu >= 0.0

    # Now compute each LD law:
    c1, c2, c3, c4 = fit_non_linear(mu, I0)
    a = fit_linear(mu[idx], I0[idx])
    u1, u2 = fit_quadratic(mu[idx], I0[idx])
    b1, b2, b3 = fit_three_parameter(mu[idx], I0[idx])
    l1, l2 = fit_logarithmic(mu[idx], I0[idx])
    e1, e2 = fit_exponential(mu[idx], I0[idx])
    s1, s2 = fit_square_root(mu[idx], I0[idx])
    # Make this correction:
    if model == 'PQS':
        c1, c2, c3, c4 = fit_non_linear(mu[idx], I0[idx])

    # Stack all LD coefficients into one single tuple:
    LDC = a, u1, u2, b1, b2, b3, c1, c2, c3, c4, l1, l2, e1, e2, s1, s2

    return LDC


def compute(Teff=None, grav=None, metal=None, vturb=-1, RF=None, FT=None, min_w=None,
            max_w=None, name='', interpolation_order=1, atlas_correction=True,
            photon_correction=True):
    """
    Compute limb-darkening coefficients.

    Parameters
    ----------
    Teff: Float
       Effective temperature of the star (K).
    grav: Float
       log_g of the star (cgs).
    metal: Float
       Metallicity of the star.
    vturb: Float
       Turbulent velocity in the star (km/s)
    RF: String
       A standard response function or filename of a response
       function under the response_functions folder.
    FT: String
       Limb-darkening fitting technique model.  Select one or more
       (comma separated, no blank spaces) model from the following list:
          A17:  LDs using ATLAS with all its 17 angles
          A100: LDs using ATLAS models interpolating 100 mu-points with a
                cubic spline (i.e., like Claret & Bloemen, 2011)
          AS:   LDs using ATLAS with 15 angles for linear, quadratic and
                three-parameter laws, bit 17 angles for the non-linear
                law (i.e., like Sing, 2010)
          P:    LDs using PHOENIX models (Husser et al., 2013).
          PS:   LDs using PHOENIX models using the methods of Sing (2010).
          PQS:  LDs using PHOENIX quasi-spherical models (mu>=0.1 only)
          P100: LDs using PHOENIX models and interpolating 100 mu-points
                with cubic spline (i.e., like Claret & Bloemen, 2011)
    min_w: Float
       Minimum wavelength to integrate (if None, use the minimum wavelength
       of the response function).
    max_w: Float
       Maximum wavelength to integrate (if None, use the maximum wavelength
       of the response function).
    name: String
       Name of the object we are working on (to write in ofile).
    interpolation_order: Integer
       Degree of the spline interpolation order.
    atlas_correction: Bool
       If True, convert ATLAS intensities using c/lambda**2 (ATLAS
       intensities are given per frequency).
    photon_correction: Bool
       If True, apply photon counting correction (lambda/hc).

    Returns
    -------
    LDC: 1D list
       Each element in this list contains a tuple of all the LD laws
       for a given parameter set.  The tuples of LD laws contain:
       The linear (a), quadratic (u1, u2), three-parameter (b1, b2, b3),
       non-linear (c1, c2, c3, c4), logarithmic (l1, l2),
       exponential (e1, e2), and square-root laws (s1, s2).

    Example
    -------
    import get_lds as lds
    ldc1 = lds.lds(ifile="input_files/example_input_file.dat")
    ldc2 = lds.lds(5500.0, 4.5, 0.0, -1, "KpHiRes", "A100,P100")
    """

    if Teff is None or grav is None or metal is None or RF is None or FT is None:
        print('Invalid input parameters.  Either define ifile, or '
              'define Teff, grav, metal, RF, and FT.')
        return None
    input_set = [[name, RF, FT, metal, grav, Teff, vturb, min_w, max_w]]

    # Compute LDCs for each input set:
    LDC = []
    for i in np.arange(len(input_set)):
        iset = input_set[i] + [atlas_correction, photon_correction, interpolation_order]
        models = iset[2].split(',')
        for model in models:
            iset[2] = model
            LDC.append(calc_lds(*iset))

    print('         \t • LD calculation finished without problems.\n')

    if os.path.exists(rootdir + '/atlas_models'):
        subprocess.run(f'rm -rf {rootdir}/atlas_models', shell=True, check=True)

    return LDC
