import mechanicalsoup

from urllib.request import urlopen
from bs4 import BeautifulSoup
from collections import namedtuple


def show_html(url):
    """Show html webpage

    Parameters
    ----------
    url : str
        Url of webpage to open.

    Returns
    -------
    TYPE
        Description
    """
    html = urlopen(url)

    return html.read()


def find_input(url, verbose=False):
    """Find inputs in given webpage

    Parameters
    ----------
    url : str
        Url of webpage to scrab.
    verbose : bool, optional (default is False)
        Show the whole list of options.
    """
    # finds the input tags in the HTML
    bs = BeautifulSoup(show_html(url), 'html.parser')
    search = bs.find_all('input')
    results = []
    i = 0
    for result in search:
        if verbose:
            print(f'{i} {result}\n')
            input_value = result.get('value')
            print(f'{input_value}\n')
        results.append(result)
        i += 1


def find_observatory(observatory='', url='https://astro.swarthmore.edu/transits/transits.cgi'):
    """Look for available observatories

    Parameters
    ----------
    url : str, optional (default is swarthmore webpage)
        Url of swarthmore webpage.
    observatory : str, optional (default is "Siding")
        String pattern to look for observatory.
    verbose : bool, optional (default is False)
        Show the whole list of options.

    Returns
    -------
    int
        Index of observatory.

    """
    # finds the input tags in the HTML
    bs = BeautifulSoup(show_html(url), 'html.parser')
    search = bs.find_all('option')
    observatories = []

    for result in search:
        obs = result['value'].split(';')[-1]
        obs = obs.split(',')[0]

        if len(observatories) <= 71:
            observatories.append(obs)

    if not observatory:
        for j, obs in enumerate(observatories):
            print(f'{j} {obs}\n')

        observatory = input('Observatory not provided. Select one from the previous list: ')

    for i, obs in enumerate(observatories):
        if observatory in obs:
            outputs = namedtuple('outputs', 'idx name')
            return outputs(i, obs)
        else:
            continue


def configure_transit_finder(url='https://astro.swarthmore.edu/transits/transits.cgi',
                             database='exoplanets', starting_date='today', days_to_print=1,
                             days_in_past=0, min_start_elevation=30, elevation_conector='or',
                             min_end_elevation=30, min_transit_depth=5, max_magnitude=11,
                             observatory=''):
    """Configure swarthmore webpage to look for transits.

    Parameters
    ----------
    url : str, optional (default is Swarthmore webpage)
        Swarthmore webpage to look for transits.
    database: str, optional (default is 'exoplanets')
        The specific database to query
    starting_date : str, optional  (default is 'today')
        Starting date to look for transits. Either "today" or "mm-dd-yyyy" (ex. "08-20-2021")
    days_to_print : int, optional (default is 5)
        Number of days from provided date to show transits.
    days_in_past : str, optional (default is 0)
        Number of days to the past of provided date.
    min_start_elevation : int, optional (default is 30)
        Minimum elevation at start of transit to look for transits.
    elevation_conector : str, optional (default is "or")
        Conditional for minimum and maximum elevation. It can be "or" and "and"
    min_end_elevation : int, optional (default is 30)
        Minimum elevation at end of transit to look for transits.
    min_transit_depth : int, optional (default is 5)
        Minimum transit depth to filter transits.
    max_magnitude : int, optional (default is 11)
        Maximum stellar magnitude to filter transits.
    observatory : str, optional (default is "Siding")
        String pattern to look for observatory.

    Returns
    -------
    str
        Link to the CSV table for downloading.

    """

    browser = mechanicalsoup.Browser()

    page = browser.get(url)

    load_html = page.soup

    form = load_html.select('form')[0]

    # Select observatory
    observatory = find_observatory(observatory=observatory)
    form.select('option')[observatory.idx]['selected'] = 'selected'

    # Target List
    db_flag = 0
    if database == 'tois':
        db_flag = 2
    form.select('input')[0]['value'] = db_flag

    # Use UTC [1]
    form.select('input')[12]['value'] = 1

    # Start date
    form.select('input')[15]['value'] = starting_date  # It can be ex. "mm-dd-yyyy"

    # Days to print
    form.select('input')[16]['value'] = days_to_print

    # Days in past
    form.select('input')[17]['value'] = days_in_past

    # Minimum start elevation
    form.select('input')[18]['value'] = min_start_elevation

    # And/or
    form.select('input')[19]['value'] = elevation_conector

    # Minimum end elevation
    form.select('input')[21]['value'] = min_end_elevation

    # Minimum Depth
    form.select('input')[28]['value'] = min_transit_depth

    # Maximum Magnitude
    form.select('input')[29]['value'] = max_magnitude

    # CSV table
    form.select('input')[32]['value'] = 2

    # Click Submit
    form.select('input')[36]['value'] = 'Submit'

    charge_page = browser.submit(form, page.url)

    return charge_page.url
