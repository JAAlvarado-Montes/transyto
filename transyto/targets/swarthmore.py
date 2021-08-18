import mechanicalsoup

from urllib.request import urlopen
from bs4 import BeautifulSoup


def show_html(URL_input):
    html = urlopen(URL_input)

    return html.read()


def find_input(URL_input, verbose=False):
    # finds the input tags in the HTML
    bs = BeautifulSoup(show_html(URL_input), 'html.parser')
    search = bs.find_all('input')
    results = []
    i = 0
    for result in search:
        if verbose:
            print(f"{i} {result}", '\n')
            input_value = result.get('value')
            print(input_value, '\n')
        results.append(result)
        i += 1


def find_observatory(URL_input, observatory="Siding", verbose=False):
    # finds the input tags in the HTML
    bs = BeautifulSoup(show_html(URL_input), 'html.parser')
    search = bs.find_all('option')
    observatories = []

    for result in search:
        if verbose:
            print(result, '\n')
            input_value = result.get('value')
            print(input_value, '\n')
        observatories.append(result["value"])

    for i, obs in enumerate(observatories):
        if observatory in obs:
            return i
        else:
            continue


def configure_transit_finder(url="https://astro.swarthmore.edu/transits/transits.cgi",
                             database="exoplanets", time="UTC", start_date="today",
                             days_to_print=1, days_in_past=0, min_start_elevation=30,
                             elevation_conector="or", min_end_elevation=30, min_transit_depth=5,
                             max_magnitude=11, table_type="HTML", observatory="Siding"):

    browser = mechanicalsoup.Browser()

    page = browser.get(url)

    load_html = page.soup

    form = load_html.select("form")[0]

    # Select observatory
    observatory = find_observatory(url, observatory=observatory)
    form.select("option")[observatory]["selected"] = "selected"

    # Target List
    db_flag = 0
    if database == "tois":
        db_flag = 2
    form.select("input")[0]["value"] = db_flag

    # Use UTC [1] or Local Time [0]
    time_flag = 1
    if time == "Local":
        time_flag = 0
    form.select("input")[11]["value"] = time_flag

    # Start date
    form.select("input")[14]["value"] = start_date  # It can be ex. "mm-dd-yyyy"

    # Days to print
    form.select("input")[15]["value"] = days_to_print

    # Days in past
    form.select("input")[16]["value"] = days_in_past

    # Minimum start elevation
    form.select("input")[17]["value"] = min_start_elevation

    # And/or
    form.select("input")[19]["value"] = elevation_conector

    # Minimum end elevation
    form.select("input")[20]["value"] = min_end_elevation

    # Minimum Depth
    form.select("input")[27]["value"] = min_transit_depth

    # Maximum Magnitude
    form.select("input")[28]["value"] = max_magnitude

    # HTML or CSV table
    table_flag = 1
    if table_type == "CSV":
        table_flag = 2
    form.select("input")[31]["value"] = table_flag

    # Click Submit
    form.select("input")[35]["value"] = "Submit"

    charge_page = browser.submit(form, page.url)

    return charge_page.url


if __name__ == "__main__":
    link = set_transit_webpage()
    print(link)
    # url = "https://astro.swarthmore.edu/transits/transits.cgi"
    # find_input(url)
