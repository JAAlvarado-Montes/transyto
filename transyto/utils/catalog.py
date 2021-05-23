from contextlib import suppress

import requests


planeturl = "https://exo.mast.stsci.edu/api/v0.1/exoplanets/"
dvurl = "https://exo.mast.stsci.edu/api/v0.1/dvdata/tess/"
header = {}


class StarData:
    """Data from MAST catalog"""

    def __init__(self, star_id):
        """Initialized Data class"""

        self.star_id = star_id

    def query_from_mast(self):
        """Query data from MAST catalog

        Parameters
        ----------
        star_id : string
            Name of target star

        Returns
        Properties of the system
        ------------------
        """

        url = planeturl + f"{self.star_id} b" + "/properties/"

        r = requests.get(url=url, headers=header)

        self.planet_prop = r.json()

        with suppress(IndexError):
            return self.planet_prop[0]
