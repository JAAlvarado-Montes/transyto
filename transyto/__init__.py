#!/usr/bin/env python
# -*- coding: utf-8 -*-
###########################################################
#
# .######.#####...####..##..##..####..##..##.######..####..#
# ...##...##..##.##..##.###.##.##......####....##...##..##.#
# ...##...#####..######.##.###..####....##.....##...##..##.#
# ...##...##..##.##..##.##..##.....##...##.....##...##..##.#
# ...##...##..##.##..##.##..##..####....##.....##....####..#
# .........................................................#
#                                                          #
# Transyto                                                 #
# Transit photometry                                       #
#                                                          #
############################################################
# Jaime A. Alvarado-Montes (C), 2020                       #
############################################################

from __future__ import absolute_import

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
MPLSTYLE = os.path.join(PACKAGEDIR, 'data', 'transyto.mplstyle')

# By default Matplotlib is configured to work with a graphical user interface
# which may require an X11 connection (i.e. a display).  When no display is
# available, errors may occur.  In this case, we default to the robust Agg backend.
import platform
if platform.system() == 'Linux' and os.environ.get('DISPLAY', '') == '':
    import matplotlib
    matplotlib.use('Agg')

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

from .version import __version__
from .transyto import *
from transyto.utils import *
