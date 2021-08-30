# -*- coding: utf-8 -*-
"""
Application
===========
"""

import dash
import dash_bootstrap_components
import os
from flask import Flask

__author__ = 'Alex Forsythe, Gayle McAdams, Thomas Mansencal, Nick Shaw'
__copyright__ = ('Copyright (C) 2020-2021 '
                 'Academy of Motion Picture Arts and Sciences')
__license__ = 'Academy of Motion Picture Arts and Sciences License Terms'
__maintainer__ = 'Academy of Motion Picture Arts and Sciences'
__email__ = 'acessupport@oscars.org'
__status__ = 'Production'

__application_name__ = 'AMPAS - Apps'

__major_version__ = '0'
__minor_version__ = '1'
__change_version__ = '0'
__version__ = '.'.join(
    (__major_version__,
     __minor_version__,
     __change_version__))  # yapf: disable

__all__ = ['SERVER', 'SERVER_URL', 'APP']

SERVER = Flask(__name__)
"""
*Flask* server hosting the *Dash* app.

SERVER : Flask
"""

SERVER_URL = os.environ.get('AMPAS_APPS_SERVER')
"""
Server url used to construct permanent links for the individual apps.

SERVER_URL : unicode
"""

APP = dash.Dash(
    __application_name__,
    external_scripts=os.environ.get('AMPAS_APPS_JS', '').split(','),
    external_stylesheets=[dash_bootstrap_components.themes.LITERA] +
    os.environ.get('AMPAS_APPS_CSS', '').split(','),
    server=SERVER)
"""
*Dash* app.

APP : Dash
"""

APP.config['suppress_callback_exceptions'] = True
