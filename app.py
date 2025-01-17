"""
Application
===========
"""

import os

import dash
import dash_bootstrap_components
from flask import Flask

__author__ = "Alex Forsythe, Gayle McAdams, Thomas Mansencal, Nick Shaw"
__copyright__ = "Copyright 2020 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__application_name__ = "AMPAS - Apps"

__major_version__ = "0"
__minor_version__ = "1"
__change_version__ = "0"
__version__ = f"{__major_version__}.{__minor_version__}.{__change_version__}"

__all__ = ["SERVER", "SERVER_URL", "APP"]

SERVER = Flask(__name__)
"""
*Flask* server hosting the *Dash* app.

SERVER : Flask
"""

SERVER_URL = os.environ.get("AMPAS_APPS_SERVER")
"""
Server url used to construct permanent links for the individual apps.

SERVER_URL : str
"""

APP = dash.Dash(
    __application_name__,
    external_scripts=os.environ.get("AMPAS_APPS_JS", "").split(","),
    external_stylesheets=[
        dash_bootstrap_components.themes.BOOTSTRAP,
        *os.environ.get("AMPAS_APPS_CSS", "").split(","),
    ],
    server=SERVER,
)
"""
*Dash* app.

APP : Dash
"""

APP.config["suppress_callback_exceptions"] = True
