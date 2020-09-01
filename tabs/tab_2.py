#!/usr/bin/env python3

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash_table import DataTable
from dash.dependencies import Input, Output
from dash_core_components import Dropdown
from dash_html_components import H5
import pandas as pd

import colour
from colour import ILLUMINANT_SDS

ILLUMINANTS_OPTIONS = [{
    'label': key,
    'value': key
} for key in sorted(ILLUMINANT_SDS.keys())]

params = [
    'Wavelength', 'R Sensitivity', 'G Sensitivity', 'B Sensitivity'
]

tab_2_layout = html.Div([
    H5(children='Illuminant'),
    Dropdown(
        id='illuminant',
        options=ILLUMINANTS_OPTIONS,
        value=ILLUMINANTS_OPTIONS[0]['value'],
        clearable=False,
        className='app-widget'
    ),
    H5(children='Camera System Spectral Sensitivities (Copy & Paste)'), 
        DataTable(
            id='sensitivities',
            columns=(
                [{'id': p, 'name': p} for p in params]
            ),
            data=[
            dict(Model=i, **{param: 0 for param in params})
            for i in range(1, 5)
            ],
            editable=True
        ),
])